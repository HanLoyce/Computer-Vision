import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common.video_utils import create_mp4_writer


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def next_test_name(project_dir: Path) -> str:
    max_index = 0
    for p in project_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if not name.startswith("test"):
            continue
        suffix = name[4:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return f"test{max_index + 1}"


def select_best_from_train_dir(train_dir: Path) -> Path | None:
    if not train_dir.exists() or not train_dir.is_dir():
        return None

    direct_best = train_dir / "best.pt"
    if direct_best.exists():
        return direct_best

    direct_weights_best = train_dir / "weights" / "best.pt"
    if direct_weights_best.exists():
        return direct_weights_best

    best_candidates: list[tuple[int, Path]] = []
    for child in train_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if not name.startswith("train-"):
            continue
        suffix = name.split("train-", maxsplit=1)[1]
        if not suffix.isdigit():
            continue
        candidate = child / "weights" / "best.pt"
        if candidate.exists():
            best_candidates.append((int(suffix), candidate))

    if best_candidates:
        best_candidates.sort(key=lambda item: item[0])
        return best_candidates[-1][1]

    return None


def resolve_model_path(model_arg: str) -> Path:
    requested = Path(model_arg)
    if not requested.is_absolute():
        requested = repo_root() / requested
    if requested.exists():
        if requested.is_dir():
            selected = select_best_from_train_dir(requested)
            if selected is not None:
                return selected
            raise SystemExit(f"模型目录中没有可用权重: {requested}")
        return requested

    if requested.name.lower() == "best.pt" and requested.parent.exists():
        selected = select_best_from_train_dir(requested.parent)
        if selected is not None:
            requested.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, requested)
            print(f"[OK] 已从 {selected} 复制模型到 {requested}")
            return requested

    search_roots = [repo_root() / "runs" / "detect", repo_root() / "runs"]
    candidates: list[Path] = []
    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in search_root.glob("**/best.pt"):
            candidate = candidate.resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            candidates.append(candidate)

    if not candidates:
        raise SystemExit(f"模型不存在: {requested}")

    def model_score(path: Path) -> tuple[int, int, int]:
        parts = [part.lower() for part in path.parts]
        return (
            0 if "train_model" in parts else 1,
            0 if path.parent.name.lower() == "weights" else 1,
            len(path.parts),
        )

    source = sorted(candidates, key=model_score)[0]
    requested.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, requested)
    print(f"[OK] 已从 {source} 复制模型到 {requested}")
    return requested


def parse_line_arg(line_arg: str, frame_w: int, frame_h: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if not line_arg:
        return (0, frame_h // 2), (frame_w - 1, frame_h // 2)

    parts = [part.strip() for part in line_arg.split(",")]
    if len(parts) != 4:
        raise SystemExit("--count_line 格式应为 x1,y1,x2,y2")

    try:
        x1, y1, x2, y2 = (int(float(value)) for value in parts)
    except ValueError as exc:
        raise SystemExit(f"--count_line 参数解析失败: {line_arg}") from exc

    return (x1, y1), (x2, y2)


def line_side(point: tuple[float, float], start: tuple[int, int], end: tuple[int, int]) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 2026)
    bgr = rng.integers(64, 256, size=3).tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="train_model", help="检测模型路径或训练目录，如 train_model、train_model/train-1、train_model/train-1/weights/best.pt")
    ap.add_argument("--source", type=str, default="input", help="测试输入视频或目录")
    ap.add_argument("--tracker", type=str, default="trackers/botsort_reid.yaml", help="tracker 配置文件")
    ap.add_argument("--project", type=str, default="output", help="输出根目录")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    ap.add_argument("--device", type=str, default="0", help="设备，如 0 或 cpu")
    ap.add_argument("--imgsz", type=int, default=960, help="推理尺寸")
    ap.add_argument(
        "--count_line",
        type=str,
        default="",
        help="越线计数线，格式为 x1,y1,x2,y2；留空时默认使用画面中线",
    )
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = repo_root() / source_path
    if not source_path.exists():
        raise SystemExit(f"测试输入不存在: {source_path}")

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = repo_root() / project_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = next_test_name(project_dir)

    source_video = source_path.is_file() and source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if not source_video:
        raise SystemExit("当前越线计数版本仅支持单个视频文件作为 --source")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {source_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_dir = project_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{source_path.stem}_count.mp4"
    writer = create_mp4_writer(output_video, fps=fps, size_wh=(frame_w, frame_h))
    line_start, line_end = parse_line_arg(args.count_line, frame_w, frame_h)

    model = YOLO(str(model_path))
    results = model.track(
        source=str(source_path),
        tracker=args.tracker,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        persist=True,
        stream=True,
        verbose=False,
    )

    prev_side_by_id: dict[int, float] = {}
    cross_count = 0
    name_map = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    for r in results:
        frame = r.orig_img.copy()
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2, cv2.LINE_AA)

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            clss = r.boxes.cls.int().cpu().tolist()
            ids = (
                r.boxes.id.int().cpu().tolist()
                if r.boxes.id is not None
                else [-1] * len(clss)
            )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                track_id = int(ids[i])
                class_id = int(clss[i])
                color = class_color(class_id)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if track_id >= 0:
                    current_side = line_side(center, line_start, line_end)
                    prev_side = prev_side_by_id.get(track_id)
                    if prev_side is not None and prev_side != 0 and current_side != 0:
                        if prev_side * current_side < 0:
                            cross_count += 1
                    if current_side != 0:
                        prev_side_by_id[track_id] = current_side

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                cls_name = name_map.get(class_id, str(class_id))
                label = f"#{track_id}:{cls_name}" if track_id >= 0 else cls_name
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.rectangle(frame, (8, 8), (180, 42), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"cross count: {cross_count}",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"[OK] 跟踪结果目录: {output_dir}")
    print(f"[OK] 越线计数视频: {output_video}")
    print(f"[OK] line crossing count: {cross_count}")


if __name__ == "__main__":
    main()
