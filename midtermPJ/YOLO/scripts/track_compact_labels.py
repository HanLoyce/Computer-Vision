import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common.video_utils import create_mp4_writer
from common.visdrone import SHORT_CLASS_NAMES



def id_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 2026)
    bgr = rng.integers(64, 256, size=3).tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def choose_label_rect(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text_w: int,
    text_h: int,
    frame_w: int,
    frame_h: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    pad = 2
    candidates = [
        (x1, y1 - text_h - 2 * pad - 2),  # box 上方
        (x1, y1 + 2),  # box 内顶部
        (x1, y2 + 2),  # box 下方
    ]

    for lx, ly in candidates:
        lx = max(0, min(lx, frame_w - text_w - 2 * pad - 1))
        ly = max(0, min(ly, frame_h - text_h - 2 * pad - 1))
        rect = (lx, ly, lx + text_w + 2 * pad, ly + text_h + 2 * pad)
        if all(not rect_intersects(rect, r) for r in occupied):
            return rect

    # 都冲突时，退化为上方位置（至少保证可见）
    lx = max(0, min(x1, frame_w - text_w - 2 * pad - 1))
    ly = max(0, min(y1 - text_h - 2 * pad - 2, frame_h - text_h - 2 * pad - 1))
    return lx, ly, lx + text_w + 2 * pad, ly + text_h + 2 * pad


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="检测模型权重或目录，如 train_model、train_model/train-1、train_model/train-1/weights/best.pt")
    ap.add_argument("--source", type=str, required=True, help="输入视频路径")
    ap.add_argument("--tracker", type=str, default="trackers/botsort_reid.yaml", help="tracker yaml 路径")
    ap.add_argument("--out", type=str, default="output/compact_track.mp4", help="输出视频路径")
    ap.add_argument("--conf", type=float, default=0.30, help="检测置信度阈值")
    ap.add_argument("--device", type=str, default="0", help="设备，如 0/cpu")
    ap.add_argument("--line_width", type=int, default=1, help="框线粗细")
    ap.add_argument(
        "--output_mode",
        type=str,
        default="id",
        choices=["id_cls", "id", "cls"],
        help="输出模式：id_cls(输出ID+类别)、id(只输出ID)、cls(只输出类别)",
    )
    ap.add_argument(
        "--min_label_area",
        type=int,
        default=900,
        help="仅当框面积>=该值时显示标签（减少密集场景遮挡）",
    )
    ap.add_argument("--font_scale", type=float, default=0.36, help="标签字体大小")
    ap.add_argument(
        "--count_line",
        type=str,
        default="",
        help="越线计数线，格式为 x1,y1,x2,y2；留空时默认使用画面中线",
    )
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"source 不存在: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = Path(args.out)
    writer = create_mp4_writer(out_path, fps=fps, size_wh=(w, h))
    line_start, line_end = parse_line_arg(args.count_line, w, h)

    model = YOLO(args.model)
    name_map = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    prev_side_by_id: dict[int, float] = {}
    cross_count = 0

    results = model.track(
        source=str(src),
        tracker=args.tracker,
        conf=args.conf,
        device=args.device,
        persist=True,
        stream=True,
        verbose=False,
    )

    for r in results:
        frame = r.orig_img.copy()
        occupied: list[tuple[int, int, int, int]] = []

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
                cls_name = name_map.get(clss[i], str(clss[i]))
                short_cls = SHORT_CLASS_NAMES.get(cls_name, cls_name[:4])
                track_id = int(ids[i])
                color = id_color(track_id if track_id >= 0 else i)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if track_id >= 0:
                    current_side = line_side(center, line_start, line_end)
                    prev_side = prev_side_by_id.get(track_id)
                    if prev_side is not None and prev_side != 0 and current_side != 0:
                        if prev_side * current_side < 0:
                            cross_count += 1
                    if current_side != 0:
                        prev_side_by_id[track_id] = current_side

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.line_width, cv2.LINE_AA)

                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                if box_area < args.min_label_area:
                    continue

                if args.output_mode == "id":
                    label = f"#{track_id}" if track_id >= 0 else "#?"
                elif args.output_mode == "id_cls":
                    label = f"#{track_id}:{short_cls}" if track_id >= 0 else short_cls
                else:  # cls
                    label = short_cls

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 1)
                rx1, ry1, rx2, ry2 = choose_label_rect(
                    x1, y1, x2, y2, tw, th, w, h, occupied
                )
                occupied.append((rx1, ry1, rx2, ry2))

                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    label,
                    (rx1 + 2, ry2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    args.font_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        count_text = f"cross count: {cross_count}"
        cv2.rectangle(frame, (8, 8), (170, 42), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            count_text,
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"[OK] compact result saved to: {out_path}")
    print(f"[OK] line crossing count: {cross_count}")


if __name__ == "__main__":
    main()

