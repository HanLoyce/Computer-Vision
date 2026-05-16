import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from common.video_utils import read_frame_size
from common.visdrone import DET_CLASS_NAMES, find_det_split_dirs, find_image_path


def _visdrone_line_to_yolo(line: str, img_w: int, img_h: int) -> tuple[int, float, float, float, float] | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 8:
        return None

    x, y, w, h = map(float, parts[0:4])
    score = float(parts[4])
    category = int(float(parts[5]))

    # VisDrone DET: ignored regions/category==0 或 w/h<=0 直接跳过
    if category <= 0 or w <= 1 or h <= 1:
        return None

    # 训练集里 score 通常为 1；如果存在 0 也可跳过
    if score <= 0:
        return None

    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h

    # YOLO 类别从 0 开始
    cls = category - 1
    return cls, xc, yc, ww, hh


def convert_split(raw_root: Path, out_root: Path, split: str) -> None:
    img_dir, ann_dir = find_det_split_dirs(raw_root, split)

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        raise FileNotFoundError(f"annotations 为空：{ann_dir}")

    for ann_path in tqdm(ann_files, desc=f"convert {split}"):
        stem = ann_path.stem
        img_path = find_image_path(img_dir, stem)
        if img_path is None:
            continue

        img_w, img_h = read_frame_size(img_path)

        yolo_lines: list[str] = []
        with ann_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                converted = _visdrone_line_to_yolo(line, img_w, img_h)
                if converted is None:
                    continue
                cls, xc, yc, ww, hh = converted
                yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        # labels：即使没有 bbox，也要写空文件（YOLO 训练需要）
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        # images：用硬链接/复制都行。Windows 硬链接权限不稳定，这里复制更稳（但占空间）
        out_img_path = out_img_dir / img_path.name
        if not out_img_path.exists():
            out_img_path.write_bytes(img_path.read_bytes())


def write_yaml(out_root: Path, yaml_path: Path) -> None:
    cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(DET_CLASS_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="data/raw", help="解压后的 VisDrone 原始根目录")
    ap.add_argument("--out_root", type=str, default="data/visdrone_yolo", help="输出 YOLO 格式根目录")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 默认做 train/val（test 没标注通常不需要转）
    convert_split(raw_root, out_root, "train")
    convert_split(raw_root, out_root, "val")

    yaml_path = Path("visdrone.yaml").resolve()
    write_yaml(out_root, yaml_path)

    print(f"[OK] 转换完成：{out_root}")
    print(f"[OK] 已生成：{yaml_path}")
    print("下一步训练示例：yolo detect train data=visdrone.yaml model=yolov8n.pt epochs=20 imgsz=640 batch=16")


if __name__ == "__main__":
    main()

