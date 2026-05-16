import argparse
from pathlib import Path

from common.io_utils import ensure_dir, resolve_images_dir
from common.video_utils import write_images_to_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="序列图片目录（按文件名排序写入视频）")
    ap.add_argument("--out", type=str, required=True, help="输出 mp4 路径")
    ap.add_argument("--fps", type=float, default=25.0, help="输出帧率")
    ap.add_argument("--glob", type=str, default="*.jpg", help="图片匹配模式（例如 *.jpg 或 *.png）")
    args = ap.parse_args()

    images_dir = resolve_images_dir(Path(args.images_dir))
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    img_paths = sorted(images_dir.glob(args.glob))
    if not img_paths:
        raise SystemExit(f"未找到图片：{images_dir} / {args.glob}")

    wrote_frames = write_images_to_video(img_paths, out_path, fps=args.fps)
    print(f"[OK] wrote video: {out_path} ({wrote_frames} frames, fps={args.fps})")


if __name__ == "__main__":
    main()

