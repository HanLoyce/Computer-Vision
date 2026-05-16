import argparse
from collections import defaultdict
from pathlib import Path

from common.io_utils import resolve_images_dir
from common.video_utils import write_images_to_video
from common.visdrone import parse_det_image_name


def choose_sequence(img_paths: list[Path], need_frames: int, prefer_seq: str | None) -> tuple[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in img_paths:
        parsed = parse_det_image_name(p)
        if parsed is None:
            continue
        seq, _ = parsed
        groups[seq].append(p)

    if not groups:
        raise SystemExit("未能从文件名解析出序列信息（seq_id, frame_id）。")

    def sort_key(p: Path) -> int:
        parsed = parse_det_image_name(p)
        return parsed[1] if parsed else 0

    if prefer_seq is not None:
        if prefer_seq not in groups:
            raise SystemExit(f"指定的 --seq_id={prefer_seq} 不存在。可用序列数：{len(groups)}")
        seq = prefer_seq
        frames = sorted(groups[seq], key=sort_key)
        return seq, frames

    # 选帧数足够的序列；否则选帧最多的
    best_seq = None
    best_frames: list[Path] = []
    for seq, frames in groups.items():
        frames_sorted = sorted(frames, key=sort_key)
        if len(frames_sorted) >= need_frames:
            return seq, frames_sorted
        if len(frames_sorted) > len(best_frames):
            best_seq = seq
            best_frames = frames_sorted

    assert best_seq is not None
    return best_seq, best_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images_dir",
        type=str,
        default="data/raw/VisDrone2019-DET-train/images",
        help="VisDrone DET 图片目录（建议 train/images）",
    )
    ap.add_argument("--out", type=str, default="output/visdrone_20s.mp4", help="输出 mp4 路径")
    ap.add_argument("--seconds", type=float, default=20.0, help="视频时长（建议 10~30 秒）")
    ap.add_argument("--fps", type=float, default=25.0, help="输出帧率")
    ap.add_argument("--glob", type=str, default="*.jpg", help="图片匹配模式")
    ap.add_argument("--seq_id", type=str, default=None, help="指定序列 id（例如 0000002），不指定则自动选择")
    args = ap.parse_args()

    try:
        images_dir = resolve_images_dir(Path(args.images_dir))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    img_paths = sorted(images_dir.glob(args.glob))
    if not img_paths:
        raise SystemExit(f"未找到图片：{images_dir} / {args.glob}")

    need_frames = int(round(args.seconds * args.fps))
    if need_frames <= 0:
        raise SystemExit("--seconds 和 --fps 需为正数")

    seq, frames = choose_sequence(img_paths, need_frames=need_frames, prefer_seq=args.seq_id)

    out_path = Path(args.out)
    write_images_to_video(frames, out_path, fps=args.fps, required_frames=need_frames)

    print(f"[OK] 选用序列 seq_id={seq}，总帧数可用={len(frames)}，输出={out_path}")
    print("下一步跟踪示例：python scripts/run_test_tracking.py --model train_model --source input")


if __name__ == "__main__":
    main()

