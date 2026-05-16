from __future__ import annotations

from pathlib import Path

import cv2


def read_frame_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def create_mp4_writer(output_path: Path, fps: float, size_wh: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size_wh)
    return writer


def write_images_to_video(
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    required_frames: int | None = None,
) -> int:
    if not image_paths:
        raise ValueError("image_paths is empty")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise FileNotFoundError(f"Failed to read image: {image_paths[0]}")

    height, width = first.shape[:2]
    writer = create_mp4_writer(output_path, fps=fps, size_wh=(width, height))

    frame_count = 0
    index = 0
    max_frames = required_frames if required_frames is not None else len(image_paths)

    while frame_count < max_frames:
        path = image_paths[index]
        image = cv2.imread(str(path))
        if image is not None:
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(image)
            frame_count += 1

        index += 1
        if index >= len(image_paths):
            if required_frames is None:
                break
            index = 0

    writer.release()
    return frame_count
