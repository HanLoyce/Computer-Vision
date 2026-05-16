from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_images_dir(images_dir: Path) -> Path:
    """Resolve images directory and handle one-level nested extraction layout."""
    if images_dir.exists():
        return images_dir

    parent = images_dir.parent
    nested = list(parent.glob("*/images"))
    if nested:
        return nested[0]

    raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
