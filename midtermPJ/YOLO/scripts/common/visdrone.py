from __future__ import annotations

from pathlib import Path


DET_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


SHORT_CLASS_NAMES = {
    "pedestrian": "ped",
    "people": "ppl",
    "bicycle": "bike",
    "car": "car",
    "van": "van",
    "truck": "trk",
    "tricycle": "tri",
    "awning-tricycle": "atri",
    "bus": "bus",
    "motor": "moto",
}


def parse_det_image_name(image_path: Path) -> tuple[str, int] | None:
    """Parse VisDrone DET image name to (sequence_id, frame_id)."""
    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        frame_id = int(parts[1])
    except ValueError:
        return None
    return parts[0], frame_id


def find_det_split_dirs(raw_root: Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, annotations_dir) for a split in a robust way."""
    split_lower = split.lower()
    candidates = [
        raw_root / f"VisDrone2019-DET-{split_lower}",
        raw_root / f"VisDrone2019-DET-{split_lower.capitalize()}",
        raw_root / f"VisDrone2019-DET-{split_lower.upper()}",
    ]

    nested = list(raw_root.rglob(f"VisDrone2019-DET-{split_lower}"))
    candidates.extend(nested[:10])

    for root in candidates:
        if not root.exists():
            continue
        images_dir = root / "images"
        ann_dir = root / "annotations"
        if images_dir.exists() and ann_dir.exists():
            return images_dir, ann_dir

    raise FileNotFoundError(
        f"Cannot find images/annotations for VisDrone split '{split_lower}'. "
        f"Please check extracted folders under: {raw_root}"
    )


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    """Find matching image file by common extensions."""
    for ext in (".jpg", ".png", ".jpeg"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None
