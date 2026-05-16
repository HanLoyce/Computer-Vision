import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


class OxfordPetSegDataset(Dataset):
    """Oxford-IIIT Pet trimap segmentation dataset (3 classes)."""

    def __init__(
        self,
        image_root: Path,
        mask_root: Path,
        id_list: List[str],
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.image_root = image_root
        self.mask_root = mask_root
        self.id_list = id_list
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.id_list[idx]
        image_path = self.image_root / f"{sample_id}.jpg"
        mask_path = self.mask_root / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - 0.5) / 0.5
        image_np = np.transpose(image_np, (2, 0, 1))

        mask_np = np.asarray(mask, dtype=np.int64)
        # Original trimap label set is {1, 2, 3}; map to {0, 1, 2}
        mask_np = np.clip(mask_np - 1, 0, 2)

        image_t = torch.from_numpy(image_np)
        mask_t = torch.from_numpy(mask_np)
        return image_t, mask_t


def _read_ids(split_file: Path) -> List[str]:
    ids: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: image_id class_id species_id breed_id
            sample_id = line.split()[0]
            if sample_id.startswith("._"):
                continue
            ids.append(sample_id)
    return ids


def _split_train_val_ids(
    trainval_ids: List[str],
    val_ratio: float,
) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    ids = list(trainval_ids)
    if len(ids) < 2:
        raise ValueError("trainval split requires at least 2 samples.")

    random.shuffle(ids)

    val_count = max(1, int(round(len(ids) * val_ratio)))
    if val_count >= len(ids):
        val_count = len(ids) - 1

    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    return train_ids, val_ids


def _build_dataset_from_ids(
    image_root: Path,
    mask_root: Path,
    id_list: List[str],
    image_size: int,
    augment: bool,
) -> OxfordPetSegDataset:
    return OxfordPetSegDataset(
        image_root=image_root,
        mask_root=mask_root,
        id_list=id_list,
        image_size=image_size,
        augment=augment,
    )


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    print(f"Downloading {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path)


def _extract_tar_gz(archive_path: Path, dataset_root: Path) -> None:
    print(f"Extracting {archive_path} -> {dataset_root}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dataset_root)


def _has_complete_dataset(dataset_dir: Path) -> bool:
    images_dir = dataset_dir / "images"
    trimaps_dir = dataset_dir / "annotations" / "trimaps"
    train_split = dataset_dir / "annotations" / "trainval.txt"
    test_split = dataset_dir / "annotations" / "test.txt"

    has_images = images_dir.exists() and any(images_dir.glob("*.jpg"))
    has_trimaps = trimaps_dir.exists() and any(trimaps_dir.glob("*.png"))
    has_splits = train_split.exists() and test_split.exists()
    return has_images and has_trimaps and has_splits


def _resolve_dataset_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "oxford-iiit-pet",
        dataset_root / "OxfordPetSegDataset",
    ]
    for candidate in candidates:
        if _has_complete_dataset(candidate):
            return candidate
    return dataset_root / "oxford-iiit-pet"


def _ensure_dataset_ready(dataset_root: Path) -> Path:
    dataset_dir = _resolve_dataset_dir(dataset_root)
    if _has_complete_dataset(dataset_dir):
        return dataset_dir

    images_dir = dataset_dir / "images"
    trimaps_dir = dataset_dir / "annotations" / "trimaps"
    train_split = dataset_dir / "annotations" / "trainval.txt"
    test_split = dataset_dir / "annotations" / "test.txt"

    has_images = images_dir.exists() and any(images_dir.glob("*.jpg"))
    has_trimaps = trimaps_dir.exists() and any(trimaps_dir.glob("*.png"))
    has_splits = train_split.exists() and test_split.exists()

    if has_images and has_trimaps and has_splits:
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_archive = dataset_dir / "images.tar.gz"
    annotations_archive = dataset_dir / "annotations.tar.gz"

    if not has_images:
        _download_file(IMAGES_URL, images_archive)
        _extract_tar_gz(images_archive, dataset_dir)

    if not (has_trimaps and has_splits):
        _download_file(ANNOTATIONS_URL, annotations_archive)
        _extract_tar_gz(annotations_archive, dataset_dir)

    has_images = images_dir.exists() and any(images_dir.glob("*.jpg"))
    has_trimaps = trimaps_dir.exists() and any(trimaps_dir.glob("*.png"))
    has_splits = train_split.exists() and test_split.exists()
    if not (has_images and has_trimaps and has_splits):
        raise FileNotFoundError(
            "Oxford-IIIT Pet dataset is still incomplete after download. "
            f"Expected images in {images_dir} and annotations in {trimaps_dir}."
        )
    return dataset_dir


def build_dataloaders(
    dataset_root: Path,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    dataset_dir = _ensure_dataset_ready(dataset_root)

    annotations_root = dataset_dir / "annotations"
    image_root = dataset_dir / "images"
    mask_root = annotations_root / "trimaps"

    trainval_ids = _read_ids(annotations_root / "trainval.txt")
    train_ids, val_ids = _split_train_val_ids(trainval_ids, val_ratio)

    # Keep official test split fully independent from train/val pipeline.
    test_ids = set(_read_ids(annotations_root / "test.txt"))
    if test_ids.intersection(train_ids) or test_ids.intersection(val_ids):
        raise RuntimeError("Data leakage detected: test split overlaps with train/val splits.")

    train_set = OxfordPetSegDataset(
        image_root=image_root,
        mask_root=mask_root,
        id_list=train_ids,
        image_size=image_size,
        augment=True,
    )
    val_set = OxfordPetSegDataset(
        image_root=image_root,
        mask_root=mask_root,
        id_list=val_ids,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_test_dataloader(
    dataset_root: Path,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    dataset_dir = _ensure_dataset_ready(dataset_root)

    annotations_root = dataset_dir / "annotations"
    image_root = dataset_dir / "images"
    mask_root = annotations_root / "trimaps"

    test_ids = _read_ids(annotations_root / "test.txt")
    test_set = _build_dataset_from_ids(
        image_root=image_root,
        mask_root=mask_root,
        id_list=test_ids,
        image_size=image_size,
        augment=False,
    )

    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_official_split_counts(dataset_root: Path) -> Tuple[int, int]:
    dataset_dir = _ensure_dataset_ready(dataset_root)
    annotations_root = dataset_dir / "annotations"
    trainval_count = len(_read_ids(annotations_root / "trainval.txt"))
    test_count = len(_read_ids(annotations_root / "test.txt"))
    return trainval_count, test_count
