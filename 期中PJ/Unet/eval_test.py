from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import build_test_dataloader, get_official_split_counts
from losses import build_criterion
from metrics import SegmentationMetric
from model import UNet


LOSS_LABELS = {
    "ce": "Cross-Entropy",
    "dice": "Dice",
    "combined": "CE + Dice",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_eval(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    metric = SegmentationMetric(num_classes=3)
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)
            metric.update(logits, targets)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, metric.miou()


def load_checkpoint(model: UNet, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def plot_comparison(results: list[dict], save_path: Path) -> None:
    labels = [item["label"] for item in results]
    values = [float(item["test_miou"]) for item in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"][: len(labels)])
    ax.set_ylabel("Test mIoU")
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.set_title("Test mIoU Comparison")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(results: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "test_summary.csv"
    json_path = output_dir / "test_summary.json"

    csv_fields = ["loss_name", "label", "test_loss", "test_miou", "checkpoint"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in csv_fields})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    device = get_device()
    root = repo_root()
    results_dir = root / "results"
    dataset_root = root.parent / "data"

    trainval_count, test_count = get_official_split_counts(dataset_root)
    test_loader = build_test_dataloader(
        dataset_root=dataset_root,
        image_size=256,
        batch_size=8,
        num_workers=0,
    )

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Official split counts: trainval={trainval_count}, test={test_count}")

    results: list[dict] = []
    for loss_name, label in LOSS_LABELS.items():
        checkpoint_path = results_dir / loss_name / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = UNet(in_channels=3, num_classes=3, base_channels=32).to(device)
        criterion = build_criterion(loss_name, num_classes=3).to(device)
        load_checkpoint(model, checkpoint_path, device)

        test_loss, test_miou = run_eval(model, test_loader, criterion, device)
        print(f"{label}: test_loss={test_loss:.4f}, test_mIoU={test_miou:.4f}")

        results.append(
            {
                "loss_name": loss_name,
                "label": label,
                "test_loss": round(test_loss, 6),
                "test_miou": round(test_miou, 6),
                "checkpoint": str(checkpoint_path),
            }
        )

    save_summary(results, results_dir)
    plot_comparison(results, results_dir / "test_miou_comparison.png")

    print("\n=== Final Summary ===")
    for item in results:
        print(f"{item['label']}: test_mIoU={float(item['test_miou']):.4f}")
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()