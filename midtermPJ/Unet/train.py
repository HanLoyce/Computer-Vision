from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

try:
	import wandb
except ImportError:
	wandb = None

from data import build_dataloaders, build_test_dataloader, get_official_split_counts
from losses import build_criterion
from metrics import SegmentationMetric
from model import UNet


LOSS_LABELS = {
	"ce": "Cross-Entropy",
	"dice": "Dice",
	"combined": "CE + Dice",
}

VAL_RATIO = 0.2


class WandbLogger:
	def __init__(self, args: argparse.Namespace):
		self.enabled = bool(getattr(args, "use_wandb", False))
		self.active = False
		self.run = None

		if not self.enabled:
			return
		if wandb is None:
			print("[W&B] wandb 未安装，已跳过记录。请先执行: pip install wandb")
			return

		self.run = wandb.init(
			project=getattr(args, "wandb_project", "Unet-OxfordPets"),
			name=getattr(args, "wandb_name", None),
			config=vars(args),
		)
		self.active = True

	def log(self, data: dict, step: int | None = None) -> None:
		if self.active:
			if step is None:
				wandb.log(data)
			else:
				wandb.log(data, step=step)

	def image(self, fig, name: str) -> None:
		if self.active:
			wandb.log({name: wandb.Image(fig)})

	def table(self, columns: list[str], rows: list[list], name: str) -> None:
		if self.active:
			tbl = wandb.Table(columns=columns, data=rows)
			wandb.log({name: tbl})

	def finish(self) -> None:
		if self.active:
			wandb.finish()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train U-Net on Oxford-IIIT Pet segmentation.")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path(__file__).resolve().parent.parent / "data",
		help="Dataset root containing oxford-iiit-pet/ or images/ and annotations/.",
	)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--image-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--base-channels", type=int, default=32)
	parser.add_argument(
		"--loss-mode",
		choices=["ce", "dice", "combined", "all"],
		default="all",
		help="Train with one loss or run all three experiments.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "results",
		help="Directory for checkpoints, plots, and summary files.",
	)
	parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
	parser.add_argument("--wandb-project", type=str, default="Unet-OxfordPets", help="W&B project name")
	parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
	return parser.parse_args()


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def run_epoch(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	criterion: nn.Module,
	device: torch.device,
	optimizer: Adam | None = None,
) -> tuple[float, float]:
	is_train = optimizer is not None
	model.train(mode=is_train)

	metric = SegmentationMetric(num_classes=3)
	running_loss = 0.0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		with torch.set_grad_enabled(is_train):
			logits = model(images)
			loss = criterion(logits, targets)

			if is_train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		running_loss += loss.item() * images.size(0)
		metric.update(logits.detach(), targets.detach())

	avg_loss = running_loss / len(loader.dataset)
	return avg_loss, metric.miou()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
	return (image * 0.5 + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def save_prediction_visual(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
	save_path: Path,
	title: str,
) -> None:
	model.eval()
	images, targets = next(iter(loader))
	num_samples = min(3, images.size(0))
	images = images[:num_samples].to(device)
	targets = targets[:num_samples].to(device)
	logits = model(images)
	preds = torch.argmax(logits, dim=1)

	fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
	if num_samples == 1:
		axes = axes[None, :]

	for row in range(num_samples):
		axes[row, 0].imshow(denormalize_image(images[row]).permute(1, 2, 0).cpu().numpy())
		axes[row, 0].set_title("Image")
		axes[row, 1].imshow(targets[row].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
		axes[row, 1].set_title("Ground Truth")
		axes[row, 2].imshow(preds[row].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
		axes[row, 2].set_title("Prediction")
		for col in range(3):
			axes[row, col].axis("off")

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def build_curves_figure(history: dict[str, list[float]], title: str):
	epochs = range(1, len(history["train_loss"]) + 1)
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	axes[0].plot(epochs, history["train_loss"], label="Train Loss")
	axes[0].plot(epochs, history["val_loss"], label="Val Loss")
	axes[0].set_title("Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].legend()

	axes[1].plot(epochs, history["train_miou"], label="Train mIoU")
	axes[1].plot(epochs, history["val_miou"], label="Val mIoU")
	axes[1].set_title("mIoU")
	axes[1].set_xlabel("Epoch")
	axes[1].legend()

	fig.suptitle(title)
	fig.tight_layout()
	return fig


def plot_curves(history: dict[str, list[float]], save_path: Path, title: str) -> None:
	fig = build_curves_figure(history, title)
	fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def log_plot_curves(
	history: dict[str, list[float]],
	title: str,
	logger: WandbLogger | None,
	image_name: str | None = None,
) -> None:
	if logger is None or not logger.active:
		return
	fig = build_curves_figure(history, title)
	logger.image(fig, image_name or title)
	plt.close(fig)



def plot_combined_loss_curves(results: list[dict[str, float | str]], output_dir: Path, logger: WandbLogger | None = None) -> None:
	"""Plot each loss function curve separately and keep a combined local summary."""
	if not results or "history" not in results[0]:
		return

	colors = ["#4C78A8", "#F58518", "#54A24B"]
	for idx, item in enumerate(results):
		if "history" not in item:
			continue
		history = item["history"]
		label = item.get("label", item.get("loss_name", f"Loss {idx}"))
		loss_name = str(item.get("loss_name", f"loss_{idx}"))
		fig = build_curves_figure(history, f"{label} Training Curves")
		per_loss_path = output_dir / f"{loss_name}_curves.png"
		fig.savefig(str(per_loss_path), dpi=150, bbox_inches="tight")
		if logger is not None and logger.active:
			logger.image(fig, f"{loss_name}_training_curves")
		plt.close(fig)

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))

	for idx, item in enumerate(results):
		if "history" not in item:
			continue
		history = item["history"]
		label = item.get("label", item.get("loss_name", f"Loss {idx}"))
		epochs = range(1, len(history.get("val_loss", [])) + 1)

		axes[0].plot(epochs, history.get("val_loss", []), label=label, marker="o", color=colors[idx % len(colors)])
		axes[1].plot(epochs, history.get("val_miou", []), label=label, marker="o", color=colors[idx % len(colors)])

	axes[0].set_title("Validation Loss - Combined")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")
	axes[0].legend()

	axes[1].set_title("Validation mIoU - Combined")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("mIoU")
	axes[1].legend()

	fig.tight_layout()
	save_path = output_dir / "combined_loss_curves.png"
	fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

	plt.close(fig)


def plot_comparison(results: list[dict[str, float | str]], save_path: Path, logger: WandbLogger | None = None) -> None:
	labels = [str(item["label"]) for item in results]
	val_values = [float(item["best_val_miou"]) for item in results]
	test_values = [float(item.get("best_test_miou", 0.0)) for item in results]

	x_positions = list(range(len(labels)))
	width = 0.35

	fig, ax = plt.subplots(figsize=(8, 4))
	val_bars = ax.bar([x - width / 2 for x in x_positions], val_values, width=width, label="Best Val mIoU", color="#4C78A8")
	test_bars = ax.bar([x + width / 2 for x in x_positions], test_values, width=width, label="Test mIoU", color="#54A24B")
	ax.set_ylabel("mIoU")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(labels)
	combined_values = val_values + test_values
	ax.set_ylim(0.0, max(combined_values) * 1.15 if combined_values else 1.0)
	ax.set_title("Val/Test mIoU Comparison")
	ax.legend()
	for bar, value in list(zip(val_bars, val_values)) + list(zip(test_bars, test_values)):
		ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")

	fig.tight_layout()
	fig.savefig(save_path, dpi=150, bbox_inches="tight")
	if logger is not None and logger.active:
		logger.image(fig, "miou_comparison")
		logger.table(
			["loss_name", "label", "best_val_miou", "best_test_miou", "best_epoch"],
			[[item["loss_name"], item["label"], item["best_val_miou"], item.get("best_test_miou", ""), item["best_epoch"]] for item in results],
			"miou_comparison_table",
		)
	plt.close(fig)


def save_summary(results: list[dict[str, float | str]], output_dir: Path, logger: WandbLogger | None = None) -> None:
	csv_path = output_dir / "summary.csv"
	json_path = output_dir / "summary.json"

	csv_fields = ["loss_name", "label", "best_val_miou", "best_test_miou", "best_epoch", "checkpoint"]
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=csv_fields)
		writer.writeheader()
		for item in results:
			row = {k: item[k] for k in csv_fields if k in item}
			writer.writerow(row)

	with json_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	if logger is not None and logger.active:
		logger.table(
			csv_fields,
			[[item.get(k, "") for k in csv_fields] for item in results],
			"summary_table",
		)


def train_single_experiment(
	args: argparse.Namespace,
	loss_name: str,
	device: torch.device,
	logger: WandbLogger | None = None,
	log_curves_to_wandb: bool = True,
) -> dict[str, float | str]:
	label = LOSS_LABELS[loss_name]
	print(f"\n=== Training with {label} ===")

	model = UNet(in_channels=3, num_classes=3, base_channels=args.base_channels).to(device)
	criterion = build_criterion(loss_name, num_classes=3).to(device)
	optimizer = Adam(model.parameters(), lr=args.lr)

	experiment_dir = args.output_dir / loss_name
	experiment_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = experiment_dir / "best.pt"

	history = {
		"train_loss": [],
		"val_loss": [],
		"train_miou": [],
		"val_miou": [],
	}
	best_val_miou = -1.0
	best_epoch = 0
	last_val_loader: torch.utils.data.DataLoader | None = None

	for epoch in range(1, args.epochs + 1):
		train_loader, val_loader = build_dataloaders(
			dataset_root=args.dataset_root,
			image_size=args.image_size,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			val_ratio=VAL_RATIO,
		)
		last_val_loader = val_loader

		train_loss, train_miou = run_epoch(model, train_loader, criterion, device, optimizer)
		val_loss, val_miou = run_epoch(model, val_loader, criterion, device)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_miou"].append(train_miou)
		history["val_miou"].append(val_miou)

		if epoch % 5 == 0:
			print(
				f"{label} | Epoch {epoch:02d}/{args.epochs} | "
				f"train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} | "
				f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
			)

		if logger is not None and logger.active:
			logger.log(
				{
					f"{loss_name}/train_loss": train_loss,
					f"{loss_name}/val_loss": val_loss,
					f"{loss_name}/train_miou": train_miou,
					f"{loss_name}/val_miou": val_miou,
					f"{loss_name}/epoch": epoch,
				},
			)

		if val_miou > best_val_miou:
			best_val_miou = val_miou
			best_epoch = epoch
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"epoch": epoch,
					"val_miou": val_miou,
					"loss_name": loss_name,
					"args": vars(args),
				},
				checkpoint_path,
			)

	plot_curves(history, experiment_dir / "curves.png", f"{label} Training Curves")
	if log_curves_to_wandb:
		log_plot_curves(history, f"{label} Training Curves", logger, image_name=f"{loss_name}_training_curves")

	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model_state_dict"])
	if last_val_loader is None:
		raise RuntimeError("Validation loader is not initialized.")
	save_prediction_visual(
		model,
		last_val_loader,
		device,
		experiment_dir / "predictions.png",
		f"{label} Predictions",
	)

	test_loader = build_test_dataloader(
		dataset_root=args.dataset_root,
		image_size=args.image_size,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	test_loss, test_miou = run_epoch(model, test_loader, criterion, device)
	print(f"{label} | Test | test_loss={test_loss:.4f} test_mIoU={test_miou:.4f}")

	history_path = experiment_dir / "history.json"
	with history_path.open("w", encoding="utf-8") as f:
		json.dump(history, f, indent=2)

	return {
		"loss_name": loss_name,
		"label": label,
		"best_val_miou": round(best_val_miou, 6),
		"best_test_miou": round(test_miou, 6),
		"best_epoch": best_epoch,
		"checkpoint": str(checkpoint_path),
		"history": history,
	}


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	device = get_device()
	logger = WandbLogger(args)

	trainval_count, test_count = get_official_split_counts(args.dataset_root)
	val_count = max(1, int(round(trainval_count * VAL_RATIO)))
	if val_count >= trainval_count:
		val_count = trainval_count - 1
	train_count = trainval_count - val_count

	experiment_losses = ["ce", "dice", "combined"] if args.loss_mode == "all" else [args.loss_mode]

	print(f"Using device: {device}")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	print(f"Official split counts: trainval={trainval_count}, test={test_count}")
	print(f"Per-epoch random split: train={train_count}, val={val_count} (from trainval, val_ratio={VAL_RATIO})")

	results: list[dict[str, float | str]] = []
	for loss_name in experiment_losses:
		results.append(
			train_single_experiment(
				args,
				loss_name,
				device,
				logger=logger,
				log_curves_to_wandb=args.loss_mode != "all",
			)
		)

	save_summary(results, args.output_dir, logger=logger)
	if len(results) > 1:
		plot_combined_loss_curves(results, args.output_dir, logger=logger)
		plot_comparison(results, args.output_dir / "miou_comparison.png", logger=logger)

	if logger is not None:
		logger.finish()

	print("\n=== Final Summary ===")
	for item in results:
		print(
			f"{item['label']}: best_val_mIoU={float(item['best_val_miou']):.4f}, "
			f"test_mIoU={float(item.get('best_test_miou', 0.0)):.4f} "
			f"at epoch {item['best_epoch']}"
		)
	print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
	main()
