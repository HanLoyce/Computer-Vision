import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet

try:
    import wandb
except ImportError:
    wandb = None
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


NUM_CLASSES = 37
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "result" / "experiment_results.json"
DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data"

TRAINING_CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "val_ratio": 0.2,
    "weight_decay": 5e-4,
    "momentum": 0.9,
}

EXPERIMENT_CONFIG = {
    "backbone": "resnet18",
    "attention_model": "se_resnet18",
    "baseline": {
        "epochs": 10,
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
    },
    "hyperparam": {
        "epochs": [6, 12, 20],
        "head_lrs": [5e-3, 1e-2, 2e-2],
        "backbone_ratio": 0.1,
    },
    "ablation": {
        "epochs": 10,
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
    },
    "attention": {
        "epochs": 10,
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
    },
}


class WandbLogger:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "use_wandb", False) or getattr(args, "use_swanlab", False))
        self.active = False
        self.run = None

        if not self.enabled:
            return
        if wandb is None:
            print("[wandb] wandb 未安装，已跳过可视化记录。请先执行: pip install wandb")
            return

        try:
            run_name = f"imagenet-exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = {
                "project": "ImageNet-OxfordPets",
                "name": run_name,
                "config": vars(args),
                "reinit": True,
            }

            self.run = wandb.init(**init_kwargs)
            self.active = True
            print(f"[wandb] 已启用，run={run_name}")
        except Exception as exc:
            self.active = False
            print(f"[wandb] 初始化失败，已降级为仅本地训练。原因: {exc}")

    def log(self, data: Dict, step: Optional[int] = None) -> None:
        if not self.active:
            return
        try:
            if step is None:
                wandb.log(data)
            else:
                wandb.log(data, step=step)
        except Exception as exc:
            self.active = False
            print(f"[wandb] 日志写入失败，后续已停止上报。原因: {exc}")

    def log_figure(self, fig, name: Optional[str] = None) -> None:
        if not self.active:
            return
        try:
            wandb.log({name or "figure": wandb.Image(fig)})
        except Exception as exc:
            print(f"[wandb] log_figure 失败: {exc}")

    def log_series(self, prefix: str, series: Dict[str, List[float]]) -> None:
        if not self.active:
            return
        try:
            payload = {f"{prefix}/{k}": list(v) for k, v in series.items()}
            wandb.log(payload)
        except Exception as exc:
            print(f"[wandb] log_series 失败: {exc}")

    def finish(self) -> None:
        if not self.active:
            return
        try:
            if self.run is not None:
                wandb.finish()
        except Exception:
            pass
        self.active = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class OxfordPetsDataModule:
    def __init__(
        self,
        root: str = "./data",
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.root = root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

    @staticmethod
    def _is_valid_pet_dataset_dir(path: Path) -> bool:
        ann_dir = path / "annotations"
        img_dir = path / "images"
        if not ann_dir.is_dir() or not img_dir.is_dir():
            return False
        required = [ann_dir / "trainval.txt", ann_dir / "test.txt"]
        if not all(p.exists() for p in required):
            return False
        has_images = any(img_dir.glob("*.jpg"))
        return has_images

    def _resolve_dataset_parent(self) -> Tuple[Path, bool]:
        root = Path(self.root).expanduser().resolve()

        candidate_dirs = [
            root / "oxford-iiit-pet",
            root,
            SCRIPT_DIR.parent / "data" / "oxford-iiit-pet",
            SCRIPT_DIR.parent / "OxfordPetSegDataset" / "oxford-iiit-pet",
        ]

        for dataset_dir in candidate_dirs:
            if dataset_dir.name == "oxford-iiit-pet" and self._is_valid_pet_dataset_dir(dataset_dir):
                return dataset_dir.parent, False

        # Fall back to torchvision default layout under the configured root.
        return root, True

    def _build_transforms(self):
        train_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(self.img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return train_tf, eval_tf

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Build raw datasets without transform so that train/val can have different transforms.
        dataset_parent, need_download = self._resolve_dataset_parent()
        if need_download:
            print(f"[Data] 未检测到完整本地数据，将下载到: {(dataset_parent / 'oxford-iiit-pet').resolve()}")
        else:
            print(f"[Data] 使用本地数据: {(dataset_parent / 'oxford-iiit-pet').resolve()}")

        trainval_raw = OxfordIIITPet(
            root=str(dataset_parent),
            split="trainval",
            target_types="category",
            download=need_download,
            transform=None,
        )
        test_raw = OxfordIIITPet(
            root=str(dataset_parent),
            split="test",
            target_types="category",
            download=need_download,
            transform=None,
        )

        val_len = int(len(trainval_raw) * self.val_ratio)
        train_len = len(trainval_raw) - val_len
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(trainval_raw, [train_len, val_len], generator=generator)

        train_tf, eval_tf = self._build_transforms()
        train_ds = TransformSubset(train_subset, train_tf)
        val_ds = TransformSubset(val_subset, eval_tf)
        test_ds = TransformSubset(test_raw, eval_tf)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = self.pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        return x * weight


class SEResNet(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            stage_channels = [64, 128, 256, 512]
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            stage_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone for SEResNet: {backbone_name}")

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.se1 = SEBlock(stage_channels[0])
        self.se2 = SEBlock(stage_channels[1])
        self.se3 = SEBlock(stage_channels[2])
        self.se4 = SEBlock(stage_channels[3])
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    if model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    if model_name in {"se_resnet18", "se_resnet34"}:
        backbone = model_name.replace("se_", "")
        return SEResNet(backbone_name=backbone, pretrained=pretrained, num_classes=NUM_CLASSES)

    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(
    model: nn.Module,
    pretrained: bool,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    momentum: float,
) -> optim.Optimizer:
    if not pretrained:
        return optim.SGD(model.parameters(), lr=head_lr, momentum=momentum, weight_decay=weight_decay)

    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.SGD(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        momentum=momentum,
        weight_decay=weight_decay,
    )


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def train_one_experiment(
    task_name: str,
    model_name: str,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    momentum: float,
    device: torch.device,
    wandb_logger: Optional[WandbLogger] = None,
) -> Dict:
    model = build_model(model_name=model_name, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        pretrained=pretrained,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    
    # history for plotting
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "task": task_name,
                    "model": model_name,
                    "pretrained": int(pretrained),
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                },
                step=epoch,
            )

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "model": model_name,
        "pretrained": pretrained,
        "epochs": epochs,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "best_val_acc": round(best_val_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "best_state": best_state,
        "history": history,
    }


def _ensure_result_dir(output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.parent


def plot_learning_curve(history: Dict[str, List[float]], out_file: Path, title: str = "Learning Curve", wandb_logger: Optional[WandbLogger] = None) -> None:
    epochs = list(range(1, len(history.get("train_acc", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_acc", []), label="train_acc", marker="o")
    axes[0].plot(epochs, history.get("val_acc", []), label="val_acc", marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_loss", []), label="train_loss", marker="o")
    axes[1].plot(epochs, history.get("val_loss", []), label="val_loss", marker="o")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(str(out_file), dpi=150)
    # upload the rendered figure before closing it
    if wandb_logger is not None and getattr(wandb_logger, "active", False):
        try:
            wandb_logger.log_figure(fig, name=title)
            wandb_logger.log_series(title, {"train_acc": history.get("train_acc", []), "val_acc": history.get("val_acc", []), "train_loss": history.get("train_loss", []), "val_loss": history.get("val_loss", [])})
        except Exception as exc:
            print(f"[wandb] baseline plot log failed: {exc}")
    plt.close(fig)


def save_hyperparam_table(results: List[Dict], out_dir: Path, wandb_logger: Optional[WandbLogger] = None) -> None:
    # filter only hyperparam task rows
    df = pd.DataFrame(results)
    hp_df = df[df["task"] == "hyperparam"].copy()
    if hp_df.empty:
        return
    cols = ["epochs", "backbone_lr", "head_lr", "best_val_acc", "test_acc"]
    hp_df = hp_df[cols]
    csv_path = out_dir / "hyperparam_results.csv"
    hp_df.to_csv(csv_path, index=False)

    # also save a simple table image
    fig, ax = plt.subplots(figsize=(8, max(2, len(hp_df) * 0.3 + 1)))
    ax.axis("off")
    table = ax.table(cellText=np.round(hp_df.values, 6).astype(str), colLabels=hp_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(str(out_dir / "hyperparam_results_table.png"), dpi=150)
    if wandb_logger is not None and getattr(wandb_logger, "active", False):
        try:
            wandb_logger.log_figure(fig, name="hyperparam_results_table")
            wandb_logger.log({"hyperparam/table": wandb.Table(dataframe=hp_df)})
        except Exception as exc:
            print(f"[wandb] hyperparam table log failed: {exc}")
    plt.close(fig)


def plot_ablation_curves_and_bar(pretrained_res: Dict, scratch_res: Dict, out_dir: Path, wandb_logger: Optional[WandbLogger] = None) -> None:
    h1 = pretrained_res.get("history", {})
    h2 = scratch_res.get("history", {})
    # plot val_acc curves
    fig, ax = plt.subplots(figsize=(6, 4))
    if h1 and "val_acc" in h1:
        ax.plot(range(1, len(h1["val_acc"]) + 1), h1["val_acc"], label="pretrained", marker="o")
    if h2 and "val_acc" in h2:
        ax.plot(range(1, len(h2["val_acc"]) + 1), h2["val_acc"], label="scratch", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Pretraining Ablation: Val Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / "ablation_val_acc_curves.png"), dpi=150)
    if wandb_logger is not None and getattr(wandb_logger, "active", False):
        try:
            wandb_logger.log_figure(fig, name="ablation_val_acc_curves")
            wandb_logger.log_series("ablation_val_acc", {"pretrained_val_acc": h1.get("val_acc", []), "scratch_val_acc": h2.get("val_acc", [])})
        except Exception as exc:
            print(f"[wandb] ablation curves log failed: {exc}")
    plt.close(fig)

    # bar chart for test acc
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["pretrained", "scratch"]
    vals = [pretrained_res.get("test_acc", 0.0), scratch_res.get("test_acc", 0.0)]
    ax.bar(labels, vals, color=["tab:blue", "tab:orange"]) 
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Ablation: Test Accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(str(out_dir / "ablation_test_acc_bar.png"), dpi=150)
    if wandb_logger is not None and getattr(wandb_logger, "active", False):
        try:
            wandb_logger.log_figure(fig, name="ablation_test_acc_bar")
            wandb_logger.log_series("ablation_test_acc", {"pretrained_test_acc": [pretrained_res.get("test_acc", 0.0)], "scratch_test_acc": [scratch_res.get("test_acc", 0.0)]})
        except Exception as exc:
            print(f"[wandb] ablation bar log failed: {exc}")
    plt.close(fig)


def plot_compare_with_baseline(target_res: Dict, baseline_res: Dict, out_dir: Path, name: str = "attention", wandb_logger: Optional[WandbLogger] = None) -> None:
    h_t = target_res.get("history", {})
    h_b = baseline_res.get("history", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    if h_b and "val_acc" in h_b:
        ax.plot(range(1, len(h_b["val_acc"]) + 1), h_b["val_acc"], label="baseline", marker="o")
    if h_t and "val_acc" in h_t:
        ax.plot(range(1, len(h_t["val_acc"]) + 1), h_t["val_acc"], label=name, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title(f"{name.capitalize()} vs Baseline: Val Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / f"{name}_vs_baseline_val_acc.png"), dpi=150)
    if wandb_logger is not None and getattr(wandb_logger, "active", False):
        try:
            wandb_logger.log_figure(fig, name=f"{name}_vs_baseline")
            wandb_logger.log_series(f"{name}_vs_baseline", {"baseline_val_acc": h_b.get("val_acc", []), f"{name}_val_acc": h_t.get("val_acc", [])})
        except Exception as exc:
            print(f"[wandb] compare plot log failed: {exc}")
    plt.close(fig)


def run_all_experiments(args) -> List[Dict]:
    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = OxfordPetsDataModule(
        root=args.data_root,
        img_size=TRAINING_CONFIG["img_size"],
        batch_size=TRAINING_CONFIG["batch_size"],
        num_workers=TRAINING_CONFIG["num_workers"],
        val_ratio=TRAINING_CONFIG["val_ratio"],
        seed=DEFAULT_SEED,
    )
    train_loader, val_loader, test_loader = data.get_loaders()
    wandb_logger = WandbLogger(args)

    # ensure result directory
    result_dir = _ensure_result_dir(args.output)

    results: List[Dict] = []
    hp_results: List[Dict] = []
    best_result: Optional[Dict] = None

    def maybe_update_best(candidate: Dict) -> None:
        nonlocal best_result
        if best_result is None or float(candidate["best_val_acc"]) > float(best_result["best_val_acc"]):
            best_result = dict(candidate)
            save_best_model_checkpoint(candidate, result_dir)

    print("\n==== (1) Baseline: pretrained ResNet fine-tuning ====")
    baseline_result = train_one_experiment(
        task_name="baseline",
        model_name=EXPERIMENT_CONFIG["backbone"],
        pretrained=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EXPERIMENT_CONFIG["baseline"]["epochs"],
        backbone_lr=EXPERIMENT_CONFIG["baseline"]["backbone_lr"],
        head_lr=EXPERIMENT_CONFIG["baseline"]["head_lr"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        momentum=TRAINING_CONFIG["momentum"],
        device=device,
        wandb_logger=wandb_logger,
    )
    baseline_record = {"task": "baseline", **baseline_result}
    maybe_update_best(baseline_record)
    baseline_record.pop("best_state", None)
    results.append(baseline_record)
    print(f"Baseline test_acc={baseline_result['test_acc']:.4f}")

    # save baseline learning curve
    try:
        plot_learning_curve(baseline_result.get("history", {}), result_dir / "baseline_learning_curve.png", title="Baseline Learning Curve", wandb_logger=wandb_logger)
        print(f"Saved baseline plot to {result_dir / 'baseline_learning_curve.png'}")
    except Exception as exc:
        print(f"Failed to save baseline plot: {exc}")

    print("\n==== (2) Hyperparameter analysis ====")
    for epoch in EXPERIMENT_CONFIG["hyperparam"]["epochs"]:
        for lr in EXPERIMENT_CONFIG["hyperparam"]["head_lrs"]:
            print(f"\nRunning hyperparam combo: epochs={epoch}, head_lr={lr}")
            hp_result = train_one_experiment(
                task_name="hyperparam",
                model_name=EXPERIMENT_CONFIG["backbone"],
                pretrained=True,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epoch,
                backbone_lr=lr * EXPERIMENT_CONFIG["hyperparam"]["backbone_ratio"],
                head_lr=lr,
                weight_decay=TRAINING_CONFIG["weight_decay"],
                momentum=TRAINING_CONFIG["momentum"],
                device=device,
                wandb_logger=wandb_logger,
            )
            hp_record = {"task": "hyperparam", **hp_result}
            maybe_update_best(hp_record)
            hp_record.pop("best_state", None)
            results.append(hp_record)
            hp_results.append(hp_record.copy())

    print("\n==== (3) Pretraining ablation ====")
    ablation_pretrained = train_one_experiment(
        task_name="ablation_pretrained",
        model_name=EXPERIMENT_CONFIG["backbone"],
        pretrained=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EXPERIMENT_CONFIG["ablation"]["epochs"],
        backbone_lr=EXPERIMENT_CONFIG["ablation"]["backbone_lr"],
        head_lr=EXPERIMENT_CONFIG["ablation"]["head_lr"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        momentum=TRAINING_CONFIG["momentum"],
        device=device,
        wandb_logger=wandb_logger,
    )
    ablation_pretrained_record = {"task": "ablation_pretrained", **ablation_pretrained}
    maybe_update_best(ablation_pretrained_record)
    ablation_pretrained_record.pop("best_state", None)
    results.append(ablation_pretrained_record)

    ablation_scratch = train_one_experiment(
        task_name="ablation_scratch",
        model_name=EXPERIMENT_CONFIG["backbone"],
        pretrained=False,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EXPERIMENT_CONFIG["ablation"]["epochs"],
        backbone_lr=EXPERIMENT_CONFIG["ablation"]["head_lr"],
        head_lr=EXPERIMENT_CONFIG["ablation"]["head_lr"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        momentum=TRAINING_CONFIG["momentum"],
        device=device,
        wandb_logger=wandb_logger,
    )
    ablation_scratch_record = {"task": "ablation_scratch", **ablation_scratch}
    maybe_update_best(ablation_scratch_record)
    ablation_scratch_record.pop("best_state", None)
    results.append(ablation_scratch_record)

    delta = ablation_pretrained["test_acc"] - ablation_scratch["test_acc"]
    print(f"Pretrained - Scratch test_acc improvement: {delta:.4f}")

    # save ablation plots
    try:
        plot_ablation_curves_and_bar(ablation_pretrained, ablation_scratch, result_dir, wandb_logger=wandb_logger)
        print(f"Saved ablation plots to {result_dir}")
    except Exception as exc:
        print(f"Failed to save ablation plots: {exc}")

    print("\n==== (4) Attention model comparison ====")
    attention_result = train_one_experiment(
        task_name="attention",
        model_name=EXPERIMENT_CONFIG["attention_model"],
        pretrained=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EXPERIMENT_CONFIG["attention"]["epochs"],
        backbone_lr=EXPERIMENT_CONFIG["attention"]["backbone_lr"],
        head_lr=EXPERIMENT_CONFIG["attention"]["head_lr"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        momentum=TRAINING_CONFIG["momentum"],
        device=device,
        wandb_logger=wandb_logger,
    )
    attention_record = {"task": "attention", **attention_result}
    maybe_update_best(attention_record)
    attention_record.pop("best_state", None)
    results.append(attention_record)

    attention_gain = attention_result["test_acc"] - baseline_result["test_acc"]
    print(f"Attention - Baseline test_acc improvement: {attention_gain:.4f}")
    # save attention vs baseline plot
    try:
        plot_compare_with_baseline(attention_result, baseline_result, result_dir, name="attention", wandb_logger=wandb_logger)
        print(f"Saved attention vs baseline plot to {result_dir / 'attention_vs_baseline_val_acc.png'}")
    except Exception as exc:
        print(f"Failed to save attention vs baseline plot: {exc}")

    # save hyperparam table (after attention so all hp runs complete)
    try:
        save_hyperparam_table(results, result_dir, wandb_logger=wandb_logger)
        print(f"Saved hyperparam results to {result_dir}")
    except Exception as exc:
        print(f"Failed to save hyperparam table: {exc}")
    if wandb_logger is not None:
        wandb_logger.log(
            {
                "summary/baseline_test_acc": baseline_result["test_acc"],
                "summary/attention_test_acc": attention_result["test_acc"],
                "summary/attention_gain": attention_gain,
                "summary/pretrained_vs_scratch_gain": delta,
            }
        )
        wandb_logger.finish()

    return results


def print_result_table(results: List[Dict]) -> None:
    print("\n================ Experiment Summary ================")
    header = (
        f"{'task':<20} {'model':<12} {'pretrained':<10} {'epochs':<8} "
        f"{'backbone_lr':<12} {'head_lr':<10} {'val_acc':<10} {'test_acc':<10}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['task']:<20} {row['model']:<12} {str(row['pretrained']):<10} "
            f"{row['epochs']:<8} {row['backbone_lr']:<12.6f} {row['head_lr']:<10.6f} "
            f"{row['best_val_acc']:<10.4f} {row['test_acc']:<10.4f}"
        )


def save_results(results: List[Dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output.resolve()}")


def save_best_model_checkpoint(result: Dict, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    target_path = result_dir / "best_model.pt"
    torch.save(
        {
            "task": result["task"],
            "model_name": result["model"],
            "pretrained": result["pretrained"],
            "epochs": result["epochs"],
            "backbone_lr": result["backbone_lr"],
            "head_lr": result["head_lr"],
            "weight_decay": result["weight_decay"],
            "momentum": result["momentum"],
            "best_val_acc": result["best_val_acc"],
            "test_acc": result["test_acc"],
            "model_state_dict": result["best_state"],
        },
        target_path,
    )
    print(f"Best model saved to: {target_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet fine-tuning and ablation experiments")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Dataset directory")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Result json path")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb experiment visualization")
    parser.add_argument("--use-swanlab", action="store_true", help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()
    results = run_all_experiments(args)
    print_result_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
