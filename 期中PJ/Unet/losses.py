import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Multiclass Dice Loss from raw logits."""
    probs = F.softmax(logits, dim=1)

    targets_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * targets_oh).sum(dims)
    cardinality = probs.sum(dims) + targets_oh.sum(dims)

    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_per_class.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return dice_loss(logits, targets, num_classes=self.num_classes, smooth=self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int = 3, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_criterion(loss_name: str, num_classes: int = 3) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "dice":
        return DiceLoss(num_classes=num_classes)
    if loss_name == "combined":
        return CombinedLoss(num_classes=num_classes)
    raise ValueError(f"Unsupported loss name: {loss_name}")
