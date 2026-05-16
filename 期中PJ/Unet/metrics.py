import numpy as np
import torch


class SegmentationMetric:
    def __init__(self, num_classes: int = 3) -> None:
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.hist.fill(0)

    def _fast_hist(self, true: np.ndarray, pred: np.ndarray) -> np.ndarray:
        mask = (true >= 0) & (true < self.num_classes)
        hist = np.bincount(
            self.num_classes * true[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        trues = targets.cpu().numpy()
        for true, pred in zip(trues, preds):
            self.hist += self._fast_hist(true, pred)

    def miou(self) -> float:
        iou = np.diag(self.hist) / (
            self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + 1e-10
        )
        return float(np.nanmean(iou))
