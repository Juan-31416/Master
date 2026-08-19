"""Common evaluation metrics for BEV perception, planning, and segmentation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def occupancy_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, F1 and IoU for binary occupancy maps."""
    probs = torch.sigmoid(pred_logits)
    pred = (probs >= threshold).int()
    target = target.int()

    tp = int(((pred == 1) & (target == 1)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    tn = int(((pred == 0) & (target == 0)).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(acc),
    }


def per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> List[float]:
    """Compute IoU for each class from integer masks."""
    pred = pred.view(-1)
    target = target.view(-1)

    ious: List[float] = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            ious.append(float("nan"))
            continue
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou = (inter / (union + 1e-8)).item()
        ious.append(float(iou))
    return ious


def multiclass_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Compute mean IoU across classes."""
    class_ious = per_class_iou(pred, target, num_classes, ignore_index)
    clean = [x for x in class_ious if not np.isnan(x)]
    return float(np.mean(clean)) if clean else 0.0


def binary_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute binary F1 score for tensors in {0,1}."""
    pred = pred.int()
    target = target.int()
    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float((2 * precision * recall / (precision + recall + 1e-8)).item())


def multiclass_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged multiclass F1 score."""
    scores = []
    for cls in range(num_classes):
        p = (pred == cls).int()
        t = (target == cls).int()
        scores.append(binary_f1(p, t))
    return float(np.mean(scores))
