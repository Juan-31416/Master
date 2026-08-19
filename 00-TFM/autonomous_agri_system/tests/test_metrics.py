"""Unit test examples for metric implementations."""

import torch

from utils.metrics import multiclass_iou, occupancy_metrics


def test_occupancy_metrics_perfect_prediction() -> None:
    logits = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    m = occupancy_metrics(logits, target)
    assert m["f1"] > 0.99
    assert m["iou"] > 0.99


def test_multiclass_iou_simple_case() -> None:
    pred = torch.tensor([[[0, 1], [1, 2]]])
    target = torch.tensor([[[0, 1], [2, 2]]])
    miou = multiclass_iou(pred, target, num_classes=3)
    assert 0.0 <= miou <= 1.0
