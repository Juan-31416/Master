"""Evaluate segmentation model on test split.

Example:
    python -m identification.evaluate_segmentation \
        --config configs/identification.yaml \
        --checkpoint checkpoints/identification/best_segmentation.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from identification.dataset import build_dataloaders
from identification.deeplabv3plus import build_deeplabv3plus
from utils.config import load_config
from utils.metrics import multiclass_f1, multiclass_iou, per_class_iou
from utils.visualization import save_bev_triplet


def _extract_logits(model_out: torch.Tensor | dict) -> torch.Tensor:
    return model_out["out"] if isinstance(model_out, dict) else model_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/identification.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/identification_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = build_dataloaders(
        root=cfg["data"]["dataset_root"],
        batch_size=cfg["train"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=cfg["train"].get("num_workers", 0),
    )

    model = build_deeplabv3plus(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"].get("encoder_name", "efficientnet-b4"),
        pretrained=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    f1s = []
    class_ious_accum = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = _extract_logits(model(img))
            pred = logits.argmax(dim=1)

            ious.append(multiclass_iou(pred, mask, cfg["model"]["num_classes"]))
            f1s.append(multiclass_f1(pred, mask, cfg["model"]["num_classes"]))
            class_ious_accum.append(
                per_class_iou(pred, mask, cfg["model"]["num_classes"])
            )

            if i < 5:
                # Reuse generic utility for side-by-side qualitative output.
                pred_np = pred[0].cpu().numpy().astype(np.float32)
                gt_np = mask[0].cpu().numpy().astype(np.float32)
                confidence = torch.softmax(logits[0], dim=0).max(dim=0).values.cpu().numpy()
                save_bev_triplet(gt_np, pred_np, confidence, out_dir / f"sample_{i}.png")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    mean_class_ious = np.nanmean(np.asarray(class_ious_accum), axis=0) if class_ious_accum else np.array([])

    print("=== Segmentation Evaluation ===")
    print(f"mIoU: {mean_iou:.4f}")
    print(f"F1  : {mean_f1:.4f}")
    for idx, val in enumerate(mean_class_ious):
        print(f"Class {idx} IoU: {float(val):.4f}")
    print(f"Qualitative outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
