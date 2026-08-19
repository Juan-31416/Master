"""Training script for DeepLabV3+ agricultural segmentation.

Example:
    python -m identification.train_segmentation --config configs/identification.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from identification.dataset import build_dataloaders
from identification.deeplabv3plus import build_deeplabv3plus
from utils.config import load_config
from utils.metrics import multiclass_f1, multiclass_iou


def _extract_logits(model_out: torch.Tensor | dict) -> torch.Tensor:
    return model_out["out"] if isinstance(model_out, dict) else model_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/identification.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = build_dataloaders(
        root=cfg["data"]["dataset_root"],
        batch_size=cfg["train"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=cfg["train"].get("num_workers", 0),
    )

    model = build_deeplabv3plus(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"].get("encoder_name", "efficientnet-b4"),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = _extract_logits(model(img))
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        ious = []
        f1s = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                logits = _extract_logits(model(img))
                loss = criterion(logits, mask)
                val_loss += float(loss.item())

                pred = logits.argmax(dim=1)
                ious.append(multiclass_iou(pred, mask, cfg["model"]["num_classes"]))
                f1s.append(multiclass_f1(pred, mask, cfg["model"]["num_classes"]))

        scheduler.step()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        miou = float(sum(ious) / max(1, len(ious)))
        f1 = float(sum(f1s) / max(1, len(f1s)))

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train={train_loss:.4f} val={val_loss:.4f} mIoU={miou:.4f} F1={f1:.4f}")

        if miou > best:
            best = miou
            ckpt = out_dir / "best_segmentation.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt)
            print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
