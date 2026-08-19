"""Train BEV transformer on synthetic multimodal data.

Example:
    python -m perception.train_bev --config configs/perception.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from perception.bev_transformer import BEVConfig, BEVTransformer
from perception.data_generator import SyntheticBEVConfig, SyntheticBEVDataset, split_indices
from utils.config import load_config
from utils.metrics import occupancy_metrics


def build_dataloaders(cfg: Dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = SyntheticBEVConfig(
        num_samples=cfg["data"]["num_samples"],
        image_size=cfg["data"]["image_size"],
        bev_size=cfg["model"]["bev_size"],
        semantic_classes=cfg["model"]["semantic_classes"],
    )
    dataset = SyntheticBEVDataset(data_cfg, seed=cfg["train"].get("seed", 42))
    train_idx, val_idx, _ = split_indices(len(dataset))

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    return train_loader, val_loader


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    loss_occ = bce(outputs["occupancy"], batch["occupancy"])
    loss_sem = ce(outputs["semantic"], batch["semantic"])
    loss_det = mse(outputs["detection"], batch["detection"])
    loss_conf = bce(outputs["confidence"], batch["confidence"])
    return loss_occ + loss_sem + 0.2 * loss_det + 0.3 * loss_conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/perception.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = BEVConfig(
        bev_size=cfg["model"]["bev_size"],
        bev_channels=cfg["model"]["bev_channels"],
        semantic_classes=cfg["model"]["semantic_classes"],
    )
    model = BEVTransformer(model_cfg).to(device)

    train_loader, val_loader = build_dataloaders(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    out_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["camera"], batch["lidar"])
            loss = compute_loss(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        model.eval()
        val_loss = 0.0
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["camera"], batch["lidar"])
                loss = compute_loss(outputs, batch)
                val_loss += float(loss.item())
                val_metrics.append(occupancy_metrics(outputs["occupancy"], batch["occupancy"]))

        train_loss = running / max(1, len(train_loader))
        val_loss = val_loss / max(1, len(val_loader))
        mean_iou = sum(m["iou"] for m in val_metrics) / max(1, len(val_metrics))

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train={train_loss:.4f} val={val_loss:.4f} occ_iou={mean_iou:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = out_dir / "best_bev.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
