"""Evaluate trained BEV model on synthetic test split.

Example:
    python -m perception.evaluate_bev --config configs/perception.yaml --checkpoint checkpoints/bev/best_bev.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from perception.bev_transformer import BEVConfig, BEVTransformer
from perception.data_generator import SyntheticBEVConfig, SyntheticBEVDataset, split_indices
from utils.config import load_config
from utils.metrics import multiclass_iou, occupancy_metrics
from utils.visualization import save_bev_triplet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/perception.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/perception_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = BEVConfig(
        bev_size=cfg["model"]["bev_size"],
        bev_channels=cfg["model"]["bev_channels"],
        semantic_classes=cfg["model"]["semantic_classes"],
    )
    model = BEVTransformer(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = SyntheticBEVDataset(
        SyntheticBEVConfig(
            num_samples=cfg["data"]["num_samples"],
            image_size=cfg["data"]["image_size"],
            bev_size=cfg["model"]["bev_size"],
            semantic_classes=cfg["model"]["semantic_classes"],
        ),
        seed=cfg["train"].get("seed", 42),
    )
    _, _, test_idx = split_indices(len(dataset))
    loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg["train"]["batch_size"], shuffle=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    occ_scores = []
    sem_scores = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["camera"], batch["lidar"])

            occ_scores.append(occupancy_metrics(outputs["occupancy"], batch["occupancy"]))
            sem_pred = outputs["semantic"].argmax(dim=1)
            sem_iou = multiclass_iou(sem_pred, batch["semantic"], num_classes=cfg["model"]["semantic_classes"])
            sem_scores.append(sem_iou)

            if i < 5:
                occ_map = torch.sigmoid(outputs["occupancy"][0, 0]).cpu().numpy()
                sem_map = sem_pred[0].cpu().numpy()
                conf = torch.sigmoid(outputs["confidence"][0, 0]).cpu().numpy()
                save_bev_triplet(occ_map, sem_map, conf, save_dir / f"sample_{i}.png")

    mean_occ_iou = sum(m["iou"] for m in occ_scores) / max(1, len(occ_scores))
    mean_occ_f1 = sum(m["f1"] for m in occ_scores) / max(1, len(occ_scores))
    mean_sem_iou = sum(sem_scores) / max(1, len(sem_scores))

    print("=== BEV Evaluation ===")
    print(f"Occupancy IoU: {mean_occ_iou:.4f}")
    print(f"Occupancy F1 : {mean_occ_f1:.4f}")
    print(f"Semantic mIoU: {mean_sem_iou:.4f}")
    print(f"Qualitative outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
