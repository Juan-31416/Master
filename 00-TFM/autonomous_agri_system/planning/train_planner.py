"""Train neural trajectory scorer with synthetic labels.

Example:
    python -m planning.train_planner --config configs/planning.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from planning.local_planner import LatticePlanner, PlannerState
from planning.trajectory_scorer_net import TrajectoryScorerNet
from utils.config import load_config


class SyntheticPlanningDataset(Dataset):
    """Generates occupancy/goal with trajectory candidates and quality labels."""

    def __init__(self, n_samples: int = 1000, bev_size: int = 200, seed: int = 7) -> None:
        self.n_samples = n_samples
        self.bev_size = bev_size
        self.rng = np.random.default_rng(seed)
        self.lattice = LatticePlanner(horizon=20, dt=0.2)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        del idx
        occ = np.zeros((self.bev_size, self.bev_size), dtype=np.float32)
        for _ in range(self.rng.integers(2, 10)):
            x = self.rng.integers(10, self.bev_size - 10)
            y = self.rng.integers(10, self.bev_size - 10)
            occ[y - 3 : y + 3, x - 3 : x + 3] = 1.0

        ego = PlannerState(20.0, 20.0, 0.0, 1.0)
        goal = np.array([self.rng.uniform(140, 190), self.rng.uniform(140, 190), 0.0], dtype=np.float32)
        traj = self.lattice.generate(ego, speeds=[0.8, 1.1, 1.4], curvatures=[-0.2, -0.1, 0.0, 0.1, 0.2])

        scores = []
        for t in traj:
            end = t[-1, :2]
            dist = np.linalg.norm(end - goal[:2])
            penalty = 0.0
            for pt in t:
                x = int(np.clip(round(pt[0]), 0, self.bev_size - 1))
                y = int(np.clip(round(pt[1]), 0, self.bev_size - 1))
                penalty += occ[y, x] * 20.0
            scores.append(np.exp(-(dist / 80.0)) - penalty)
        y = np.asarray(scores, dtype=np.float32)

        bev = np.stack([occ, 1.0 - occ], axis=0)
        ego_goal = np.array([ego.x, ego.y, ego.heading, goal[0], goal[1], goal[2]], dtype=np.float32)

        return (
            torch.from_numpy(bev),
            torch.from_numpy(ego_goal),
            torch.from_numpy(traj),
            torch.from_numpy(y),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/planning.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticPlanningDataset(
        n_samples=cfg["data"]["num_samples"],
        bev_size=cfg["data"]["bev_size"],
        seed=cfg["train"].get("seed", 7),
    )
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = TrajectoryScorerNet(bev_channels=2, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.MSELoss()

    out_dir = Path(cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr = 0.0
        for bev, ego_goal, traj, target in train_loader:
            bev, ego_goal, traj, target = bev.to(device), ego_goal.to(device), traj.to(device), target.to(device)
            pred = model(bev, ego_goal, traj)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss.item())

        model.eval()
        va = 0.0
        with torch.no_grad():
            for bev, ego_goal, traj, target in val_loader:
                bev, ego_goal, traj, target = bev.to(device), ego_goal.to(device), traj.to(device), target.to(device)
                pred = model(bev, ego_goal, traj)
                va += float(loss_fn(pred, target).item())

        tr /= max(1, len(train_loader))
        va /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train={tr:.4f} val={va:.4f}")

        if va < best:
            best = va
            ckpt = out_dir / "best_planner_scorer.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt)
            print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
