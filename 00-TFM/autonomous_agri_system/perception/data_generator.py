"""Synthetic multimodal data generator for BEV perception training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticBEVConfig:
    num_samples: int = 2000
    image_size: int = 256
    bev_size: int = 200
    semantic_classes: int = 6


class SyntheticBEVDataset(Dataset):
    """Creates synthetic camera/LiDAR-like tensors with BEV labels."""

    def __init__(self, cfg: SyntheticBEVConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.cfg.num_samples

    def _draw_obstacles(self, grid: np.ndarray) -> None:
        n_obj = self.rng.integers(5, 18)
        h, w = grid.shape
        for _ in range(n_obj):
            cx = int(self.rng.integers(10, w - 10))
            cy = int(self.rng.integers(10, h - 10))
            rx = int(self.rng.integers(2, 10))
            ry = int(self.rng.integers(2, 10))
            x0, x1 = max(0, cx - rx), min(w, cx + rx)
            y0, y1 = max(0, cy - ry), min(h, cy + ry)
            grid[y0:y1, x0:x1] = 1.0

    def _create_semantic(self, occupancy: np.ndarray) -> np.ndarray:
        semantic = np.zeros_like(occupancy, dtype=np.int64)
        semantic[occupancy > 0.5] = self.rng.integers(1, self.cfg.semantic_classes)

        row_spacing = 12
        for y in range(0, occupancy.shape[0], row_spacing):
            semantic[y : y + 2, :] = 2  # crop rows

        weed_mask = (self.rng.random(occupancy.shape) < 0.02) & (occupancy < 0.5)
        semantic[weed_mask] = 3
        return semantic

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        del index
        bev_occ = np.zeros((self.cfg.bev_size, self.cfg.bev_size), dtype=np.float32)
        self._draw_obstacles(bev_occ)
        semantic = self._create_semantic(bev_occ)

        # synthetic camera and lidar observations correlated with occupancy
        camera = self.rng.normal(0.45, 0.2, size=(3, self.cfg.image_size, self.cfg.image_size)).astype(np.float32)
        lidar = self.rng.normal(0.0, 0.2, size=(1, self.cfg.image_size, self.cfg.image_size)).astype(np.float32)

        occ_resized = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(bev_occ)[None, None],
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        )
        lidar += occ_resized[None, :, :] * 1.2

        detection_target = np.array([50.0, 50.0, 20.0, 20.0, 0.0, 1.0], dtype=np.float32)
        confidence_target = 1.0 - bev_occ

        return {
            "camera": torch.from_numpy(camera),
            "lidar": torch.from_numpy(lidar),
            "occupancy": torch.from_numpy(bev_occ[None, :, :]),
            "semantic": torch.from_numpy(semantic),
            "detection": torch.from_numpy(detection_target),
            "confidence": torch.from_numpy(confidence_target[None, :, :]),
        }


def split_indices(n_samples: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test split indices."""
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    ds = SyntheticBEVDataset(SyntheticBEVConfig(num_samples=3))
    sample = ds[0]
    print({k: tuple(v.shape) for k, v in sample.items()})
