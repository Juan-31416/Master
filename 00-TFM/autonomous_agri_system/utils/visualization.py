"""Visualization helpers for debug plots and qualitative analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_bev_triplet(
    occupancy: np.ndarray,
    semantic: np.ndarray,
    confidence: Optional[np.ndarray],
    out_path: str | Path,
) -> None:
    """Save occupancy/semantic/(optional)confidence maps side-by-side."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = 3 if confidence is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 2:
        axes = np.asarray(axes)

    axes[0].imshow(occupancy, cmap="gray")
    axes[0].set_title("Occupancy")
    axes[1].imshow(semantic, cmap="tab20")
    axes[1].set_title("Semantic")
    if confidence is not None:
        axes[2].imshow(confidence, cmap="viridis")
        axes[2].set_title("Confidence")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_set(
    occupancy: np.ndarray,
    trajectories: Iterable[np.ndarray],
    best_idx: int,
    out_path: str | Path,
) -> None:
    """Plot candidate trajectories over occupancy map."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(occupancy, cmap="Greys", origin="lower")

    for idx, traj in enumerate(trajectories):
        color = "lime" if idx == best_idx else "deepskyblue"
        lw = 2.5 if idx == best_idx else 1.0
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=lw)

    ax.set_title("Trajectory candidates")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
