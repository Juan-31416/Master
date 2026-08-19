"""Global coverage planning with boustrophedon/zigzag strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class CoveragePlan:
    waypoints: np.ndarray
    estimated_length: float
    estimated_coverage: float


class BoustrophedonPlanner:
    """Creates a zigzag coverage path over a traversable grid map."""

    def __init__(self, row_step: int = 8) -> None:
        self.row_step = row_step

    def plan(self, traversable: np.ndarray) -> CoveragePlan:
        """Generate alternating left-right sweeps.

        Args:
            traversable: Binary map where 1 = free, 0 = blocked.

        Returns:
            CoveragePlan with waypoints in grid coordinates.
        """
        h, w = traversable.shape
        points: List[Tuple[float, float]] = []

        y_values = list(range(0, h, self.row_step))
        for i, y in enumerate(y_values):
            valid_x = np.where(traversable[y] > 0)[0]
            if len(valid_x) == 0:
                continue
            x_min, x_max = float(valid_x.min()), float(valid_x.max())
            if i % 2 == 0:
                points.extend([(x_min, float(y)), (x_max, float(y))])
            else:
                points.extend([(x_max, float(y)), (x_min, float(y))])

        if not points:
            return CoveragePlan(waypoints=np.zeros((0, 2)), estimated_length=0.0, estimated_coverage=0.0)

        arr = np.asarray(points, dtype=np.float32)
        seg = np.diff(arr, axis=0)
        path_len = float(np.linalg.norm(seg, axis=1).sum())

        swept = min(len(y_values) * self.row_step * w, int(traversable.sum()))
        coverage = swept / (traversable.sum() + 1e-8)

        return CoveragePlan(waypoints=arr, estimated_length=path_len, estimated_coverage=float(coverage))


if __name__ == "__main__":
    grid = np.ones((120, 160), dtype=np.uint8)
    grid[:, 70:75] = 0
    planner = BoustrophedonPlanner(row_step=10)
    plan = planner.plan(grid)
    print(f"Waypoints: {plan.waypoints.shape}, length={plan.estimated_length:.2f}, coverage={plan.estimated_coverage:.2%}")
