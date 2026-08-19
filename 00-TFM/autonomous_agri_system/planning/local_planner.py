"""Local planning: lattice generation + ML scoring + classical safety filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from planning.trajectory_scorer_net import TrajectoryScorerNet


@dataclass
class VehicleParams:
    max_curvature: float = 0.25
    safety_margin: int = 2


@dataclass
class PlannerState:
    x: float
    y: float
    heading: float
    speed: float


class LatticePlanner:
    """Generates candidate trajectories using simple curvature primitives."""

    def __init__(self, horizon: int = 20, dt: float = 0.2) -> None:
        self.horizon = horizon
        self.dt = dt

    def generate(self, state: PlannerState, speeds: List[float], curvatures: List[float]) -> np.ndarray:
        candidates = []
        for v in speeds:
            for kappa in curvatures:
                x, y, theta = state.x, state.y, state.heading
                traj = []
                for _ in range(self.horizon):
                    x += v * np.cos(theta) * self.dt
                    y += v * np.sin(theta) * self.dt
                    theta += v * kappa * self.dt
                    traj.append([x, y, theta])
                candidates.append(np.asarray(traj, dtype=np.float32))
        return np.stack(candidates, axis=0)


def _safe_index(v: float, bound: int) -> int:
    return int(np.clip(round(v), 0, bound - 1))


class HybridLocalPlanner:
    """Select best safe trajectory by combining ML score and classical costs."""

    def __init__(
        self,
        scorer_net: TrajectoryScorerNet,
        vehicle_params: VehicleParams | None = None,
        alpha: float = 0.65,
    ) -> None:
        self.scorer_net = scorer_net
        self.vehicle_params = vehicle_params or VehicleParams()
        self.alpha = alpha
        self.lattice = LatticePlanner()

    def is_collision_free(self, traj: np.ndarray, occupancy: np.ndarray) -> bool:
        h, w = occupancy.shape
        margin = self.vehicle_params.safety_margin
        for pt in traj:
            x = _safe_index(pt[0], w)
            y = _safe_index(pt[1], h)
            x0, x1 = max(0, x - margin), min(w, x + margin + 1)
            y0, y1 = max(0, y - margin), min(h, y + margin + 1)
            if occupancy[y0:y1, x0:x1].max() > 0.5:
                return False
        return True

    def classical_cost(self, traj: np.ndarray, goal: Tuple[float, float]) -> float:
        end = traj[-1, :2]
        dist_goal = np.linalg.norm(end - np.asarray(goal, dtype=np.float32))
        heading_change = np.abs(np.diff(traj[:, 2])).mean()
        curvature_penalty = float(heading_change)
        return float(dist_goal + 2.0 * curvature_penalty + 1e-3)

    def select(self, occupancy: np.ndarray, ego: PlannerState, goal: Tuple[float, float]) -> tuple[np.ndarray, int, np.ndarray]:
        candidates = self.lattice.generate(ego, speeds=[0.8, 1.1, 1.4], curvatures=[-0.2, -0.1, 0.0, 0.1, 0.2])

        bev = np.stack([occupancy, 1.0 - occupancy], axis=0).astype(np.float32)
        bev_t = torch.from_numpy(bev)[None]
        ego_goal = torch.tensor([[ego.x, ego.y, ego.heading, goal[0], goal[1], 0.0]], dtype=torch.float32)
        traj_t = torch.from_numpy(candidates)[None]

        with torch.no_grad():
            ml_scores = self.scorer_net(bev_t, ego_goal, traj_t)[0].numpy()

        best_idx = -1
        best_score = -1e9
        for idx, (traj, ml) in enumerate(zip(candidates, ml_scores)):
            if not self.is_collision_free(traj, occupancy):
                continue
            cc = self.classical_cost(traj, goal)
            score = self.alpha * float(ml) + (1.0 - self.alpha) * (1.0 / cc)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            # emergency fallback: standstill trajectory
            stop = np.repeat(np.array([[ego.x, ego.y, ego.heading]], dtype=np.float32), repeats=10, axis=0)
            return stop, -1, candidates

        return candidates[best_idx], best_idx, candidates


if __name__ == "__main__":
    occ = np.zeros((200, 200), dtype=np.float32)
    occ[90:110, 95:105] = 1.0
    planner = HybridLocalPlanner(TrajectoryScorerNet())
    best, idx, all_traj = planner.select(occ, PlannerState(30, 30, 0.1, 1.0), goal=(180, 180))
    print("best idx", idx, "best shape", best.shape, "candidates", all_traj.shape)
