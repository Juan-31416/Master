"""Prototype MPC-like controller for kinematic bicycle tracking.

This implementation is educational and solver-free. It approximates MPC by
sampling control candidates over a short horizon and choosing the minimum cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MPCParams:
    wheelbase: float = 1.2
    dt: float = 0.1
    horizon: int = 12
    steer_limit: float = 0.45
    accel_limit: float = 1.2


@dataclass
class VehicleState:
    x: float
    y: float
    heading: float
    speed: float


class KinematicBicycleMPC:
    """Simple sampled MPC controller using kinematic bicycle dynamics."""

    def __init__(self, params: MPCParams | None = None) -> None:
        self.params = params or MPCParams()

    def step_dynamics(self, state: VehicleState, steer: float, accel: float) -> VehicleState:
        p = self.params
        steer = float(np.clip(steer, -p.steer_limit, p.steer_limit))
        accel = float(np.clip(accel, -p.accel_limit, p.accel_limit))

        x = state.x + state.speed * np.cos(state.heading) * p.dt
        y = state.y + state.speed * np.sin(state.heading) * p.dt
        heading = state.heading + (state.speed / p.wheelbase) * np.tan(steer) * p.dt
        speed = max(0.0, state.speed + accel * p.dt)
        return VehicleState(x, y, heading, speed)

    def _trajectory_cost(self, traj: np.ndarray, ref: np.ndarray) -> float:
        n = min(len(traj), len(ref))
        pos_err = np.linalg.norm(traj[:n, :2] - ref[:n, :2], axis=1).mean()
        heading_err = np.abs(traj[:n, 2] - ref[:n, 2]).mean()
        speed_err = np.abs(traj[:n, 3] - ref[:n, 3]).mean() if ref.shape[1] > 3 else 0.0
        return float(pos_err + 0.3 * heading_err + 0.2 * speed_err)

    def solve(self, state: VehicleState, reference: np.ndarray) -> Tuple[float, float]:
        """Return control action (steer, accel) for the current time step."""
        steer_grid = np.linspace(-self.params.steer_limit, self.params.steer_limit, 9)
        accel_grid = np.linspace(-self.params.accel_limit, self.params.accel_limit, 7)

        best_cost = float("inf")
        best_action = (0.0, 0.0)

        for steer in steer_grid:
            for accel in accel_grid:
                s = state
                rollout = []
                for _ in range(self.params.horizon):
                    s = self.step_dynamics(s, float(steer), float(accel))
                    rollout.append([s.x, s.y, s.heading, s.speed])
                rollout_arr = np.asarray(rollout, dtype=np.float32)
                cost = self._trajectory_cost(rollout_arr, reference)
                effort = 0.05 * (abs(steer) + abs(accel))
                total = cost + effort
                if total < best_cost:
                    best_cost = total
                    best_action = (float(steer), float(accel))

        return best_action


if __name__ == "__main__":
    mpc = KinematicBicycleMPC()
    s0 = VehicleState(x=0.0, y=0.0, heading=0.0, speed=1.0)
    ref = np.array([[i * 0.1, 0.0, 0.0, 1.0] for i in range(20)], dtype=np.float32)
    u = mpc.solve(s0, ref)
    print("control", u)
