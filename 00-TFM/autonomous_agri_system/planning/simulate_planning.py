"""Simulation loop for global+local planning and MPC trajectory tracking.

Example:
    python -m planning.simulate_planning --config configs/planning.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from planning.global_planner import BoustrophedonPlanner
from planning.local_planner import HybridLocalPlanner, PlannerState
from planning.mpc_controller import KinematicBicycleMPC, VehicleState
from planning.trajectory_scorer_net import TrajectoryScorerNet
from utils.config import load_config
from utils.visualization import plot_trajectory_set


def build_synthetic_field(size: int = 200) -> np.ndarray:
    occ = np.zeros((size, size), dtype=np.float32)
    occ[80:120, 96:104] = 1.0
    occ[40:55, 40:55] = 1.0
    occ[145:165, 140:160] = 1.0
    return occ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/planning.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="outputs/planning_sim")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    occ = build_synthetic_field(cfg["data"]["bev_size"])
    traversable = (1.0 - occ).astype(np.uint8)

    global_planner = BoustrophedonPlanner(row_step=cfg["global"]["row_step"])
    global_plan = global_planner.plan(traversable)

    scorer = TrajectoryScorerNet()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        scorer.load_state_dict(state["model"])
    scorer.eval()

    local = HybridLocalPlanner(scorer_net=scorer, alpha=cfg["local"]["alpha"])
    mpc = KinematicBicycleMPC()

    if len(global_plan.waypoints) < 2:
        raise RuntimeError("Global planner failed to generate waypoints.")

    ego = PlannerState(x=float(global_plan.waypoints[0, 0]), y=float(global_plan.waypoints[0, 1]), heading=0.0, speed=1.0)
    next_goal = tuple(global_plan.waypoints[min(1, len(global_plan.waypoints)-1)])

    best_traj, best_idx, candidates = local.select(occ, ego, next_goal)
    plot_trajectory_set(occ, candidates, best_idx if best_idx >= 0 else 0, save_dir / "trajectory_selection.png")

    state = VehicleState(ego.x, ego.y, ego.heading, ego.speed)
    ref = np.concatenate([best_traj, np.ones((best_traj.shape[0], 1), dtype=np.float32)], axis=1)

    rollout = []
    for _ in range(30):
        steer, accel = mpc.solve(state, ref)
        state = mpc.step_dynamics(state, steer, accel)
        rollout.append([state.x, state.y, state.heading, state.speed])

    rollout = np.asarray(rollout, dtype=np.float32)
    np.save(save_dir / "mpc_rollout.npy", rollout)
    np.save(save_dir / "global_waypoints.npy", global_plan.waypoints)

    print("=== Planning Simulation Summary ===")
    print(f"Global coverage estimate: {global_plan.estimated_coverage:.2%}")
    print(f"Global path length: {global_plan.estimated_length:.2f} cells")
    print(f"Selected local trajectory index: {best_idx}")
    print(f"Artifacts saved in: {save_dir}")


if __name__ == "__main__":
    main()
