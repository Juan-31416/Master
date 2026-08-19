"""Unit test examples for local hybrid planner."""

import numpy as np

from planning.local_planner import HybridLocalPlanner, PlannerState
from planning.trajectory_scorer_net import TrajectoryScorerNet


def test_local_planner_returns_trajectory() -> None:
    occ = np.zeros((200, 200), dtype=np.float32)
    occ[95:105, 95:105] = 1.0
    planner = HybridLocalPlanner(TrajectoryScorerNet())

    best, idx, candidates = planner.select(occ, PlannerState(20, 20, 0.0, 1.0), goal=(180, 180))
    assert best.ndim == 2 and best.shape[1] == 3
    assert candidates.ndim == 3
    assert isinstance(idx, int)
