"""Unit test examples for global planning behavior."""

import numpy as np

from planning.global_planner import BoustrophedonPlanner


def test_boustrophedon_generates_waypoints() -> None:
    traversable = np.ones((80, 120), dtype=np.uint8)
    traversable[:, 50:55] = 0
    planner = BoustrophedonPlanner(row_step=8)
    plan = planner.plan(traversable)
    assert plan.waypoints.shape[1] == 2
    assert plan.estimated_length > 0
    assert 0 <= plan.estimated_coverage <= 1.2
