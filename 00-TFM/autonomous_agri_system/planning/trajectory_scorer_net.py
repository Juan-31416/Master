"""Neural trajectory scorer used by the local hybrid planner."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryScorerNet(nn.Module):
    """Scores candidate trajectories given BEV map and ego/goal states.

    Input shapes:
        bev: (B, C, H, W)
        ego_goal: (B, 6) -> [ego_x, ego_y, ego_heading, goal_x, goal_y, goal_heading]
        trajectories: (B, N, T, 3)
    Output:
        scores: (B, N)
    """

    def __init__(self, bev_channels: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(inplace=True))
        self.traj_encoder = nn.GRU(input_size=3, hidden_size=64, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bev: torch.Tensor, ego_goal: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        b, n, t, d = trajectories.shape
        assert d == 3, "Trajectory must be (x, y, heading)."

        bev_feat = self.bev_encoder(bev)
        state_feat = self.state_encoder(ego_goal)

        traj_flat = trajectories.reshape(b * n, t, d)
        _, h = self.traj_encoder(traj_flat)
        traj_feat = h[-1].reshape(b, n, 64)

        bev_expand = bev_feat[:, None, :].expand(-1, n, -1)
        state_expand = state_feat[:, None, :].expand(-1, n, -1)

        fused = torch.cat([bev_expand, state_expand, traj_feat], dim=-1)
        scores = self.head(fused).squeeze(-1)
        return scores


if __name__ == "__main__":
    model = TrajectoryScorerNet()
    bev = torch.randn(2, 2, 200, 200)
    ego_goal = torch.randn(2, 6)
    traj = torch.randn(2, 24, 20, 3)
    out = model(bev, ego_goal, traj)
    print(out.shape)
