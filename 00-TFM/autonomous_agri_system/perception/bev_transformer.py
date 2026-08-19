"""BEV transformer for multimodal agricultural perception.

The implementation is intentionally compact and educational for thesis work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class BEVConfig:
    camera_channels: int = 3
    lidar_channels: int = 1
    bev_size: int = 200
    bev_channels: int = 128
    semantic_classes: int = 6
    detection_dim: int = 6


class ConvEncoder(nn.Module):
    """Small CNN encoder used for camera and LiDAR feature extraction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion block for camera and LiDAR tokens."""

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.cam_to_lidar = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.lidar_to_cam = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cam_tokens: torch.Tensor, lidar_tokens: torch.Tensor) -> torch.Tensor:
        cam_q, _ = self.cam_to_lidar(cam_tokens, lidar_tokens, lidar_tokens)
        lidar_q, _ = self.lidar_to_cam(lidar_tokens, cam_tokens, cam_tokens)
        fused = 0.5 * (cam_q + lidar_q)
        return self.norm(fused)


class BEVTransformer(nn.Module):
    """BEV perception model with multi-task output heads.

    Inputs:
        camera: (B, 3, H, W)
        lidar_bev: (B, 1, H, W)

    Outputs:
        dict with occupancy logits, semantic logits, detection vector, confidence map
    """

    def __init__(self, cfg: BEVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.bev_channels

        self.camera_encoder = ConvEncoder(cfg.camera_channels, c)
        self.lidar_encoder = ConvEncoder(cfg.lidar_channels, c)
        self.fusion = CrossAttentionFusion(embed_dim=c)

        self.decode = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.occupancy_head = nn.Conv2d(c // 2, 1, kernel_size=1)
        self.semantic_head = nn.Conv2d(c // 2, cfg.semantic_classes, kernel_size=1)
        self.confidence_head = nn.Conv2d(c // 2, 1, kernel_size=1)

        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c // 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cfg.detection_dim),
        )

    def forward(self, camera: torch.Tensor, lidar_bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        cam_feat = self.camera_encoder(camera)
        lidar_feat = self.lidar_encoder(lidar_bev)

        b, c, h, w = cam_feat.shape
        cam_tokens = cam_feat.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_feat.flatten(2).transpose(1, 2)
        fused_tokens = self.fusion(cam_tokens, lidar_tokens)

        fused = fused_tokens.transpose(1, 2).reshape(b, c, h, w)
        decoded = self.decode(fused)

        occupancy = nn.functional.interpolate(
            self.occupancy_head(decoded),
            size=(self.cfg.bev_size, self.cfg.bev_size),
            mode="bilinear",
            align_corners=False,
        )
        semantic = nn.functional.interpolate(
            self.semantic_head(decoded),
            size=(self.cfg.bev_size, self.cfg.bev_size),
            mode="bilinear",
            align_corners=False,
        )
        confidence = nn.functional.interpolate(
            self.confidence_head(decoded),
            size=(self.cfg.bev_size, self.cfg.bev_size),
            mode="bilinear",
            align_corners=False,
        )
        detection = self.detection_head(decoded)

        return {
            "occupancy": occupancy,
            "semantic": semantic,
            "confidence": confidence,
            "detection": detection,
        }


if __name__ == "__main__":
    config = BEVConfig()
    model = BEVTransformer(config)
    cam = torch.randn(2, 3, 256, 256)
    lidar = torch.randn(2, 1, 256, 256)
    out = model(cam, lidar)
    for k, v in out.items():
        print(k, tuple(v.shape))
