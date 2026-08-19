"""Perception module with BEV transformer and synthetic data tools."""

from .bev_transformer import BEVTransformer
from .data_generator import SyntheticBEVDataset

__all__ = ["BEVTransformer", "SyntheticBEVDataset"]
