"""Dataset and augmentations for crop/disease/weed segmentation.

Expected dataset structure:

root/
  images/
    train/*.jpg|png
    val/*.jpg|png
    test/*.jpg|png
  masks/
    train/*.png
    val/*.png
    test/*.png

Image and mask basenames must match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _list_files(directory: Path) -> List[Path]:
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        files.extend(sorted(directory.glob(ext)))
    return sorted(files)


class SegmentationDataset(Dataset):
    """Generic semantic segmentation dataset for user-provided agricultural data."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"

        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected directories not found: {self.image_dir} and {self.mask_dir}"
            )

        self.images = _list_files(self.image_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() < 0.3:
            angle = np.random.uniform(-12, 12)
            h, w = image.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-15, 15)
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask.astype(np.int64)),
        }


def build_dataloaders(
    root: str | Path,
    batch_size: int,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from split folders."""
    train_ds = SegmentationDataset(root=root, split="train", image_size=image_size, augment=True)
    val_ds = SegmentationDataset(root=root, split="val", image_size=image_size, augment=False)
    test_ds = SegmentationDataset(root=root, split="test", image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
