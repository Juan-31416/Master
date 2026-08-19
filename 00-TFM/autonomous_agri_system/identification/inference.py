"""Inference script for crop/disease/weed segmentation.

Example:
    python -m identification.inference \
        --config configs/identification.yaml \
        --checkpoint checkpoints/identification/best_segmentation.pt \
        --input_dir /path/to/images \
        --output_dir outputs/inference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from identification.deeplabv3plus import build_deeplabv3plus
from utils.config import load_config


def _extract_logits(model_out: torch.Tensor | dict) -> torch.Tensor:
    return model_out["out"] if isinstance(model_out, dict) else model_out


def _colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors[mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/identification.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_deeplabv3plus(
        num_classes=cfg["model"]["num_classes"],
        encoder_name=cfg["model"].get("encoder_name", "efficientnet-b4"),
        pretrained=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(sorted(in_dir.glob(ext)))

    if not image_paths:
        raise RuntimeError(f"No images found in {in_dir}")

    size = tuple(cfg["data"]["image_size"])

    with torch.no_grad():
        for path in image_paths:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]
            resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]).to(device)

            logits = _extract_logits(model(tensor))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pred_up = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            color = _colorize_mask(pred_up, cfg["model"]["num_classes"])
            overlay = cv2.addWeighted(image, 0.65, color, 0.35, 0)

            cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), pred_up)
            cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved predictions for {len(image_paths)} images in {out_dir}")


if __name__ == "__main__":
    main()
