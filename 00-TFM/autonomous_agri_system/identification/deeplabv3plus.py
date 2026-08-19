"""DeepLabV3+ for agricultural segmentation with EfficientNet backbone."""

from __future__ import annotations

import torch
from torch import nn


def build_deeplabv3plus(num_classes: int, encoder_name: str = "efficientnet-b4", pretrained: bool = True) -> nn.Module:
    """Build DeepLabV3+ model.

    Preferred backend is segmentation_models_pytorch (SMP). If SMP is not
    installed, fallback to torchvision DeepLabV3 (ResNet backbone).
    """
    try:
        import segmentation_models_pytorch as smp

        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
        return model
    except Exception:
        from torchvision.models.segmentation import deeplabv3_resnet50

        model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        return model


if __name__ == "__main__":
    net = build_deeplabv3plus(num_classes=5)
    x = torch.randn(2, 3, 512, 512)
    y = net(x)
    if isinstance(y, dict):
        y = y["out"]
    print(y.shape)
