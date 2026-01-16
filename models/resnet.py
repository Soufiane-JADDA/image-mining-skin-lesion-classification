# resnet.py
"""
ResNet utilities for HAM10000 classification.

This module provides:
- A ResNet50-based classifier adapted for multi-class skin lesion classification
- Options for ImageNet pretraining, freezing the backbone, and adding dropout

Typical usage:
    from resnet import build_resnet50
    model = build_resnet50(num_classes=7, pretrained=True, freeze_backbone=False, dropout=0.2)
"""

from typing import Optional
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier wrapper.

    It uses a ResNet50 backbone and replaces the final FC layer with:
        [Dropout (optional)] -> Linear(in_features, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Load backbone
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)

        # Optionally freeze all backbone parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Replace classification head
        in_features = self.backbone.fc.in_features

        head_layers = []
        if dropout and dropout > 0:
            head_layers.append(nn.Dropout(p=dropout))
        head_layers.append(nn.Linear(in_features, num_classes))

        self.backbone.fc = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.backbone(x)


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Convenience function to build a ResNet50 classifier.

    Args:
        num_classes: number of output classes.
        pretrained: load ImageNet pretrained weights if True.
        freeze_backbone: if True, train only the final classification head.
        dropout: dropout probability before the final linear layer.

    Returns:
        A torch.nn.Module ready for training.
    """
    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick sanity check
    model = build_resnet50(num_classes=7, pretrained=True, freeze_backbone=False, dropout=0.2)
    print(model)
