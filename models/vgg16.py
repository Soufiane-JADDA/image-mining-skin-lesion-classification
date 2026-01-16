# vgg16.py
"""
VGG16 model utilities for HAM10000 classification.

This file provides:
- A builder for VGG16 (ImageNet pretrained by default)
- A replaced classifier head for num_classes
- Optional freezing strategies (train head only vs fine-tune deeper layers)

Notes:
- VGG16 is heavier in parameters than ResNet50 and can overfit more easily.
- VGG16 does NOT have residual connections; training from scratch is harder.
- Fine-tuning from ImageNet weights is strongly recommended.

Usage:
    from vgg16 import get_vgg16
    model = get_vgg16(num_classes=7, pretrained=True, trainable="classifier")

Trainable options:
- "classifier": train only the classifier head
- "features5": unfreeze the last conv block (block5) + classifier
- "all": fine-tune entire network
"""

from typing import Literal

import torch.nn as nn
from torchvision import models


TrainableMode = Literal["classifier", "features5", "all"]


def get_vgg16(
    num_classes: int,
    pretrained: bool = True,
    trainable: TrainableMode = "all",
    dropout: float = 0.5,
) -> nn.Module:
    """
    Build a VGG16 classifier for HAM10000.

    Args:
        num_classes: Number of target classes.
        pretrained: If True, load ImageNet-pretrained weights.
        trainable: Which parts of the model to unfreeze.
        dropout: Dropout probability used in the classifier head.

    Returns:
        A VGG16 model with a classifier adapted to num_classes.
    """
    if pretrained:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg16(weights=None)

    # Replace the classifier last layer with num_classes output
    # Default VGG16 classifier structure:
    # [Linear(25088->4096), ReLU, Dropout, Linear(4096->4096), ReLU, Dropout, Linear(4096->1000)]
    in_features = model.classifier[-1].in_features

    # Optionally adjust dropout in classifier (keep same layer sizes)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, num_classes),
    )

    # Set trainable strategy
    set_vgg16_trainable(model, trainable=trainable)

    return model


def set_vgg16_trainable(model: nn.Module, trainable: TrainableMode = "all") -> None:
    """
    Set which layers are trainable in VGG16.

    Args:
        model: VGG16 model.
        trainable:
            - "classifier": train only classifier layers
            - "features5": train block5 conv layers + classifier
            - "all": train all parameters
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    if trainable == "classifier":
        return

    if trainable == "features5":
        # Unfreeze the last convolutional block (block5)
        # VGG16 features is a Sequential of conv/relu/pool layers.
        # Block5 corresponds roughly to the last ~7 layers before the final MaxPool.
        # A robust way: unfreeze last N layers in features.
        last_n = 7
        for layer in list(model.features.children())[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        return

    if trainable == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown trainable mode: {trainable}")


if __name__ == "__main__":
    m = get_vgg16(num_classes=7, pretrained=True, trainable="features5")
    print("VGG16 ready. Trainable params:",
          sum(p.requires_grad for p in m.parameters()),
          "/",
          sum(1 for _ in m.parameters()))
