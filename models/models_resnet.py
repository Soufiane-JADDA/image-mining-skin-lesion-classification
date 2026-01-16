# models_resnet.py
"""
ResNet models for HAM10000 classification.

This file contains:
- A helper function to build a ResNet50 model for fine-tuning
  on the HAM10000 dataset.

The idea:
- Start from an ImageNet-pretrained ResNet50
- Replace the final fully-connected layer by a new classifier
  with `num_classes` outputs
- Optionally: freeze some layers or fine-tune all of them
"""

from typing import Optional

import torch.nn as nn
from torchvision import models


def get_resnet50_finetune(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ResNet50 model adapted for HAM10000 classification.

    Args:
        num_classes (int):
            Number of target classes (output neurons in the classifier head).
        pretrained (bool):
            If True, load ImageNet-pretrained weights.
        freeze_backbone (bool):
            If True, freeze all convolutional layers (only train the final FC layer).

    Returns:
        nn.Module:
            A ResNet50 model ready to be trained on HAM10000.
    """
    # Load ResNet50 backbone
    if pretrained:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet = models.resnet50(weights=None)

    # Optionally freeze the backbone parameters
    if freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer with a new classifier
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)

    return resnet


if __name__ == "__main__":
    # Simple debug usage example
    model = get_resnet50_finetune(num_classes=7, pretrained=True, freeze_backbone=False)
    print(model)
