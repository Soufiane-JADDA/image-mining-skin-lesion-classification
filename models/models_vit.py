# models_vit.py
"""
Vision Transformer (ViT) model builders for HAM10000 classification.

This file provides:
- ViT-Base/16 from scratch (no pretrained weights)
- ViT-Base/16 fine-tuning (ImageNet pretrained weights)

Dependencies:
- timm (recommended): pip install timm

Notes:
- ViT from scratch usually needs more data, stronger augmentation,
  longer training, and careful regularization.
- ViT fine-tuning is typically much stronger on medical imaging
  when data is limited.
"""

import torch.nn as nn
import timm


def get_vit_base_patch16_scratch(num_classes: int) -> nn.Module:
    """
    Create ViT-Base Patch16 model from scratch (no pretrained weights).

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ViT model with a classification head for num_classes.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def get_vit_base_patch16_finetune(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    Create ViT-Base Patch16 model for fine-tuning (ImageNet pretrained).

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freeze all layers except the classifier head.

    Returns:
        nn.Module: ViT model ready for fine-tuning.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )

    # Optionally freeze all parameters except the classification head
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze the head parameters
        for name, param in model.head.named_parameters():
            param.requires_grad = True

    return model


def get_vit_embeddings_backbone(pretrained: bool = True) -> nn.Module:
    """
    Create a ViT backbone model that outputs embeddings instead of class logits.
    This is used for the "ViT embeddings + SVM" pipeline.

    Args:
        pretrained (bool): If True, use ImageNet pretrained weights.

    Returns:
        nn.Module: ViT model returning a feature vector per image (e.g., 768-dim).
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=0,  # num_classes=0 makes timm return features instead of logits
    )
    return model


if __name__ == "__main__":
    # Debug checks
    scratch = get_vit_base_patch16_scratch(num_classes=7)
    finetune = get_vit_base_patch16_finetune(num_classes=7)
    backbone = get_vit_embeddings_backbone(pretrained=True)

    print("ViT scratch head:", scratch.head)
    print("ViT finetune head:", finetune.head)
    print("ViT backbone output features:", backbone)
