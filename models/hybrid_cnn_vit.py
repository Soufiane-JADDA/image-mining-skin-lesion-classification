# hybrid_cnn_vit.py
"""
Hybrid CNN + ViT model for HAM10000 classification.

Core idea:
1) Use a CNN backbone (e.g., ResNet50) to extract a feature map of shape [B, C, H, W]
2) Treat each spatial location (h, w) as a "patch" of size 1x1 (on the feature map)
   -> sequence length = H * W tokens
3) Project CNN channel dimension C to Transformer dimension d_model
4) Add a learnable CLS token and learnable positional embeddings
5) Pass tokens through a Transformer encoder
6) Use CLS output for classification

This implements the "patch size 1x1" concept at the CNN feature-map level.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class HybridConfig:
    """Configuration for the CNN+ViT hybrid model."""
    d_model: int = 768
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # If you use standard ResNet50 with 224x224 inputs,
    # the output feature map is typically 7x7 (so 49 tokens).
    # If you change input size or backbone, this may change.
    expected_hw_tokens: int = 49

    # CNN backbone options
    pretrained_backbone: bool = True
    freeze_backbone: bool = False


class CNNViTHybrid(nn.Module):
    """
    CNN + Transformer hybrid classifier.

    Inputs:
        x: [B, 3, 224, 224]

    Outputs:
        logits: [B, num_classes]
    """

    def __init__(self, num_classes: int, cfg: Optional[HybridConfig] = None):
        super().__init__()
        self.cfg = cfg or HybridConfig()

        # -----------------------
        # CNN backbone (ResNet50)
        # -----------------------
        if self.cfg.pretrained_backbone:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(weights=None)

        # Keep layers up to the last convolutional block (exclude avgpool and fc)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])

        # ResNet50 last conv block outputs 2048 channels
        self.cnn_out_channels = 2048

        # Optionally freeze CNN backbone
        if self.cfg.freeze_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # --------------------------------
        # Projection from CNN channels -> d_model
        # --------------------------------
        self.proj = nn.Linear(self.cnn_out_channels, self.cfg.d_model)

        # -----------------------
        # Transformer Encoder
        # -----------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            batch_first=True,  # important: input shape [B, T, D]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)

        # -----------------------
        # CLS token and position embedding
        # -----------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))

        # Positional embedding length: 1 (CLS) + H*W tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.cfg.expected_hw_tokens, self.cfg.d_model)
        )

        self.pos_drop = nn.Dropout(self.cfg.dropout)

        # -----------------------
        # Classification head
        # -----------------------
        self.head = nn.Linear(self.cfg.d_model, num_classes)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize learnable parameters (CLS token and pos embedding)."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # proj and head use default init; you can customize if needed.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
        1) Extract CNN feature map: [B, C, H, W]
        2) Flatten spatial positions into tokens: [B, H*W, C]
        3) Project tokens to transformer dimension: [B, H*W, D]
        4) Add CLS token -> [B, 1 + H*W, D]
        5) Add positional embeddings and run transformer
        6) Classify using CLS output
        """
        # CNN feature map
        feat = self.cnn(x)  # [B, 2048, H, W]
        B, C, H, W = feat.shape

        # Flatten spatial dimensions to token sequence
        tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, 2048]

        # Project to transformer dimension
        tokens = self.proj(tokens)  # [B, H*W, d_model]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 1+H*W, d_model]

        # Handle positional embeddings safely if H*W differs from expected
        needed_len = tokens.size(1)
        if needed_len != self.pos_embed.size(1):
            # If feature map size changes, we adapt by interpolating pos embedding
            # for the patch tokens part, keeping CLS token separate.
            tokens = self._add_pos_embed_with_interpolation(tokens, H, W)
        else:
            tokens = tokens + self.pos_embed[:, :needed_len, :]

        tokens = self.pos_drop(tokens)

        # Transformer encoder
        out = self.transformer(tokens)  # [B, 1+H*W, d_model]

        # CLS output
        cls_out = out[:, 0, :]  # [B, d_model]

        # Classification logits
        logits = self.head(cls_out)  # [B, num_classes]
        return logits

    def _add_pos_embed_with_interpolation(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Add positional embeddings with interpolation if the number of tokens differs.

        This is useful if:
        - input resolution changes
        - backbone changes feature map resolution

        We keep:
        - pos_embed[:, 0] for CLS
        and interpolate the patch embeddings to [H, W].
        """
        B, T, D = tokens.shape
        assert T == 1 + H * W, "Token length does not match 1 + H*W."

        # Separate CLS and patch tokens
        cls_tok = tokens[:, :1, :]       # [B, 1, D]
        patch_tok = tokens[:, 1:, :]     # [B, H*W, D]

        # Pos embedding: separate CLS and patch parts from stored pos_embed
        pos_cls = self.pos_embed[:, :1, :]  # [1, 1, D]
        pos_patch = self.pos_embed[:, 1:, :]  # [1, N, D] where N=expected_hw_tokens

        # Infer original grid size from stored embedding length
        N = pos_patch.size(1)
        orig_size = int(N ** 0.5)
        if orig_size * orig_size != N:
            # If not a perfect square, fallback to slicing
            # (this should not happen for standard ResNet50 7x7 output)
            pos = torch.cat([pos_cls, pos_patch[:, :H * W, :]], dim=1)
            return tokens + pos[:, :T, :]

        # Reshape to 2D grid: [1, D, orig, orig]
        pos_patch_2d = pos_patch.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)

        # Interpolate to [H, W]
        pos_patch_2d = torch.nn.functional.interpolate(
            pos_patch_2d,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten back to tokens: [1, H*W, D]
        pos_patch_new = pos_patch_2d.permute(0, 2, 3, 1).reshape(1, H * W, D)

        # Add embeddings
        cls_tok = cls_tok + pos_cls
        patch_tok = patch_tok + pos_patch_new

        return torch.cat([cls_tok, patch_tok], dim=1)


def get_cnn_vit_hybrid(num_classes: int, cfg: Optional[HybridConfig] = None) -> nn.Module:
    """
    Factory function to create the hybrid CNN+ViT model.

    Args:
        num_classes (int): number of classes
        cfg (HybridConfig, optional): model configuration

    Returns:
        nn.Module: CNNViTHybrid instance
    """
    return CNNViTHybrid(num_classes=num_classes, cfg=cfg)


if __name__ == "__main__":
    # Debug run
    model = get_cnn_vit_hybrid(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # expected: [2, 7]
