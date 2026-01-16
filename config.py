# config.py
"""
Global configuration file for HAM10000 + ViT/ResNet experiments.

This file centralizes:
- Paths (images, labels)
- Data split ratios
- Image preprocessing configuration
- Training hyperparameters
- Device selection (CPU / GPU)
"""

import torch
from pathlib import Path

# =========================
# Paths
# =========================

# Root project directory (you can adapt this)
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory where HAM10000 images are stored
# Example: PROJECT_ROOT / "data" / "HAM10000_images"
IMAGE_DIR = PROJECT_ROOT / "HAM5000/HAM500_images"

# CSV file containing image names and labels
# It should at least contain columns: "image", "label"
LABELS_CSV = PROJECT_ROOT / "HAM5000/HAM500_metadata.csv"


# =========================
# Data split configuration
# =========================

# Train / validation split
TRAIN_RATIO = 0.6   # 60% for training
VAL_RATIO = 0.4     # 40% for validation (you can later re-split into val/test if needed)

# For reproducibility
RANDOM_STATE = 42


# =========================
# Image & preprocessing
# =========================

# Input image size (height, width)
IMAGE_SIZE = 224

# Normalization values (ImageNet statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation parameters (for documentation purposes only)
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "max_rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.0,
    },
    # You can extend this dict if you want to document more transforms
}


# =========================
# DataLoader configuration
# =========================

BATCH_SIZE_VIT = 4
BATCH_SIZE_CNN = 16

NUM_WORKERS = 4      # adapt depending on your CPU
PIN_MEMORY = True    # helpful when using GPU

# Mapping from string labels (HAM10000) to integer class IDs
LABEL_MAPPING = {
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
}


# =========================
# Training hyperparameters
# =========================

# Global default values (you can override per-model in train scripts)
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# For some models (e.g. ViT from scratch) you might want more epochs
EPOCHS_VIT_SCRATCH = 10

# You can define different learning rates for different experiments if needed
LEARNING_RATE_VIT_FINETUNE = 1e-4
LEARNING_RATE_RESNET_FINETUNE = 1e-4
LEARNING_RATE_VGG16_FINETUNE = 1e-4
LEARNING_RATE_HYBRID = 3e-4


# =========================
# Device configuration
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Logging / output
# =========================

# Directory where you will save checkpoints, logs, curves, etc.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File names templates (you can use these in your training scripts)
CHECKPOINT_TEMPLATE = "checkpoint_{model_name}.pth"
HISTORY_TEMPLATE = "history_{model_name}.pt"
FIGURE_LOSS_TEMPLATE = "loss_{model_name}.png"
FIGURE_AUC_TEMPLATE = "auc_{model_name}.png"


if __name__ == "__main__":
    # Simple debug to print config when you run this file directly
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("IMAGE_DIR:", IMAGE_DIR)
    print("LABELS_CSV:", LABELS_CSV)
    print("DEVICE:", DEVICE)
    print("OUTPUT_DIR:", OUTPUT_DIR)
