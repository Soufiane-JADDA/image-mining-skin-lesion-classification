# train.py
"""
Main training script for HAM10000 classification experiments.

This script:
- Loads the HAM10000 dataset and creates DataLoaders
- Builds one of the following models:
    * ResNet50 fine-tuning
    * ViT-Base/16 from scratch
    * ViT-Base/16 fine-tuning
    * Hybrid CNN + ViT model
- Trains the model for a given number of epochs
- Computes Loss and AUC (macro, multi-class) on train and validation sets
- Saves:
    * model checkpoint
    * training history
    * loss and AUC curves as PNG files

Usage example:

    python train.py --model resnet50 --num_classes 7
    python train.py --model vit_scratch --num_classes 7
    python train.py --model vit_finetune --num_classes 7
    python train.py --model hybrid --num_classes 7
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    OUTPUT_DIR,
    EPOCHS,
    EPOCHS_VIT_SCRATCH,
    LEARNING_RATE,
    LEARNING_RATE_VIT_FINETUNE,
    LEARNING_RATE_RESNET_FINETUNE,
    LEARNING_RATE_VGG16_FINETUNE,
    LEARNING_RATE_HYBRID,
    WEIGHT_DECAY,
    CHECKPOINT_TEMPLATE,
    HISTORY_TEMPLATE,
    FIGURE_LOSS_TEMPLATE,
    FIGURE_AUC_TEMPLATE,
    BATCH_SIZE_CNN, BATCH_SIZE_VIT,
)
from datasets import create_dataloaders
from models.models_resnet import get_resnet50_finetune

# These imports will work once you create the corresponding files
try:
    from models.models_vit import (
        get_vit_base_patch16_scratch,
        get_vit_base_patch16_finetune,
    )
except ImportError:
    get_vit_base_patch16_scratch = None
    get_vit_base_patch16_finetune = None

try:
    from models.vgg16 import get_vgg16
except ImportError:
    get_vgg16 = None

try:
    from models.hybrid_cnn_vit import get_cnn_vit_hybrid
except ImportError:
    get_cnn_vit_hybrid = None


def choose_batch_size(model_name: str) -> int:
    """
    Choose batch size depending on the model type.
    """
    model_name = model_name.lower()

    if model_name in ("vit_scratch", "vit_finetune", "hybrid"):
        return BATCH_SIZE_VIT
    else:
        # resnet50, vgg16
        return BATCH_SIZE_CNN



# ============================================
#   Training & Evaluation Helpers
# ============================================

from torch.cuda.amp import autocast, GradScaler  # make sure this import exists

from torch.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Train the model for one epoch.

    Returns:
        avg_loss (float): mean training loss over the epoch
        auc (float): macro AUC (multi-class, one-vs-rest)
    """
    model.train()
    scaler = GradScaler("cuda")

    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward + loss
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward + optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    lb = LabelBinarizer()
    try:
        auc = roc_auc_score(
            lb.fit_transform(all_labels),
            all_probs,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        auc = float("nan")

    return avg_loss, auc




def eval_one_epoch(model, loader, device, criterion):
    """
    Evaluate the model on a given DataLoader.

    Returns:
        avg_loss (float): mean validation loss
        auc (float): macro AUC (multi-class, one-vs-rest)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
 
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)


            total_loss += loss.item() * images.size(0)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(all_labels)
    try:
        auc = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return avg_loss, auc


def plot_curves(history: Dict[str, Any], model_name: str, output_dir: str):
    """
    Plot loss and AUC curves and save them as PNG files.

    Args:
        history (dict): dictionary with keys:
            - train_loss
            - val_loss
            - train_auc
            - val_auc
        model_name (str): name of the model (used in file names)
        output_dir (str): directory where PNG files will be saved
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # ----- Loss curve -----
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    loss_path = os.path.join(output_dir, FIGURE_LOSS_TEMPLATE.format(model_name=model_name))
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # ----- AUC curve -----
    plt.figure()
    plt.plot(epochs, history["train_auc"], label="Train AUC")
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC (macro)")
    plt.title(f"{model_name} - AUC")
    plt.legend()
    auc_path = os.path.join(output_dir, FIGURE_AUC_TEMPLATE.format(model_name=model_name))
    plt.savefig(auc_path, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve to: {loss_path}")
    print(f"Saved AUC curve to:  {auc_path}")


# ============================================
#   Model Factory
# ============================================

def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Build the requested model based on its name.

    Supported models:
        - 'resnet50'
        - 'vit_scratch'
        - 'vit_finetune'
        - 'hybrid'
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = get_resnet50_finetune(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=False,
        )

    elif model_name == "vit_scratch":
        if get_vit_base_patch16_scratch is None:
            raise ImportError("models_vit.py with get_vit_base_patch16_scratch is missing.")
        model = get_vit_base_patch16_scratch(num_classes=num_classes)

    elif model_name == "vit_finetune":
        if get_vit_base_patch16_finetune is None:
            raise ImportError("models_vit.py with get_vit_base_patch16_finetune is missing.")
        model = get_vit_base_patch16_finetune(num_classes=num_classes)

    elif model_name == "hybrid":
        if get_cnn_vit_hybrid is None:
            raise ImportError("models_hybrid.py with get_cnn_vit_hybrid is missing.")
        model = get_cnn_vit_hybrid(num_classes=num_classes)

    elif model_name == "vgg16":
        model = get_vgg16(
            num_classes=num_classes,
            pretrained=True,
            trainable="all",   # or "classifier" / "features5"
            dropout=0.5
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def choose_hyperparams(model_name: str, epochs: int = None, lr: float = None):
    """
    Choose default hyperparameters depending on the model type,
    with possibility to override from CLI.
    """
    model_name = model_name.lower()

    # Default epochs
    if epochs is None:
        if model_name == "vit_scratch":
            epochs = EPOCHS_VIT_SCRATCH
        else:
            epochs = EPOCHS

    # Default learning rate
    if lr is None:
        if model_name == "resnet50":
            lr = LEARNING_RATE_RESNET_FINETUNE
        elif model_name in ("vit_scratch", "vit_finetune"):
            lr = LEARNING_RATE_VIT_FINETUNE
        elif model_name == "hybrid":
            lr = LEARNING_RATE_HYBRID
        elif model_name == "vgg16":
            lr = LEARNING_RATE_VGG16_FINETUNE
        else:
            lr = LEARNING_RATE

    return epochs, lr


# ============================================
#   Main Training Function
# ============================================

def run_training(
    model_name: str,
    num_classes: int,
    epochs: int = None,
    lr: float = None,
    weight_decay: float = WEIGHT_DECAY,
):
    """
    High-level function that:
    - Creates DataLoaders
    - Builds the model
    - Trains and evaluates per epoch
    - Saves checkpoint, history, and curves
    """
    # Create dataloaders
    batch_size = choose_batch_size(model_name)
    print(f"Using batch size: {batch_size}")

    train_loader, val_loader, train_df, val_df = create_dataloaders(batch_size=batch_size) 
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Build model
    model = build_model(model_name, num_classes=num_classes)
    model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(model)

    # Select hyperparameters
    epochs, lr = choose_hyperparams(model_name, epochs=epochs, lr=lr)
    print(f"Training for {epochs} epochs with lr={lr}, weight_decay={weight_decay}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # History dict
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
    }

    best_val_auc = -float("inf")
    best_model_state = None

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, DEVICE, criterion)
        val_loss, val_auc = eval_one_epoch(model, val_loader, DEVICE, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Track best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()

    # Save best checkpoint
    model_name_clean = model_name.lower()
    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        CHECKPOINT_TEMPLATE.format(model_name=model_name_clean),
    )
    if best_model_state is not None:
        torch.save(best_model_state, checkpoint_path)
        print(f"Saved best model checkpoint to: {checkpoint_path}")
    else:
        print("Warning: best_model_state is None, no checkpoint saved.")

    # Save history
    history_path = os.path.join(
        OUTPUT_DIR,
        HISTORY_TEMPLATE.format(model_name=model_name_clean),
    )
    torch.save(history, history_path)
    print(f"Saved training history to: {history_path}")

    # Plot curves
    plot_curves(history, model_name_clean, OUTPUT_DIR)


# ============================================
#   CLI Interface
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train HAM10000 models (ResNet / ViT / Hybrid).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "vgg16", "vit_scratch", "vit_finetune", "hybrid"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of classes in the HAM10000 classification task.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (if None, use default per model).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (if None, use default per model).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Weight decay (L2 regularization), default={WEIGHT_DECAY}.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        model_name=args.model,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
