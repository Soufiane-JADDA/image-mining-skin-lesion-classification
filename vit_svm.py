# vit_svm.py
"""
ViT Embeddings + SVM for HAM10000 classification.

Goal:
- Use a Vision Transformer (ViT) as a feature extractor (no MLP classification head)
- Extract embeddings (typically the CLS token / pooled features)
- Train an SVM classifier on these embeddings
- Evaluate performance (AUC macro, accuracy, confusion matrix optional)

Important notes:
- This is NOT end-to-end training with backprop for the classifier.
- The ViT is used in eval mode to generate fixed embeddings.
- The SVM is trained using scikit-learn.

Expected project files:
- config.py
- datasets.py

Usage example:
    python vit_svm.py --num_classes 7 --kernel linear
    python vit_svm.py --num_classes 7 --kernel rbf --C 3.0 --gamma scale
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

import timm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEVICE, OUTPUT_DIR, BATCH_SIZE_VIT

from datasets import create_dataloaders


# ============================================
#   Embedding Extraction
# ============================================

@torch.no_grad()
def extract_vit_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ViT embeddings for all samples in a dataloader.

    Args:
        model: ViT backbone returning features (no classifier head).
        dataloader: PyTorch DataLoader.
        device: "cuda" or "cpu".

    Returns:
        X: numpy array of embeddings [N, D]
        y: numpy array of labels [N]
    """
    model.eval()
    model.to(device)

    all_feats = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device, non_blocking=True)

        feats = model(images)  # shape usually [B, 768] for ViT-Base
        feats = feats.detach().cpu().numpy()

        all_feats.append(feats)
        all_labels.append(labels.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def build_vit_backbone(model_name: str = "vit_base_patch16_224", pretrained: bool = True) -> torch.nn.Module:
    """
    Build a ViT model that outputs embeddings instead of classification logits.

    In timm:
    - num_classes=0 removes the classifier head and returns features.

    Args:
        model_name: timm model name.
        pretrained: whether to load ImageNet weights.

    Returns:
        ViT backbone (feature extractor).
    """
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return backbone


# ============================================
#   SVM Training & Evaluation
# ============================================

def train_and_evaluate_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    kernel: str = "linear",
    C: float = 1.0,
    gamma: str = "scale",
) -> None:
    """
    Train an SVM classifier on embeddings and evaluate.

    We use:
    - StandardScaler (important for SVM)
    - SVC(probability=True) so we can compute AUC

    Args:
        X_train, y_train: train embeddings and labels.
        X_val, y_val: validation embeddings and labels.
        num_classes: number of target classes.
        kernel: "linear" or "rbf" (others possible).
        C: SVM regularization.
        gamma: kernel coefficient for RBF/poly/sigmoid (ignored for linear).
    """
    # Pipeline: scaling + SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)),
    ])

    print("\nTraining SVM...")
    clf.fit(X_train, y_train)

    print("Evaluating SVM...")
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)  # shape [N, num_classes]

    acc = accuracy_score(y_val, y_pred)

    # AUC macro (multi-class, one-vs-rest)
    lb = LabelBinarizer()
    y_val_bin = lb.fit_transform(y_val)
    auc = roc_auc_score(y_val_bin, y_proba, average="macro", multi_class="ovr")

    print(f"\nSVM Results (kernel={kernel})")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC (macro OVR): {auc:.4f}")

    # Save metrics to a text file
    out_path = os.path.join(OUTPUT_DIR, f"vit_svm_results_{kernel}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"SVM kernel: {kernel}\n")
        f.write(f"C: {C}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"AUC (macro OVR): {auc:.6f}\n")

    print(f"Saved results to: {out_path}")


# ============================================
#   Main
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="ViT Embeddings + SVM for HAM10000")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--vit_model", type=str, default="vit_base_patch16_224", help="timm ViT model name.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ViT.")
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"], help="SVM kernel.")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter.")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM gamma (for rbf).")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build dataloaders (train/val split already handled in datasets.py)
    train_loader, val_loader, _, _ = create_dataloaders(batch_size=BATCH_SIZE_VIT)

    # Build ViT backbone that outputs embeddings (no MLP head)
    vit_backbone = build_vit_backbone(model_name=args.vit_model, pretrained=args.pretrained)

    print(f"Device: {DEVICE}")
    print(f"ViT backbone: {args.vit_model} | pretrained={args.pretrained}")
    print("Extracting train embeddings...")
    X_train, y_train = extract_vit_embeddings(vit_backbone, train_loader, DEVICE)

    print("Extracting validation embeddings...")
    X_val, y_val = extract_vit_embeddings(vit_backbone, val_loader, DEVICE)

    print(f"Train embeddings: {X_train.shape}, Val embeddings: {X_val.shape}")

    # Train and evaluate SVM
    train_and_evaluate_svm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=args.num_classes,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
