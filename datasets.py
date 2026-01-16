# datasets.py
"""
Dataset and DataLoader utilities for the HAM10000 experiments.

This file contains:
- HAMDataset class (PyTorch Dataset)
- Data augmentation & preprocessing pipelines
- Train/Validation split
- Functions to create DataLoaders

It is adapted to a labels.csv file with the following columns:
    lesion_id ; image_id ; label

We do NOT modify the CSV file. Instead, we:
- Read it with a ';' separator
- Use 'image_id' as the image filename column
- Use 'label' as the class column
"""

import os 
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    IMAGE_DIR,
    LABELS_CSV,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_STATE,
    BATCH_SIZE_CNN,
    BATCH_SIZE_VIT,
    NUM_WORKERS,
    PIN_MEMORY,
    LABEL_MAPPING
)


# ============================================
#   HAM Dataset Class
# ============================================

class HAMDataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion dataset.

    This version expects a DataFrame with at least:
    - image_id : filename of the lesion image (without path, possibly without extension)
    - label    : class ID (integer or string)

    If label is a string and not already numeric, you should
    map it to integers BEFORE creating this dataset or adapt
    the code to include a mapping.
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def _resolve_image_path(self, img_name: str) -> str:
        """
        Resolve the true image path, trying common extensions if needed.

        - If img_name already has an extension and exists, return it.
        - If not, try adding .jpg, .jpeg, .png until a file is found.
        - If nothing is found, raise a clear error.
        """
        import os

        # Direct path (maybe img_name already has extension)
        direct_path = os.path.join(self.img_dir, img_name)
        if os.path.exists(direct_path):
            return direct_path

        # If there is no extension, try common ones
        root, ext = os.path.splitext(img_name)
        if ext == "":
            for candidate_ext in [".jpg", ".jpeg", ".png"]:
                candidate_name = root + candidate_ext
                candidate_path = os.path.join(self.img_dir, candidate_name)
                if os.path.exists(candidate_path):
                    return candidate_path

        # If everything fails, raise an explicit error
        raise FileNotFoundError(
            f"Could not find image for '{img_name}' in directory '{self.img_dir}'. "
            f"Tried with extensions: '', '.jpg', '.jpeg', '.png'."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Use 'image_id' column from CSV as base name
        img_name = str(row["image_id"])
        img_path = self._resolve_image_path(img_name)

        # Open image and convert to RGB
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        # ----- Label handling -----
        raw_label = row["label"]

        # If label is a string (e.g. 'mel', 'nv', ...), map it to an integer
        if isinstance(raw_label, str):
            if raw_label not in LABEL_MAPPING:
                raise ValueError(
                    f"Unknown label '{raw_label}' encountered. "
                    f"Expected one of: {list(LABEL_MAPPING.keys())}"
                )
            label = LABEL_MAPPING[raw_label]
        else:
            # If it's already numeric, just convert to int
            label = int(raw_label)

        # Apply transforms (augmentation + normalization)
        if self.transform:
            image = self.transform(image)

        return image, label




# ============================================
#   Transforms (Augmentation + Normalization)
# ============================================

def get_train_transform():
    """Return the augmentation + preprocessing pipeline for training images."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    """Return the preprocessing pipeline for validation/test images."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================
#   DataLoader Builder
# ============================================

def create_dataloaders(batch_size: int):
    """
    Create train and validation DataLoaders.

    Steps:
    1. Load CSV with ';' separator
    2. Train/validation split with stratification on 'label'
    3. Apply transforms
    4. Build DataLoaders
    """

    # Load the CSV file with semicolon separator
    df = pd.read_csv(LABELS_CSV, sep=";")

    # Safety check: ensure required columns exist
    required_cols = {"image_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labels file: {missing}. "
                         f"Found columns: {list(df.columns)}")

    # Stratified split to preserve class distribution
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        train_size=TRAIN_RATIO,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # Create datasets
    train_dataset = HAMDataset(
        train_df, IMAGE_DIR, transform=get_train_transform()
    )
    val_dataset = HAMDataset(
        val_df, IMAGE_DIR, transform=get_val_transform()
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, train_df, val_df


# ============================================
#   Debug Run
# ============================================

if __name__ == "__main__":
    print("Building dataloaders...")
    train_loader, val_loader, train_df, val_df = create_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("Train label distribution:")
    print(train_df["label"].value_counts())
    print("Val label distribution:")
    print(val_df["label"].value_counts())
