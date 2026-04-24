# src/dataloader.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


class SkinDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["target"], dtype=torch.float32)
        return img, label


def _safe_read_csv(csv_path):
    return pd.read_csv(csv_path, low_memory=False)


def _subset_with_min_positives(df, max_rows, seed, min_pos=50):
    if max_rows is None or len(df) <= max_rows:
        return df

    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]

    pos_keep = min(len(pos), min_pos)
    neg_keep = max(0, min(len(neg), max_rows - pos_keep))

    pos_s = pos.sample(n=pos_keep, random_state=seed) if pos_keep > 0 else pos
    neg_s = neg.sample(n=neg_keep, random_state=seed) if neg_keep > 0 else neg

    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def _group_stratified_split(df, val_split, seed, group_col):
    y = df["target"].astype(int).values
    groups = df[group_col].astype(str).fillna("NA").values

    n_splits = max(2, int(round(1.0 / val_split)))

    class_counts = np.bincount(y)
    min_class = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if min_class < n_splits:
        return None

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_split = None
    best_diff = float("inf")

    for train_idx, val_idx in sgkf.split(df, y, groups):
        actual_val_frac = len(val_idx) / len(df)
        diff = abs(actual_val_frac - val_split)
        if diff < best_diff:
            best_diff = diff
            best_split = (train_idx, val_idx)

    if best_split is None:
        return None

    train_idx, val_idx = best_split
    actual_val_frac = len(val_idx) / len(df)

    if actual_val_frac < 0.1 or actual_val_frac > 0.5:
        print(f"[WARN] Group split gave {actual_val_frac:.1%} val — too skewed, falling back.")
        return None

    print(f"[INFO] Group split val fraction: {actual_val_frac:.1%} (target: {val_split:.1%})")
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _make_group_id(df, lesion_col="lesion_id"):
    lesion = df[lesion_col].astype(str)
    is_missing = (
        df[lesion_col].isna()
        | lesion.str.lower().isin(["nan", "none", ""])
    )

    # fall back to isic_id (unique per image) or row index
    if "isic_id" in df.columns:
        fallback = df["isic_id"].astype(str)
    else:
        fallback = df.index.astype(str)

    group_id = np.where(is_missing, fallback, lesion)

    n_real = (~is_missing).sum()
    n_fallback = is_missing.sum()
    print(f"[INFO] Group IDs: {n_real:,} from {lesion_col}, {n_fallback:,} from fallback (isic_id)")

    return group_id


def get_train_transforms(img_size):
    """Stronger augmentations for better generalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),     # cutout-style
    ])


def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_datasets(
    csv_path,
    img_size=224,
    batch_size=32,
    val_split=0.2,
    seed=42,
    num_workers=0,
    max_rows=None,
    group_col="lesion_id",
    save_val_csv=True,
):
    csv_path = Path(csv_path)
    df = _safe_read_csv(csv_path)

    if "target" not in df.columns or "image_path" not in df.columns:
        raise ValueError("CSV must contain columns: target, image_path")

    df = df.dropna(subset=["image_path", "target"]).copy()
    df["target"] = df["target"].astype(int)

    print(f"Total rows in CSV: {len(df):,}")
    print(f"Malignant: {int(df['target'].sum()):,} ({df['target'].mean() * 100:.2f}%)")
    print(f"Benign:    {int((1 - df['target']).sum()):,}")

    if max_rows is not None:
        df = _subset_with_min_positives(df, int(max_rows), seed=seed, min_pos=50)
        print(f"[INFO] Using subset: {len(df):,} rows (max_rows={max_rows})")
        print(f"[INFO] Subset malignant: {int(df['target'].sum()):,}")

    # build a clean group_id: use lesion_id when it exists, otherwise fall back
    # to isic_id (unique per image) so NaN lesion_ids don't form one giant group
    train_df = val_df = None
    if group_col in df.columns:
        df["group_id"] = _make_group_id(df, group_col)
        split = _group_stratified_split(df, val_split=val_split, seed=seed, group_col="group_id")
        if split is not None:
            train_df, val_df = split

    if train_df is None:
        print("[INFO] Using stratified random split (no group leakage prevention)")
        train_df, val_df = train_test_split(
            df, test_size=val_split, stratify=df["target"], random_state=seed
        )

    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(f"  Train malignant: {int(train_df['target'].sum()):,}")
    print(f"  Val malignant:   {int(val_df['target'].sum()):,}")

    if save_val_csv:
        out_path = csv_path.parent / "val_split.csv"
        val_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved val split -> {out_path}")

    # datasets
    train_dataset = SkinDataset(train_df, transform=get_train_transforms(img_size))
    val_dataset = SkinDataset(val_df, transform=get_val_transforms(img_size))

    # weighted sampler
    labels = train_df["target"].values.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # persistent_workers keeps workers alive between epochs (faster)
    persist = num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persist,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persist,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, train_df, val_df