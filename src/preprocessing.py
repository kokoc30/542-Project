# src/preprocessing.py
"""
ISIC 2024 — Preprocessing & EDA. Run this before training.
Merges CSVs, checks class balance, scans images, saves train_merged.csv.
Usage: python src/preprocessing.py [--experiment v4] [--no_show]
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = RAW_DIR / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_no_show = False  # set by --no_show flag in main()

sns.set_style("whitegrid")


# --- STEP 1: Load & Merge ---
def load_and_merge():
    print("=" * 60)
    print("  STEP 1: Loading & Merging CSVs")
    print("=" * 60)

    train_csv = RAW_DIR / "train.csv"
    meta_csv = RAW_DIR / "metadata.csv"

    if not train_csv.exists():
        print(f"  ERROR: {train_csv} not found!")
        sys.exit(1)
    if not meta_csv.exists():
        print(f"  ERROR: {meta_csv} not found!")
        sys.exit(1)

    train = pd.read_csv(train_csv, low_memory=False)
    meta = pd.read_csv(meta_csv, low_memory=False)

    print(f"  train.csv:    {train.shape[0]:,} rows, {train.shape[1]} cols")
    print(f"  metadata.csv: {meta.shape[0]:,} rows, {meta.shape[1]} cols")

    # normalize label column
    if "malignant" in train.columns:
        train = train.rename(columns={"malignant": "target"})
    if "target" not in train.columns:
        print("  ERROR: No 'target' or 'malignant' column found!")
        sys.exit(1)

    # merge
    df = train.merge(meta, on="isic_id", how="left")
    df["image_path"] = df["isic_id"].apply(lambda x: str(IMAGE_DIR / f"{x}.jpg"))

    print(f"  Merged: {df.shape[0]:,} rows, {df.shape[1]} cols")
    print(f"  Columns: {df.columns.tolist()}")

    return df


# --- STEP 2: Class Distribution ---
def analyze_class_distribution(df):
    print("\n" + "=" * 60)
    print("  STEP 2: Class Distribution Analysis")
    print("=" * 60)

    counts = df["target"].value_counts().sort_index()
    n_benign = int(counts.get(0, 0))
    n_malignant = int(counts.get(1, 0))
    total = len(df)

    print(f"  Benign (0):    {n_benign:>8,}  ({n_benign/total*100:.2f}%)")
    print(f"  Malignant (1): {n_malignant:>8,}  ({n_malignant/total*100:.2f}%)")
    print(f"  Imbalance ratio: 1 malignant per {n_benign // max(n_malignant, 1)} benign")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Benign (0)", "Malignant (1)"]

    axes[0].bar(labels, [n_benign, n_malignant], color=colors)
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([n_benign, n_malignant]):
        axes[0].text(i, v + total * 0.01, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie([n_benign, n_malignant], labels=["Benign", "Malignant"],
                autopct="%1.2f%%", colors=colors, startangle=90,
                explode=(0, 0.1))
    axes[1].set_title("Class Distribution (%)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  Saved -> {FIGURES_DIR / 'class_distribution.png'}")
    if not _no_show:
        plt.show()

    ratio = n_benign / max(n_malignant, 1)
    summary_path = FIGURES_DIR / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total images: {total}\n")
        f.write(f"Malignant: {n_malignant}  ({n_malignant/total*100:.1f}%)\n")
        f.write(f"Benign:    {n_benign}  ({n_benign/total*100:.1f}%)\n")
        f.write(f"Imbalance ratio (neg/pos): {ratio:.2f}\n")
    print(f"  Saved -> {summary_path}")


# --- STEP 3: Metadata Exploration ---
def analyze_metadata(df):
    print("\n" + "=" * 60)
    print("  STEP 3: Metadata Exploration")
    print("=" * 60)

    # missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        print("\n  Columns with missing values:")
        for col, pct in missing_pct.head(15).items():
            print(f"    {col}: {pct:.1f}%")
    else:
        print("\n  No missing values!")

    # age distribution by class
    age_col = None
    for candidate in ["age_approx", "age"]:
        if candidate in df.columns:
            age_col = candidate
            break

    if age_col:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[df["target"] == 0][age_col].hist(
            bins=30, alpha=0.6, label="Benign", color="#2ecc71", ax=ax, density=True
        )
        df[df["target"] == 1][age_col].hist(
            bins=30, alpha=0.6, label="Malignant", color="#e74c3c", ax=ax, density=True
        )
        ax.set_title("Age Distribution by Class")
        ax.set_xlabel("Age")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved -> {FIGURES_DIR / 'age_distribution.png'}")
        if not _no_show:
            plt.show()

    # categorical features vs target
    cat_candidates = [
        "sex", "anatom_site_general", "anatom_site_general_challenge",
        "tbp_lv_location_simple",
    ]
    found_cats = [c for c in cat_candidates if c in df.columns and df[c].nunique() <= 15]

    if found_cats:
        n_plots = min(len(found_cats), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            col = found_cats[i]
            ct = pd.crosstab(df[col], df["target"], normalize="index")
            ct.plot(kind="barh", stacked=True, ax=axes[i], color=["#2ecc71", "#e74c3c"])
            axes[i].set_title(f"{col}")
            axes[i].legend(["Benign", "Malignant"], loc="lower right", fontsize=8)

        plt.suptitle("Malignancy Rate by Metadata Group", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "metadata_by_target.png", dpi=150, bbox_inches="tight")
        print(f"  Saved -> {FIGURES_DIR / 'metadata_by_target.png'}")
        if not _no_show:
            plt.show()


# --- STEP 4: Image Quality Check ---
def check_images(df, sample_n=1000):
    print("\n" + "=" * 60)
    print("  STEP 4: Image Quality Check")
    print("=" * 60)

    # check for missing images
    print("\n  Checking for missing images...")
    missing_mask = ~df["image_path"].apply(os.path.exists)
    n_missing = missing_mask.sum()
    print(f"  Missing images: {n_missing:,} / {len(df):,}")

    if n_missing > 0:
        print(f"  WARNING: {n_missing} images not found in {IMAGE_DIR}")

    # sample images to check sizes and detect corrupt files
    print(f"\n  Sampling {sample_n} images for size/quality analysis...")
    sample_df = df[~missing_mask].sample(n=min(sample_n, (~missing_mask).sum()), random_state=42)

    widths, heights = [], []
    corrupt_files = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="  Checking"):
        try:
            img = Image.open(row["image_path"])
            img.verify()    # checks for corruption
            img = Image.open(row["image_path"])     # reopen after verify
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception as e:
            corrupt_files.append(row["image_path"])

    if corrupt_files:
        print(f"\n  WARNING: {len(corrupt_files)} corrupt images found!")
        for f in corrupt_files[:5]:
            print(f"    {f}")
    else:
        print(f"\n  All {len(sample_df)} sampled images are valid.")

    if widths:
        print(f"\n  Image dimensions (sample of {len(widths)}):")
        print(f"    Width  — min: {min(widths)}, max: {max(widths)}, median: {np.median(widths):.0f}")
        print(f"    Height — min: {min(heights)}, max: {max(heights)}, median: {np.median(heights):.0f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(widths, bins=30, color="steelblue", edgecolor="white")
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width (px)")

        axes[1].hist(heights, bins=30, color="coral", edgecolor="white")
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height (px)")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "image_sizes.png", dpi=150, bbox_inches="tight")
        print(f"  Saved -> {FIGURES_DIR / 'image_sizes.png'}")
        if not _no_show:
            plt.show()

    return missing_mask, corrupt_files


# --- STEP 5: Sample Images ---
def show_sample_images(df, n=8):
    print("\n" + "=" * 60)
    print("  STEP 5: Sample Images")
    print("=" * 60)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6))

    for row_idx, (label, title, color) in enumerate(
        [(0, "BENIGN", "#2ecc71"), (1, "MALIGNANT", "#e74c3c")]
    ):
        subset = df[df["target"] == label]
        if len(subset) == 0:
            continue
        samples = subset.sample(n=min(n, len(subset)), random_state=42)

        for col_idx, (_, row) in enumerate(samples.iterrows()):
            try:
                img = Image.open(row["image_path"])
                axes[row_idx, col_idx].imshow(img)
            except Exception:
                axes[row_idx, col_idx].text(0.5, 0.5, "ERROR", ha="center")
            axes[row_idx, col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=12,
                                                  fontweight="bold", color=color)

    plt.suptitle("Sample Lesion Images", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
    print(f"  Saved -> {FIGURES_DIR / 'sample_images.png'}")
    if not _no_show:
        plt.show()


# --- STEP 6: Clean & Save ---
def clean_and_save(df, missing_mask, corrupt_files):
    print("\n" + "=" * 60)
    print("  STEP 6: Cleaning & Saving")
    print("=" * 60)

    original_len = len(df)

    # remove rows with missing images
    df = df[~missing_mask].copy()

    # remove corrupt images (if any found in sample)
    if corrupt_files:
        df = df[~df["image_path"].isin(corrupt_files)].copy()

    removed = original_len - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows (missing/corrupt images)")

    print(f"  Final dataset: {len(df):,} rows")
    print(f"    Benign:    {int((df['target'] == 0).sum()):,}")
    print(f"    Malignant: {int((df['target'] == 1).sum()):,}")

    out_path = PROCESSED_DIR / "train_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    return df


# --- STEP 7: Summary ---
def print_summary(df):
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE — SUMMARY")
    print("=" * 60)

    n_total = len(df)
    n_mal = int(df["target"].sum())
    n_ben = n_total - n_mal

    print(f"""
    Total samples:     {n_total:,}
    Benign:            {n_ben:,} ({n_ben/n_total*100:.2f}%)
    Malignant:         {n_mal:,} ({n_mal/n_total*100:.2f}%)
    Imbalance ratio:   1:{n_ben // max(n_mal, 1)}
    Columns:           {df.shape[1]}

    Saved files:
      data/processed/train_merged.csv
      {FIGURES_DIR}/class_distribution.png
      {FIGURES_DIR}/age_distribution.png
      {FIGURES_DIR}/metadata_by_target.png
      {FIGURES_DIR}/image_sizes.png
      {FIGURES_DIR}/sample_images.png
      {FIGURES_DIR}/dataset_summary.txt

    Next step:
      python src/train.py --experiment {FIGURES_DIR.name} --epochs 15 --fine_tune --fine_tune_epochs 10 --batch_size 128 --num_workers 8
    """)


# --- MAIN ---
def main():
    global FIGURES_DIR, _no_show

    parser = argparse.ArgumentParser(description="ISIC 2024 — Preprocessing & EDA")
    parser.add_argument("--experiment", type=str, default="v4",
                        help="Experiment name; outputs go to reports/figures/<experiment>/")
    parser.add_argument("--no_show", action="store_true",
                        help="Skip plt.show() (useful for headless/server runs)")
    args = parser.parse_args()

    _no_show = args.no_show
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / args.experiment
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + " ISIC 2024 — Skin Lesion Preprocessing ".center(60, "="))
    print()

    df = load_and_merge()
    analyze_class_distribution(df)
    analyze_metadata(df)
    missing_mask, corrupt_files = check_images(df)
    show_sample_images(df)
    df = clean_and_save(df, missing_mask, corrupt_files)
    print_summary(df)


if __name__ == "__main__":
    main()