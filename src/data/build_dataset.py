from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

IMAGES_DIR = RAW / "images"
TRAIN_CSV  = RAW / "train.csv"
META_CSV   = RAW / "metadata.csv"
OUT_CSV    = PROC / "train_merged.csv"

def pick_group_col(df: pd.DataFrame) -> str:
    # prefer patient identifier if it exists
    for c in df.columns:
        cl = c.lower()
        if "patient" in cl or cl in {"mrn", "subject_id"}:
            return c
    return "lesion_id" if "lesion_id" in df.columns else ""

def find_image_path(isic_id: str) -> str:
    # try common extensions
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMAGES_DIR / f"{isic_id}{ext}"
        if p.exists():
            return str(p)
    # default (so you see what's missing)
    return str(IMAGES_DIR / f"{isic_id}.jpg")

def main():
    PROC.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    meta  = pd.read_csv(META_CSV)

    # normalize label column name
    if "malignant" in train.columns:
        train = train.rename(columns={"malignant": "target"})
    if "target" not in train.columns:
        raise ValueError("Expected 'malignant' or 'target' in train.csv")

    df = train.merge(meta, on="isic_id", how="left")

    df["image_path"] = df["isic_id"].apply(find_image_path)

    # sanity check: do first 50 images exist?
    missing = [p for p in df["image_path"].head(50) if not Path(p).exists()]
    if missing:
        print("WARNING: Some image paths not found (check extension or folder):")
        for m in missing[:10]:
            print(" ", m)

    group_col = pick_group_col(df)
    if group_col:
        print(f"Group split column detected: {group_col}")
    else:
        print("No patient/lesion group column detected.")

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  rows={len(df):,}  cols={df.shape[1]}")

if __name__ == "__main__":
    main()
