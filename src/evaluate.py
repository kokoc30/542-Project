# src/evaluate.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
)
from tqdm import tqdm
from PIL import Image

from dataloader import SkinDataset
from train import SkinModel


# ─────────────────────────────────────────────
# TEST TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
class TTADataset(Dataset):
    def __init__(self, dataframe, img_size=256):
        self.df = dataframe.reset_index(drop=True)
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), norm,
            ]),
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(), norm,
            ]),
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(), norm,
            ]),
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(), norm,
            ]),
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=(15, 15)),
                transforms.ToTensor(), norm,
            ]),
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(row["target"], dtype=torch.float32)
        images = torch.stack([t(img) for t in self.tta_transforms])
        return images, label


def predict_with_tta(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images_batch, labels in tqdm(loader, desc="TTA Predicting"):
            batch_size, n_tta, C, H, W = images_batch.shape
            preds_sum = torch.zeros(batch_size)
            for i in range(n_tta):
                imgs = images_batch[:, i].to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(imgs)
                preds_sum += torch.sigmoid(outputs).cpu()
            avg_preds = preds_sum / n_tta
            all_preds.extend(avg_preds.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# THRESHOLD HELPERS
# ─────────────────────────────────────────────
def find_best_f1(y_true, y_prob):
    """Find threshold that maximizes F1 Score."""
    best_f1, best_thr = 0.0, 0.5
    best_prec, best_rec = 0.0, 0.0

    for t in np.linspace(0.001, 0.999, 500):
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
            best_prec, best_rec = prec, rec

    return best_f1, best_thr, best_prec, best_rec


def target_recall_threshold(y_true, y_prob, target_recall=0.90):
    thresholds = np.linspace(0.99, 0.001, 500)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        rec = tp / (tp + fn + 1e-9)
        if rec >= target_recall:
            return t, rec
    return thresholds[-1], 1.0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default="auto",
                        help="float, 'auto' (best F1), or 'recall90'")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b1",
                                 "efficientnet_b2", "convnext_tiny"])
    parser.add_argument("--experiment", type=str, default="v1")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Experiment: {args.experiment}")
    print(f"[INFO] TTA: {'enabled (5 views)' if args.tta else 'disabled'}")

    ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = ROOT / "checkpoints" / args.experiment / "best_model.pth"
    FIG_DIR = ROOT / "reports" / "figures" / args.experiment
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # ── 1. Load model ──
    model = SkinModel(model_name=args.model, weights="none", dropout=0.4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded: {MODEL_PATH}")

    # ── 2. Load val split ──
    val_csv = ROOT / "data" / "processed" / "val_split.csv"
    val_df = pd.read_csv(val_csv, low_memory=False)
    y_true = val_df["target"].astype(int).values
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    print(f"[INFO] Val: {len(val_df):,} total | {n_neg:,} benign | {n_pos:,} malignant")

    # ── 3. Get predictions ──
    if args.tta:
        tta_dataset = TTADataset(val_df, img_size=args.img_size)
        tta_loader = DataLoader(tta_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        y_prob, y_true = predict_with_tta(model, tta_loader, device)
    else:
        val_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_dataset = SkinDataset(val_df, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        all_preds = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Predicting"):
                images = images.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(images)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
        y_prob = np.array(all_preds)

    # ── 4. Core Metrics ──
    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    f1_score, f1_thr, f1_prec, f1_rec = find_best_f1(y_true, y_prob)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {args.experiment}")
    print(f"{'='*60}")
    print(f"")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  PR-AUC:   {ap:.4f}")
    if args.tta:
        print(f"  TTA:      5-view average")
    print(f"")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  F1 SCORE:  {f1_score:.4f}                          │")
    print(f"  │  (at threshold = {f1_thr:.3f})                   │")
    print(f"  │                                             │")
    print(f"  │  Precision: {f1_prec:.4f}  (correct when it      │")
    print(f"  │                          says 'cancer')     │")
    print(f"  │  Recall:    {f1_rec:.4f}  (cancers it found)     │")
    print(f"  │                                             │")
    print(f"  │  F1 = 2 × Precision × Recall                │")
    print(f"  │       ─────────────────────                 │")
    print(f"  │         Precision + Recall                  │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"{'='*60}")

    # ── 5. Threshold selection ──
    thr_str = str(args.threshold).lower()
    if thr_str == "auto":
        thr = f1_thr
        print(f"\n[INFO] Using best F1 threshold: {thr:.3f} (F1={f1_score:.4f})")
    elif thr_str.startswith("recall"):
        target = int(thr_str.replace("recall", "")) / 100.0
        thr, actual_rec = target_recall_threshold(y_true, y_prob, target_recall=target)
        print(f"\n[INFO] Target recall={target:.0%}: thr={thr:.3f}, actual recall={actual_rec:.4f}")
    else:
        thr = float(args.threshold)

    y_pred = (y_prob >= thr).astype(int)

    # ── 6. Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1_at_thr = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)

    print(f"\n  Confusion Matrix (threshold={thr:.3f}):")
    print(f"  ┌────────────┬──────────────┬──────────────┐")
    print(f"  │            │ Pred Benign  │ Pred Malig.  │")
    print(f"  ├────────────┼──────────────┼──────────────┤")
    print(f"  │ True Benign│ TN={tn:>8,} │ FP={fp:>8,} │")
    print(f"  │ True Malig.│ FN={fn:>8,} │ TP={tp:>8,} │")
    print(f"  └────────────┴──────────────┴──────────────┘")
    print(f"")
    print(f"  Sensitivity (Recall):  {sensitivity:.4f}  ({tp}/{tp+fn} malignant caught)")
    print(f"  Specificity:           {specificity:.4f}  ({tn}/{tn+fp} benign correct)")
    print(f"  Precision:             {precision:.4f}  ({tp}/{tp+fp} predicted malig. correct)")
    print(f"  F1 Score:              {f1_at_thr:.4f}")

    # ── 7. ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUROC={auroc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {args.experiment}" + (" (TTA)" if args.tta else ""))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {FIG_DIR / 'roc_curve.png'}")
    plt.show()

    # ── 8. PR Curve ──
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(rec_arr, prec_arr, lw=2, label=f"PR (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {args.experiment}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {FIG_DIR / 'pr_curve.png'}")
    plt.show()

    # ── 9. Confusion Matrix Plot ──
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (thr={thr:.3f}, F1={f1_at_thr:.4f})")
    plt.colorbar()
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {FIG_DIR / 'confusion_matrix.png'}")
    plt.show()

    # ── 10. Classification Report ──
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Malignant"], digits=4))

    # ── 11. Summary ──
    print(f"{'='*60}")
    print(f"  FINAL SUMMARY — {args.experiment}")
    print(f"{'='*60}")
    print(f"  AUROC:          {auroc:.4f}")
    print(f"  PR-AUC:         {ap:.4f}")
    print(f"  F1 Score:       {f1_score:.4f}")
    print(f"  Cancers caught: {tp}/{tp+fn} ({sensitivity:.1%})")
    print(f"  False alarms:   {fp}/{tn+fp} ({fp/(tn+fp)*100:.2f}%)")
    print(f"  All plots → {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()