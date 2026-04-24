# src/train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from dataloader import load_datasets


# --- F1 SCORE HELPER ---
def compute_best_f1(y_true, y_prob):
    """Scan thresholds and return the one that gives the best F1 score."""
    best_f1, best_thr = 0.0, 0.5
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for t in np.linspace(0.01, 0.99, 200):
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return best_f1, best_thr


# --- MODEL ---
class SkinModel(nn.Module):
    """Pretrained CNN backbone + dropout + 1 output node for binary classification."""

    def __init__(self, model_name="efficientnet_b0", weights="imagenet", dropout=0.4):
        super().__init__()
        self.model_name = model_name

        if HAS_TIMM and weights != "none":
            print(f"[INFO] Using {model_name} (pretrained on ImageNet)")
            self.base = timm.create_model(model_name, pretrained=True, num_classes=0)
            feat_dim = self.base.num_features
        elif HAS_TIMM:
            print(f"[INFO] Using {model_name} (random weights)")
            self.base = timm.create_model(model_name, pretrained=False, num_classes=0)
            feat_dim = self.base.num_features
        else:
            print("[WARN] timm not installed — using simple CNN")
            self.base = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            feat_dim = 128

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        print("[INFO] Base frozen (only head is trainable)")

    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True
        print("[INFO] Base unfrozen (full fine-tuning)")

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)


# --- TRAINING / VALIDATION ---
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    epoch_loss = running_loss / len(loader.dataset)
    try:
        auroc = roc_auc_score(all_labels, all_preds)
        prauc = average_precision_score(all_labels, all_preds)
    except ValueError:
        auroc, prauc = 0.0, 0.0

    best_f1, _ = compute_best_f1(all_labels, all_preds)

    return epoch_loss, auroc, prauc, best_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        auroc = roc_auc_score(all_labels, all_preds)
        prauc = average_precision_score(all_labels, all_preds)
    except ValueError:
        auroc, prauc = 0.0, 0.0

    best_f1, _ = compute_best_f1(all_labels, all_preds)

    return epoch_loss, auroc, prauc, best_f1


# --- TRAINING PHASE ---
def run_training_phase(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, scaler, epochs, ckpt_path, phase_name="Training",
    patience=5, scheduler_type="plateau",
):
    history = {"loss": [], "auroc": [], "prauc": [], "f1": [],
               "val_loss": [], "val_auroc": [], "val_prauc": [], "val_f1": [],
               "lr": []}
    best_prauc = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"  {phase_name}: {epochs} epochs")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        start = time.time()

        t_loss, t_auroc, t_prauc, t_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        v_loss, v_auroc, v_prauc, v_f1 = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(t_loss)
        history["auroc"].append(t_auroc)
        history["prauc"].append(t_prauc)
        history["f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_auroc"].append(v_auroc)
        history["val_prauc"].append(v_prauc)
        history["val_f1"].append(v_f1)
        history["lr"].append(lr)

        print(f"Epoch {epoch}/{epochs} ({elapsed:.0f}s) lr={lr:.1e}  "
              f"loss={t_loss:.4f} auroc={t_auroc:.4f} prauc={t_prauc:.4f} f1={t_f1:.4f}  "
              f"val_loss={v_loss:.4f} val_auroc={v_auroc:.4f} val_prauc={v_prauc:.4f} val_f1={v_f1:.4f}",
              end="")

        if v_prauc > best_prauc:
            best_prauc = v_prauc
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print(f"  ★ BEST (saved)")
        else:
            patience_counter += 1
            print()

        old_lr = optimizer.param_groups[0]["lr"]
        if scheduler_type == "plateau":
            scheduler.step(v_prauc)
        else:
            scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  [INFO] LR reduced: {old_lr:.1e} → {new_lr:.1e}")

        if patience_counter >= patience:
            print(f"[INFO] Early stopping after {patience} epochs without improvement.")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            break

    return history


# --- PLOTTING ---
def plot_history(history, out_path, no_show=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0, 0].plot(epochs, history["loss"], marker="o", markersize=3, label="train")
    axes[0, 0].plot(epochs, history["val_loss"], marker="o", markersize=3, label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["auroc"], marker="o", markersize=3, label="train")
    axes[0, 1].plot(epochs, history["val_auroc"], marker="o", markersize=3, label="val")
    axes[0, 1].set_title("AUROC")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["prauc"], marker="o", markersize=3, label="train")
    axes[1, 0].plot(epochs, history["val_prauc"], marker="o", markersize=3, label="val")
    axes[1, 0].set_title("PR-AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["f1"], marker="o", markersize=3, label="train", color="green")
    axes[1, 1].plot(epochs, history["val_f1"], marker="o", markersize=3, label="val", color="red")
    axes[1, 1].set_title("F1 Score (Best Threshold)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved training curves -> {out_path}")
    if not no_show:
        plt.show()


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b1",
                                 "efficientnet_b2", "convnext_tiny"])

    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)

    parser.add_argument("--experiment", type=str, default="v4",
                        help="Experiment name: saves to checkpoints/<exp>/ and figures/<exp>/")
    parser.add_argument("--no_show", action="store_true",
                        help="Skip plt.show() (useful for headless/server runs)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        print("[INFO] cuDNN benchmark: enabled")

    print(f"[INFO] Experiment: {args.experiment}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Image size: {args.img_size}x{args.img_size}")
    print(f"[INFO] Batch size: {args.batch_size}")

    ROOT = Path(__file__).resolve().parents[1]
    CSV_PATH = ROOT / "data" / "processed" / "train_merged.csv"

    CKPT_DIR = ROOT / "checkpoints" / args.experiment
    FIG_DIR = ROOT / "reports" / "figures" / args.experiment
    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # ── 1. Load data ──
    train_loader, val_loader, train_df, val_df = load_datasets(
        csv_path=CSV_PATH,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        max_rows=args.max_rows,
        group_col="lesion_id",
        save_val_csv=True,
    )

    # ── 2. Build model ──
    model = SkinModel(model_name=args.model, weights=args.weights, dropout=0.4).to(device)
    model.freeze_base()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {total:,} total, {trainable:,} trainable")

    # ── 3. Loss — plain BCE (sampler already balances classes) ──
    n_neg = int((train_df["target"] == 0).sum())
    n_pos = int((train_df["target"] == 1).sum())
    print(f"[INFO] Train set — neg: {n_neg:,}  pos: {n_pos:,}  ratio: {n_neg/max(n_pos,1):.1f}:1")
    print(f"[INFO] pos_weight disabled — WeightedRandomSampler is handling imbalance")

    criterion = nn.BCEWithLogitsLoss()

    # ── 4. Phase 1: train head only ──
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_path = str(CKPT_DIR / "best_model.pth")

    history = run_training_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, scaler, epochs=args.epochs, ckpt_path=best_path,
        phase_name="Phase 1: Head Only",
        scheduler_type="plateau",
    )

    # ── 5. Phase 2: fine-tune ──
    if args.fine_tune:
        print("\n[INFO] Fine-tuning: unfreezing backbone...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.unfreeze_base()

        ft_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Trainable after unfreeze: {ft_trainable:,}")

        ft_optimizer = optim.AdamW(model.parameters(), lr=args.fine_tune_lr, weight_decay=1e-4)
        ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            ft_optimizer, T_max=args.fine_tune_epochs, eta_min=1e-7
        )

        ft_history = run_training_phase(
            model, train_loader, val_loader, criterion, ft_optimizer, ft_scheduler,
            device, scaler, epochs=args.fine_tune_epochs, ckpt_path=best_path,
            phase_name="Phase 2: Fine-Tune (Cosine LR)",
            patience=7,
            scheduler_type="cosine",
        )

        for k in history:
            history[k] = history[k] + ft_history[k]

    # ── 6. Plot + history CSV ──
    plot_history(history, str(FIG_DIR / "training_curves.png"), no_show=args.no_show)

    csv_path = FIG_DIR / "train_history.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["epoch", "loss", "auroc", "prauc", "val_loss", "val_auroc", "val_prauc", "lr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history["loss"])):
            writer.writerow({
                "epoch": i + 1,
                "loss": history["loss"][i],
                "auroc": history["auroc"][i],
                "prauc": history["prauc"][i],
                "val_loss": history["val_loss"][i],
                "val_auroc": history["val_auroc"][i],
                "val_prauc": history["val_prauc"][i],
                "lr": history["lr"][i],
            })
    print(f"[INFO] Saved train history -> {csv_path}")

    # ── 7. Save ──
    final_path = CKPT_DIR / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n[INFO] Saved final model -> {final_path}")
    print(f"[INFO] Best model -> {best_path}")
    print(f"[INFO] Figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()