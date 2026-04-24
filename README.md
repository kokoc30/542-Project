# ISIC 2024 Skin Lesion Classification

Binary image classification: **0 = benign**, **1 = malignant**.

The dataset is extremely imbalanced (~98% benign), so accuracy is not a useful metric here. We use AUROC, PR-AUC, and recall-based thresholds instead.

---

## Project Structure

```
C:\isic-skin-cancer
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src
│   ├── preprocessing.py
│   ├── dataloader.py
│   ├── train.py
│   ├── evaluate.py
│   └── data
│       └── build_dataset.py
│
├── data
│   ├── raw
│   │   ├── train.csv
│   │   ├── metadata.csv
│   │   └── images
│   │       └── .gitkeep        <- put local ISIC images here (not in GitHub)
│   │
│   └── processed
│       ├── train_merged.csv
│       └── val_split.csv
│
├── checkpoints
│   └── v3_KOKO
│       ├── best_model.pth
│       └── final_model.pth
│
└── reports
    └── figures
        └── v3_KOKO
            ├── class_distribution.png
            ├── image_sizes.png
            ├── sample_images.png
            ├── training_curves.png
            ├── roc_curve.png
            ├── pr_curve.png
            ├── confusion_matrix_auto.png
            ├── confusion_matrix_recall90.png
            ├── metrics_auto.txt
            └── metrics_recall90.txt
```

---

## Main Scripts

| Script | What it does |
|---|---|
| `preprocessing.py` | Merges CSVs, checks class balance, checks image quality, saves `train_merged.csv` |
| `dataloader.py` | Builds dataset, train/val split, transforms, and dataloaders |
| `train.py` | Trains EfficientNet, saves best checkpoint to `checkpoints/` |
| `evaluate.py` | Loads best checkpoint, runs inference, saves metrics and plots |

---

## Final Experiment: v3_KOKO

| Metric | Value |
|---|---|
| AUROC | 0.9370 |
| PR-AUC | 0.0174 |
| Auto threshold | 0.999 |
| Auto F1 | 0.0326 |
| Auto recall | 0.5823 |
| Recall90 threshold | 0.677 |
| Recall90 recall | 0.9114 |

---

## How to Run

**1. Preprocess**
```bash
python src/preprocessing.py
```

**2. Train**
```bash
python src/train.py --experiment v3_KOKO --epochs 15 --fine_tune --fine_tune_epochs 10 --batch_size 128 --num_workers 8 --img_size 256
```

**3. Evaluate**
```bash
# best F1 threshold
python src/evaluate.py --experiment v3_KOKO --threshold auto --batch_size 128 --img_size 256 --num_workers 8

# threshold targeting 90% recall
python src/evaluate.py --experiment v3_KOKO --threshold recall90 --batch_size 128 --img_size 256 --num_workers 8
```

---

## Notes

- Raw images are **not included in GitHub**. Place local ISIC images inside `data/raw/images/`.
- Results are from a class project and are not intended for medical use.
