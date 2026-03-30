# Brain Tumour Segmentation (U-Net)

Binary segmentation of brain tumours in MRI scans using a U-Net trained from scratch.
Developed as a Kaggle notebook project.

## Overview

Given an axial brain MRI slice, the model predicts a binary mask identifying tumour tissue
pixel by pixel.

### Architecture — U-Net

Encoder-decoder with skip connections (Ronneberger et al., 2015):

```
Input [3, 256, 256]
  └─ Encoder: DoubleConv + MaxPool  →  64 / 128 / 256 / 512 feature maps
  └─ Bottleneck:                    →  1024 feature maps
  └─ Decoder: Upsample + skip + DoubleConv  →  512 / 256 / 128 / 64
  └─ Head: Conv2d 64→1, sigmoid
Output [1, 256, 256]  — tumour probability per pixel
```

Skip connections carry fine spatial detail from encoder to decoder,
enabling sharp tumour boundary predictions.

### Loss function

Combined BCE + Dice loss:
- **BCE** — stable per-pixel gradients throughout training
- **Dice loss** — directly optimises the overlap metric, robust to class imbalance
  (most pixels are healthy background)

### Metrics

- **Dice coefficient** — primary metric: `2|pred ∩ mask| / (|pred| + |mask|)`
- **IoU** — stricter overlap: `|pred ∩ mask| / |pred ∪ mask|`

## Dataset

[LGG MRI Segmentation — Mateusz Buda (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

- 110 patients, ~3,929 axial MRI slices (256×256 RGB TIFF)
- Paired binary tumour masks (single-channel TIFF, pixels ∈ {0, 255})
- Split at **patient level** to avoid data leakage: 70 / 15 / 15 train / val / test

## Project Structure

```
brain-tumor-segmentation/
├── src/
│   ├── dataset.py    # Data loading, patient-level split, joint augmentation
│   ├── model.py      # U-Net (DoubleConv, Down, Up blocks)
│   ├── train.py      # Training loop, combined loss, ReduceLROnPlateau
│   └── evaluate.py   # Inference, Dice/IoU metrics, prediction grid
├── data/             # Dataset (not tracked) — place kaggle_3m/ here
├── results/          # Checkpoints and plots (not tracked)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/marina15rodriguez/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it as:
```
data/
  kaggle_3m/
    TCGA_CS_4941_19960909/
      TCGA_CS_4941_19960909_1.tif
      TCGA_CS_4941_19960909_1_mask.tif
      ...
    ...
```

## Usage

```bash
cd src

# Train
python train.py --epochs 30 --batch-size 16 --lr 1e-3

# Evaluate
python evaluate.py --checkpoint ../results/best_model.pth
```

Results (checkpoint, training curves, prediction grid) are saved to `results/`.

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Split strategy | Patient-level | Slice-level would leak same-patient data across splits |
| Loss | BCE + Dice | BCE stabilises early training; Dice handles class imbalance |
| LR scheduler | ReduceLROnPlateau | Adapts to convergence speed without fixing num epochs upfront |
| Checkpoint criterion | Highest val Dice | Directly optimises the evaluation metric |
| Mask interpolation | NEAREST | Bilinear would corrupt binary labels at boundaries |

## Authors

Marina Rodríguez — Brain Tumour Segmentation, Kaggle
