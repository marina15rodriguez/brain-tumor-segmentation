# Model Improvements and Performance Progression

This document tracks the iterative improvements made to the brain tumour segmentation model,
explaining what was changed, why, and what effect each change had on the test set.

## Dataset

LGG MRI Segmentation (Mateusz Buda, Kaggle) — 110 patients, ~3,929 axial MRI slices.
Test set: 589 slices total, of which **230 have a tumour** and 359 are empty (no tumour present).
All metrics below are reported on **tumour slices only** — see note on empty masks at the end.

---

## Version 1 — Scratch U-Net, 30 epochs

**Architecture:** U-Net built from scratch (encoder + decoder, ~31M parameters, all randomly initialised)
**Loss:** BCE + Dice
**Scheduler:** ReduceLROnPlateau (patience=5)
**Val Dice:** computed over all slices including empty ones (inflated metric)

| Metric | Value |
|---|---|
| Mean Dice (tumour) | 0.592 |
| Median Dice | 0.727 |
| 25th percentile | 0.286 |
| Best epoch | 18 |

**Problem identified:** The validation Dice was computed over all slices, including empty ones.
Empty slices score Dice ≈ 0 even when the model correctly predicts an empty mask, because
the Dice formula is undefined when both prediction and ground truth are zero. This dragged the
metric down and caused training to stop early (epoch 18) thinking improvement had stalled,
when the model had not yet converged.

---

## Version 2 — Scratch U-Net, 50 epochs + fixed val Dice

**What changed:**
- Validation Dice now computed on **tumour slices only** (non-empty masks)
- Checkpoint saved on this honest metric
- Default epochs increased from 30 to 50

**Why it helped:** With the correct metric, the scheduler no longer reduced the LR prematurely,
and the checkpoint was selected based on actual segmentation quality rather than an inflated
all-slices average. Training continued to epoch 48 before converging.

| Metric | Value | Change |
|---|---|---|
| Mean Dice (tumour) | 0.683 | +0.091 |
| Median Dice | 0.836 | +0.109 |
| 25th percentile | 0.526 | +0.240 |
| Best epoch | 48 | — |

---

## Version 3 — Pretrained ResNet34 encoder

**What changed:**
- Replaced the scratch U-Net encoder with a **ResNet34 pretrained on ImageNet**
  using the `segmentation-models-pytorch` library
- Decoder and skip connections unchanged
- **Differential learning rates**: encoder gets 10× lower LR than decoder
  (encoder: `5e-5`, decoder: `5e-4`) to avoid destroying pretrained features

**Why it helped:** The scratch encoder had to learn to detect edges, textures, and fine
structures entirely from 2,719 training slices. The pretrained ResNet34 already knows how
to detect these from 1.2M ImageNet images. This is especially important for small tumours,
where subtle intensity patterns are hard to learn from scratch. The 25th percentile —
the metric that captures hard cases — improved the most.

| Metric | Value | Change |
|---|---|---|
| Mean Dice (tumour) | 0.777 | +0.094 |
| Median Dice | 0.888 | +0.052 |
| 25th percentile | 0.743 | +0.217 |
| Best epoch | 42 | — |

---

## Version 4 — Weighted BCE + Test-Time Augmentation (TTA)

**What changed (training):** Replaced standard BCE with **weighted BCE** (`pos_weight=10`):
tumour pixels are penalised 10× more than background pixels when missed.

**What changed (inference):** Added **test-time augmentation**: each image is predicted
twice (original + horizontally flipped), the flipped prediction is mirrored back, and the
two probability maps are averaged before thresholding.

**Why weighted BCE helped:** Standard BCE treats all pixels equally. Since tumour pixels
are only ~2–5% of the image, the model could achieve low BCE loss by predicting background
everywhere. The positive weight forces the model to pay attention to the rare tumour region.

**Why TTA helped:** Averaging two predictions cancels out asymmetric spurious activations —
false positives that appear on one side of the image but not its mirror. This reduced false
positives on empty slices and sharpened tumour boundary predictions, at the cost of one
extra forward pass per batch and zero retraining.

| Metric | Value | Change |
|---|---|---|
| Mean Dice (tumour) | 0.796 | +0.019 |
| Median Dice | 0.889 | +0.001 |
| 25th percentile | 0.771 | +0.028 |
| Best epoch | 30 | — |

---

## Full progression summary

| Version | Mean Dice | Median Dice | 25th %ile |
|---|---|---|---|
| Scratch U-Net, 30 ep | 0.592 | 0.727 | 0.286 |
| Scratch U-Net, 50 ep + fixed metric | 0.683 | 0.836 | 0.526 |
| + ResNet34 pretrained encoder | 0.777 | 0.888 | 0.743 |
| + Weighted BCE + TTA | **0.796** | **0.889** | **0.771** |

---

## Note on empty masks and the Dice metric

359 of the 589 test slices contain no tumour at all. When both the ground truth mask and the
model prediction are all zeros, the Dice formula produces ≈ 0 (not 1) due to the ε term:

```
Dice = (2 * 0 + ε) / (0 + 0 + ε) ≈ 0
```

This means a model that correctly predicts "no tumour" on an empty slice still scores 0.
Reporting mean Dice over all slices therefore underestimates true performance.
All metrics in this document exclude empty slices. Both all-slices and tumour-only
numbers are reported by `evaluate.py` for completeness.
