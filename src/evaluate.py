"""Evaluation and visualisation for brain MRI tumour segmentation.

Loads the best checkpoint, runs inference on the test set, and reports:
  - Mean Dice coefficient
  - Mean IoU (Intersection over Union)
  - Dice distribution (median, quartiles)
  - Grid of sample predictions saved to results/

Usage:
    cd src
    python evaluate.py
    python evaluate.py --checkpoint ../results/best_model.pth --data-dir ../data
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import create_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Dice coefficient (after thresholding).

    Args:
        pred:   [B, 1, H, W] float in [0, 1]
        target: [B, 1, H, W] binary {0., 1.}

    Returns:
        Tensor of shape [B] with per-sample Dice scores.
    """
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union        = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample IoU (Intersection over Union) after thresholding.

    Args:
        pred:   [B, 1, H, W] float in [0, 1]
        target: [B, 1, H, W] binary {0., 1.}

    Returns:
        Tensor of shape [B] with per-sample IoU scores.
    """
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union        = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return (intersection + eps) / (union + eps)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load U-Net weights from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}  "
          f"(epoch {checkpoint.get('epoch', '?')}, "
          f"val Dice {checkpoint.get('val_dice', '?'):.4f})")
    return model


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model on all batches and collect results.

    Test-Time Augmentation (TTA): when tta=True, each image is predicted
    twice — original and horizontally flipped. The flipped prediction is
    mirrored back and the two probability maps are averaged before thresholding.
    This costs one extra forward pass per batch but typically adds 1-2 Dice points.

    Returns:
        images:      [N, 3, H, W]  — normalised input images
        masks:       [N, 1, H, W]  — ground truth masks
        predictions: [N, 1, H, W]  — raw probabilities in [0, 1]
        dices:       [N]           — per-sample Dice scores
    """
    all_images  = []
    all_masks   = []
    all_preds   = []
    all_dices   = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            # Forward pass on original
            preds = model(images)

            if tta:
                # Forward pass on horizontally flipped images
                images_flipped = torch.flip(images, dims=[3])        # flip W axis
                preds_flipped  = model(images_flipped)
                preds_flipped  = torch.flip(preds_flipped, dims=[3]) # flip prediction back

                # Average the two probability maps
                preds = (preds + preds_flipped) / 2.0

            dices = dice_coefficient(preds, masks)

            all_images.append(images.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_dices.append(dices.cpu().numpy())

    return (
        np.concatenate(all_images,  axis=0),
        np.concatenate(all_masks,   axis=0),
        np.concatenate(all_preds,   axis=0),
        np.concatenate(all_dices,   axis=0),
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _denormalise(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation for display.

    Args:
        image: [3, H, W] float32

    Returns:
        [H, W, 3] uint8 in [0, 255]
    """
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = image * std + mean           # undo normalisation
    img  = np.clip(img, 0, 1)
    img  = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)      # [H, W, 3]


def plot_predictions(
    images:      np.ndarray,
    masks:       np.ndarray,
    predictions: np.ndarray,
    dices:       np.ndarray,
    n_samples:   int = 6,
    output_path: Path | None = None,
) -> None:
    """Save a grid of n_samples examples: image | ground truth | prediction | overlay.

    Samples are chosen to span the Dice range (best, middle, worst).
    """
    n_samples = min(n_samples, len(dices))

    # Pick samples that span the Dice range
    sorted_idx = np.argsort(dices)
    step       = max(1, len(sorted_idx) // n_samples)
    indices    = sorted_idx[::step][:n_samples]

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.5 * n_samples),
                              tight_layout=True)
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI slice", "Ground truth", "Prediction (≥0.5)", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        img_rgb  = _denormalise(images[idx])          # [H, W, 3]
        gt_mask  = masks[idx, 0]                      # [H, W]
        pred_raw = predictions[idx, 0]                # [H, W] probabilities
        pred_bin = (pred_raw > 0.5).astype(np.float32)

        # Overlay: image with predicted mask in green
        overlay = img_rgb.copy().astype(np.float32)
        overlay[pred_bin == 1, 0] = 255   # R
        overlay[pred_bin == 1, 1] = 50    # G
        overlay[pred_bin == 1, 2] = 50    # B — red tint on prediction
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axes[row, 0].imshow(img_rgb)
        axes[row, 1].imshow(gt_mask,  cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(f"Dice={dices[idx]:.3f}", fontsize=9)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Predictions grid saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on brain MRI test set")
    parser.add_argument("--checkpoint",  type=str, default="../results/best_model.pth")
    parser.add_argument("--data-dir",    type=str, default=None)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir",  type=str, default="../results")
    parser.add_argument("--n-samples",   type=int, default=6,
                        help="Number of example predictions to visualise")
    parser.add_argument("--no-tta",      action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data — we only need the test split
    _, _, test_loader = build_dataloaders(
        data_root   = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # Model
    model = load_model_from_checkpoint(Path(args.checkpoint), device)

    # Inference
    use_tta = not args.no_tta
    print(f"Running inference on test set (TTA: {'on' if use_tta else 'off'})...")
    images, masks, predictions, dices = run_inference(model, test_loader, device, tta=use_tta)

    # IoU
    preds_t  = torch.from_numpy(predictions)
    masks_t  = torch.from_numpy(masks)
    ious     = iou_score(preds_t, masks_t).numpy()

    # Empty-mask mask: slices where the ground truth has no tumour at all
    non_empty = masks.sum(axis=(1, 2, 3)) > 0

    # Report
    print(f"\nTest results on {len(dices)} slices "
          f"({non_empty.sum()} with tumour, {(~non_empty).sum()} empty):")

    print(f"\n  -- All slices --")
    print(f"  Mean Dice : {dices.mean():.4f}")
    print(f"  Mean IoU  : {ious.mean():.4f}")
    print(f"  Dice distribution:")
    print(f"    min    : {dices.min():.4f}")
    print(f"    25th % : {np.percentile(dices, 25):.4f}")
    print(f"    median : {np.median(dices):.4f}")
    print(f"    75th % : {np.percentile(dices, 75):.4f}")
    print(f"    max    : {dices.max():.4f}")

    print(f"\n  -- Tumour slices only (empty masks excluded) --")
    print(f"  Mean Dice : {dices[non_empty].mean():.4f}")
    print(f"  Mean IoU  : {ious[non_empty].mean():.4f}")
    print(f"  Dice distribution:")
    print(f"    min    : {dices[non_empty].min():.4f}")
    print(f"    25th % : {np.percentile(dices[non_empty], 25):.4f}")
    print(f"    median : {np.median(dices[non_empty]):.4f}")
    print(f"    75th % : {np.percentile(dices[non_empty], 75):.4f}")
    print(f"    max    : {dices[non_empty].max():.4f}")

    # Visualise
    plot_predictions(
        images, masks, predictions, dices,
        n_samples   = args.n_samples,
        output_path = output_dir / "predictions.png",
    )


if __name__ == "__main__":
    main()
