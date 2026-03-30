"""Training loop for brain MRI tumour segmentation (U-Net).

Loss:      BCE + Dice (combined)
Optimiser: Adam
Scheduler: ReduceLROnPlateau on val Dice (patience=5, factor=0.5)
Checkpoint: saved on highest validation Dice (tumour slices only)

Usage:
    cd src
    python train.py
    python train.py --epochs 50 --batch-size 16 --lr 1e-3 --output-dir ../results
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_dataloaders
from model import create_model, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = 2 * |pred * target| / (|pred| + |target| + eps)
    Loss = 1 - Dice
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred and target: [B, 1, H, W]
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice         = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice loss, weighted equally."""

    def __init__(self):
        super().__init__()
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(pred, target) + self.dice(pred, target)


# ---------------------------------------------------------------------------
# Dice metric (non-differentiable, threshold at 0.5)
# ---------------------------------------------------------------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute mean Dice coefficient over a batch (after thresholding)."""
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union        = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice         = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimiser.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation. Returns (mean_loss, mean_dice_tumour_only).

    Dice is computed only on slices that have a tumour (non-empty mask).
    Empty slices return Dice~0 even when correctly predicted all-background,
    which inflates the metric artificially.
    """
    model.eval()
    total_loss  = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = criterion(preds, masks)
            total_loss += loss.item() * images.size(0)

            # Per-sample Dice, then keep only non-empty masks
            pred_bin     = (preds > 0.5).float()
            intersection = (pred_bin * masks).sum(dim=(1, 2, 3))
            union        = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            per_sample   = (2.0 * intersection + 1e-6) / (union + 1e-6)  # [B]
            non_empty    = masks.sum(dim=(1, 2, 3)) > 0                  # [B] bool
            if non_empty.any():
                dice_scores.append(per_sample[non_empty].cpu())

    mean_dice = torch.cat(dice_scores).mean().item() if dice_scores else 0.0
    return total_loss / len(loader.dataset), mean_dice


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path: Path, metadata: dict) -> None:
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    val_dices:    list[float],
    output_path:  Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

    ax1.plot(epochs, train_losses, label="train loss")
    ax1.plot(epochs, val_losses,   label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Training and validation loss")
    ax1.legend()
    ax1.grid(linestyle=":")

    ax2.plot(epochs, val_dices, color="tab:green", label="val Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice coefficient")
    ax2.set_title("Validation Dice coefficient")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(linestyle=":")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on brain MRI segmentation")
    parser.add_argument("--data-dir",    type=str,   default=None)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--output-dir",  type=str,   default="../results")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = build_dataloaders(
        data_root   = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    # Model
    model = create_model().to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters — total: {total:,} | trainable: {trainable:,}")

    # Loss, optimiser, scheduler
    criterion = CombinedLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )
    # mode="max" because we track val Dice (higher is better)

    # Training loop
    train_losses, val_losses, val_dices = [], [], []
    best_dice    = 0.0
    best_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        train_loss          = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_dice  = validate(model, val_loader, criterion, device)
        scheduler.step(val_dice)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        lr_now = optimiser.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"val Dice: {val_dice:.4f} | "
              f"lr: {lr_now:.2e}")

        if val_dice > best_dice:
            best_dice  = val_dice
            best_epoch = epoch
            save_checkpoint(model, output_dir / "best_model.pth", {
                "epoch":      epoch,
                "val_dice":   val_dice,
                "val_loss":   val_loss,
                "lr":         args.lr,
                "batch_size": args.batch_size,
                "seed":       args.seed,
            })

    print(f"\nBest val Dice: {best_dice:.4f} at epoch {best_epoch}")

    plot_training_curves(
        train_losses, val_losses, val_dices,
        output_path=output_dir / "training_curves.png",
    )


if __name__ == "__main__":
    main()
