"""Data loading and augmentation for brain MRI tumour segmentation.

Dataset: LGG MRI Segmentation (Mateusz Buda, Kaggle)
  - 110 patients, ~3,929 axial MRI slices
  - Each slice has a corresponding binary tumour mask
  - Images: RGB TIFF, 256x256
  - Masks:  single-channel TIFF, pixel in {0, 255}

Split strategy: patient-level 70 / 15 / 15  (train / val / test)
to avoid data leakage across splits.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test fraction = 1 - 0.70 - 0.15 = 0.15


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def find_data_root(start: Path | None = None) -> Path:
    """Walk common sub-paths to locate the dataset root folder.

    Expected layout:
        <root>/
          <patient_folder>/
            <patient>_<n>.tif        <- MRI slice
            <patient>_<n>_mask.tif   <- binary mask

    Returns the first directory that contains patient sub-folders with .tif files.
    """
    base = Path(start) if start else Path(__file__).resolve().parents[1]
    candidates = [
        base / "data" / "kaggle_3m",
        base / "data" / "lgg-mri-segmentation" / "kaggle_3m",
        base / "data",
    ]
    for c in candidates:
        if c.is_dir() and any(
            p.is_dir() and list(p.glob("*.tif"))
            for p in c.iterdir() if p.is_dir()
        ):
            return c
    raise FileNotFoundError(
        "Dataset root not found. "
        "Place the kaggle_3m folder under data/ in the project root."
    )


def collect_pairs(data_root: Path) -> list[dict]:
    """Scan data_root and return a list of dicts with keys:
        'image'   : Path to the MRI slice (.tif)
        'mask'    : Path to the binary mask (_mask.tif)
        'patient' : patient folder name (used for splitting)
    """
    pairs = []
    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        for img_path in sorted(patient_dir.glob("*.tif")):
            if img_path.stem.endswith("_mask"):
                continue                        # skip mask files here
            mask_path = img_path.with_name(img_path.stem + "_mask.tif")
            if not mask_path.exists():
                continue
            pairs.append({
                "image":   img_path,
                "mask":    mask_path,
                "patient": patient,
            })
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BrainMRIDataset(Dataset):
    """Loads (image, mask) pairs for binary tumour segmentation.

    Each __getitem__ returns:
        image : FloatTensor [3, H, W]  — normalised RGB
        mask  : FloatTensor [1, H, W]  — binary {0.0, 1.0}
    """

    def __init__(self, pairs: list[dict], augment: bool = False):
        self.pairs   = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item  = self.pairs[idx]
        image = Image.open(item["image"]).convert("RGB")
        mask  = Image.open(item["mask"]).convert("L")   # grayscale {0, 255}

        image, mask = self._joint_transform(image, mask)

        # Image → normalised float tensor [3, H, W]
        image = TF.to_tensor(image)                             # [3, H, W] in [0, 1]
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        # Mask → binary float tensor [1, H, W]
        mask = torch.from_numpy(np.array(mask)).float()         # [H, W] values {0, 255}
        mask = (mask > 127).float().unsqueeze(0)                # [1, H, W] values {0., 1.}

        return image, mask

    # ------------------------------------------------------------------
    def _joint_transform(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Apply identical spatial transforms to image and mask."""

        # Always resize
        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.NEAREST)
        # NEAREST for masks: we must not interpolate between 0 and 255

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Random rotation +-15 degrees
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            # Colour jitter on image only (does not affect mask)
            image = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )(image)

        return image, mask


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_root: Path | None = None,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders with patient-level split.

    Args:
        data_root:   path to kaggle_3m folder (auto-detected if None)
        batch_size:  number of samples per batch
        num_workers: parallel workers for data loading
        seed:        random seed for reproducible patient split

    Returns:
        train_loader, val_loader, test_loader
    """
    if data_root is None:
        data_root = find_data_root()
    data_root = Path(data_root)

    all_pairs = collect_pairs(data_root)
    if not all_pairs:
        raise RuntimeError(f"No image/mask pairs found under {data_root}")

    # Patient-level split
    patients = sorted({p["patient"] for p in all_pairs})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_train = int(len(patients) * TRAIN_FRAC)
    n_val   = int(len(patients) * VAL_FRAC)

    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train: n_train + n_val])
    test_patients  = set(patients[n_train + n_val:])

    train_pairs = [p for p in all_pairs if p["patient"] in train_patients]
    val_pairs   = [p for p in all_pairs if p["patient"] in val_patients]
    test_pairs  = [p for p in all_pairs if p["patient"] in test_patients]

    print(f"Patients — train: {len(train_patients):3d} | val: {len(val_patients):3d} | test: {len(test_patients):3d}")
    print(f"Slices   — train: {len(train_pairs):4d} | val: {len(val_pairs):4d} | test: {len(test_pairs):4d}")

    train_ds = BrainMRIDataset(train_pairs, augment=True)
    val_ds   = BrainMRIDataset(val_pairs,   augment=False)
    test_ds  = BrainMRIDataset(test_pairs,  augment=False)

    loader_kws = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kws)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kws)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kws)

    return train_loader, val_loader, test_loader
