"""Model loading and inference logic for the API.

This module is the bridge between the trained checkpoint and the web layer.
It handles:
  - Loading the model from disk once at startup
  - Preprocessing an uploaded image into a tensor
  - Running inference (with TTA)
  - Post-processing the output into a PNG mask and summary statistics
"""

import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Add src/ to path so we can import from the training modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from model import create_model
from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Model — loaded once when the API starts, reused for every request
# ---------------------------------------------------------------------------
_model = None
_device = None


def load_model(checkpoint_path: str | Path) -> None:
    """Load the U-Net checkpoint into memory.

    Called once at API startup. Stores model in module-level variables
    so every request reuses the same loaded model without re-reading disk.
    """
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=_device)

    _model = create_model().to(_device)
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()

    print(f"Model loaded from {checkpoint_path} on {_device}")
    print(f"  Checkpoint epoch : {checkpoint.get('epoch', '?')}")
    print(f"  Val Dice         : {checkpoint.get('val_dice', '?'):.4f}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised model-ready tensor.

    Args:
        image: PIL image (any mode — converted to RGB internally)

    Returns:
        Tensor of shape [1, 3, H, W] ready for the model
    """
    image = image.convert("RGB")
    image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.BILINEAR)
    tensor = TF.to_tensor(image)                              # [3, H, W] in [0, 1]
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0)                                # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _predict_with_tta(tensor: torch.Tensor) -> torch.Tensor:
    """Run model with horizontal-flip TTA.

    Args:
        tensor: [1, 3, H, W] input tensor on CPU

    Returns:
        [1, 1, H, W] probability map on CPU
    """
    tensor = tensor.to(_device)

    with torch.no_grad():
        pred          = _model(tensor)
        tensor_flip   = torch.flip(tensor, dims=[3])
        pred_flip     = torch.flip(_model(tensor_flip), dims=[3])

    return ((pred + pred_flip) / 2.0).cpu()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _mask_to_png_base64(mask: np.ndarray) -> str:
    """Convert a binary [H, W] numpy mask to a base64-encoded PNG string.

    The PNG has pixel values 0 (black = background) or 255 (white = tumour).
    Base64 encoding lets us include binary image data inside a JSON response.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _overlay_to_png_base64(
    original: Image.Image, mask: np.ndarray
) -> str:
    """Create a red-overlay visualisation and return as base64 PNG.

    The original MRI slice is shown in greyscale with the predicted
    tumour region highlighted in red.
    """
    original_rgb = original.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    arr     = np.array(original_rgb).astype(np.float32)
    overlay = arr.copy()

    # Red tint on tumour pixels
    overlay[mask == 1, 0] = 255   # R channel max
    overlay[mask == 1, 1] = 50    # G channel low
    overlay[mask == 1, 2] = 50    # B channel low

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def run_prediction(image_bytes: bytes) -> dict:
    """Run full inference pipeline on an uploaded image.

    Args:
        image_bytes: raw bytes of the uploaded image file

    Returns:
        dict with keys:
            tumour_detected  (bool)   — whether any tumour was found
            tumour_fraction  (float)  — fraction of pixels predicted as tumour
            mask_png         (str)    — base64-encoded PNG of the binary mask
            overlay_png      (str)    — base64-encoded PNG of the overlay
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Load image
    original = Image.open(io.BytesIO(image_bytes))

    # Preprocess → predict → threshold
    tensor      = _preprocess(original)
    prob_map    = _predict_with_tta(tensor)           # [1, 1, H, W]
    mask        = (prob_map[0, 0] > 0.5).numpy()     # [H, W] bool

    tumour_fraction = float(mask.mean())
    tumour_detected = tumour_fraction > 0.001         # at least 0.1% of pixels

    return {
        "tumour_detected": tumour_detected,
        "tumour_fraction": round(tumour_fraction, 4),
        "mask_png":        _mask_to_png_base64(mask.astype(np.uint8)),
        "overlay_png":     _overlay_to_png_base64(original, mask.astype(np.uint8)),
    }
