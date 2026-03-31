"""U-Net with pretrained ResNet34 encoder for brain tumour segmentation.

Architecture:
  Encoder: ResNet34 pretrained on ImageNet (frozen feature extractor initially)
  Decoder: U-Net decoder with skip connections (trained from scratch)
  Head:    Conv2d -> sigmoid, single output channel (tumour probability)

We use the segmentation-models-pytorch (smp) library which provides
ready-made encoder-decoder architectures with pretrained ImageNet encoders.

Reference: Ronneberger et al. (2015) "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI.
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def create_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    """Return a U-Net with a pretrained ResNet34 encoder.

    Args:
        encoder_name:    timm/smp encoder name (default: resnet34)
        encoder_weights: pretrained weights (default: imagenet, use None for scratch)
        in_channels:     number of input channels (3 for RGB)

    Returns:
        smp.Unet model with sigmoid activation
    """
    model = smp.Unet(
        encoder_name     = encoder_name,
        encoder_weights  = encoder_weights,
        in_channels      = in_channels,
        classes          = 1,
        activation       = "sigmoid",
    )
    return model


def get_parameter_groups(model: nn.Module) -> tuple[list, list]:
    """Separate encoder and decoder parameters for differential learning rates.

    Returns:
        encoder_params, decoder_params
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = (
        list(model.decoder.parameters()) +
        list(model.segmentation_head.parameters())
    )
    return encoder_params, decoder_params


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
