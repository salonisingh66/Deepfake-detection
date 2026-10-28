"""Deterministic image and face crop preprocessing for DeepShield AI.

Transforms raw RGB facial crops into normalized tensors conforming to model requirements.
"""

from typing import List, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from config.model_config import ModelMetadata, AVAILABLE_MODELS, DEFAULT_MODEL_ID


def get_torchvision_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> transforms.Compose:
    """Return standard deterministic PyTorch torchvision transform pipeline."""
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_face_crop(
    crop_rgb: np.ndarray,
    metadata: ModelMetadata,
    device: str = "cpu"
) -> torch.Tensor:
    """Preprocess a single RGB face crop numpy array into a 4D batch tensor [1, 3, H, W]."""
    pil_img = Image.fromarray(crop_rgb)
    transform = get_torchvision_transforms(
        input_size=metadata.input_size,
        mean=metadata.normalization_mean,
        std=metadata.normalization_std
    )
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor


def preprocess_face_batch(
    crops_rgb: List[np.ndarray],
    metadata: ModelMetadata,
    device: str = "cpu"
) -> torch.Tensor:
    """Preprocess a list of RGB crops into a single batched tensor [B, 3, H, W]."""
    if not crops_rgb:
        raise ValueError("Cannot preprocess an empty batch of facial crops.")
        
    transform = get_torchvision_transforms(
        input_size=metadata.input_size,
        mean=metadata.normalization_mean,
        std=metadata.normalization_std
    )
    tensors = [transform(Image.fromarray(crop)) for crop in crops_rgb]
    batch = torch.stack(tensors, dim=0).to(device)
    return batch
