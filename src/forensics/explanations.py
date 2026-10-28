"""Visual forensic explanations (Attention Maps & Grad-CAM).

Generates visual overlays highlighting spatial regions of model sensitivity.
Clearly documented: highlighted areas represent model sensitivity, NOT conclusive proof of manipulation.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils.logging_config import logger


def generate_vit_attention_map(
    model: nn.Module,
    inputs: dict,
    original_image_rgb: np.ndarray,
    alpha: float = 0.5
) -> Optional[np.ndarray]:
    """Generate visual explanation overlay from Vision Transformer attention weights.
    
    Extracts multi-head self-attention from the [CLS] token to the 14x14 image patches in the final layer.
    """
    try:
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs, output_attentions=True)
            
        if not hasattr(outputs, "attentions") or not outputs.attentions:
            logger.warning("Model outputs do not contain attention weights.")
            return None

        # Last layer attention: shape [batch, num_heads, num_tokens, num_tokens]
        last_attentions = outputs.attentions[-1]
        
        # Self-attention from [CLS] (token 0) to all patch tokens (1 to 197)
        cls_attn = last_attentions[0, :, 0, 1:]  # [num_heads, 196]
        mean_attn = cls_attn.mean(dim=0).reshape(1, 1, 14, 14)  # [1, 1, 14, 14]

        # Interpolate to match original image dimensions
        orig_h, orig_w = original_image_rgb.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            mean_attn,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        heatmap = upsampled.squeeze().cpu().numpy()
        # Min-max normalization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert to 8-bit heatmap colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlay = cv2.addWeighted(original_image_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
        return overlay

    except Exception as e:
        logger.error(f"Failed to generate ViT attention overlay: {e}")
        return None


def generate_cnn_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    original_image_rgb: np.ndarray,
    target_class_idx: int = 1,
    alpha: float = 0.5
) -> Optional[np.ndarray]:
    """Generate classic Grad-CAM visualization for convolutional backbones (e.g., EfficientNet)."""
    try:
        activations = []
        gradients = []

        def fwd_hook(mod, inp, out):
            activations.append(out)

        def bwd_hook(mod, gin, gout):
            gradients.append(gout[0])

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        # Forward pass
        model.eval()
        output = model(input_tensor)
        score = output[0, target_class_idx]

        # Backward pass
        model.zero_grad()
        score.backward(retain_graph=False)

        h_fwd.remove()
        h_bwd.remove()

        if not activations or not gradients:
            return None

        act = activations[0].detach()  # [1, C, H, W]
        grad = gradients[0].detach()    # [1, C, H, W]

        # Global average pooling on gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = torch.relu(cam)

        orig_h, orig_w = original_image_rgb.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            cam,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )

        heatmap = upsampled.squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_image_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
        return overlay

    except Exception as e:
        logger.error(f"Failed to generate CNN Grad-CAM overlay: {e}")
        return None
