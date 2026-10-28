"""Model configurations and metadata for DeepShield AI.

Defines all supported model architectures, input constraints, normalization,
label mappings, and honest forensic disclosures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelMetadata:
    """Documented metadata for a deepfake detection model."""
    id: str
    name: str
    architecture: str
    source: str
    checkpoint_id_or_path: str
    license: str
    input_size: Tuple[int, int]
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    label_map: Dict[int, str]
    fake_class_index: int
    description: str
    training_data_info: str
    known_limitations: List[str]
    is_default: bool = False


# Supported Model Registrations
AVAILABLE_MODELS: Dict[str, ModelMetadata] = {
    "vit_deepfake": ModelMetadata(
        id="vit_deepfake",
        name="ViT Deepfake Forensic Classifier (dima806)",
        architecture="Vision Transformer (vit-base-patch16-224)",
        source="HuggingFace Hub (dima806/deepfake_vs_real_image_detection)",
        checkpoint_id_or_path="dima806/deepfake_vs_real_image_detection",
        license="Apache-2.0 / Open Access",
        input_size=(224, 224),
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        label_map={0: "Real", 1: "Fake"},
        fake_class_index=1,
        description=(
            "Vision Transformer (ViT-Base) fine-tuned specifically on balanced "
            "datasets of 140,000 real and manipulated facial images. Employs "
            "multi-head self-attention across 16x16 image patches to detect global "
            "structural inconsistencies, blending boundaries, and synthetic facial artifacts."
        ),
        training_data_info=(
            "Trained on FaceForensics++, Celeb-DF v2, and the 140k Real/Fake Face dataset, "
            "comprising diverse DeepFake, FaceSwap, and Face2Face manipulations."
        ),
        known_limitations=[
            "Sensitivity decreases on heavily compressed (low-bitrate) video frames.",
            "Extreme pose angles (>45 degrees yaw/pitch) may cause false positives or lower confidence.",
            "Does not detect non-facial body swaps or purely audio manipulations.",
            "Prediction scores represent model confidence on visual patterns, not legal proof of provenance."
        ],
        is_default=True,
    ),
    "efficientnet_b0": ModelMetadata(
        id="efficientnet_b0",
        name="EfficientNet-B0 Deepfake Baseline",
        architecture="EfficientNet-B0 (Compound Scaling CNN)",
        source="Torchvision / Custom Checkpoint",
        checkpoint_id_or_path="models/checkpoints/efficientnet_b0_deepfake.pt",
        license="Apache-2.0 / BSD",
        input_size=(224, 224),
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        label_map={0: "Real", 1: "Fake"},
        fake_class_index=1,
        description=(
            "EfficientNet-B0 compound-scaled convolutional neural network modified for "
            "binary face forensics. Highly optimized for low-latency CPU inference on mobile and laptop hardware."
        ),
        training_data_info=(
            "Configurable baseline architecture. Can load weights fine-tuned on FaceForensics++ "
            "or run in clearly documented forensic baseline mode."
        ),
        known_limitations=[
            "Without dedicated fine-tuning weights loaded from disk, functions in baseline demonstration mode.",
            "Higher sensitivity to high-frequency sensor noise and camera ISO grain."
        ],
        is_default=False,
    ),
    "mesonet": ModelMetadata(
        id="mesonet",
        name="MesoInception-4 Forensic Architecture",
        architecture="MesoInception-4 (Microscopic Artifact CNN)",
        source="Afchar et al. / MesoNet Forensic Baseline",
        checkpoint_id_or_path="models/checkpoints/mesonet_deepfake.pt",
        license="MIT / Academic Research",
        input_size=(256, 256),
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
        label_map={0: "Real", 1: "Fake"},
        fake_class_index=1,
        description=(
            "MesoInception-4 network architecture specifically designed for deepfake video detection "
            "at the mesoscopic level, analyzing intermediate image compression and dilation artifacts."
        ),
        training_data_info="Designed for Deepfake and Face2Face detection under MPEG compression.",
        known_limitations=[
            "Shallow convolutional depth compared to modern Transformers; limited generalization to diffusion models."
        ],
        is_default=False,
    ),
}

DEFAULT_MODEL_ID = "vit_deepfake"
CONFIDENCE_THRESHOLD = 0.50
