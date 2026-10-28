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


# Supported Model Registration (Standardized on single high-accuracy Vision Transformer)
AVAILABLE_MODELS: Dict[str, ModelMetadata] = {
    "vit_deepfake": ModelMetadata(
        id="vit_deepfake",
        name="ViT Deepfake Forensic Classifier",
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
}

DEFAULT_MODEL_ID = "vit_deepfake"
CONFIDENCE_THRESHOLD = 0.50
