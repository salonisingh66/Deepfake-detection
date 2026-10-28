"""Inference Execution Engine for DeepShield AI.

Runs bounded batch and single-face deepfake detection with timing benchmarks,
softmax normalization, and validated class mappings.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from PIL import Image

from config.model_config import CONFIDENCE_THRESHOLD, ModelMetadata
from config.settings import settings
from src.models.model_loader import LoadedModelContainer, ModelLoader
from src.models.preprocessing import preprocess_face_batch, preprocess_face_crop
from src.utils.logging_config import logger


@dataclass
class FacePredictionResult:
    """Structured inference result for an individual face crop."""
    prediction: str            # "Real" or "Fake"
    confidence: float          # Probability of predicted class [0.0 - 1.0]
    fake_score: float          # Explicit probability of manipulation [0.0 - 1.0]
    real_score: float          # Explicit probability of authenticity [0.0 - 1.0]
    probabilities: Dict[str, float]
    inference_time_ms: float
    model_id: str
    face_index: int = 0


class InferenceEngine:
    """Executes CPU-optimized inference using loaded deepfake models."""

    def __init__(self, loader: Optional[ModelLoader] = None):
        self.loader = loader or ModelLoader.get_instance()

    def predict_face(
        self,
        crop_rgb: np.ndarray,
        model_id: str,
        threshold: float = CONFIDENCE_THRESHOLD,
        face_index: int = 0
    ) -> FacePredictionResult:
        """Run inference on a single RGB face crop."""
        container = self.loader.load_model(model_id)
        if not container.is_ready:
            raise RuntimeError(f"Cannot run inference: {container.error_message}")

        metadata = container.metadata
        model = container.model

        t_start = time.perf_counter()
        with torch.inference_mode():
            if model_id == "vit_deepfake" and container.processor is not None:
                # Use HF processor
                pil_img = Image.fromarray(crop_rgb)
                inputs = container.processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                # Use standard tensor transform
                tensor = preprocess_face_crop(crop_rgb, metadata, device=settings.DEVICE)
                logits = model(tensor)

            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        t_ms = (time.perf_counter() - t_start) * 1000.0

        # Map probabilities using verified model label_map
        fake_idx = metadata.fake_class_index
        real_idx = 1 - fake_idx if len(metadata.label_map) == 2 else 0

        fake_prob = float(probs[fake_idx])
        real_prob = float(probs[real_idx])

        # Classification decision based on configurable threshold
        is_fake = fake_prob >= threshold
        predicted_label = "Fake" if is_fake else "Real"
        confidence = fake_prob if is_fake else real_prob

        prob_dict = {
            metadata.label_map.get(0, "Class_0"): float(probs[0]),
            metadata.label_map.get(1, "Class_1"): float(probs[1])
        }

        return FacePredictionResult(
            prediction=predicted_label,
            confidence=round(confidence, 4),
            fake_score=round(fake_prob, 4),
            real_score=round(real_prob, 4),
            probabilities=prob_dict,
            inference_time_ms=round(t_ms, 2),
            model_id=model_id,
            face_index=face_index
        )

    def predict_batch(
        self,
        crops_rgb: List[np.ndarray],
        model_id: str,
        threshold: float = CONFIDENCE_THRESHOLD,
        batch_size: int = settings.INFERENCE_BATCH_SIZE
    ) -> List[FacePredictionResult]:
        """Run batch inference on multiple crops in bounded chunks."""
        if not crops_rgb:
            return []

        results: List[FacePredictionResult] = []
        for i in range(0, len(crops_rgb), batch_size):
            chunk = crops_rgb[i : i + batch_size]
            for local_idx, crop in enumerate(chunk):
                global_idx = i + local_idx
                res = self.predict_face(
                    crop,
                    model_id=model_id,
                    threshold=threshold,
                    face_index=global_idx
                )
                results.append(res)
        return results
