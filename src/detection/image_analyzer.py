"""End-to-End Image Deepfake Forensic Analyzer for DeepShield AI.

Coordinates validation, cryptographic hashing, metadata extraction,
multi-face detection, model inference, and visual explanation overlay.
"""

from dataclasses import dataclass, field
import io
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from config.model_config import CONFIDENCE_THRESHOLD, DEFAULT_MODEL_ID, AVAILABLE_MODELS
from config.settings import settings
from src.detection.face_detector import DetectedFace, FaceDetector
from src.forensics.explanations import generate_vit_attention_map
from src.forensics.hashing import compute_file_fingerprints
from src.forensics.metadata import extract_image_metadata
from src.models.inference import FacePredictionResult, InferenceEngine
from src.models.model_loader import ModelLoader
from src.utils.logging_config import logger
from src.utils.validation import validate_image_bytes


@dataclass
class ImageAnalysisResult:
    """Complete forensic findings for an analyzed image."""
    filename: str
    is_valid: bool
    status: str                         # "SUCCESS", "NO_FACES_DETECTED", "VALIDATION_FAILED", "ERROR"
    error_message: Optional[str] = None
    prediction: str = "Indeterminate"   # "Real" or "Fake"
    confidence: float = 0.0             # [0.0 - 1.0]
    fake_score: float = 0.0             # [0.0 - 1.0]
    model_name: str = ""
    model_id: str = ""
    sha256: str = ""
    md5: str = ""
    size_bytes: int = 0
    dimensions: str = ""
    faces_detected_count: int = 0
    face_predictions: List[FacePredictionResult] = field(default_factory=list)
    annotated_image_rgb: Optional[np.ndarray] = None
    explanation_overlay_rgb: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class ImageAnalyzer:
    """Forensic pipeline for static image inspection."""

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        inference_engine: Optional[InferenceEngine] = None
    ):
        self.face_detector = face_detector or FaceDetector()
        self.engine = inference_engine or InferenceEngine()

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = CONFIDENCE_THRESHOLD,
        generate_explanation: bool = True
    ) -> ImageAnalysisResult:
        """Execute full forensic analysis pipeline on uploaded image bytes."""
        t_start = time.perf_counter()

        # 1. Validation
        is_valid, val_err, meta_info = validate_image_bytes(image_bytes, filename)
        if not is_valid:
            return ImageAnalysisResult(
                filename=filename,
                is_valid=False,
                status="VALIDATION_FAILED",
                error_message=val_err,
                execution_time_ms=round((time.perf_counter() - t_start) * 1000.0, 2)
            )

        # 2. Cryptographic Fingerprints & Metadata
        fingerprints = compute_file_fingerprints(image_bytes)
        img_meta = extract_image_metadata(image_bytes)
        img_meta.update(meta_info)

        # 3. Decode Image into RGB NumPy Array
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                image_rgb = np.array(pil_img.convert("RGB"))
        except Exception as e:
            return ImageAnalysisResult(
                filename=filename,
                is_valid=False,
                status="ERROR",
                error_message=f"Failed to decode image pixels: {str(e)}",
                sha256=fingerprints["sha256"],
                md5=fingerprints["md5"]
            )

        # 4. Face Detection
        faces = self.face_detector.detect_faces(image_rgb)
        if not faces:
            logger.info(f"No face detected in {filename}.")
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return ImageAnalysisResult(
                filename=filename,
                is_valid=True,
                status="NO_FACES_DETECTED",
                error_message=(
                    "No human facial region was identified in this image. "
                    "DeepShield AI requires detectable facial features (jawline, eyes, nose) "
                    "for deepfake forensic classification. Full-image background analysis is unsupported."
                ),
                prediction="No Face Detected",
                confidence=0.0,
                fake_score=0.0,
                model_name=AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[DEFAULT_MODEL_ID]).name,
                model_id=model_id,
                sha256=fingerprints["sha256"],
                md5=fingerprints["md5"],
                size_bytes=len(image_bytes),
                dimensions=f"{image_rgb.shape[1]}x{image_rgb.shape[0]} px",
                faces_detected_count=0,
                metadata=img_meta,
                execution_time_ms=round(t_ms, 2)
            )

        # 5. Run Model Inference on Detected Faces
        face_predictions: List[FacePredictionResult] = []
        labels_for_annotation = []

        for idx, face in enumerate(faces):
            pred = self.engine.predict_face(
                crop_rgb=face.crop_rgb,
                model_id=model_id,
                threshold=threshold,
                face_index=idx
            )
            face_predictions.append(pred)
            labels_for_annotation.append(f"{pred.prediction} ({pred.fake_score*100:.1f}%)")

        # 6. Overall Image Verdict Aggregation
        # If any face in the image is classified as Fake, flag image with the highest fake score
        highest_fake_pred = max(face_predictions, key=lambda p: p.fake_score)
        if highest_fake_pred.fake_score >= threshold:
            overall_prediction = "Fake"
            overall_confidence = highest_fake_pred.fake_score
            overall_fake_score = highest_fake_pred.fake_score
        else:
            overall_prediction = "Real"
            overall_confidence = 1.0 - highest_fake_pred.fake_score
            overall_fake_score = highest_fake_pred.fake_score

        # 7. Annotate Image with Bounding Boxes
        box_color = (255, 50, 50) if overall_prediction == "Fake" else (50, 200, 50)
        annotated_rgb = self.face_detector.annotate_image(
            image_rgb=image_rgb,
            faces=faces,
            labels=labels_for_annotation,
            color=box_color
        )

        # 8. Visual Explanation (Attention Overlay on Primary Face Crop)
        explanation_overlay = None
        if generate_explanation and model_id == "vit_deepfake":
            primary_crop = faces[0].crop_rgb
            try:
                container = self.engine.loader.load_model(model_id)
                if container.is_ready and container.processor is not None:
                    pil_crop = Image.fromarray(primary_crop)
                    inputs = container.processor(images=pil_crop, return_tensors="pt")
                    inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
                    explanation_overlay = generate_vit_attention_map(
                        model=container.model,
                        inputs=inputs,
                        original_image_rgb=primary_crop
                    )
            except Exception as e:
                logger.warning(f"Could not generate visual explanation for {filename}: {e}")

        t_ms = (time.perf_counter() - t_start) * 1000.0
        model_meta = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[DEFAULT_MODEL_ID])

        return ImageAnalysisResult(
            filename=filename,
            is_valid=True,
            status="SUCCESS",
            error_message=None,
            prediction=overall_prediction,
            confidence=round(overall_confidence, 4),
            fake_score=round(overall_fake_score, 4),
            model_name=model_meta.name,
            model_id=model_id,
            sha256=fingerprints["sha256"],
            md5=fingerprints["md5"],
            size_bytes=len(image_bytes),
            dimensions=f"{image_rgb.shape[1]}x{image_rgb.shape[0]} px",
            faces_detected_count=len(faces),
            face_predictions=face_predictions,
            annotated_image_rgb=annotated_rgb,
            explanation_overlay_rgb=explanation_overlay,
            metadata=img_meta,
            execution_time_ms=round(t_ms, 2)
        )
