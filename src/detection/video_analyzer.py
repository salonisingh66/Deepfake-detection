"""Video Deepfake Analyzer for DeepShield AI.

Implements streaming frame sampling, CPU memory bounds, OpenCV resource cleanup,
and per-frame facial inference with timeline telemetry.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np

from config.model_config import CONFIDENCE_THRESHOLD, DEFAULT_MODEL_ID, AVAILABLE_MODELS
from config.settings import settings
from src.detection.aggregation import VideoAggregationSummary, aggregate_video_predictions
from src.detection.face_detector import FaceDetector
from src.forensics.hashing import compute_file_fingerprints
from src.forensics.metadata import extract_video_metadata
from src.models.inference import InferenceEngine
from src.utils.logging_config import logger
from src.utils.validation import validate_video_file


@dataclass
class VideoFrameAnalysis:
    """Individual frame analysis record."""
    frame_index: int
    timestamp: float           # in seconds
    face_detected: bool
    prediction: str            # "Real", "Fake", "No Face"
    fake_score: Optional[float]
    confidence: Optional[float]
    face_crop_rgb: Optional[np.ndarray] = None
    annotated_frame_rgb: Optional[np.ndarray] = None


@dataclass
class VideoAnalysisResult:
    """Complete forensic findings for an analyzed video stream."""
    filename: str
    is_valid: bool
    status: str                         # "SUCCESS", "VALIDATION_FAILED", "ERROR"
    error_message: Optional[str] = None
    prediction: str = "Indeterminate"   # "Real", "Fake", "No Faces Detected"
    confidence: float = 0.0
    summary: Optional[VideoAggregationSummary] = None
    model_name: str = ""
    model_id: str = ""
    sha256: str = ""
    md5: str = ""
    size_bytes: int = 0
    duration_seconds: float = 0.0
    fps: float = 0.0
    total_video_frames: int = 0
    frames_sampled_count: int = 0
    frame_results: List[VideoFrameAnalysis] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class VideoAnalyzer:
    """End-to-end video forensic analysis processor."""

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        inference_engine: Optional[InferenceEngine] = None
    ):
        self.face_detector = face_detector or FaceDetector()
        self.engine = inference_engine or InferenceEngine()

    def analyze_video(
        self,
        video_path: Union[str, Path],
        filename: Optional[str] = None,
        model_id: str = DEFAULT_MODEL_ID,
        sample_interval: int = settings.DEFAULT_FRAME_SAMPLE_INTERVAL,
        max_frames: int = settings.MAX_FRAMES_TO_ANALYZE,
        threshold: float = CONFIDENCE_THRESHOLD,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> VideoAnalysisResult:
        """Stream video frames, run inference on detected faces, and aggregate results."""
        t_start = time.perf_counter()
        path = Path(video_path)
        display_name = filename or path.name

        # 1. Validation
        is_valid, val_err, meta_info = validate_video_file(path, filename=display_name)
        if not is_valid:
            return VideoAnalysisResult(
                filename=display_name,
                is_valid=False,
                status="VALIDATION_FAILED",
                error_message=val_err,
                execution_time_ms=round((time.perf_counter() - t_start) * 1000.0, 2)
            )

        # 2. Cryptographic Fingerprints & Metadata
        fingerprints = compute_file_fingerprints(path)
        video_meta = extract_video_metadata(path)
        video_meta.update(meta_info)

        # 3. Stream frames safely using OpenCV
        cap = cv2.VideoCapture(str(path))
        frame_records: List[VideoFrameAnalysis] = []
        frames_sampled_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps_safe = fps if fps > 0 else 25.0

        try:
            curr_frame_idx = 0
            while cap.isOpened() and frames_sampled_count < max_frames:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                # Sample every Nth frame
                if curr_frame_idx % sample_interval == 0:
                    frames_sampled_count += 1
                    timestamp_sec = round(curr_frame_idx / fps_safe, 2)

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    detected_faces = self.face_detector.detect_faces(frame_rgb)

                    if detected_faces:
                        # Primary face crop
                        primary_face = detected_faces[0]
                        pred = self.engine.predict_face(
                            crop_rgb=primary_face.crop_rgb,
                            model_id=model_id,
                            threshold=threshold,
                            face_index=0
                        )

                        # Downscale thumbnail crop for UI preview memory efficiency
                        thumb = cv2.resize(primary_face.crop_rgb, (128, 128))

                        # Annotate frame
                        annotated = self.face_detector.annotate_image(
                            image_rgb=frame_rgb,
                            faces=[primary_face],
                            labels=[f"{pred.prediction} ({pred.fake_score*100:.1f}%)"],
                            color=(255, 50, 50) if pred.prediction == "Fake" else (50, 200, 50)
                        )
                        # Downscale annotated frame for memory
                        annotated_small = cv2.resize(annotated, (320, int(320 * (annotated.shape[0]/annotated.shape[1]))))

                        frame_records.append(VideoFrameAnalysis(
                            frame_index=curr_frame_idx,
                            timestamp=timestamp_sec,
                            face_detected=True,
                            prediction=pred.prediction,
                            fake_score=pred.fake_score,
                            confidence=pred.confidence,
                            face_crop_rgb=thumb,
                            annotated_frame_rgb=annotated_small
                        ))
                    else:
                        frame_records.append(VideoFrameAnalysis(
                            frame_index=curr_frame_idx,
                            timestamp=timestamp_sec,
                            face_detected=False,
                            prediction="No Face",
                            fake_score=None,
                            confidence=None,
                            face_crop_rgb=None,
                            annotated_frame_rgb=None
                        ))

                    if progress_callback:
                        progress_callback(
                            min(1.0, frames_sampled_count / max_frames),
                            frames_sampled_count,
                            max_frames
                        )

                curr_frame_idx += 1

        except Exception as e:
            logger.error(f"Error during video frame processing of {display_name}: {e}")
            return VideoAnalysisResult(
                filename=display_name,
                is_valid=False,
                status="ERROR",
                error_message=f"Video frame analysis failed: {str(e)}",
                sha256=fingerprints["sha256"],
                md5=fingerprints["md5"]
            )
        finally:
            cap.release()

        # 4. Statistical Score Aggregation
        aggregation_dicts = [
            {
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "face_detected": f.face_detected,
                "fake_score": f.fake_score,
                "prediction": f.prediction
            }
            for f in frame_records
        ]
        summary = aggregate_video_predictions(aggregation_dicts, threshold=threshold)

        t_ms = (time.perf_counter() - t_start) * 1000.0
        model_meta = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[DEFAULT_MODEL_ID])

        return VideoAnalysisResult(
            filename=display_name,
            is_valid=True,
            status="SUCCESS",
            error_message=None,
            prediction=summary.overall_prediction,
            confidence=summary.confidence,
            summary=summary,
            model_name=model_meta.name,
            model_id=model_id,
            sha256=fingerprints["sha256"],
            md5=fingerprints["md5"],
            size_bytes=path.stat().st_size if path.exists() else 0,
            duration_seconds=video_meta.get("duration_seconds", 0.0),
            fps=fps_safe,
            total_video_frames=total_frames,
            frames_sampled_count=len(frame_records),
            frame_results=frame_records,
            metadata=video_meta,
            execution_time_ms=round(t_ms, 2)
        )
