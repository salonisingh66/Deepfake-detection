"""FastAPI Inference Service for DeepShield AI.

Provides RESTful endpoints for image and video deepfake detection with
strict input validation, security controls, and JSON serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL_ID
from config.settings import settings
from src.detection.image_analyzer import ImageAnalyzer
from src.detection.video_analyzer import VideoAnalyzer
from src.models.model_loader import ModelLoader
from src.utils.file_utils import safe_temp_file, sanitize_filename
from src.utils.logging_config import logger

app = FastAPI(
    title="DeepShield AI - Deepfake Forensic API",
    description="REST API for automated image and video facial manipulation forensics.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global analyzer instances
model_loader = ModelLoader.get_instance()
image_analyzer = ImageAnalyzer()
video_analyzer = VideoAnalyzer()


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_ready: bool
    default_model: str


class ModelInfoItem(BaseModel):
    id: str
    name: str
    architecture: str
    source: str
    license: str
    input_size: List[int]
    label_map: Dict[int, str]
    known_limitations: List[str]


class ImagePredictionResponse(BaseModel):
    filename: str
    status: str
    prediction: str
    confidence: float
    fake_score: float
    faces_detected: int
    sha256: str
    dimensions: str
    model_name: str
    execution_time_ms: float
    metadata: Dict[str, Any]


class VideoPredictionResponse(BaseModel):
    filename: str
    status: str
    prediction: str
    confidence: float
    mean_frame_score: float
    median_frame_score: float
    score_variation_std: float
    fake_frame_ratio: float
    frames_sampled: int
    frames_with_faces: int
    duration_seconds: float
    fps: float
    sha256: str
    model_name: str
    execution_time_ms: float
    aggregation_notes: List[str]


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Verify API operational health and model readiness status."""
    is_ready, _ = model_loader.is_model_ready(DEFAULT_MODEL_ID)
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.VERSION,
        model_ready=is_ready,
        default_model=DEFAULT_MODEL_ID
    )


@app.get("/model-info", tags=["Models"])
def get_model_info():
    """Retrieve specifications and architectural metadata for all available models."""
    models_data = []
    for m in AVAILABLE_MODELS.values():
        models_data.append({
            "id": m.id,
            "name": m.name,
            "architecture": m.architecture,
            "source": m.source,
            "license": m.license,
            "input_size": list(m.input_size),
            "label_map": m.label_map,
            "known_limitations": m.known_limitations,
            "is_default": m.is_default
        })
    return {"models": models_data, "default_model_id": DEFAULT_MODEL_ID}


@app.post("/predict/image", response_model=ImagePredictionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    model_id: str = Form(DEFAULT_MODEL_ID, description="Target model ID")
):
    """Analyze an uploaded image for facial deepfake manipulation."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_id '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    safe_name = sanitize_filename(file.filename or "upload.jpg")
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if len(content) > settings.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds maximum limit of {settings.MAX_IMAGE_SIZE_BYTES / (1024*1024):.0f} MB."
        )

    res = image_analyzer.analyze_image(
        image_bytes=content,
        filename=safe_name,
        model_id=model_id,
        generate_explanation=False
    )

    if not res.is_valid:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=res.error_message)

    return ImagePredictionResponse(
        filename=res.filename,
        status=res.status,
        prediction=res.prediction,
        confidence=res.confidence,
        fake_score=res.fake_score,
        faces_detected=res.faces_detected_count,
        sha256=res.sha256,
        dimensions=res.dimensions,
        model_name=res.model_name,
        execution_time_ms=res.execution_time_ms,
        metadata=res.metadata
    )


@app.post("/predict/video", response_model=VideoPredictionResponse, tags=["Inference"])
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI)"),
    model_id: str = Form(DEFAULT_MODEL_ID, description="Target model ID"),
    sample_interval: int = Form(settings.DEFAULT_FRAME_SAMPLE_INTERVAL, ge=1, le=60),
    max_frames: int = Form(settings.MAX_FRAMES_TO_ANALYZE, ge=5, le=100)
):
    """Sample and analyze video frames for facial manipulation."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_id '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    safe_name = sanitize_filename(file.filename or "upload.mp4")
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded video file is empty.")

    if len(content) > settings.MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video size exceeds limit of {settings.MAX_VIDEO_SIZE_BYTES / (1024*1024):.0f} MB."
        )

    ext = Path(safe_name).suffix.lower()
    with safe_temp_file(suffix=ext) as temp_video_path:
        with open(temp_video_path, "wb") as f:
            f.write(content)

        res = video_analyzer.analyze_video(
            video_path=temp_video_path,
            filename=safe_name,
            model_id=model_id,
            sample_interval=sample_interval,
            max_frames=max_frames
        )

    if not res.is_valid:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=res.error_message)

    summary = res.summary
    return VideoPredictionResponse(
        filename=res.filename,
        status=res.status,
        prediction=res.prediction,
        confidence=res.confidence,
        mean_frame_score=summary.mean_score if summary else 0.0,
        median_frame_score=summary.median_score if summary else 0.0,
        score_variation_std=summary.score_std if summary else 0.0,
        fake_frame_ratio=summary.fake_frame_ratio if summary else 0.0,
        frames_sampled=summary.frames_sampled if summary else 0,
        frames_with_faces=summary.frames_with_faces if summary else 0,
        duration_seconds=res.duration_seconds,
        fps=res.fps,
        sha256=res.sha256,
        model_name=res.model_name,
        execution_time_ms=res.execution_time_ms,
        aggregation_notes=summary.aggregation_notes if summary else []
    )
