"""Application Settings and Configuration for DeepShield AI.

All paths, limits, and operational parameters are centralized here.
Configured for efficient CPU-only inference on standard hardware (12GB RAM).
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class AppSettings:
    """Core application settings."""
    APP_NAME: str = "DeepShield AI"
    APP_TAGLINE: str = "Deepfake Detection & Digital Forensics Analysis System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("APP_ENV", "production")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    EVAL_DATA_DIR: Path = BASE_DIR / "data" / "evaluation"
    MODELS_DIR: Path = BASE_DIR / "models"
    CHECKPOINTS_DIR: Path = BASE_DIR / "models" / "checkpoints"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    TEMP_DIR: Path = BASE_DIR / "data" / "temp"
    DATABASE_PATH: Path = BASE_DIR / "data" / "deepshield_history.db"

    # Security & Input Validation Limits
    MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB
    MAX_IMAGE_DIMENSION: int = 4096  # Max width or height
    MIN_FACE_CROP_SIZE: int = 40  # Minimum face width/height in pixels for reliable inference
    
    # Supported File Extensions
    ALLOWED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    ALLOWED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
    
    # Video Processing Parameters (CPU Optimized)
    DEFAULT_FRAME_SAMPLE_INTERVAL: int = 15  # Sample every 15th frame
    MIN_FRAME_SAMPLE_INTERVAL: int = 1
    MAX_FRAME_SAMPLE_INTERVAL: int = 60
    MAX_FRAMES_TO_ANALYZE: int = 60  # Upper bound on sampled frames to protect CPU & memory
    MAX_VIDEO_DURATION_SECONDS: int = 300  # Max 5 minutes video
    
    # Batch Analysis
    MAX_BATCH_IMAGE_COUNT: int = 20
    
    # Computation / Hardware
    DEVICE: str = "cpu"  # Explicitly CPU
    NUM_WORKERS: int = 0
    INFERENCE_BATCH_SIZE: int = 8  # Bounded batching for CPU memory efficiency

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))


# Initialize default settings instance
settings = AppSettings()

# Ensure required runtime directories exist safely
for path in (
    settings.DATA_DIR,
    settings.SAMPLES_DIR,
    settings.EVAL_DATA_DIR,
    settings.MODELS_DIR,
    settings.CHECKPOINTS_DIR,
    settings.REPORTS_DIR,
    settings.TEMP_DIR,
):
    path.mkdir(parents=True, exist_ok=True)
