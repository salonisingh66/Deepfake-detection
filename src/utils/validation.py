"""Input validation and security verification for DeepShield AI.

Ensures uploaded files conform to strict security, format, size,
and dimension constraints before processing.
"""

import io
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config.settings import settings
from src.utils.logging_config import logger


class ValidationError(Exception):
    """Raised when an uploaded input fails security or format validation."""
    pass


def validate_image_bytes(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = settings.MAX_IMAGE_SIZE_BYTES,
    max_dimension: int = settings.MAX_IMAGE_DIMENSION
) -> Tuple[bool, Optional[str], Dict]:
    """Validate raw image bytes for security, size, and valid decodability.
    
    Returns:
        (is_valid, error_message, metadata_dict)
    """
    if not file_bytes:
        return False, "Uploaded image is empty (0 bytes).", {}
        
    size_bytes = len(file_bytes)
    if size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        curr_mb = size_bytes / (1024 * 1024)
        return False, f"File size ({curr_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.0f} MB).", {}

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        return (
            False,
            f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}",
            {}
        )

    # Validate image decoding safely using PIL
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()  # Verifies integrity without decoding entire image yet
            
        # Re-open to read properties (verify leaves stream exhausted)
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.size
            img_format = img.format or ext.lstrip(".").upper()
            mode = img.mode

        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions: {width}x{height}", {}

        if width > max_dimension or height > max_dimension:
            return (
                False,
                f"Image dimensions ({width}x{height}) exceed maximum allowed limit of {max_dimension}px.",
                {}
            )

        metadata = {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "size_bytes": size_bytes,
            "extension": ext
        }
        return True, None, metadata

    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        logger.warning(f"Image decoding failed for {filename}: {e}")
        return False, f"Failed to decode image content: file may be corrupted or not a valid image.", {}
    except Exception as e:
        logger.error(f"Unexpected error validating image {filename}: {e}")
        return False, f"Unexpected validation error: {str(e)}", {}


def validate_video_file(
    video_path: Union[str, Path],
    filename: Optional[str] = None,
    max_size_bytes: int = settings.MAX_VIDEO_SIZE_BYTES,
    max_duration_seconds: int = settings.MAX_VIDEO_DURATION_SECONDS
) -> Tuple[bool, Optional[str], Dict]:
    """Validate video file integrity, format, dimensions, and duration using OpenCV.
    
    Returns:
        (is_valid, error_message, metadata_dict)
    """
    path = Path(video_path)
    display_name = filename or path.name

    if not path.exists():
        return False, f"Video file not found at path: {path}", {}

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return False, "Uploaded video is empty (0 bytes).", {}

    if size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        curr_mb = size_bytes / (1024 * 1024)
        return False, f"Video file size ({curr_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.0f} MB).", {}

    ext = path.suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        return (
            False,
            f"Unsupported video extension '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}",
            {}
        )

    # Safely probe using OpenCV
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "Could not open video stream. File may be corrupted or codec unsupported.", {}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        # Check if basic read succeeds on first frame
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            return False, "Unable to read initial video frame. Stream may be empty or corrupted.", {}

        # Calculate duration
        duration_seconds = 0.0
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps

        if duration_seconds > max_duration_seconds:
            return (
                False,
                f"Video duration ({duration_seconds:.1f}s) exceeds maximum allowed limit of {max_duration_seconds}s.",
                {}
            )

        metadata = {
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": round(fps, 2) if fps > 0 else 0.0,
            "duration_seconds": round(duration_seconds, 2),
            "fourcc": fourcc_str or "Unknown",
            "size_bytes": size_bytes,
            "extension": ext
        }
        return True, None, metadata

    except Exception as e:
        logger.error(f"Error inspecting video {display_name}: {e}")
        return False, f"Failed to inspect video stream: {str(e)}", {}
    finally:
        cap.release()
