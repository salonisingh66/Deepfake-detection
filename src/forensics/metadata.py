"""Forensic metadata extraction and provenance inspection.

Extracts container, camera, software, and compression characteristics
from image and video files. Distinguishes checked-and-absent data from errors.
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
from PIL import Image, ExifTags

from src.utils.logging_config import logger


def extract_image_metadata(
    image_source: Union[bytes, str, Path]
) -> Dict[str, Any]:
    """Extract technical and forensic metadata from an image.
    
    Includes dimensions, color space, EXIF data, software tags, and forensic notes.
    """
    metadata: Dict[str, Any] = {
        "media_type": "image",
        "dimensions": None,
        "format": None,
        "mode": None,
        "has_exif": False,
        "exif_tags": {},
        "software_or_device": None,
        "creation_timestamp": None,
        "forensic_observations": [],
    }

    try:
        if isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        else:
            img = Image.open(str(image_source))

        metadata["dimensions"] = f"{img.width}x{img.height} px"
        metadata["format"] = img.format or "Unknown"
        metadata["mode"] = img.mode

        # Check for EXIF metadata
        raw_exif = None
        try:
            raw_exif = img.getexif()
        except Exception:
            raw_exif = None

        if raw_exif and len(raw_exif) > 0:
            metadata["has_exif"] = True
            parsed_tags = {}
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Truncate overly long binary tags
                val_str = str(value)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "... [truncated]"
                parsed_tags[tag_name] = val_str

            metadata["exif_tags"] = parsed_tags

            # Extract common software or device signatures
            device_parts = []
            if "Make" in parsed_tags:
                device_parts.append(parsed_tags["Make"])
            if "Model" in parsed_tags:
                device_parts.append(parsed_tags["Model"])
            if device_parts:
                metadata["software_or_device"] = " ".join(device_parts)

            if "Software" in parsed_tags:
                software_str = parsed_tags["Software"]
                metadata["software_or_device"] = (
                    f"{metadata['software_or_device']} (Software: {software_str})"
                    if metadata["software_or_device"]
                    else software_str
                )

            if "DateTime" in parsed_tags:
                metadata["creation_timestamp"] = parsed_tags["DateTime"]
            elif "DateTimeOriginal" in parsed_tags:
                metadata["creation_timestamp"] = parsed_tags["DateTimeOriginal"]

            metadata["forensic_observations"].append(
                "EXIF metadata is present in container. Note: EXIF data can be easily edited or preserved during editing."
            )
        else:
            metadata["has_exif"] = False
            metadata["forensic_observations"].append(
                "EXIF metadata was checked and found to be absent. "
                "Caveat: Most social media platforms (WhatsApp, Twitter, Instagram) automatically strip EXIF. "
                "Absence of EXIF does NOT indicate deepfake generation."
            )

        # Transparency or alpha channel check
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            metadata["forensic_observations"].append(
                "Image contains an alpha channel (transparency). Common in graphic composites and sticker overlays."
            )

        img.close()

    except Exception as e:
        logger.error(f"Error extracting image metadata: {e}")
        metadata["forensic_observations"].append(f"Metadata parsing encountered an issue: {str(e)}")

    return metadata


def extract_video_metadata(
    video_path: Union[str, Path]
) -> Dict[str, Any]:
    """Extract technical and container metadata from a video file.
    
    Includes codec FourCC, dimensions, frame rate, duration, and forensic observations.
    """
    path = Path(video_path)
    metadata: Dict[str, Any] = {
        "media_type": "video",
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "dimensions": None,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "fourcc_codec": "Unknown",
        "forensic_observations": [],
    }

    if not path.exists():
        metadata["forensic_observations"].append("Target video file does not exist on disk.")
        return metadata

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            metadata["forensic_observations"].append("Could not open video stream via OpenCV.")
            return metadata

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        metadata["dimensions"] = f"{width}x{height} px"
        metadata["fps"] = round(fps, 2) if fps > 0 else 0.0
        metadata["frame_count"] = frame_count
        metadata["fourcc_codec"] = fourcc_str or "Unknown"

        duration = 0.0
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
        metadata["duration_seconds"] = round(duration, 2)

        # Forensic observations
        metadata["forensic_observations"].append(
            f"Container reporting {frame_count} frames at {metadata['fps']} FPS (~{metadata['duration_seconds']}s)."
        )
        if fourcc_str:
            metadata["forensic_observations"].append(f"Reported FourCC codec tag: '{fourcc_str}'.")

        if metadata["fps"] > 0 and metadata["fps"] not in (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0):
            metadata["forensic_observations"].append(
                f"Non-standard frame rate detected ({metadata['fps']} FPS). Often indicates screen recording, re-encoding, or variable frame-rate rendering."
            )

    except Exception as e:
        logger.error(f"Error extracting video metadata from {path}: {e}")
        metadata["forensic_observations"].append(f"OpenCV metadata inspection error: {str(e)}")
    finally:
        cap.release()

    return metadata
