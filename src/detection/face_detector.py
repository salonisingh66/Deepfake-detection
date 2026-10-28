"""Dedicated Face Detection and Bounding Box Extraction Module.

Handles multi-face detection, margin expansion for boundary artifacts,
minimum size thresholds, and CPU-efficient face cropping.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config.settings import settings
from src.utils.logging_config import logger


@dataclass(frozen=True)
class FaceBoundingBox:
    """Coordinates and dimensions of a detected facial region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def box_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectedFace:
    """A detected face with its extracted crop and coordinates."""
    bbox: FaceBoundingBox
    crop_rgb: np.ndarray
    original_size: Tuple[int, int]
    face_index: int


class FaceDetector:
    """Robust, CPU-efficient face detector using OpenCV Haar Cascades with multi-scale inspection."""

    def __init__(
        self,
        min_face_size: int = settings.MIN_FACE_CROP_SIZE,
        margin_percentage: float = 0.20
    ):
        self.min_face_size = min_face_size
        self.margin_percentage = margin_percentage
        
        # Load frontal and profile face cascades
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        
        self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        
        if self.frontal_cascade.empty():
            logger.error("Failed to load OpenCV frontal face Haar Cascade.")
        else:
            logger.debug("Initialized FaceDetector with OpenCV Haar Cascade.")

    def detect_faces(self, image_rgb: np.ndarray) -> List[DetectedFace]:
        """Detect all valid faces in an RGB image.
        
        Returns list of DetectedFace objects sorted by area (largest to smallest).
        """
        if image_rgb is None or image_rgb.size == 0:
            return []

        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Histogram equalization for contrast invariance
        gray_eq = cv2.equalizeHist(gray)

        # Detect frontal faces
        raw_faces = self.frontal_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected_boxes: List[FaceBoundingBox] = []
        if len(raw_faces) > 0:
            for (x, y, fw, fh) in raw_faces:
                detected_boxes.append(FaceBoundingBox(x=int(x), y=int(y), width=int(fw), height=int(fh), confidence=0.95))
        else:
            # Fallback to profile cascade if no frontal faces found
            profile_faces = self.profile_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(self.min_face_size, self.min_face_size)
            )
            for (x, y, fw, fh) in profile_faces:
                detected_boxes.append(FaceBoundingBox(x=int(x), y=int(y), width=int(fw), height=int(fh), confidence=0.85))

        if not detected_boxes:
            return []

        # Sort by area descending (primary subject first)
        detected_boxes.sort(key=lambda b: b.area, reverse=True)

        faces: List[DetectedFace] = []
        for idx, box in enumerate(detected_boxes):
            # Apply margin padding to capture jawline, hairline, and blending seams
            margin_x = int(box.width * self.margin_percentage)
            margin_y = int(box.height * self.margin_percentage)

            x1 = max(0, box.x - margin_x)
            y1 = max(0, box.y - margin_y)
            x2 = min(w, box.x + box.width + margin_x)
            y2 = min(h, box.y + box.height + margin_y)

            crop = image_rgb[y1:y2, x1:x2].copy()
            if crop.shape[0] < self.min_face_size or crop.shape[1] < self.min_face_size:
                continue

            padded_box = FaceBoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1, confidence=box.confidence)
            faces.append(DetectedFace(
                bbox=padded_box,
                crop_rgb=crop,
                original_size=(h, w),
                face_index=idx
            ))

        return faces

    def annotate_image(
        self,
        image_rgb: np.ndarray,
        faces: List[DetectedFace],
        labels: Optional[List[str]] = None,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw bounding boxes and optional labels on an image."""
        annotated = image_rgb.copy()
        for idx, face in enumerate(faces):
            box = face.bbox
            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                2
            )
            label = labels[idx] if labels and idx < len(labels) else f"Face #{idx+1}"
            cv2.putText(
                annotated,
                label,
                (box.x, max(15, box.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
        return annotated
