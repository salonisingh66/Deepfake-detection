"""Video Deepfake Forensic Score Aggregation Strategy.

Combines sequential frame-level model predictions into an honest,
documented video-level forensic assessment.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class VideoAggregationSummary:
    """Documented aggregation summary for video deepfake analysis."""
    frames_sampled: int
    frames_with_faces: int
    frames_analyzed: int
    mean_score: float
    median_score: float
    score_std: float
    fake_frame_ratio: float
    highest_frame_score: float
    lowest_frame_score: float
    overall_prediction: str         # "Fake", "Real", or "Inconclusive"
    confidence: float
    aggregation_notes: List[str] = field(default_factory=list)


def aggregate_video_predictions(
    frame_results: List[Dict[str, Any]],
    threshold: float = 0.50
) -> VideoAggregationSummary:
    """Aggregate frame-level detection scores using robust statistical measures.
    
    Args:
        frame_results: List of dicts with keys:
            'frame_index', 'timestamp', 'face_detected', 'fake_score', 'prediction'
        threshold: Score threshold above which a frame is considered manipulated.
    """
    total_sampled = len(frame_results)
    if total_sampled == 0:
        return VideoAggregationSummary(
            frames_sampled=0,
            frames_with_faces=0,
            frames_analyzed=0,
            mean_score=0.0,
            median_score=0.0,
            score_std=0.0,
            fake_frame_ratio=0.0,
            highest_frame_score=0.0,
            lowest_frame_score=0.0,
            overall_prediction="Inconclusive",
            confidence=0.0,
            aggregation_notes=["No frames were sampled from the video stream."]
        )

    # Filter to frames where faces were detected and scored
    valid_scores = [
        f["fake_score"] for f in frame_results
        if f.get("face_detected", False) and f.get("fake_score") is not None
    ]

    frames_with_faces = len(valid_scores)
    notes: List[str] = []

    if frames_with_faces == 0:
        notes.append("No facial regions were detected across any sampled frames.")
        return VideoAggregationSummary(
            frames_sampled=total_sampled,
            frames_with_faces=0,
            frames_analyzed=0,
            mean_score=0.0,
            median_score=0.0,
            score_std=0.0,
            fake_frame_ratio=0.0,
            highest_frame_score=0.0,
            lowest_frame_score=0.0,
            overall_prediction="No Faces Detected",
            confidence=0.0,
            aggregation_notes=notes
        )

    scores_arr = np.array(valid_scores)
    mean_val = float(np.mean(scores_arr))
    median_val = float(np.median(scores_arr))
    std_val = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0
    highest_val = float(np.max(scores_arr))
    lowest_val = float(np.min(scores_arr))

    fake_frames_count = sum(1 for s in valid_scores if s >= threshold)
    fake_frame_ratio = fake_frames_count / frames_with_faces

    # Forensic Decision Rules:
    # 1. In facial deepfakes, manipulations may appear only in subsets of frames (face-swap glitching).
    # 2. However, consecutive frames are temporally correlated; they are NOT independent statistical trials.
    # 3. If a significant proportion of analyzed faces (> 30%) or sustained high confidence (> 0.75 median)
    #    indicates manipulation, the video is classified as manipulated.
    notes.append(
        "Correlated Frame Caveat: Consecutive frames in video streams are temporally correlated. "
        "Frame-level averages represent statistical central tendency across samples, "
        "not a calibrated Bayesian probability of the entire video."
    )

    if fake_frame_ratio >= 0.35 or median_val >= threshold:
        overall_prediction = "Fake"
        confidence = max(median_val, fake_frame_ratio)
        notes.append(
            f"{fake_frames_count}/{frames_with_faces} analyzed frames ({fake_frame_ratio*100:.1f}%) "
            f"exceeded manipulation threshold ({threshold})."
        )
    else:
        overall_prediction = "Real"
        confidence = 1.0 - median_val
        notes.append(
            f"Majority of analyzed frames ({100 - fake_frame_ratio*100:.1f}%) "
            f"exhibit authentic facial characteristics (median fake score: {median_val:.3f})."
        )

    if std_val > 0.25:
        notes.append(
            f"High score variation (std: {std_val:.3f}) observed across frames. "
            "Suggests intermittent manipulation artifacts, changing lighting, or head rotation."
        )

    return VideoAggregationSummary(
        frames_sampled=total_sampled,
        frames_with_faces=frames_with_faces,
        frames_analyzed=frames_with_faces,
        mean_score=round(mean_val, 4),
        median_score=round(median_val, 4),
        score_std=round(std_val, 4),
        fake_frame_ratio=round(fake_frame_ratio, 4),
        highest_frame_score=round(highest_val, 4),
        lowest_frame_score=round(lowest_val, 4),
        overall_prediction=overall_prediction,
        confidence=round(confidence, 4),
        aggregation_notes=notes
    )
