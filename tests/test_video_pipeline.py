"""Integration and unit tests for video deepfake analysis pipeline."""

from pathlib import Path
import pytest
from src.detection.video_analyzer import VideoAnalyzer


@pytest.fixture
def video_analyzer():
    return VideoAnalyzer()


def test_video_pipeline_with_face(video_analyzer):
    vid_path = Path("data/samples/sample_video_with_face.mp4")
    assert vid_path.exists(), "Sample video with face must exist"

    res = video_analyzer.analyze_video(
        video_path=vid_path,
        sample_interval=5,
        max_frames=10
    )
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert res.frames_sampled_count > 0
    assert res.summary is not None
    assert res.summary.frames_with_faces > 0
    assert res.prediction in ("Real", "Fake")
    assert len(res.frame_results) > 0
    # Check that individual frame records have timestamps and face flags
    f0 = res.frame_results[0]
    assert hasattr(f0, "timestamp")
    assert hasattr(f0, "prediction")


def test_video_pipeline_no_face(video_analyzer):
    vid_no_face = Path("data/samples/sample_video_no_face.mp4")
    assert vid_no_face.exists(), "Sample video without face must exist"

    res = video_analyzer.analyze_video(
        video_path=vid_no_face,
        sample_interval=5,
        max_frames=10
    )
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert res.prediction == "No Faces Detected"
    assert res.summary.frames_with_faces == 0


def test_video_pipeline_invalid_path(video_analyzer):
    res = video_analyzer.analyze_video("non_existent_file.mp4")
    assert res.is_valid is False
    assert res.status == "VALIDATION_FAILED"
    assert "not found" in res.error_message
