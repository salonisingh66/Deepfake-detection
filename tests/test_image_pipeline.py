"""Integration and unit tests for the image deepfake analysis pipeline."""

from pathlib import Path
import pytest
from src.detection.image_analyzer import ImageAnalyzer
from src.models.model_loader import ModelLoader


@pytest.fixture
def analyzer():
    return ImageAnalyzer()


def test_image_pipeline_no_face(analyzer):
    no_face_path = Path("data/samples/sample_no_face.jpg")
    assert no_face_path.exists(), "Sample no-face image must exist"
    
    with open(no_face_path, "rb") as f:
        content = f.read()

    res = analyzer.analyze_image(
        image_bytes=content,
        filename="sample_no_face.jpg",
        generate_explanation=False
    )
    assert res.is_valid is True
    assert res.status == "NO_FACES_DETECTED"
    assert res.faces_detected_count == 0
    assert "No human facial region" in res.error_message
    assert len(res.sha256) == 64


def test_image_pipeline_with_face(analyzer):
    face_path = Path("data/samples/sample_face_real.jpg")
    assert face_path.exists(), "Sample face image must exist"

    with open(face_path, "rb") as f:
        content = f.read()

    res = analyzer.analyze_image(
        image_bytes=content,
        filename="sample_face_real.jpg",
        generate_explanation=True
    )
    assert res.is_valid is True
    assert res.status == "SUCCESS"
    assert res.faces_detected_count >= 1
    assert res.prediction in ("Real", "Fake")
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.fake_score <= 1.0
    assert res.annotated_image_rgb is not None
    assert res.execution_time_ms > 0


def test_image_pipeline_corrupt_data(analyzer):
    corrupt_bytes = b"CORRUPTED_NON_IMAGE_DATA_000"
    res = analyzer.analyze_image(
        image_bytes=corrupt_bytes,
        filename="bad.png"
    )
    assert res.is_valid is False
    assert res.status == "VALIDATION_FAILED"
    assert "Failed to decode" in res.error_message


def test_missing_model_handling(analyzer):
    loader = ModelLoader.get_instance()
    is_ready, err = loader.is_model_ready("non_existent_model_id")
    assert is_ready is False
    assert "Unknown model ID" in err
