"""API endpoint tests using FastAPI TestClient."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "DeepShield AI"
    assert data["model_ready"] is True


def test_api_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    assert data["default_model_id"] == "vit_deepfake"


def test_api_predict_image_success():
    img_path = Path("data/samples/sample_face_real.jpg")
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post(
            "/predict/image",
            files={"file": ("sample_face_real.jpg", f, "image/jpeg")},
            data={"model_id": "vit_deepfake"}
        )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert res["prediction"] in ("Real", "Fake")
    assert 0.0 <= res["fake_score"] <= 1.0
    assert len(res["sha256"]) == 64


def test_api_predict_image_empty():
    response = client.post(
        "/predict/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_predict_image_invalid_format():
    response = client.post(
        "/predict/image",
        files={"file": ("malicious.sh", b"echo 'hack'", "text/plain")}
    )
    assert response.status_code == 422


def test_api_predict_video_success():
    vid_path = Path("data/samples/sample_video_with_face.mp4")
    assert vid_path.exists()

    with open(vid_path, "rb") as f:
        response = client.post(
            "/predict/video",
            files={"file": ("sample_video_with_face.mp4", f, "video/mp4")},
            data={"model_id": "vit_deepfake", "sample_interval": 5, "max_frames": 5}
        )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert res["prediction"] in ("Real", "Fake")
    assert res["frames_sampled"] > 0
