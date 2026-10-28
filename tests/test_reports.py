"""Unit tests for forensic PDF report generation."""

from pathlib import Path
from src.forensics.report_generator import MANDATORY_DISCLAIMER, generate_forensic_pdf


def test_generate_image_forensic_pdf(tmp_path):
    report_data = {
        "report_id": "DS-TEST-001",
        "timestamp": "2026-10-28 05:00:00 UTC",
        "filename": "suspect_image.jpg",
        "media_type": "IMAGE",
        "prediction": "Fake",
        "confidence": 0.942,
        "fake_score": 0.942,
        "model_name": "ViT Deepfake Forensic Classifier",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "md5": "d41d8cd98f00b204e9800998ecf8427e",
        "size_bytes": 1048576,
        "metadata_observations": [
            "Image dimensions: 512x512 px",
            "EXIF metadata was checked and found to be absent."
        ],
        "limitations": [
            "Sensitivity decreases on heavily compressed video frames."
        ]
    }

    out_file = tmp_path / "test_report.pdf"
    pdf_bytes = generate_forensic_pdf(report_data, output_path=out_file)

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert out_file.exists()
    assert out_file.stat().st_size == len(pdf_bytes)


def test_generate_video_forensic_pdf(tmp_path):
    report_data = {
        "report_id": "DS-TEST-002",
        "timestamp": "2026-10-28 05:00:00 UTC",
        "filename": "suspect_video.mp4",
        "media_type": "VIDEO",
        "prediction": "Real",
        "confidence": 0.825,
        "fake_score": 0.175,
        "model_name": "ViT Deepfake Forensic Classifier",
        "sha256": "f4b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b899",
        "md5": "a41d8cd98f00b204e9800998ecf8427a",
        "size_bytes": 5242880,
        "metadata_observations": ["25 FPS, duration 2.5s"],
        "video_stats": {
            "frames_sampled": 10,
            "frames_with_faces": 10,
            "mean_score": 0.18,
            "median_score": 0.175,
            "score_std": 0.04,
            "fake_frame_ratio": 0.0
        },
        "frame_predictions": [
            {"frame_index": i * 15, "timestamp": i * 0.5, "face_detected": True, "prediction": "Real", "fake_score": 0.18}
            for i in range(10)
        ]
    }

    pdf_bytes = generate_forensic_pdf(report_data)
    assert len(pdf_bytes) > 1500
    assert pdf_bytes.startswith(b"%PDF-")
    assert "sole basis for legal" in MANDATORY_DISCLAIMER
