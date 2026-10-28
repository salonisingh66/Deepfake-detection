"""Unit tests for video aggregation and statistical score calculation."""

from src.detection.aggregation import aggregate_video_predictions


def test_empty_frame_results():
    summary = aggregate_video_predictions([])
    assert summary.frames_sampled == 0
    assert summary.frames_with_faces == 0
    assert summary.overall_prediction == "Inconclusive"


def test_frames_without_faces():
    frames = [
        {"frame_index": 0, "timestamp": 0.0, "face_detected": False, "fake_score": None, "prediction": "No Face"},
        {"frame_index": 15, "timestamp": 0.5, "face_detected": False, "fake_score": None, "prediction": "No Face"},
    ]
    summary = aggregate_video_predictions(frames)
    assert summary.frames_sampled == 2
    assert summary.frames_with_faces == 0
    assert summary.overall_prediction == "No Faces Detected"


def test_predominantly_real_video():
    frames = [
        {"frame_index": i * 10, "timestamp": i * 0.33, "face_detected": True, "fake_score": 0.15 + (i * 0.02), "prediction": "Real"}
        for i in range(10)
    ]
    summary = aggregate_video_predictions(frames, threshold=0.50)
    assert summary.frames_sampled == 10
    assert summary.frames_with_faces == 10
    assert summary.overall_prediction == "Real"
    assert summary.fake_frame_ratio == 0.0
    assert summary.mean_score < 0.50
    assert any("Correlated Frame Caveat" in note for note in summary.aggregation_notes)


def test_manipulated_video():
    frames = [
        {"frame_index": i * 10, "timestamp": i * 0.33, "face_detected": True, "fake_score": 0.85 - (i * 0.01), "prediction": "Fake"}
        for i in range(10)
    ]
    summary = aggregate_video_predictions(frames, threshold=0.50)
    assert summary.frames_sampled == 10
    assert summary.frames_with_faces == 10
    assert summary.overall_prediction == "Fake"
    assert summary.fake_frame_ratio == 1.0
    assert summary.median_score >= 0.70


def test_intermittent_score_variation():
    frames = [
        {"frame_index": 0, "timestamp": 0.0, "face_detected": True, "fake_score": 0.10, "prediction": "Real"},
        {"frame_index": 10, "timestamp": 0.3, "face_detected": True, "fake_score": 0.95, "prediction": "Fake"},
        {"frame_index": 20, "timestamp": 0.6, "face_detected": True, "fake_score": 0.15, "prediction": "Real"},
        {"frame_index": 30, "timestamp": 0.9, "face_detected": True, "fake_score": 0.90, "prediction": "Fake"},
    ]
    summary = aggregate_video_predictions(frames, threshold=0.50)
    assert summary.score_std > 0.25
    assert any("High score variation" in note for note in summary.aggregation_notes)
