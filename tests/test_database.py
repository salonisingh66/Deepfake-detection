"""Unit tests for SQLite analysis history database."""

from pathlib import Path
import pytest
from src.database.db import get_db_connection, init_db
from src.database.history import (
    clear_all_history,
    delete_record_by_id,
    get_analysis_records,
    get_record_by_id,
    save_analysis_record,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_deepshield.db"
    init_db(db_file)
    return db_file


def test_db_initialization_and_schema(temp_db):
    conn = get_db_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_history'")
    assert cursor.fetchone() is not None

    # Verify no raw media/blob columns exist
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [row["name"] for row in cursor.fetchall()]
    assert "image_bytes" not in columns
    assert "video_bytes" not in columns
    assert "face_crop" not in columns
    conn.close()


def test_save_and_retrieve_record(temp_db):
    rec_id = save_analysis_record(
        filename="test_face.jpg",
        file_type="image",
        sha256="abc123456789",
        model_name="ViT Deepfake Classifier",
        prediction="Fake",
        confidence_score=0.885,
        fake_score=0.885,
        faces_detected=1,
        execution_time_ms=120.5,
        status="SUCCESS",
        metadata_summary={"dimensions": "512x512"},
        db_path=temp_db
    )
    assert rec_id.startswith("DS-")

    retrieved = get_record_by_id(rec_id, db_path=temp_db)
    assert retrieved is not None
    assert retrieved["filename"] == "test_face.jpg"
    assert retrieved["prediction"] == "Fake"
    assert retrieved["confidence_score"] == 0.885
    assert retrieved["metadata_summary"]["dimensions"] == "512x512"


def test_filter_and_search_records(temp_db):
    save_analysis_record(
        filename="person_a.jpg",
        file_type="image",
        sha256="hash111",
        model_name="ViT",
        prediction="Real",
        confidence_score=0.92,
        fake_score=0.08,
        faces_detected=1,
        execution_time_ms=100.0,
        db_path=temp_db
    )
    save_analysis_record(
        filename="person_b.mp4",
        file_type="video",
        sha256="hash222",
        model_name="ViT",
        prediction="Fake",
        confidence_score=0.78,
        fake_score=0.78,
        faces_detected=1,
        execution_time_ms=500.0,
        db_path=temp_db
    )

    # Filter by prediction
    real_recs = get_analysis_records(prediction_filter="Real", db_path=temp_db)
    assert len(real_recs) == 1
    assert real_recs[0]["filename"] == "person_a.jpg"

    # Search by filename
    search_recs = get_analysis_records(search_query="person_b", db_path=temp_db)
    assert len(search_recs) == 1
    assert search_recs[0]["file_type"] == "video"


def test_delete_and_clear_history(temp_db):
    id1 = save_analysis_record(
        filename="sample1.jpg", file_type="image", sha256="h1",
        model_name="M", prediction="Real", confidence_score=0.9, fake_score=0.1,
        faces_detected=1, execution_time_ms=50.0, db_path=temp_db
    )
    id2 = save_analysis_record(
        filename="sample2.jpg", file_type="image", sha256="h2",
        model_name="M", prediction="Fake", confidence_score=0.9, fake_score=0.9,
        faces_detected=1, execution_time_ms=50.0, db_path=temp_db
    )

    assert delete_record_by_id(id1, db_path=temp_db) is True
    assert get_record_by_id(id1, db_path=temp_db) is None
    assert get_record_by_id(id2, db_path=temp_db) is not None

    cleared = clear_all_history(db_path=temp_db)
    assert cleared == 1
    assert len(get_analysis_records(db_path=temp_db)) == 0
