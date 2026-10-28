"""Analysis History CRUD operations for DeepShield AI.

Provides safe record insertion, search/filtering, and record deletion.
Privacy guarantee: Raw media payloads, frames, or biometric vectors are NEVER stored.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.database.db import get_db_connection
from src.utils.logging_config import logger


def save_analysis_record(
    filename: str,
    file_type: str,
    sha256: str,
    model_name: str,
    prediction: str,
    confidence_score: float,
    fake_score: float,
    faces_detected: int,
    execution_time_ms: float,
    status: str = "SUCCESS",
    metadata_summary: Optional[Dict[str, Any]] = None,
    record_id: Optional[str] = None,
    db_path: Optional[Path] = None
) -> str:
    """Insert a new analysis record into the database. Returns the unique record ID."""
    rec_id = record_id or f"DS-{uuid.uuid4().hex[:12].upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta_json = json.dumps(metadata_summary or {})

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO analysis_history (
                    id, timestamp, filename, file_type, sha256,
                    model_name, prediction, confidence_score, fake_score,
                    faces_detected, execution_time_ms, status, metadata_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id, timestamp, filename, file_type, sha256,
                    model_name, prediction, confidence_score, fake_score,
                    faces_detected, execution_time_ms, status, meta_json
                )
            )
        logger.debug(f"Saved analysis record {rec_id} for file {filename}")
        return rec_id
    finally:
        conn.close()


def get_analysis_records(
    search_query: Optional[str] = None,
    prediction_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieve history records with optional search and filters."""
    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM analysis_history WHERE 1=1"
        params: List[Any] = []

        if search_query:
            query += " AND (filename LIKE ? OR sha256 LIKE ? OR id LIKE ?)"
            term = f"%{search_query.strip()}%"
            params.extend([term, term, term])

        if prediction_filter and prediction_filter.upper() != "ALL":
            query += " AND UPPER(prediction) = ?"
            params.append(prediction_filter.upper())

        if file_type_filter and file_type_filter.upper() != "ALL":
            query += " AND UPPER(file_type) = ?"
            params.append(file_type_filter.upper())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            rec = dict(row)
            try:
                rec["metadata_summary"] = json.loads(rec["metadata_summary"]) if rec["metadata_summary"] else {}
            except Exception:
                rec["metadata_summary"] = {}
            results.append(rec)
        return results
    finally:
        conn.close()


def get_record_by_id(record_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single analysis record by its unique ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            return None
        rec = dict(row)
        try:
            rec["metadata_summary"] = json.loads(rec["metadata_summary"]) if rec["metadata_summary"] else {}
        except Exception:
            rec["metadata_summary"] = {}
        return rec
    finally:
        conn.close()


def delete_record_by_id(record_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a single analysis record by its ID."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_history(db_path: Optional[Path] = None) -> int:
    """Clear all analysis history entries. Returns the number of records removed."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history")
            return cursor.rowcount
    finally:
        conn.close()
