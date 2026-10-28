"""SQLite Database connection and schema initializer.

Ensures persistent thread-safe storage of audit records without media storage.
"""

from pathlib import Path
import sqlite3
from typing import Optional

from config.settings import settings
from src.utils.logging_config import logger


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an active connection to the SQLite database with row factory enabled."""
    target_path = Path(db_path or settings.DATABASE_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables with indexing for performant search."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    fake_score REAL NOT NULL,
                    faces_detected INTEGER DEFAULT 0,
                    execution_time_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata_summary TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_timestamp
                ON analysis_history (timestamp DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_sha256
                ON analysis_history (sha256);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_prediction
                ON analysis_history (prediction);
            """)
        logger.info(f"Analysis database initialized at {db_path or settings.DATABASE_PATH}")
    finally:
        conn.close()
