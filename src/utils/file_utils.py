"""File manipulation and temporary file management utilities for DeepShield AI.

Ensures safe file isolation, prevents path traversal, and guarantees temporary cleanup.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Generator, Optional, Union
from contextlib import contextmanager

from config.settings import settings
from src.utils.logging_config import logger


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and remove unsafe characters."""
    # Strip any directory components
    safe_name = os.path.basename(filename)
    # Remove null bytes and path separators
    safe_name = safe_name.replace("\x00", "").replace("/", "").replace("\\", "")
    if not safe_name:
        safe_name = f"unnamed_{uuid.uuid4().hex[:8]}"
    return safe_name


def is_safe_path(target_path: Union[str, Path], allowed_base: Union[str, Path]) -> bool:
    """Check if target_path resolves strictly within allowed_base (prevent directory traversal)."""
    try:
        resolved_target = Path(target_path).resolve()
        resolved_base = Path(allowed_base).resolve()
        return resolved_base in resolved_target.parents or resolved_target == resolved_base
    except (ValueError, RuntimeError):
        return False


@contextmanager
def safe_temp_file(
    suffix: str = "",
    prefix: str = "deepshield_tmp_",
    dir_path: Optional[Path] = None
) -> Generator[Path, None, None]:
    """Context manager providing a safe temporary file path, guaranteed to be removed on exit."""
    target_dir = dir_path or settings.TEMP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    unique_name = f"{prefix}{uuid.uuid4().hex}{suffix}"
    temp_file = target_dir / unique_name
    
    try:
        yield temp_file
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")


def cleanup_directory(directory: Path, max_age_seconds: Optional[int] = None) -> int:
    """Clean up files in a temporary directory.
    
    Returns the count of files successfully removed.
    """
    if not directory.exists() or not directory.is_dir():
        return 0
        
    removed_count = 0
    now = os.time() if hasattr(os, "time") else 0
    import time
    now = time.time()
    
    for item in directory.glob("*"):
        if item.is_file():
            try:
                if max_age_seconds is not None:
                    file_age = now - item.stat().st_mtime
                    if file_age < max_age_seconds:
                        continue
                item.unlink()
                removed_count += 1
            except OSError as err:
                logger.debug(f"Failed to remove {item}: {err}")
    return removed_count
