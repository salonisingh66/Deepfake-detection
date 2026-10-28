"""Unit tests for input validation and security checks."""

import io
from pathlib import Path
import pytest
from PIL import Image

from config.settings import settings
from src.utils.validation import validate_image_bytes, validate_video_file


def create_test_image_bytes(format="JPEG", size=(100, 100), color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_validate_valid_images():
    for fmt, ext in [("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp")]:
        b = create_test_image_bytes(format=fmt)
        valid, err, meta = validate_image_bytes(b, f"test{ext}")
        assert valid is True
        assert err is None
        assert meta["width"] == 100
        assert meta["height"] == 100


def test_validate_unsupported_extension():
    b = create_test_image_bytes()
    valid, err, _ = validate_image_bytes(b, "test.exe")
    assert valid is False
    assert "Unsupported file extension" in err


def test_validate_empty_image():
    valid, err, _ = validate_image_bytes(b"", "empty.jpg")
    assert valid is False
    assert "empty" in err.lower()


def test_validate_oversized_image():
    b = create_test_image_bytes()
    valid, err, _ = validate_image_bytes(b, "test.jpg", max_size_bytes=10)
    assert valid is False
    assert "exceeds maximum allowed limit" in err


def test_validate_corrupt_image():
    corrupt_bytes = b"NOT_AN_IMAGE_HEADER_RANDOM_DATA_123456789"
    valid, err, _ = validate_image_bytes(corrupt_bytes, "corrupt.jpg")
    assert valid is False
    assert "Failed to decode" in err


def test_validate_oversized_dimensions():
    b = create_test_image_bytes(size=(200, 200))
    valid, err, _ = validate_image_bytes(b, "large.png", max_dimension=150)
    assert valid is False
    assert "exceed maximum allowed limit" in err


def test_validate_video_file(tmp_path):
    # Missing file
    valid, err, _ = validate_video_file(tmp_path / "missing.mp4")
    assert valid is False
    assert "not found" in err

    # Empty file
    empty_f = tmp_path / "empty.mp4"
    empty_f.write_bytes(b"")
    valid, err, _ = validate_video_file(empty_f)
    assert valid is False
    assert "empty" in err

    # Unsupported extension
    unsupp = tmp_path / "video.txt"
    unsupp.write_bytes(b"dummy")
    valid, err, _ = validate_video_file(unsupp)
    assert valid is False
    assert "Unsupported video extension" in err
