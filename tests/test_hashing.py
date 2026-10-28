"""Unit tests for cryptographic hashing and forensic fingerprints."""

import hashlib
from src.forensics.hashing import (
    compute_file_fingerprints,
    compute_sha256_bytes,
    compute_sha256_file,
)


def test_compute_sha256_bytes():
    data = b"DeepShield AI Test Content"
    expected = hashlib.sha256(data).hexdigest()
    computed = compute_sha256_bytes(data)
    assert computed == expected


def test_compute_sha256_file(tmp_path):
    test_f = tmp_path / "test_file.bin"
    content = b"Forensic Cryptographic Fingerprint Verification 12345" * 1000
    test_f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    computed = compute_sha256_file(test_f)
    assert computed == expected


def test_compute_file_fingerprints_bytes_and_file(tmp_path):
    data = b"Sample bytes for fingerprinting"
    fp_bytes = compute_file_fingerprints(data)

    test_f = tmp_path / "fingerprint_target.txt"
    test_f.write_bytes(data)
    fp_file = compute_file_fingerprints(test_f)

    assert fp_bytes["sha256"] == fp_file["sha256"]
    assert fp_bytes["md5"] == fp_file["md5"]
    assert len(fp_bytes["sha256"]) == 64
    assert len(fp_bytes["md5"]) == 32
