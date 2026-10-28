"""Cryptographic hashing module for forensic provenance and integrity verification.

Calculates SHA-256 and MD5 hashes in streaming chunks to maintain low memory usage.
"""

import hashlib
from pathlib import Path
from typing import Dict, Union

# 64 KB chunk size for memory-efficient streaming hash calculation
CHUNK_SIZE = 65536


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of a byte string."""
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()


def compute_sha256_file(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk using streaming chunks."""
    sha256 = hashlib.sha256()
    path = Path(file_path)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_file_fingerprints(file_source: Union[str, Path, bytes]) -> Dict[str, str]:
    """Compute forensic integrity fingerprints (SHA-256 and MD5) for a file or bytes.
    
    Returns:
        dict: {'sha256': str, 'md5': str}
    """
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    
    if isinstance(file_source, bytes):
        sha256.update(file_source)
        md5.update(file_source)
    else:
        path = Path(file_source)
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                sha256.update(chunk)
                md5.update(chunk)
                
    return {
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest()
    }
