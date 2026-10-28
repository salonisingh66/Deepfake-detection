"""Logging configuration for DeepShield AI.

Provides structured, readable console and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False

def setup_logger(
    name: str = "deepshield",
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Set up and return a configured logger."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Formatter
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

# Default logger instance
logger = setup_logger()
