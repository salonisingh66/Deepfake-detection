"""Model setup and verification script for DeepShield AI.

Verifies model weights, tests CPU inference, benchmarks latency,
and prepares the local environment for offline readiness.
"""

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# Ensure root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL_ID
from src.models.model_loader import ModelLoader
from src.models.inference import InferenceEngine
from src.utils.logging_config import setup_logger

logger = setup_logger("setup_model")


def run_model_setup(model_id: str = DEFAULT_MODEL_ID):
    """Download, cache, and verify model readiness."""
    print("=" * 60)
    print(f"DeepShield AI - Setting Up Model: {model_id}")
    print("=" * 60)

    loader = ModelLoader.get_instance()
    t0 = time.time()
    
    print(f"Loading checkpoint and configuration for '{model_id}'...")
    container = loader.load_model(model_id)

    if not container.is_ready:
        print(f"[ERROR] Model setup failed: {container.error_message}")
        sys.exit(1)

    load_duration = time.time() - t0
    print(f"[SUCCESS] Model loaded in {load_duration:.2f} seconds.")
    print(f"  - Model Name: {container.metadata.name}")
    print(f"  - Architecture: {container.metadata.architecture}")
    print(f"  - Source: {container.metadata.source}")
    print(f"  - License: {container.metadata.license}")
    print(f"  - Input Size: {container.metadata.input_size}")
    print(f"  - Label Mapping: {container.metadata.label_map}")

    # Perform a dummy inference sanity check
    print("\nRunning CPU inference sanity check on synthetic sample...")
    engine = InferenceEngine(loader=loader)
    synthetic_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    t_inf = time.time()
    res = engine.predict_face(synthetic_crop, model_id=model_id)
    latency = (time.time() - t_inf) * 1000.0

    print(f"[SUCCESS] Test inference passed in {latency:.1f} ms.")
    print(f"  - Prediction: {res.prediction}")
    print(f"  - Confidence: {res.confidence * 100:.1f}%")
    print(f"  - Fake Score: {res.fake_score:.4f}")
    print(f"  - Probabilities: {res.probabilities}")
    print("\nSetup complete. DeepShield AI is ready for CPU inference.\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    run_model_setup(target)
