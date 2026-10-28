"""Dataset evaluation engine for benchmark validation.

Runs model across labeled test sets (images organized into real/ and fake/ subfolders or CSV),
computes performance metrics, and exports CSV reports.
"""

from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

from config.model_config import CONFIDENCE_THRESHOLD, DEFAULT_MODEL_ID
from config.settings import settings
from src.detection.image_analyzer import ImageAnalyzer
from src.evaluation.metrics import EvaluationMetrics, compute_binary_metrics
from src.utils.logging_config import logger


class DatasetEvaluator:
    """Evaluates deepfake detector on labeled datasets."""

    def __init__(self, analyzer: Optional[ImageAnalyzer] = None):
        self.analyzer = analyzer or ImageAnalyzer()

    def discover_dataset_files(
        self,
        dataset_dir: Union[str, Path]
    ) -> List[Tuple[Path, int]]:
        """Scan a directory structured into 'real' and 'fake' subfolders.
        
        Returns list of (file_path, ground_truth_label) tuples:
            0 = Real, 1 = Fake.
        """
        base = Path(dataset_dir)
        samples: List[Tuple[Path, int]] = []

        real_dir = base / "real"
        fake_dir = base / "fake"

        if real_dir.exists() and real_dir.is_dir():
            for ext in settings.ALLOWED_IMAGE_EXTENSIONS:
                for img_path in real_dir.glob(f"*{ext}"):
                    samples.append((img_path, 0))

        if fake_dir.exists() and fake_dir.is_dir():
            for ext in settings.ALLOWED_IMAGE_EXTENSIONS:
                for img_path in fake_dir.glob(f"*{ext}"):
                    samples.append((img_path, 1))

        return samples

    def evaluate_directory(
        self,
        dataset_dir: Union[str, Path],
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = CONFIDENCE_THRESHOLD,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> Tuple[Optional[EvaluationMetrics], pd.DataFrame]:
        """Run evaluation on images in dataset_dir.
        
        Returns:
            (EvaluationMetrics, per_sample_results_dataframe)
        """
        samples = self.discover_dataset_files(dataset_dir)
        if not samples:
            logger.warning(f"No labeled images found in {dataset_dir} (expected 'real/' and 'fake/' folders).")
            return None, pd.DataFrame()

        if max_samples and max_samples < len(samples):
            import random
            random.seed(42)
            samples = random.sample(samples, max_samples)

        total_count = len(samples)
        logger.info(f"Starting evaluation of {total_count} labeled samples using model '{model_id}'...")

        records = []
        y_true = []
        y_pred = []
        y_scores = []

        for idx, (path, true_label) in enumerate(samples):
            try:
                with open(path, "rb") as f:
                    file_bytes = f.read()

                res = self.analyzer.analyze_image(
                    image_bytes=file_bytes,
                    filename=path.name,
                    model_id=model_id,
                    threshold=threshold,
                    generate_explanation=False  # Speed up eval
                )

                if res.status == "SUCCESS":
                    pred_label = 1 if res.prediction == "Fake" else 0
                    fake_score = res.fake_score
                else:
                    # Inconclusive / no face detected: default to non-fake or flag
                    pred_label = 0
                    fake_score = 0.50

                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(fake_score)

                records.append({
                    "filename": path.name,
                    "true_label": "Fake" if true_label == 1 else "Real",
                    "predicted_label": res.prediction,
                    "fake_score": fake_score,
                    "status": res.status,
                    "faces_detected": res.faces_detected_count,
                    "inference_time_ms": res.execution_time_ms
                })

            except Exception as e:
                logger.error(f"Error evaluating sample {path.name}: {e}")

            if progress_callback:
                progress_callback(min(1.0, (idx + 1) / total_count), idx + 1, total_count)

        df = pd.DataFrame(records)
        metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        return metrics, df
