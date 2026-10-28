"""Command-line dataset evaluation tool for DeepShield AI."""

import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL_ID
from src.evaluation.evaluate import DatasetEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepShield AI on a labeled dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset with 'real/' and 'fake/' folders")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold (default: 0.50)")
    parser.add_argument("--output-csv", type=str, default="data/evaluation/eval_results.csv", help="CSV output path")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    if not data_path.exists():
        print(f"[ERROR] Specified data directory does not exist: {data_path}")
        sys.exit(1)

    evaluator = DatasetEvaluator()
    print(f"Discovering samples in {data_path}...")
    metrics, df = evaluator.evaluate_directory(
        dataset_dir=data_path,
        model_id=args.model_id,
        threshold=args.threshold,
        max_samples=args.max_samples
    )

    if metrics is None:
        print(f"[NOTICE] No test images found in {data_path}. Ensure it contains 'real/' and 'fake/' subdirectories.")
        sys.exit(0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Samples: {metrics.total_samples} (Real: {metrics.num_real}, Fake: {metrics.num_fake})")
    print(f"Accuracy:      {metrics.accuracy * 100:.2f}%")
    print(f"Precision:     {metrics.precision * 100:.2f}%")
    print(f"Recall:        {metrics.recall * 100:.2f}%")
    print(f"F1-Score:      {metrics.f1 * 100:.2f}%")
    if metrics.roc_auc is not None:
        print(f"ROC-AUC:       {metrics.roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN={metrics.true_negative:4d}, FP={metrics.false_positive:4d}]")
    print(f"  [FN={metrics.false_negative:4d}, TP={metrics.true_positive:4d}]")

    if not df.empty and args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"\nDetailed sample results written to {out_p}")


if __name__ == "__main__":
    main()
