"""Evaluation metrics calculation for deepfake binary classification.

Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix
using scikit-learn without fabricated metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    """Rigorous evaluation metrics container."""
    total_samples: int
    num_real: int
    num_fake: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    roc_fpr: Optional[List[float]] = None
    roc_tpr: Optional[List[float]] = None


def compute_binary_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_scores: Optional[Union[List[float], np.ndarray]] = None
) -> EvaluationMetrics:
    """Compute standard classification metrics for binary deepfake detection.
    
    Label convention: 0 = Real (Negative), 1 = Fake (Positive).
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    total = len(y_t)
    num_fake = int(np.sum(y_t == 1))
    num_real = int(np.sum(y_t == 0))

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = None
    fpr_list, tpr_list = None, None
    if y_scores is not None and len(np.unique(y_t)) > 1:
        try:
            scores_arr = np.array(y_scores, dtype=float)
            auc = float(roc_auc_score(y_t, scores_arr))
            fpr, tpr, _ = roc_curve(y_t, scores_arr)
            fpr_list = [round(float(x), 4) for x in fpr]
            tpr_list = [round(float(x), 4) for x in tpr]
        except Exception:
            auc = None

    return EvaluationMetrics(
        total_samples=total,
        num_real=num_real,
        num_fake=num_fake,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(auc, 4) if auc is not None else None,
        confusion_matrix=cm.tolist(),
        true_positive=int(tp),
        false_positive=int(fp),
        true_negative=int(tn),
        false_negative=int(fn),
        roc_fpr=fpr_list,
        roc_tpr=tpr_list
    )
