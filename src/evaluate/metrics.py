import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold in [0.01, 0.99] that maximises F1 on the given labels."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 1.00, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return best_thresh


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":          roc_auc_score(y_true, y_prob),
        "fpr":              fpr,
        "fnr":              fnr,
        "confusion_matrix": cm,
    }


def print_metrics(metrics: dict, dataset_label: str = "Test", threshold: float = 0.5) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*40}")
    print(f" {dataset_label} Results")
    print(f"{'='*40}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  FPR       : {metrics['fpr']:.4f}")
    print(f"  FNR       : {metrics['fnr']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  :  {tn:>6}   {fp:>6}")
    print(f"  Actual 1  :  {fn:>6}   {tp:>6}")
    print(f"{'='*40}\n")
