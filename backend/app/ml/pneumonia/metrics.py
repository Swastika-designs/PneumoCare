"""Binary classification metrics used for training, evaluation and the dashboard."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    def safe(num, den):
        return float(num / den) if den else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    has_both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(y_true, prob)) if has_both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"normal": int((y_true == 0).sum()), "pneumonia": int((y_true == 1).sum())},
    }


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data only)."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def curve_points(y_true: np.ndarray, prob: np.ndarray, max_points: int = 200) -> dict:
    fpr, tpr, thr = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)

    def thin(*arrays):
        n = len(arrays[0])
        idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
        return [a[idx] for a in arrays]

    fpr, tpr, thr = thin(fpr, tpr, np.clip(thr, 0, 1))
    prec, rec = thin(prec, rec)
    return {
        "roc": [{"fpr": float(a), "tpr": float(b), "threshold": float(c)} for a, b, c in zip(fpr, tpr, thr)],
        "pr": [{"recall": float(a), "precision": float(b)} for a, b in zip(rec, prec)],
    }
