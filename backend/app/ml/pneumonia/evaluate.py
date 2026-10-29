"""Test-set evaluation of a trained checkpoint.

Evaluating on the training source writes ``evaluation.json`` + plots (the official results shown
in the app). Evaluating on a *different* source (e.g. the Kaggle originals) is an external check:
it writes ``evaluation_<source>.json`` and ``*_<source>.png`` and never overwrites the official files.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from app.ml.pneumonia.data import CLASSES, TorchXrayDataset, build_transforms, load_dataset
from app.ml.pneumonia.metrics import binary_metrics, curve_points
from app.ml.pneumonia.model import default_checkpoint_path, load_checkpoint
from app.ml.pneumonia.train import predict_probs


def plot_evaluation(result: dict, out_dir: Path, suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = result["metrics"]["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {result['metrics']['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix{suffix}.png", dpi=120)
    plt.close(fig)

    roc = result["curves"]["roc"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([p["fpr"] for p in roc], [p["tpr"] for p in roc], label=f"AUC = {result['metrics']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_curve{suffix}.png", dpi=120)
    plt.close(fig)


def evaluate(checkpoint: Path | None = None, source: str | None = None, batch_size: int = 64,
             max_samples: int | None = None, log=print) -> dict:
    checkpoint = checkpoint or default_checkpoint_path()
    model, meta = load_checkpoint(checkpoint)
    dataset = load_dataset(source or meta.get("data_source", "auto"))
    external = dataset.source != meta.get("data_source")
    test_split = dataset.splits["test"]
    indices = None
    if max_samples and max_samples < len(test_split):
        indices = np.sort(np.random.default_rng(0).choice(len(test_split), max_samples, replace=False))
    loader = DataLoader(TorchXrayDataset(test_split, build_transforms(meta["img_size"], train=False), indices),
                        batch_size=batch_size)
    y, prob, loss = predict_probs(model, loader, "cpu")
    threshold = float(meta.get("threshold", 0.5))
    result = {
        "checkpoint": checkpoint.name,
        "data_source": dataset.source,
        "split": "test",
        "samples": int(len(y)),
        "test_loss": loss,
        "metrics": binary_metrics(y, prob, threshold),
        "metrics_at_0_5": binary_metrics(y, prob, 0.5),
        "curves": curve_points(y, prob),
        "probability_histogram": _histogram(y, prob),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_meta": {k: meta[k] for k in ("arch", "img_size", "threshold", "threshold_method", "best_epoch",
                                            "epochs_run", "trained_at", "train_samples", "val_samples")
                       if k in meta},
    }
    out_dir = checkpoint.parent
    suffix = f"_{dataset.source}" if external else ""
    if external:
        result["note"] = (f"External evaluation: model trained on {meta.get('data_source')}, tested on the "
                          f"{dataset.source} test split. Threshold unchanged ({threshold:.4f}).")
    (out_dir / f"evaluation{suffix}.json").write_text(json.dumps(result, indent=2))
    plot_evaluation(result, out_dir, suffix)
    m = result["metrics"]
    log(f"test n={len(y)} acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} "
        f"specificity={m['specificity']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    return result


def _histogram(y: np.ndarray, prob: np.ndarray, bins: int = 20) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    normal, _ = np.histogram(prob[y == 0], bins=edges)
    pneumonia, _ = np.histogram(prob[y == 1], bins=edges)
    return [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "normal": int(normal[i]),
             "pneumonia": int(pneumonia[i])} for i in range(bins)]
