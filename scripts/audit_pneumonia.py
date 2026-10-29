"""Independently audit the pneumonia checkpoint and its reported metrics.

Checks, without trusting the project's own metric code:
  1. the dataset file matches the SHA-256 recorded at download time;
  2. the checkpoint metadata (source, sample counts, threshold) is consistent with the data;
  3. the threshold can be re-derived from the validation split (Youden J, duplicates removed);
  4. test-set metrics recomputed with plain NumPy + scikit-learn match evaluation.json;
  5. the test split was never used for training or threshold selection (no duplicates with train/val).

Usage: python scripts/audit_pneumonia.py     (writes models/pneumonia/audit.json; exit code 1 on any mismatch)
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from app.core.config import settings
from app.ml.pneumonia.data import build_transforms, leakage_report, load_dataset, medmnist_path
from app.ml.pneumonia.model import checkpoint_dir, default_checkpoint_path, load_checkpoint

TOL = 1e-6


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def probs_for(model, split, idx, img_size):
    tf = build_transforms(img_size, train=False)
    out = []
    for start in range(0, len(idx), 64):
        batch = torch.stack([tf(split.load(int(i))) for i in idx[start : start + 64]])
        out.append(torch.softmax(model(batch), 1)[:, 1].numpy())
    return np.concatenate(out)


def main() -> int:
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    manifest = json.loads((settings.raw_dir / "pneumoniamnist" / "_manifest.json").read_text())
    recorded = manifest["files"][0]["sha256"]
    actual = sha256(medmnist_path())
    check("dataset file integrity", recorded == actual, f"sha256 {actual[:16]}... (manifest {recorded[:16]}...)")

    ckpt = default_checkpoint_path()
    model, meta = load_checkpoint(ckpt)
    check(
        "checkpoint metadata complete",
        all(k in meta for k in ("arch", "classes", "img_size", "threshold")),
        f"{ckpt.name}: arch={meta['arch']} source={meta['data_source']} trained_at={meta['trained_at']}",
    )

    ds = load_dataset(meta["data_source"])
    train, val, test = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    check(
        "training sample count",
        meta["train_samples"] == len(train),
        f"meta {meta['train_samples']} vs split {len(train)}",
    )
    leaks = leakage_report(ds)
    dup_val = sorted({p["index_b"] for p in leaks["duplicate_pairs"] if p["splits"] == "train-val"})
    check(
        "validation size after duplicate removal",
        meta["val_samples"] == len(val) - len(dup_val),
        f"meta {meta['val_samples']} = {len(val)} - {len(dup_val)} duplicates",
    )
    check(
        "test split disjoint from train/val",
        leaks["confirmed_near_duplicates"]["train-test"] == 0 and leaks["confirmed_near_duplicates"]["val-test"] == 0,
        f"confirmed duplicates train-test={leaks['confirmed_near_duplicates']['train-test']}, "
        f"val-test={leaks['confirmed_near_duplicates']['val-test']}",
    )

    val_idx = np.setdiff1d(np.arange(len(val)), dup_val)
    p_val = probs_for(model, val, val_idx, meta["img_size"])
    fpr, tpr, thr = roc_curve(val.labels[val_idx], p_val)
    youden = float(thr[np.argmax(tpr - fpr)])
    check(
        "threshold re-derived from validation (Youden J)",
        abs(youden - meta["threshold"]) < 1e-4,
        f"recomputed {youden:.6f} vs stored {meta['threshold']:.6f}",
    )

    test_idx = np.arange(len(test))
    y = test.labels
    p = probs_for(model, test, test_idx, meta["img_size"])
    ev = json.loads((checkpoint_dir() / "evaluation.json").read_text())
    recomputed = {}
    for key, t in (("metrics", meta["threshold"]), ("metrics_at_0_5", 0.5)):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r = {
            "accuracy": float((pred == y).mean()),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        recomputed[key] = r
        reported = ev[key]
        diffs = {
            k: abs(r[k] - reported[k]) for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
        }
        ok = max(diffs.values()) < TOL and r["confusion_matrix"] == reported["confusion_matrix"]
        check(
            f"test metrics match evaluation.json ({key}, threshold {t:.4f})",
            ok,
            f"n={len(y)} acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
            f"spec={r['specificity']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f} cm={r['confusion_matrix']}",
        )
    check(
        "test split is the official MedMNIST/Kaggle test set",
        len(y) == 624 and int(y.sum()) == 390,
        f"{len(y)} images, {int(y.sum())} pneumonia, {int((y == 0).sum())} normal",
    )

    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt.name,
        "checkpoint_sha256": sha256(ckpt),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "recomputed": recomputed,
    }
    (checkpoint_dir() / "audit.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'ALL CHECKS PASSED' if report['all_passed'] else 'SOME CHECKS FAILED'} -> models/pneumonia/audit.json")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
