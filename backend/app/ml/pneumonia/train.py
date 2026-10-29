"""Training loop for the pneumonia classifier (CPU friendly).

Run through ``scripts/train_pneumonia.py``. Produces, in ``models/pneumonia/``:
  * ``pneumonia_resnet18.pt``  - best checkpoint (by validation ROC-AUC) + metadata
  * ``training_history.json``  - per-epoch loss / accuracy / AUC
  * ``training_curves.png``    - plots of the history
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.ml.pneumonia.data import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TorchXrayDataset,
    build_transforms,
    leakage_report,
    load_dataset,
)
from app.ml.pneumonia.metrics import binary_metrics, youden_threshold
from app.ml.pneumonia.model import build_model, checkpoint_dir, save_checkpoint


@dataclass
class TrainConfig:
    source: str = "auto"
    arch: str = "resnet18"
    img_size: int = 224
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    num_workers: int = 0
    max_train_samples: int | None = None  # subsample for quick smoke runs
    max_eval_samples: int | None = None
    drop_val_duplicates: bool = True  # remove validation images that duplicate training images
    pretrained: bool = True
    output_dir: str | None = None


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subsample(n: int, limit: int | None, seed: int) -> np.ndarray | None:
    if not limit or limit >= n:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
    y_all = np.concatenate(labels)
    return y_all, np.concatenate(probs), loss_sum / max(len(y_all), 1)


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(epochs, [h["train_accuracy"] for h in history], "o-", label="train")
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], "o-", label="validation")
    axes[1].set_title("Accuracy (threshold 0.5)")
    axes[2].plot(epochs, [h["val_roc_auc"] for h in history], "o-", color="tab:green")
    axes[2].set_title("Validation ROC-AUC")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(config: TrainConfig, log=print) -> dict:
    _seed_everything(config.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(config.output_dir) if config.output_dir else checkpoint_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.source, seed=config.seed)
    train_split, val_split = dataset.splits["train"], dataset.splits["val"]
    train_idx = _subsample(len(train_split), config.max_train_samples, config.seed)
    val_idx = _subsample(len(val_split), config.max_eval_samples, config.seed)
    removed_val = []
    if config.drop_val_duplicates:
        leaks = leakage_report(dataset)
        removed_val = sorted({p["index_b"] for p in leaks["duplicate_pairs"] if p["splits"] == "train-val"})
        if removed_val:
            base = np.arange(len(val_split)) if val_idx is None else val_idx
            val_idx = np.setdiff1d(base, removed_val)
            log(f"removed {len(removed_val)} validation images that duplicate training images")
    train_ds = TorchXrayDataset(train_split, build_transforms(config.img_size, train=True), train_idx)
    val_ds = TorchXrayDataset(val_split, build_transforms(config.img_size, train=False), val_idx)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    train_labels = train_split.labels if train_idx is None else train_split.labels[train_idx]
    counts = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    log(f"source={dataset.source} train={len(train_ds)} val={len(val_ds)} device={device} "
        f"class_counts={counts.tolist()} class_weights={class_weights.tolist()}")

    model = build_model(config.arch, pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    ckpt_path = out_dir / f"pneumonia_{config.arch}.pt"
    # Best-so-far weights go to a separate file so the API never loads a half-finished model.
    partial_path = out_dir / f"pneumonia_{config.arch}.partial.pt"
    history, best_auc, best_epoch, stale = [], -1.0, 0, 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
            if step % 20 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss_sum / seen:.4f} "
                    f"({time.time() - t0:.0f}s)")
        scheduler.step()
        y_val, p_val, val_loss = predict_probs(model, val_loader, device)
        val_m = binary_metrics(y_val, p_val, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_roc_auc": val_m["roc_auc"],
            "val_f1": val_m["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        log(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_auc={row['val_roc_auc']:.4f} ({row['seconds']}s)")
        if (row["val_roc_auc"] or 0) > best_auc:
            best_auc, best_epoch, stale = row["val_roc_auc"], epoch, 0
            save_checkpoint(partial_path, model, {"arch": config.arch, "epoch": epoch})
        else:
            stale += 1
            if stale >= config.patience:
                log(f"early stopping after epoch {epoch} (no val AUC improvement for {stale} epochs)")
                break

    # Reload the best weights, choose the operating threshold on validation data only.
    best_state = torch.load(partial_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(best_state)
    y_val, p_val, _ = predict_probs(model, val_loader, device)
    threshold = youden_threshold(y_val, p_val)
    meta = {
        "arch": config.arch,
        "classes": CLASSES,
        "img_size": config.img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "threshold": threshold,
        "threshold_method": "Youden J statistic on the validation split",
        "data_source": dataset.source,
        "data_notes": dataset.notes,
        "removed_val_duplicates": len(removed_val),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_metrics_at_threshold": binary_metrics(y_val, p_val, threshold),
        "config": asdict(config),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(time.time() - started, 1),
        "torch_version": torch.__version__,
        "device": device,
    }
    save_checkpoint(ckpt_path, model, meta)
    partial_path.unlink(missing_ok=True)
    (out_dir / "training_history.json").write_text(json.dumps({"history": history, "meta": meta}, indent=2))
    plot_history(history, out_dir / "training_curves.png")
    log(f"saved {ckpt_path} (best epoch {best_epoch}, val AUC {best_auc:.4f}, threshold {threshold:.3f})")
    return {"checkpoint": str(ckpt_path), "history": history, "meta": meta}
