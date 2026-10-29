"""Train the ResNet pneumonia classifier.

Examples:
    python scripts/train_pneumonia.py                          # full run, 5 epochs
    python scripts/train_pneumonia.py --epochs 1 --max-train 256 --max-eval 128   # quick smoke test
"""
import argparse

import _bootstrap  # noqa: F401

from app.ml.pneumonia.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="auto", choices=["auto", "kaggle", "pneumoniamnist"])
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()
    train(TrainConfig(source=a.source, arch=a.arch, img_size=a.img_size, epochs=a.epochs, batch_size=a.batch_size,
                      lr=a.lr, patience=a.patience, num_workers=a.workers, max_train_samples=a.max_train,
                      max_eval_samples=a.max_eval, pretrained=not a.no_pretrained, output_dir=a.output_dir))


if __name__ == "__main__":
    main()
