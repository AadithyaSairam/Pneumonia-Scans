#!/usr/bin/env python3
"""
Train and evaluate the pneumonia classifier.

    python train.py --data-dir path/to/chest_xray
    python train.py --data-dir path/to/chest_xray --epochs 10 --lr 1e-3

Reports a full confusion matrix and per-class precision/recall, not just
accuracy. On a dataset that is roughly 3:1 pneumonia:normal, a model that
answers "pneumonia" every time scores ~74 % accuracy and misses every healthy
patient, so accuracy on its own cannot tell you whether the model works.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim

import data as data_mod
from model import build


def evaluate(model, loader, device, n_classes=2):
    """Run the model over a loader and return (loss, confusion matrix).

    The confusion matrix is indexed [true, predicted].
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            total_loss += criterion(logits, labels).item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    return total_loss / max(n_batches, 1), confusion


def report(confusion, class_names):
    """Print accuracy, and precision/recall/F1 per class."""
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()

    print(f"\n  Accuracy: {correct}/{total} = {100 * correct / total:.2f} %")

    print("\n  Confusion matrix (rows = true, cols = predicted)")
    header = "".join(f"{n:>12}" for n in class_names)
    print(f"  {'':12}{header}")
    for i, name in enumerate(class_names):
        row = "".join(f"{confusion[i, j].item():>12}" for j in range(len(class_names)))
        print(f"  {name:12}{row}")

    print(f"\n  {'class':12}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")
    for i, name in enumerate(class_names):
        tp = confusion[i, i].item()
        predicted = confusion[:, i].sum().item()
        actual = confusion[i, :].sum().item()

        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(f"  {name:12}{precision:>12.3f}{recall:>12.3f}{f1:>12.3f}{actual:>12}")

    # On this dataset the number that matters clinically is recall on
    # PNEUMONIA: a false negative is a missed diagnosis.
    print()


def train(args):
    loaders, class_names = data_mod.loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.workers
    )
    model, device = build(n_classes=len(class_names))

    print(f"Device        : {device}")
    print(f"Classes       : {class_names}")
    for split, loader in loaders.items():
        print(f"{split:<14}: {len(loader.dataset)} images")

    # Class-weighted loss, because the splits are imbalanced. Without this the
    # model can minimise the loss by ignoring the minority class entirely.
    weights = data_mod.class_weights(loaders["train"], len(class_names)).to(device)
    print(f"Class weights : {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for images, labels in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            # .item() detaches from the graph. Appending the tensor itself
            # would keep every epoch's computation graph alive and leak
            # memory until the run dies.
            running += loss.item()

        train_loss = running / len(loaders["train"])
        val_loss, val_confusion = evaluate(model, loaders["val"], device, len(class_names))
        scheduler.step(val_loss)

        val_acc = 100 * val_confusion.diag().sum().item() / max(val_confusion.sum().item(), 1)
        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"train {train_loss:.4f}  val {val_loss:.4f}  val_acc {val_acc:.1f} %"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.checkpoint)

    print(f"\nTrained in {time.time() - start:.1f} s")
    print(f"Best checkpoint: {args.checkpoint}")

    # Evaluate the best checkpoint, not the last epoch's weights.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    _, confusion = evaluate(model, loaders["test"], device, len(class_names))

    print("\n=== Test set ===")
    report(confusion, class_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing train/ val/ test/ (see README)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pt"))
    args = parser.parse_args()

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
