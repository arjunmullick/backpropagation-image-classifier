#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import ImageDataset, DatasetConfig
from .model import MLPClassifier, ModelConfig


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean())


def main():
    parser = argparse.ArgumentParser(description="Train a simple from-scratch MLP image classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory root")
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--grayscale", type=int, default=1, help="1 for grayscale, 0 for RGB")
    parser.add_argument("--val-split", type=float, default=0.2, help="If no explicit val folder, split fraction")
    parser.add_argument("--hidden", type=str, default="128,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=1.0, help="Multiply LR by this factor at decay epochs (<=1)")
    parser.add_argument("--lr-decay-epochs", type=str, default="", help="Comma-separated epoch numbers to apply LR decay, e.g. '30,45'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/exp1")
    args = parser.parse_args()

    ds_cfg = DatasetConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        grayscale=bool(args.grayscale),
        val_split=args.val_split,
        seed=args.seed,
    )
    ds = ImageDataset(ds_cfg)

    in_channels = 1 if ds_cfg.grayscale else 3
    input_dim = in_channels * args.img_size * args.img_size
    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]

    model = MLPClassifier(
        ModelConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=ds.num_classes,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
        )
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "classes": ds.classes,
        "config": vars(args),
    }

    best_val_acc = -1.0
    best_path = save_dir / "best_model.npz"

    # Parse decay epochs
    decay_epochs = set()
    if args.lr_decay_epochs.strip():
        try:
            decay_epochs = {int(x) for x in args.lr_decay_epochs.split(',') if x.strip()}
        except Exception:
            decay_epochs = set()

    for epoch in range(1, args.epochs + 1):
        # Step-wise LR decay
        if epoch in decay_epochs and args.lr_decay and args.lr_decay > 0:
            model.cfg.lr *= args.lr_decay
        # Train
        train_losses = []
        train_accs = []
        for X, y in tqdm(ds.batches("train", args.batch_size, shuffle=True), desc=f"Epoch {epoch}/{args.epochs} [train]"):
            stats = model.train_batch(X, y)
            train_losses.append(stats["loss"])
            train_accs.append(stats["acc"])

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        mean_train_acc = float(np.mean(train_accs)) if train_accs else 0.0

        # Val
        val_losses = []
        val_accs = []
        for X, y in tqdm(ds.batches("val", args.batch_size, shuffle=False), desc=f"Epoch {epoch}/{args.epochs} [val]"):
            logits = model.forward(X)
            loss, probs = model.criterion.forward(logits, y)
            preds = probs.argmax(axis=1)
            val_losses.append(float(loss))
            val_accs.append(accuracy(preds, y))

        mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(mean_train_acc)
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(mean_val_acc)
        history["lr"].append(model.cfg.lr)

        print(f"Epoch {epoch}: train_loss={mean_train_loss:.4f} train_acc={mean_train_acc:.4f} val_loss={mean_val_loss:.4f} val_acc={mean_val_acc:.4f}")

        # Save best
        if mean_val_acc >= best_val_acc:
            best_val_acc = mean_val_acc
            model.save(str(best_path), label_names=ds.classes)

        # Save running metrics
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(history, f)

        # Plot curves
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(history["train_acc"], label="train")
        ax[1].plot(history["val_acc"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(save_dir / "curves.png")
        plt.close(fig)

    print(f"Best val acc: {best_val_acc:.4f}. Saved to {best_path}")


if __name__ == "__main__":
    main()
