from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
import random

import numpy as np
from PIL import Image


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def _scan_single_level(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[Path, int]] = []
    for c in classes:
        for p in (root / c).rglob("*"):
            if p.is_file() and _is_image(p):
                items.append((p, class_to_idx[c]))
    return items, classes


def _scan_with_splits(root: Path) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[str]]:
    train_root = root / "train"
    val_root = root / "val"
    if train_root.exists() and val_root.exists():
        train_items, classes_train = _scan_single_level(train_root)
        val_items, classes_val = _scan_single_level(val_root)
        if classes_train != classes_val:
            raise ValueError("Train and val classes differ")
        return train_items, val_items, classes_train
    else:
        items, classes = _scan_single_level(root)
        return items, [], classes


@dataclass
class DatasetConfig:
    data_dir: str
    img_size: int = 28
    grayscale: bool = True
    val_split: float = 0.2
    seed: int = 42


class ImageDataset:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        root = Path(cfg.data_dir)
        train_items, val_items, classes = _scan_with_splits(root)
        if len(val_items) == 0 and cfg.val_split > 0:
            random.seed(cfg.seed)
            random.shuffle(train_items)
            n_val = int(len(train_items) * cfg.val_split)
            val_items = train_items[:n_val]
            train_items = train_items[n_val:]

        self.train_items = train_items
        self.val_items = val_items
        self.classes = classes
        self.num_classes = len(classes)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        if self.cfg.grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((self.cfg.img_size, self.cfg.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.cfg.grayscale:
            arr = arr[None, ...]  # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)  # (C, H, W)
        return arr

    def _to_row(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1)

    def batches(self, split: str, batch_size: int, shuffle: bool = True) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if split == "train":
            items = list(self.train_items)
        elif split == "val":
            items = list(self.val_items)
        else:
            raise ValueError("split must be 'train' or 'val'")

        if shuffle:
            random.seed(self.cfg.seed)
            random.shuffle(items)

        X_batch: List[np.ndarray] = []
        y_batch: List[int] = []
        for p, label in items:
            arr = self._load_image(p)
            X_batch.append(self._to_row(arr))
            y_batch.append(label)
            if len(X_batch) == batch_size:
                X = np.stack(X_batch, axis=0)
                y = np.asarray(y_batch, dtype=np.int64)
                yield X, y
                X_batch, y_batch = [], []
        if X_batch:
            X = np.stack(X_batch, axis=0)
            y = np.asarray(y_batch, dtype=np.int64)
            yield X, y

