#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .model import MLPClassifier


def load_and_preprocess(image_path: str, img_size: int, grayscale: bool) -> np.ndarray:
    img = Image.open(image_path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if grayscale:
        arr = arr.reshape(1, img_size, img_size)
    else:
        arr = arr.transpose(2, 0, 1)
    x = arr.reshape(1, -1)
    return x


def main():
    parser = argparse.ArgumentParser(description="Predict with a saved MLP model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved .npz model")
    parser.add_argument("--image", type=str, required=True, help="Image to classify")
    parser.add_argument("--img-size", type=int, default=28, help="Image size used during training")
    parser.add_argument("--grayscale", type=int, default=1, help="1 if grayscale was used during training")
    args = parser.parse_args()

    model, labels = MLPClassifier.load(args.model)
    x = load_and_preprocess(args.image, args.img_size, bool(args.grayscale))
    probs = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(probs))
    if labels and 0 <= pred_idx < len(labels):
        pred_label = labels[pred_idx]
    else:
        pred_label = str(pred_idx)

    print(f"Prediction: {pred_label}")
    print("Probabilities:")
    for i, p in enumerate(probs):
        name = labels[i] if i < len(labels) else str(i)
        print(f"  {name}: {p:.4f}")


if __name__ == "__main__":
    main()

