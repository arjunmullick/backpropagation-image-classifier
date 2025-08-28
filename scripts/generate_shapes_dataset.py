#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random
from PIL import Image, ImageDraw


def gen_circle(size: int, margin: int = 2):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    max_r = max(1, (size - 2 * margin) // 2 - 1)
    min_r = max(1, size // 6)
    if min_r > max_r:
        min_r = max_r
    r = random.randint(min_r, max_r)
    # ensure valid center range
    low = r + margin
    high = size - r - margin
    cx = random.randint(low, max(low, high))
    cy = random.randint(low, max(low, high))
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse(bbox, fill=255)
    # optional light noise / shift
    return img


def gen_square(size: int, margin: int = 2):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    max_side = max(1, size - 2 * margin - 1)
    min_side = max(1, size // 4)
    if min_side > max_side:
        min_side = max_side
    side = random.randint(min_side, max_side)
    x = random.randint(margin, max(margin, size - side - margin))
    y = random.randint(margin, max(margin, size - side - margin))
    bbox = (x, y, x + side, y + side)
    draw.rectangle(bbox, fill=255)
    return img


def gen_triangle(size: int, margin: int = 2):
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    # random triangle roughly centered, with side length range
    max_side = max(3, size - 2 * margin - 1)
    min_side = max(3, size // 4)
    if min_side > max_side:
        min_side = max_side
    side = random.randint(min_side, max_side)
    # base position
    x = random.randint(margin, max(margin, size - side - margin))
    y = random.randint(margin, max(margin, size - side - margin))
    # Upward pointing triangle vertices
    p1 = (x, y + side)
    p2 = (x + side // 2, y)
    p3 = (x + side, y + side)
    draw.polygon([p1, p2, p3], fill=255)
    return img


def save_image(img: Image.Image, path: Path, idx: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate a simple circles vs squares dataset")
    parser.add_argument("--out", type=str, required=True, help="Output root directory, e.g., data/shapes")
    parser.add_argument("--size", type=int, default=28, help="Image size (square)")
    parser.add_argument("--n", type=int, default=1000, help="Total images to generate")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    out_root = Path(args.out)
    if out_root.exists() and any(out_root.iterdir()):
        print(f"Warning: {out_root} already exists and is not empty; files may be overwritten.")

    n_train = int(args.n * args.train_split)
    n_val = args.n - n_train
    classes = ["circle", "square", "triangle"]

    # Generate train
    for split, count in [("train", n_train), ("val", n_val)]:
        for i in range(count):
            label = random.choice(classes)
            if label == "circle":
                img = gen_circle(args.size)
            elif label == "square":
                img = gen_square(args.size)
            else:
                img = gen_triangle(args.size)
            idx = i
            out_path = out_root / split / label / f"{idx:04d}.png"
            save_image(img, out_path, idx)

    print(f"Wrote dataset to {out_root}. Train: {n_train}, Val: {n_val}")


if __name__ == "__main__":
    main()
