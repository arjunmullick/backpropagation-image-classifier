#!/usr/bin/env python3
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import numpy as np

# Local imports
from src.model import MLPClassifier


app = Flask(__name__, static_folder="static", template_folder="templates")

RUNS_DIR = Path("runs").resolve()
_MODEL_CACHE: Dict[str, Tuple[MLPClassifier, list[str], int, bool]] = {}


def _safe_run_path(run_name: str) -> Path:
    p = (RUNS_DIR / run_name).resolve()
    if not str(p).startswith(str(RUNS_DIR)):
        raise ValueError("Invalid run path")
    return p


def _infer_img_config(model: MLPClassifier) -> Tuple[int, bool]:
    d = model.cfg.input_dim
    # Try grayscale
    s = int(math.isqrt(d))
    if s * s == d:
        return s, True
    # Try RGB
    if d % 3 == 0:
        s = int(math.isqrt(d // 3))
        if s * s * 3 == d:
            return s, False
    # Fallback
    return 28, True


def _load_model_for_run(run_name: str) -> Tuple[MLPClassifier, list[str], int, bool]:
    if run_name in _MODEL_CACHE:
        return _MODEL_CACHE[run_name]
    run_path = _safe_run_path(run_name)
    model_path = run_path / "best_model.npz"
    model, labels = MLPClassifier.load(str(model_path))
    img_size, grayscale = _infer_img_config(model)
    _MODEL_CACHE[run_name] = (model, labels, img_size, grayscale)
    return _MODEL_CACHE[run_name]


def _preprocess_image(img: Image.Image, img_size: int, grayscale: bool) -> np.ndarray:
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


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/runs")
def list_runs():
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if not d.is_dir():
                continue
            if (d / "best_model.npz").exists() and (d / "metrics.json").exists():
                runs.append(d.name)
    return jsonify({"runs": runs})


@app.get("/metrics")
def get_metrics():
    run_name = request.args.get("run")
    if not run_name:
        return jsonify({"error": "missing run"}), 400
    p = _safe_run_path(run_name)
    m = p / "metrics.json"
    if not m.exists():
        return jsonify({"error": "metrics not found"}), 404
    data = json.loads(m.read_text())
    return jsonify(data)


@app.get("/curves.png")
def get_curves_png():
    run_name = request.args.get("run")
    if not run_name:
        return jsonify({"error": "missing run"}), 400
    p = _safe_run_path(run_name)
    imgp = p / "curves.png"
    if not imgp.exists():
        return jsonify({"error": "curves image not found"}), 404
    return send_file(str(imgp), mimetype="image/png")


@app.post("/predict")
def predict():
    run_name = request.form.get("run") or request.args.get("run")
    if not run_name:
        return jsonify({"error": "missing run"}), 400

    model, labels, img_size, grayscale = _load_model_for_run(run_name)

    img: Image.Image | None = None
    if "image" in request.files:
        file = request.files["image"]
        img = Image.open(file.stream)
    else:
        # Try base64 data URL
        try:
            data_url = request.form.get("image_base64") or (request.json and request.json.get("image_base64"))
        except Exception:
            data_url = None
        if data_url and data_url.startswith("data:image"):
            comma = data_url.find(",")
            b64 = data_url[comma + 1 :]
            data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(data))

    if img is None:
        return jsonify({"error": "no image provided"}), 400

    x = _preprocess_image(img, img_size, grayscale)
    probs = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx] if labels and pred_idx < len(labels) else str(pred_idx)
    probs_list = [
        {"label": labels[i] if i < len(labels) else str(i), "p": float(p)}
        for i, p in enumerate(probs)
    ]
    return jsonify({"prediction": pred_label, "probs": probs_list, "img_size": img_size, "grayscale": grayscale})


def main():
    parser = argparse.ArgumentParser(description="Run the local demo website")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

