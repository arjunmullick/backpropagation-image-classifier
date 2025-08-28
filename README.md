# Backpropagation Image Classifier (from scratch)

A beginner-friendly, from-scratch image classifier that trains locally using only NumPy and PIL. It includes:

- A tiny dataset generator for synthetic shapes (circles vs squares vs triangles) so you can train immediately without downloads
- A minimal neural network with manual backpropagation (no PyTorch/TensorFlow)
- Simple CLIs to train and predict
- Optional OpenAI-assisted explanations of your training run (if you want to use your API key)

## Quickstart

1) Create a toy dataset (circles vs squares vs triangles):

```
python scripts/generate_shapes_dataset.py --out data/shapes --size 28 --n 1000 --train-split 0.8
```

2) Train the model (baseline):

```
# If you encounter Matplotlib cache/permission issues, prefix with:
# MPLCONFIGDIR=.cache/matplotlib

python -m src.train \
  --data-dir data/shapes \
  --img-size 28 \
  --grayscale 1 \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.05 \
  --save-dir runs/shapes-mlp
```

Longer run with decay and regularization (3 classes recommended):

```
MPLCONFIGDIR=.cache/matplotlib python3 -u -m src.train \
  --data-dir data/shapes \
  --img-size 28 --grayscale 1 \
  --epochs 60 --batch-size 64 \
  --lr 0.05 --l2 1e-4 \
  --hidden 256,128 \
  --lr-decay 0.6 --lr-decay-epochs 30,45 \
  --save-dir runs/shapes-3cls-60e-decay
```

3) Predict on a single image:

```
python -m src.predict --model runs/shapes-mlp/best_model.npz --image data/shapes/val/circle/0001.png --img-size 28 --grayscale 1
```

This prints the predicted class and probabilities.

If you want a visual, interactive demo, see Web Demo (Localhost) below.

## Project Structure

- `scripts/generate_shapes_dataset.py`: Builds a small image dataset locally (no network needed). Now generates circle, square, and triangle classes.
- `src/dataset.py`: Loads images from folders, handles train/val split, batching, and preprocessing.
- `src/layers.py`: Minimal dense layer, ReLU, and softmax cross-entropy loss (forward + backward).
- `src/model.py`: A simple MLP classifier with manual backprop and SGD updates.
- `src/train.py`: Training CLI with metrics and plots (loss/accuracy).
  - Supports L2 regularization and step-wise learning rate decay.
- `src/predict.py`: Predict CLI loading a saved model.
- `src/openai_assist.py` (optional): Uses OpenAI API to summarize/explain your training run.
- `web/`: Local website demo to visualize training and run interactive predictions.

## Dataset Format

Two options:

- Use the generator script to create `data/shapes` (recommended to start). Default classes: `circle`, `square`, `triangle`.
- Bring your own dataset organized as:

```
my_dataset/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    img1.jpg
    ...
```

If you provide a single folder, `train.py` will automatically split into train/val via `--val-split` (default 0.2). If you already have `train/` and `val/` subfolders, they will be used as-is.

## Why this project?

- Learn backprop the practical way: build, train, and inspect gradients and metrics.
- No magic: forward pass and backward pass are written by hand in a small, readable codebase.
- Uses real image data (even if synthetic) so you can see generalization.

## Optional: OpenAI-Assisted Explanations

If you want friendly summaries of your training run (e.g., “why did accuracy stall?”), you can enable `src/openai_assist.py`. Install the extra dependency and set your API key:

```
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

Then, after training, run:

```
python -c "from src.openai_assist import explain_training; print(explain_training('runs/shapes-mlp/metrics.json'))"
```

This is entirely optional—training runs 100% locally with NumPy.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`

## Web Demo (Localhost)

Run a lightweight website to visualize training and try predictions interactively.

1) Start the server (defaults to `localhost:8000`):

```
MPLCONFIGDIR=.cache/matplotlib python -m web.app --port 8000
```

2) Open your browser at `http://localhost:8000`.

3) In the UI:

- Select a run from the dropdown (e.g., `runs/shapes-mlp-20e`). The page shows live-updating training curves by polling the run’s `metrics.json` and also displays the saved `curves.png`.
- Upload an image or draw a white shape on the black canvas, then click Predict to see class probabilities.

Notes:
- The server lists runs that contain both `best_model.npz` and `metrics.json`.
- If you train in another terminal, the metrics chart will update automatically every 2 seconds.

---

**Detailed Overview**
- **Goal:** Hands-on learning of backpropagation by building a minimal image classifier from scratch using only NumPy. You’ll see how forward and backward passes work, and how gradients update weights during training.
- **Model:** A simple multilayer perceptron (MLP) that operates on flattened images. Layers are implemented manually with explicit forward and backward methods.
- **Dataset:** Synthetic circles vs squares, generated locally so you can train immediately without downloads.
- **Outputs:** Model checkpoint (`.npz`), metrics (`metrics.json`), and training curves (`curves.png`).
- **Demo:** A small Flask website to view metrics and test predictions by uploading an image or drawing.

**How It Works**
- **Forward pass:**
  - `Dense`: computes `X @ W + b`.
  - `ReLU`: passes positive activations and zeros out negatives.
  - Final logits go to softmax for probabilities; cross-entropy computes the loss.
- **Backward pass:**
  - `SoftmaxCrossEntropyLoss.backward()` returns `dLoss/dLogits` (softmax probs with 1 subtracted at the target index, normalized by batch size).
  - Each `Dense.backward()` computes `dW = X^T @ dOut`, `db = sum(dOut)`, `dX = dOut @ W^T`.
  - `ReLU.backward()` masks gradients where inputs were negative.
  - `MLPClassifier.sgd_step()` updates `W -= lr * dW` and `b -= lr * db` (with optional L2 weight decay).
- **Data pipeline:**
  - Images are loaded with PIL, resized, normalized to [0,1], optionally converted to grayscale, and flattened into vectors.

**End-to-End Steps (Detailed)**
- **1) Environment setup**
  - Optional: create a virtual env
    - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
    - Windows (PowerShell): `py -m venv .venv; .venv\Scripts\Activate.ps1`
  - Install deps: `pip install -r requirements.txt`

- **2) Generate a dataset**
  - `python scripts/generate_shapes_dataset.py --out data/shapes --size 28 --n 1000 --train-split 0.8`
  - Creates `data/shapes/train/{circle,square}` and `data/shapes/val/{circle,square}` with PNGs.

- **3) Train a baseline model**
  - `MPLCONFIGDIR=.cache/matplotlib python -m src.train --data-dir data/shapes --img-size 28 --grayscale 1 --epochs 20 --batch-size 64 --lr 0.05 --save-dir runs/shapes-mlp-20e`
  - Artifacts written under `runs/shapes-mlp-20e/`.

- **4) Try variants**
  - Larger network: add `--hidden 256,128`.
  - Regularization: add `--l2 1e-4`.
  - Learning rate: try `--lr 0.03`.

- **5) Predict**
  - `python -m src.predict --model runs/<run>/best_model.npz --image data/shapes/val/circle/0001.png --img-size 28 --grayscale 1`

- **6) Launch the web demo**
  - `MPLCONFIGDIR=.cache/matplotlib python -m web.app --port 8000`
  - Visit `http://localhost:8000` → select a run → view curves and predict.

**Troubleshooting**
- **Matplotlib cache/permission error:** prefix training or server with `MPLCONFIGDIR=.cache/matplotlib` (as shown above).
- **Relative import errors:** always run Python entrypoints with `-m`, e.g., `python -m src.train`, not `python src/train.py`.
- **Flask not found:** re-run `pip install -r requirements.txt` to ensure `Flask` is installed.
- **Port already in use:** start the server on another port (e.g., `--port 8001`).
- **Server control:** logs at `.cache/webapp.log`, PID at `.cache/webapp.pid`. Stop with `kill $(cat .cache/webapp.pid)`.

**File-by-File Guide**
- `scripts/generate_shapes_dataset.py`:
  - Generates grayscale images of circles/squares with randomized sizes/positions.
  - CLI: `--out`, `--size`, `--n`, `--train-split`, `--seed`.
- `src/dataset.py`:
  - Scans dataset folders, handles explicit `train/` + `val/` or splits a single folder.
  - Loads, resizes, normalizes images; yields batches `(X, y)` where `X` is `(N, D)` flattened.
- `src/layers.py`:
  - `Dense`: weights `W`, bias `b`, forward affine transform, backward gradients (`dW`, `db`, `dX`).
  - `ReLU`: elementwise nonlinearity with gradient mask.
  - `SoftmaxCrossEntropyLoss`: stable softmax, mean cross-entropy, and `backward()` that returns `dLogits`.
- `src/model.py`:
  - `ModelConfig`: network configuration and hyperparameters.
  - `MLPClassifier`: builds `[Dense, ReLU, ..., Dense]`, runs forward/backward, SGD updates, save/load weights.
  - `train_batch()`: forward → loss/acc → backward → step, returns metrics.
- `src/train.py`:
  - CLI for training; logs epoch metrics, saves best model, writes `metrics.json`, and `curves.png`.
  - Uses `matplotlib` with Agg backend so it doesn’t require a GUI.
- `src/predict.py`:
  - Loads a saved `.npz` model and prints predicted label + probabilities for a single image.
- `src/openai_assist.py` (optional):
  - Summarizes `metrics.json` via OpenAI’s API; requires `OPENAI_API_KEY` and `openai` installed.
- `web/app.py`:
  - Flask server. Endpoints: `/` (UI), `/runs` (list), `/metrics` (JSON), `/curves.png` (plot), `/predict` (inference).
  - Loads the selected run’s `best_model.npz`, infers image size/mode, preprocesses inputs, returns probabilities.
- `web/templates/index.html`:
  - Simple UI: run selection, live charts, image upload, drawing canvas.
- `web/static/main.js`:
  - Polls metrics every 2s; minimal canvas-based charts; handles upload/draw predictions.
- `web/static/styles.css`:
  - Minimal styling for a clean layout.

**Extending The Project**
- Add basic augmentations in `generate_shapes_dataset.py` (e.g., small rotations, translations, noise) to improve generalization.
- Implement a tiny CNN: write `Conv2D` + `MaxPool` layers and compare against the MLP on the same dataset.
- Add early stopping or learning-rate decay in `src/train.py` for more stable training.

## End-to-End Steps

- Install dependencies:
  - `pip install -r requirements.txt`
- Generate dataset:
  - `python scripts/generate_shapes_dataset.py --out data/shapes --size 28 --n 1000 --train-split 0.8`
- Train baseline MLP (128,64):
  - `MPLCONFIGDIR=.cache/matplotlib python -m src.train --data-dir data/shapes --img-size 28 --grayscale 1 --epochs 20 --batch-size 64 --lr 0.05 --save-dir runs/shapes-mlp-20e`
- Train larger MLP (256,128) with regularization and LR decay:
  - `MPLCONFIGDIR=.cache/matplotlib python -m src.train --data-dir data/shapes --img-size 28 --grayscale 1 --epochs 60 --batch-size 64 --lr 0.05 --l2 1e-4 --hidden 256,128 --lr-decay 0.6 --lr-decay-epochs 30,45 --save-dir runs/shapes-3cls-60e-decay`
- Predict on an image:
  - `python -m src.predict --model runs/<run>/best_model.npz --image path/to/image.png --img-size 28 --grayscale 1`

Notes:
- Use `python -m src.train` and `python -m src.predict` (module mode) to avoid relative import issues.
- If Matplotlib warns about cache permissions, prefix commands with `MPLCONFIGDIR=.cache/matplotlib` as shown.

## Training Options Reference

- `--hidden`: comma-separated hidden layer sizes (e.g., `128,64` or `256,128`).
- `--epochs`: number of training epochs.
- `--batch-size`: mini-batch size.
- `--lr`: learning rate.
- `--l2`: L2 weight decay regularization (e.g., `1e-4`).
- `--lr-decay`: multiply learning rate by this factor at specified epochs (<= 1.0). Example: `0.6`.
- `--lr-decay-epochs`: comma-separated epoch numbers where decay applies. Example: `30,45`.
- `--val-split`: when there’s no explicit `val/` folder, fraction for validation split.
- `--img-size`, `--grayscale`: image loading parameters; must match training at prediction time.

Reproducible high-accuracy recipe (3 classes):

```
MPLCONFIGDIR=.cache/matplotlib python3 -u -m src.train \
  --data-dir data/shapes --img-size 28 --grayscale 1 \
  --epochs 60 --batch-size 64 \
  --lr 0.05 --l2 1e-4 \
  --hidden 256,128 \
  --lr-decay 0.6 --lr-decay-epochs 30,45 \
  --save-dir runs/shapes-3cls-60e-decay
```

`metrics.json` contains arrays for `train_loss`, `val_loss`, `train_acc`, `val_acc`, and `lr` per epoch.

## Notes

- This is meant for learning. It’s not optimized for speed and only includes an MLP. Adding a convolution layer is a good next exercise once you’re comfortable with this.
- If you run into performance limits, try smaller images or fewer hidden units.
