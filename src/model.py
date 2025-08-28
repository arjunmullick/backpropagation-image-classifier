from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from .layers import Dense, ReLU, SoftmaxCrossEntropyLoss


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dims: List[int]
    num_classes: int
    lr: float = 0.01
    l2: float = 0.0
    seed: int | None = 42


class MLPClassifier:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        dims = [cfg.input_dim] + cfg.hidden_dims + [cfg.num_classes]
        self.layers: List[Any] = []
        rng = np.random.default_rng(cfg.seed)

        for i in range(len(dims) - 1):
            self.layers.append(Dense(dims[i], dims[i + 1], he=True, seed=int(rng.integers(0, 1_000_000))))
            if i < len(dims) - 2:
                self.layers.append(ReLU())

        self.criterion = SoftmaxCrossEntropyLoss()

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dloss_dlogits: np.ndarray):
        dout = dloss_dlogits
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                dout = layer.backward(dout)

    def sgd_step(self):
        # Apply SGD update to Dense layers
        for layer in self.layers:
            if isinstance(layer, Dense):
                # L2 regularization
                if self.cfg.l2 > 0:
                    layer.dW = layer.dW + self.cfg.l2 * layer.W
                layer.W -= self.cfg.lr * layer.dW
                layer.b -= self.cfg.lr * layer.db

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        logits = self.forward(X)
        loss, probs = self.criterion.forward(logits, y)
        # accuracy
        preds = probs.argmax(axis=1)
        acc = float((preds == y).mean())
        # backward
        dloss_dlogits = self.criterion.backward()
        self.backward(dloss_dlogits)
        self.sgd_step()
        return {"loss": float(loss), "acc": acc}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str, label_names: List[str] | None = None):
        weights = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights.append((layer.W, layer.b))
        npz_dict = {}
        for i, (W, b) in enumerate(weights):
            npz_dict[f"W_{i}"] = W
            npz_dict[f"b_{i}"] = b
        meta = {
            "cfg": {
                "input_dim": self.cfg.input_dim,
                "hidden_dims": self.cfg.hidden_dims,
                "num_classes": self.cfg.num_classes,
                "lr": self.cfg.lr,
                "l2": self.cfg.l2,
            },
            "label_names": label_names or [],
        }
        np.savez(path, **npz_dict, meta=json.dumps(meta))

    @staticmethod
    def load(path: str) -> tuple["MLPClassifier", List[str]]:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        cfg = ModelConfig(
            input_dim=meta["cfg"]["input_dim"],
            hidden_dims=list(meta["cfg"]["hidden_dims"]),
            num_classes=meta["cfg"]["num_classes"],
            lr=meta["cfg"].get("lr", 0.01),
            l2=meta["cfg"].get("l2", 0.0),
        )
        model = MLPClassifier(cfg)
        # assign weights
        dense_layers = [l for l in model.layers if isinstance(l, Dense)]
        for i, layer in enumerate(dense_layers):
            layer.W[...] = data[f"W_{i}"]
            layer.b[...] = data[f"b_{i}"]
        labels = list(meta.get("label_names", []))
        return model, labels

