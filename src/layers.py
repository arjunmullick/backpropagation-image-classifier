from __future__ import annotations
import numpy as np


class Dense:
    def __init__(self, in_dim: int, out_dim: int, weight_scale: float | None = None, he: bool = True, seed: int | None = None):
        rng = np.random.default_rng(seed)
        if he:
            # He initialization for ReLU
            w_scale = np.sqrt(2.0 / in_dim)
        else:
            w_scale = weight_scale if weight_scale is not None else 0.01
        self.W = rng.standard_normal((in_dim, out_dim)) * w_scale
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.x_cache: np.ndarray | None = None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (N, in_dim)
        self.x_cache = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout: (N, out_dim)
        assert self.x_cache is not None
        x = self.x_cache
        self.dW = x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx


class ReLU:
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.mask is not None
        return dout * self.mask


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.logits: np.ndarray | None = None
        self.targets: np.ndarray | None = None  # integer labels, shape (N,)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # Stable softmax
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        # logits: (N, C), targets: (N,)
        self.logits = logits
        self.targets = targets
        N = logits.shape[0]
        probs = self._softmax(logits)
        log_likelihood = -np.log(probs[np.arange(N), targets] + 1e-12)
        loss = log_likelihood.mean()
        return loss, probs

    def backward(self) -> np.ndarray:
        assert self.logits is not None and self.targets is not None
        N, C = self.logits.shape
        probs = self._softmax(self.logits)
        probs[np.arange(N), self.targets] -= 1.0
        probs /= N
        # gradient w.r.t logits
        return probs

