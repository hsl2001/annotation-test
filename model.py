from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def extract_windows(
    channels: NDArray[np.float32], positions: NDArray[np.int64], radius: int
) -> NDArray[np.float32]:
    width = 2 * radius + 1
    n = channels.shape[1]
    offsets = np.arange(width) - radius
    index = positions[:, None] + offsets[None, :]
    valid = (index >= 0) & (index < n)
    clipped = np.clip(index, 0, n - 1)
    windows = channels[:, clipped].transpose(1, 0, 2)
    windows *= valid[:, None, :]
    return np.ascontiguousarray(windows, dtype=np.float32)


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(
        self,
        radius: int = 26,
        hidden: int = 128,
        epochs: int = 60,
        batch: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        negatives: int = 8,
        hard_negatives: int = 5,
        seed: int = 0,
    ):
        self.radius = radius
        self.width = 2 * radius + 1
        self.hidden = hidden
        self.epochs = epochs
        self.batch = batch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negatives = negatives
        self.hard_negatives = hard_negatives
        self.seed = seed
        self.mean: NDArray[np.float64] | None = None
        self.std: NDArray[np.float64] | None = None
        self.W1: NDArray[np.float64] | None = None
        self.b1: NDArray[np.float64] | None = None
        self.W2: NDArray[np.float64] | None = None
        self.b2: NDArray[np.float64] | None = None
        self.thresholds: NDArray[np.float32] | None = None

    def _forward(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.W1 is not None and self.b1 is not None
        assert self.W2 is not None and self.b2 is not None
        hidden = np.maximum(features @ self.W1 + self.b1, 0.0)
        return _sigmoid(hidden @ self.W2 + self.b2)

    def fit(self, X: NDArray[np.float32], Y: NDArray[np.bool_]) -> Model:
        rng = np.random.default_rng(self.seed)
        _, length = X.shape
        sites = Y.shape[0]

        positives = [np.flatnonzero(Y[site]).astype(np.int64) for site in range(sites)]
        exclude = np.zeros(length, dtype=bool)
        for positions in positives:
            for offset in range(-2, 3):
                index = positions + offset
                index = index[(index >= 0) & (index < length)]
                exclude[index] = True
        background = np.flatnonzero(~exclude)

        positive_positions = np.concatenate(
            [positions for positions in positives if len(positions)]
        )
        count = min(len(background), self.negatives * max(len(positive_positions), 1))
        negative_positions = rng.choice(background, size=count, replace=False)
        positions = np.concatenate([positive_positions, negative_positions])

        labels = Y[:, positions].T.astype(np.float64)
        windows = extract_windows(X, positions, self.radius).reshape(
            len(positions), -1
        ).astype(np.float64)
        self.mean = windows.mean(axis=0)
        self.std = windows.std(axis=0) + 1e-6
        windows = (windows - self.mean) / self.std

        pos_weight = np.ones(sites)
        for site in range(sites):
            n_positive = labels[:, site].sum()
            n_negative = len(labels) - n_positive
            pos_weight[site] = min(n_negative / max(n_positive, 1.0), 100.0)

        self._train(windows, labels, pos_weight, rng)

        scores = self.predict(X)
        hard = []
        for site in range(sites):
            site_scores = scores[site].copy()
            site_scores[exclude] = -np.inf
            count = min(
                len(background), self.hard_negatives * max(len(positives[site]), 1)
            )
            hard.append(np.argsort(-site_scores)[:count])
        hard_positions = np.concatenate(hard)
        positions = np.concatenate([positions, hard_positions])
        labels = Y[:, positions].T.astype(np.float64)
        windows = extract_windows(X, positions, self.radius).reshape(
            len(positions), -1
        ).astype(np.float64)
        windows = (windows - self.mean) / self.std
        self._train(windows, labels, pos_weight, rng)

        scores = self.predict(X)
        self.thresholds = np.array(
            [
                self._best_threshold(scores[site], Y[site])
                for site in range(sites)
            ],
            dtype=np.float32,
        )
        return self

    def _best_threshold(
        self, score: NDArray[np.float32], label: NDArray[np.bool_]
    ) -> float:
        truth = np.flatnonzero(label)
        if not len(truth):
            return 1.0
        positions = np.flatnonzero(local_max(score))
        values = score[positions]
        is_true = np.isin(positions, truth)
        best_f1, best_threshold = 0.0, 0.5
        for threshold in np.linspace(0.5, 1.0, 101):
            selected = values > threshold
            true_positive = int(is_true[selected].sum())
            false_positive = int(selected.sum()) - true_positive
            false_negative = len(truth) - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator else 0.0
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(threshold)
        return best_threshold

    def _train(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.float64],
        pos_weight: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> None:
        dimension = features.shape[1]
        sites = labels.shape[1]
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / dimension), (dimension, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.hidden, sites))
        self.b2 = np.zeros(sites)

        params = [self.W1, self.b1, self.W2, self.b2]
        first = [np.zeros_like(param) for param in params]
        second = [np.zeros_like(param) for param in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = len(features)

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                index = order[start : start + self.batch]
                x = features[index]
                y = labels[index]

                pre = x @ self.W1 + self.b1
                hidden = np.maximum(pre, 0.0)
                probs = _sigmoid(hidden @ self.W2 + self.b2)

                weight = pos_weight * y + (1.0 - y)
                dlogits = (probs - y) * weight / len(index)
                dW2 = hidden.T @ dlogits + self.weight_decay * self.W2
                db2 = dlogits.sum(axis=0)
                dhidden = dlogits @ self.W2.T
                dhidden[pre <= 0] = 0.0
                dW1 = x.T @ dhidden + self.weight_decay * self.W1
                db1 = dhidden.sum(axis=0)

                step += 1
                for i, (param, grad) in enumerate(zip(params, [dW1, db1, dW2, db2])):
                    first[i] = beta1 * first[i] + (1.0 - beta1) * grad
                    second[i] = beta2 * second[i] + (1.0 - beta2) * grad * grad
                    m_hat = first[i] / (1.0 - beta1**step)
                    v_hat = second[i] / (1.0 - beta2**step)
                    param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.W1 is None or self.W2 is None:
            raise RuntimeError("model is not fitted")
        assert self.mean is not None and self.std is not None
        _, length = X.shape
        padded = np.pad(X, ((0, 0), (self.radius, self.radius)))
        out = np.zeros((self.W2.shape[1], length), dtype=np.float32)
        chunk = 100_000
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            windows = sliding_window_view(
                padded[:, start : end + 2 * self.radius], self.width, axis=1
            )
            features = windows.transpose(1, 0, 2).reshape(end - start, -1)
            features = (features.astype(np.float64) - self.mean) / self.std
            out[:, start:end] = self._forward(features).T
        return out
