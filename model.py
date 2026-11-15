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


def best_threshold(
    score: NDArray[np.float32], label: NDArray[np.bool_]
) -> float:
    truth = np.flatnonzero(label)
    if not len(truth):
        return 1.0
    positions = np.flatnonzero(local_max(score))
    if not len(positions):
        return 1.0
    return best_threshold_values(
        score[positions], np.isin(positions, truth), len(truth)
    )


def best_threshold_values(
    values: NDArray[np.float32],
    is_true: NDArray[np.bool_],
    n_truth: int,
) -> float:
    if not len(values) or not n_truth:
        return 1.0
    order = np.argsort(-values)
    true_positive = np.cumsum(is_true[order])
    false_positive = np.cumsum(~is_true[order])
    denominator = 2 * true_positive + false_positive + n_truth - true_positive
    f1 = 2 * true_positive / denominator
    best = int(np.argmax(f1))
    return float(values[order][best]) - 1e-6


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


class Model:
    def __init__(
        self,
        radius: int = 50,
        hidden: int = 256,
        epochs: int = 100,
        batch: int = 256,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        negatives: int = 8,
        hard_negatives: int = 5,
        max_pos_weight: float = 100.0,
        clip: float = 5.0,
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
        self.max_pos_weight = max_pos_weight
        self.clip = clip
        self.seed = seed
        self.mean: NDArray[np.float32] | None = None
        self.std: NDArray[np.float32] | None = None
        self.W1: NDArray[np.float32] | None = None
        self.b1: NDArray[np.float32] | None = None
        self.W2: NDArray[np.float32] | None = None
        self.b2: NDArray[np.float32] | None = None
        self.thresholds: NDArray[np.float32] | None = None

    def _forward(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.W1 is not None and self.b1 is not None
        if self.W2 is None or self.b2 is None:
            return _sigmoid(features @ self.W1 + self.b1)
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

        labels = Y[:, positions].T.astype(np.float32)
        windows = extract_windows(X, positions, self.radius).reshape(
            len(positions), -1
        )
        self.mean = windows.mean(axis=0)
        self.std = windows.std(axis=0) + 1e-6
        windows -= self.mean
        windows /= self.std

        pos_weight = np.ones(sites, dtype=np.float32)
        for site in range(sites):
            n_positive = labels[:, site].sum()
            n_negative = len(labels) - n_positive
            pos_weight[site] = min(n_negative / max(n_positive, 1.0), self.max_pos_weight)

        self._train(windows, labels, pos_weight, rng)

        if self.hard_negatives > 0:
            counts = [
                min(len(background), self.hard_negatives * max(len(positives[site]), 1))
                for site in range(sites)
            ]
            hard_positions = self._hard_negatives(X, exclude, counts)
            positions = np.concatenate([positions, hard_positions])
            labels = Y[:, positions].T.astype(np.float32)
            windows = extract_windows(X, positions, self.radius).reshape(
                len(positions), -1
            )
            windows -= self.mean
            windows /= self.std
            self._train(windows, labels, pos_weight, rng)

        self.thresholds = self._thresholds(X, Y)
        return self

    def _chunk_scores(
        self, X: NDArray[np.float32], start: int, end: int
    ) -> NDArray[np.float32]:
        assert self.W1 is not None and self.mean is not None and self.std is not None
        length = X.shape[1]
        low = max(0, start - self.radius)
        high = min(length, end + self.radius)
        left = max(0, self.radius - start)
        right = max(0, end + self.radius - length)
        padded = np.pad(X[:, low:high], ((0, 0), (left, right)))
        windows = sliding_window_view(padded, self.width, axis=1)
        features = windows.transpose(1, 0, 2).reshape(end - start, -1)
        features = (features - self.mean) / self.std
        return self._forward(features).T

    def _hard_negatives(
        self,
        X: NDArray[np.float32],
        exclude: NDArray[np.bool_],
        counts: list[int],
    ) -> NDArray[np.int64]:
        length = X.shape[1]
        sites = len(counts)
        chunk = 100_000
        best_scores = [[] for _ in range(sites)]
        best_positions = [[] for _ in range(sites)]
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            scores = self._chunk_scores(X, start, end)
            for site in range(sites):
                site_scores = scores[site].copy()
                site_scores[exclude[start:end]] = -np.inf
                count = min(counts[site], len(site_scores))
                if count <= 0:
                    continue
                index = np.argpartition(-site_scores, count - 1)[:count]
                best_scores[site].append(site_scores[index])
                best_positions[site].append(index + start)
        result = []
        for site in range(sites):
            if not best_scores[site]:
                result.append(np.zeros(0, dtype=np.int64))
                continue
            scores = np.concatenate(best_scores[site])
            positions = np.concatenate(best_positions[site])
            order = np.argsort(-scores)[: counts[site]]
            result.append(positions[order])
        return np.concatenate(result)

    def _thresholds(
        self, X: NDArray[np.float32], Y: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        length = X.shape[1]
        sites = Y.shape[0]
        chunk = 100_000
        values = [[] for _ in range(sites)]
        is_true = [[] for _ in range(sites)]
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            scores = self._chunk_scores(X, start, end)
            for site in range(sites):
                local = local_max(scores[site])
                positions = np.flatnonzero(local) + start
                values[site].append(scores[site][local])
                is_true[site].append(Y[site, positions])
        thresholds = np.empty(sites, dtype=np.float32)
        for site in range(sites):
            thresholds[site] = best_threshold_values(
                np.concatenate(values[site]),
                np.concatenate(is_true[site]),
                int(Y[site].sum()),
            )
        return thresholds

    def _best_threshold(
        self, score: NDArray[np.float32], label: NDArray[np.bool_]
    ) -> float:
        return best_threshold(score, label)

    def _train(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.float32],
        pos_weight: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> None:
        dimension = features.shape[1]
        sites = labels.shape[1]
        if self.hidden:
            self.W1 = rng.normal(0.0, np.sqrt(2.0 / dimension), (dimension, self.hidden)).astype(np.float32)
            self.b1 = np.zeros(self.hidden, dtype=np.float32)
            self.W2 = rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.hidden, sites)).astype(np.float32)
            self.b2 = np.zeros(sites, dtype=np.float32)
            params = [self.W1, self.b1, self.W2, self.b2]
        else:
            self.W1 = rng.normal(0.0, np.sqrt(1.0 / dimension), (dimension, sites)).astype(np.float32)
            self.b1 = np.zeros(sites, dtype=np.float32)
            self.W2 = None
            self.b2 = None
            params = [self.W1, self.b1]
        first = [np.zeros_like(param) for param in params]
        second = [np.zeros_like(param) for param in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = len(features)
        assert self.W1 is not None and self.b1 is not None
        W1, b1 = self.W1, self.b1
        W2, b2 = self.W2, self.b2
        if self.hidden:
            assert W2 is not None and b2 is not None

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                index = order[start : start + self.batch]
                x = features[index]
                y = labels[index]

                if self.hidden:
                    assert W2 is not None and b2 is not None
                    pre = x @ W1 + b1
                    hidden = np.maximum(pre, 0.0)
                    probs = _sigmoid(hidden @ W2 + b2)
                    weight = pos_weight * y + (1.0 - y)
                    dlogits = (probs - y) * weight / len(index)
                    dW2 = hidden.T @ dlogits + self.weight_decay * W2
                    db2 = dlogits.sum(axis=0)
                    dhidden = dlogits @ W2.T
                    dhidden[pre <= 0] = 0.0
                    dW1 = x.T @ dhidden + self.weight_decay * W1
                    db1 = dhidden.sum(axis=0)
                    grads = [dW1, db1, dW2, db2]
                else:
                    probs = _sigmoid(x @ W1 + b1)
                    weight = pos_weight * y + (1.0 - y)
                    dlogits = (probs - y) * weight / len(index)
                    dW1 = x.T @ dlogits + self.weight_decay * W1
                    db1 = dlogits.sum(axis=0)
                    grads = [dW1, db1]

                step += 1
                if self.clip:
                    norm = np.sqrt(sum(float((grad * grad).sum()) for grad in grads))
                    if norm > self.clip:
                        scale = self.clip / norm
                        grads = [grad * scale for grad in grads]
                for i, (param, grad) in enumerate(zip(params, grads)):
                    first[i] = (beta1 * first[i] + (1.0 - beta1) * grad).astype(np.float32)
                    second[i] = (beta2 * second[i] + (1.0 - beta2) * grad * grad).astype(np.float32)
                    m_hat = first[i] / (1.0 - beta1**step)
                    v_hat = second[i] / (1.0 - beta2**step)
                    param -= (
                        self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                    ).astype(np.float32)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.W1 is None:
            raise RuntimeError("model is not fitted")
        _, length = X.shape
        sites = self.W2.shape[1] if self.W2 is not None else self.W1.shape[1]
        out = np.zeros((sites, length), dtype=np.float32)
        chunk = 100_000
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            out[:, start:end] = self._chunk_scores(X, start, end)
        return out


