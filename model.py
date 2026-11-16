from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

FLOAT = torch.float32
CHUNK = 100_000


def extract_windows(
    channels: torch.Tensor, positions: torch.Tensor, radius: int
) -> torch.Tensor:
    width = 2 * radius + 1
    n = channels.shape[1]
    index = positions[:, None] + torch.arange(width) - radius
    valid = (index >= 0) & (index < n)
    windows = channels[:, index.clamp(0, n - 1)].permute(1, 0, 2)
    return windows * valid[:, None, :]


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


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
    return float(values[order][int(np.argmax(f1))]) - 1e-6


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
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.W1: torch.Tensor | None = None
        self.b1: torch.Tensor | None = None
        self.W2: torch.Tensor | None = None
        self.b2: torch.Tensor | None = None
        self.thresholds: NDArray[np.float32] | None = None

    def _logits(self, features: torch.Tensor) -> torch.Tensor:
        assert self.W1 is not None and self.b1 is not None
        hidden = features @ self.W1 + self.b1
        if self.W2 is None or self.b2 is None:
            return hidden
        return torch.relu(hidden) @ self.W2 + self.b2

    def _windows(
        self, channels: torch.Tensor, positions: NDArray[np.int64]
    ) -> torch.Tensor:
        windows = extract_windows(
            channels, torch.as_tensor(positions), self.radius
        )
        return windows.reshape(len(positions), -1)

    def fit(self, X: NDArray[np.float32], Y: NDArray[np.bool_]) -> Model:
        rng = np.random.default_rng(self.seed)
        channels = torch.as_tensor(X, dtype=FLOAT)
        truth = torch.as_tensor(Y)
        _, length = channels.shape
        sites = Y.shape[0]

        positives = [np.flatnonzero(Y[site]).astype(np.int64) for site in range(sites)]
        exclude = np.zeros(length, dtype=bool)
        for positions in positives:
            for offset in range(-2, 3):
                index = positions + offset
                exclude[index[(index >= 0) & (index < length)]] = True
        background = np.flatnonzero(~exclude)

        positive_positions = np.concatenate(
            [positions for positions in positives if len(positions)]
        )
        count = min(len(background), self.negatives * max(len(positive_positions), 1))
        positions = np.concatenate(
            [positive_positions, rng.choice(background, size=count, replace=False)]
        )

        windows = self._windows(channels, positions)
        self.mean = windows.mean(dim=0)
        self.std = windows.std(dim=0, correction=0) + 1e-6
        labels = truth[:, torch.as_tensor(positions)].t().float()

        pos_weight = torch.ones(sites, dtype=FLOAT)
        for site in range(sites):
            n_positive = float(labels[:, site].sum())
            n_negative = len(labels) - n_positive
            pos_weight[site] = min(
                n_negative / max(n_positive, 1.0), self.max_pos_weight
            )

        self._train((windows - self.mean) / self.std, labels, pos_weight, rng)

        if self.hard_negatives > 0:
            counts = [
                min(len(background), self.hard_negatives * max(len(positives[site]), 1))
                for site in range(sites)
            ]
            positions = np.concatenate(
                [positions, self._hard_negatives(channels, exclude, counts)]
            )
            windows = self._windows(channels, positions)
            labels = truth[:, torch.as_tensor(positions)].t().float()
            self._train(
                (windows - self.mean) / self.std, labels, pos_weight, rng
            )

        self.thresholds = self._thresholds(channels, truth)
        return self

    def _chunk_scores(
        self, X: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        length = X.shape[1]
        low = max(0, start - self.radius)
        high = min(length, end + self.radius)
        left = max(0, self.radius - start)
        right = max(0, end + self.radius - length)
        padded = F.pad(X[:, low:high], (left, right))
        features = F.unfold(padded[None, :, None, :], (1, self.width))[0].t()
        features = (features - self.mean) / self.std
        return torch.sigmoid(self._logits(features)).t()

    def _hard_negatives(
        self,
        X: torch.Tensor,
        exclude: NDArray[np.bool_],
        counts: list[int],
    ) -> NDArray[np.int64]:
        length = X.shape[1]
        best: list[list[tuple[NDArray[np.float32], NDArray[np.int64]]]] = [
            [] for _ in counts
        ]
        for start in range(0, length, CHUNK):
            end = min(start + CHUNK, length)
            scores = self._chunk_scores(X, start, end).numpy()
            for site in range(len(counts)):
                site_scores = scores[site].copy()
                site_scores[exclude[start:end]] = -np.inf
                count = min(counts[site], len(site_scores))
                if count <= 0:
                    continue
                index = np.argpartition(-site_scores, count - 1)[:count]
                best[site].append((site_scores[index], index + start))
        result = []
        for site, entries in enumerate(best):
            if not entries:
                result.append(np.zeros(0, dtype=np.int64))
                continue
            scores = np.concatenate([entry[0] for entry in entries])
            positions = np.concatenate([entry[1] for entry in entries])
            result.append(positions[np.argsort(-scores)[: counts[site]]])
        return np.concatenate(result)

    def _thresholds(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> NDArray[np.float32]:
        length = X.shape[1]
        sites = Y.shape[0]
        values = [[] for _ in range(sites)]
        is_true = [[] for _ in range(sites)]
        for start in range(0, length, CHUNK):
            end = min(start + CHUNK, length)
            scores = self._chunk_scores(X, start, end).numpy()
            for site in range(sites):
                local = local_max(scores[site])
                values[site].append(scores[site][local])
                is_true[site].append(Y[site, np.flatnonzero(local) + start].numpy())
        return np.array(
            [
                best_threshold_values(
                    np.concatenate(values[site]),
                    np.concatenate(is_true[site]),
                    int(Y[site].sum()),
                )
                for site in range(sites)
            ],
            dtype=np.float32,
        )

    def _train(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: torch.Tensor,
        rng: np.random.Generator,
    ) -> None:
        n, dimension = features.shape
        sites = labels.shape[1]
        if self.hidden:
            self.W1 = torch.as_tensor(
                rng.normal(0.0, np.sqrt(2.0 / dimension), (dimension, self.hidden)),
                dtype=FLOAT,
            )
            self.b1 = torch.zeros(self.hidden, dtype=FLOAT)
            self.W2 = torch.as_tensor(
                rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.hidden, sites)),
                dtype=FLOAT,
            )
            self.b2 = torch.zeros(sites, dtype=FLOAT)
        else:
            self.W1 = torch.as_tensor(
                rng.normal(0.0, np.sqrt(1.0 / dimension), (dimension, sites)),
                dtype=FLOAT,
            )
            self.b1 = torch.zeros(sites, dtype=FLOAT)
            self.W2 = self.b2 = None

        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        params = [W1, b1] if W2 is None or b2 is None else [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        weight = pos_weight * labels + (1.0 - labels)
        decay = self.weight_decay
        for _ in range(self.epochs):
            order = torch.as_tensor(rng.permutation(n))
            shuffled_x = features[order]
            shuffled_y = labels[order]
            shuffled_w = weight[order]
            for start in range(0, n, self.batch):
                x = shuffled_x[start : start + self.batch]
                y = shuffled_y[start : start + self.batch]
                w = shuffled_w[start : start + self.batch]
                pre = torch.addmm(b1, x, W1)
                if W2 is None or b2 is None:
                    dlogits = (torch.sigmoid(pre) - y) * w / len(x)
                    grads = [
                        torch.addmm(W1, x.t(), dlogits, beta=decay),
                        dlogits.sum(dim=0),
                    ]
                else:
                    hidden = torch.relu(pre)
                    logits = torch.addmm(b2, hidden, W2)
                    dlogits = (torch.sigmoid(logits) - y) * w / len(x)
                    dhidden = dlogits @ W2.t()
                    dhidden.mul_(pre > 0)
                    grads = [
                        torch.addmm(W1, x.t(), dhidden, beta=decay),
                        dhidden.sum(dim=0),
                        torch.addmm(W2, hidden.t(), dlogits, beta=decay),
                        dlogits.sum(dim=0),
                    ]
                if self.clip:
                    squares = torch.stack(
                        [torch.linalg.vector_norm(grad) ** 2 for grad in grads]
                    )
                    norm = squares.sum().sqrt()
                    if norm > self.clip:
                        grads = [grad * (self.clip / norm) for grad in grads]
                for param, grad in zip(params, grads):
                    param.grad = grad
                optimizer.step()

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.W1 is None:
            raise RuntimeError("model is not fitted")
        channels = torch.as_tensor(X, dtype=FLOAT)
        _, length = channels.shape
        sites = self.W2.shape[1] if self.W2 is not None else self.W1.shape[1]
        out = np.zeros((sites, length), dtype=np.float32)
        for start in range(0, length, CHUNK):
            end = min(start + CHUNK, length)
            out[:, start:end] = self._chunk_scores(channels, start, end).numpy()
        return out
