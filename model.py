from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden: int, sites: int):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden)
        self.fc2 = nn.Linear(hidden, sites)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(features)))


class Model:
    def __init__(
        self,
        radius: int = 50,
        hidden: int = 256,
        epochs: int = 100,
        batch: int = 1000,
        learning_rate: float = 1e-3,
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
        self.net: nn.Module | None = None
        self.thresholds: NDArray[np.float32] | None = None

    def _logits(self, features: torch.Tensor) -> torch.Tensor:
        assert self.net is not None
        return self.net(features)

    def _windows(
        self, channels: torch.Tensor, positions: NDArray[np.int64]
    ) -> torch.Tensor:
        windows = extract_windows(
            channels, torch.as_tensor(positions), self.radius
        )
        return windows.reshape(len(positions), -1)

    def _normalize(self, windows: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        return (windows - self.mean) / self.std

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

        self._train(self._normalize(windows), labels, pos_weight, rng)

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
            self._train(self._normalize(windows), labels, pos_weight, rng)

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
        with torch.no_grad():
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
        dimension = features.shape[1]
        sites = labels.shape[1]
        if self.hidden:
            self.net = MLP(dimension, self.hidden, sites)
        else:
            self.net = nn.Linear(dimension, sites)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        weight = pos_weight * labels + (1.0 - labels)
        loader = DataLoader(
            TensorDataset(features, labels, weight),
            batch_size=self.batch,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        for _ in range(self.epochs):
            for x, y, w in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    F.binary_cross_entropy_with_logits(
                        self._logits(x), y, reduction="none"
                    )
                    * w
                ).mean()
                loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                optimizer.step()

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        channels = torch.as_tensor(X, dtype=FLOAT)
        _, length = channels.shape
        sites = len(self.thresholds) if self.thresholds is not None else 0
        out = np.zeros((sites, length), dtype=np.float32)
        for start in range(0, length, CHUNK):
            end = min(start + CHUNK, length)
            out[:, start:end] = self._chunk_scores(channels, start, end).numpy()
        return out
