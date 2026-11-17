from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    periods: tuple[int, ...] = (2, 3, 4, 6, 10)
    radius: int = 20
    hidden: int = 64
    dilations: tuple[int, ...] = (1, 2, 4)
    negatives: int = 8
    hard_negatives: int = 4
    epochs: int = 20
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1
    max_pos_weight: float = 100.0
    clip: float = 5.0
    seed: int = 0
    chunk: int = 100_000


class ModelProtocol(Protocol):
    label_names: list[str]
    thresholds: NDArray[np.float32] | None

    def fit(
        self, features: NDArray[np.float32], labels: dict[str, NDArray[np.int64]]
    ) -> ModelProtocol: ...

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def save(self, path: str) -> None: ...


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


def best_threshold_values(
    values: NDArray[np.float32], is_true: NDArray[np.bool_], n_truth: int
) -> float:
    if not len(values) or not n_truth:
        return 1.0
    order = np.argsort(-values)
    true_positive = np.cumsum(is_true[order])
    false_positive = np.cumsum(~is_true[order])
    denominator = 2 * true_positive + false_positive + n_truth - true_positive
    f1 = 2 * true_positive / denominator
    return float(values[order][int(np.argmax(f1))]) - 1e-6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class SpotNet(nn.Module):
    def __init__(self, channels: int, labels: int, config: Config):
        super().__init__()
        self.input = nn.Conv1d(channels, config.hidden, 5, padding=2)
        self.norm = nn.BatchNorm1d(config.hidden)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(config.hidden, dilation, config.dropout)
                for dilation in config.dilations
            ]
        )
        self.output = nn.Conv1d(config.hidden, labels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm(self.input(x)))
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class SpotModel:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.net: SpotNet | None = None
        self.mean: NDArray[np.float32] | None = None
        self.std: NDArray[np.float32] | None = None
        self.thresholds: NDArray[np.float32] | None = None
        self.label_names: list[str] = []

    def _windows(
        self, features: NDArray[np.float32], positions: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        radius = self.config.radius
        width = 2 * radius + 1
        index = positions[:, None] + np.arange(width) - radius
        valid = (index >= 0) & (index < features.shape[1])
        windows = features[:, np.clip(index, 0, features.shape[1] - 1)].transpose(1, 0, 2)
        return windows * valid[:, None, :]

    def _train(
        self,
        features: NDArray[np.float32],
        truth: NDArray[np.bool_],
        positions: NDArray[np.int64],
    ) -> None:
        config = self.config
        assert self.net is not None and self.mean is not None and self.std is not None
        windows = (self._windows(features, positions) - self.mean) / self.std
        labels = truth[:, positions].T.astype(np.float32)
        pos_weight = torch.ones(len(self.label_names), dtype=torch.float32)
        for row in range(len(self.label_names)):
            n_pos = float(labels[:, row].sum())
            n_neg = len(labels) - n_pos
            pos_weight[row] = min(n_neg / max(n_pos, 1.0), config.max_pos_weight)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(windows), torch.as_tensor(labels)),
            batch_size=config.batch,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.net.train()
        for _ in range(config.epochs):
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.net(x)[:, :, config.radius]
                weight = pos_weight * y + (1.0 - y)
                loss = (
                    F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weight
                ).mean()
                loss.backward()
                if config.clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
                optimizer.step()

    def _score(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.net is not None and self.mean is not None and self.std is not None
        self.net.eval()
        length = features.shape[1]
        out = np.zeros((len(self.label_names), length), dtype=np.float32)
        margin = 64
        with torch.no_grad():
            for start in range(0, length, self.config.chunk):
                end = min(start + self.config.chunk, length)
                low = max(0, start - margin)
                high = min(length, end + margin)
                window = (features[:, low:high] - self.mean) / self.std
                logits = self.net(torch.as_tensor(window[None], dtype=torch.float32))[0]
                out[:, start:end] = torch.sigmoid(
                    logits[:, start - low : end - low]
                ).numpy()
        return out

    def _hard_negatives(
        self, scores: NDArray[np.float32], exclude: NDArray[np.bool_], count: int
    ) -> NDArray[np.int64]:
        max_score = np.where(exclude, -np.inf, scores.max(axis=0))
        available = int(np.isfinite(max_score).sum())
        count = min(count, available)
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        return np.argpartition(-max_score, count - 1)[:count].astype(np.int64)

    def _calibrate(
        self, features: NDArray[np.float32], truth: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        scores = self._score(features)
        thresholds = np.zeros(len(self.label_names), dtype=np.float32)
        for row in range(len(self.label_names)):
            peaks = np.flatnonzero(local_max(scores[row]))
            thresholds[row] = best_threshold_values(
                scores[row][peaks], truth[row][peaks], int(truth[row].sum())
            )
        return thresholds

    def fit(
        self, features: NDArray[np.float32], labels: dict[str, NDArray[np.int64]]
    ) -> SpotModel:
        config = self.config
        rng = np.random.default_rng(config.seed)
        length = features.shape[1]
        self.label_names = list(labels)
        truth = np.zeros((len(self.label_names), length), dtype=bool)
        for row, label in enumerate(self.label_names):
            truth[row, labels[label]] = True

        sample = rng.choice(length, size=min(length, 100_000), replace=False)
        windows = self._windows(features, sample)
        self.mean = windows.mean(axis=(0, 2))[:, None]
        self.std = windows.std(axis=(0, 2))[:, None] + 1e-6

        positives = np.flatnonzero(truth.any(axis=0))
        exclude = np.zeros(length, dtype=bool)
        for offset in range(-config.radius, config.radius + 1):
            index = positives + offset
            exclude[index[(index >= 0) & (index < length)]] = True
        background = np.flatnonzero(~exclude)
        count = min(len(background), config.negatives * max(len(positives), 1))
        negatives = rng.choice(background, size=count, replace=False)
        positions = np.concatenate([positives, negatives])

        self.net = SpotNet(features.shape[0], len(self.label_names), config)
        self._train(features, truth, positions)

        if config.hard_negatives > 0:
            scores = self._score(features)
            hard = self._hard_negatives(
                scores, exclude, config.hard_negatives * max(len(positives), 1)
            )
            positions = np.concatenate([positions, hard])
            self._train(features, truth, positions)

        self.thresholds = self._calibrate(features, truth)
        return self

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._score(features)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "state": self.net.state_dict() if self.net is not None else None,
                "mean": self.mean,
                "std": self.std,
                "thresholds": self.thresholds,
                "label_names": self.label_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> SpotModel:
        data = torch.load(path, weights_only=False)
        model = cls(data["config"])
        model.label_names = data["label_names"]
        model.net = SpotNet(
            int(data["mean"].shape[0]), len(model.label_names), model.config
        )
        model.net.load_state_dict(data["state"])
        model.mean = data["mean"]
        model.std = data["std"]
        model.thresholds = data["thresholds"]
        return model


def build_model(config: Config | None = None) -> ModelProtocol:
    return SpotModel(config)


def load_model(path: str) -> ModelProtocol:
    return SpotModel.load(path)
