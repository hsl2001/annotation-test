from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn


@dataclass
class Config:
    periods: tuple[int, ...] = (2, 3, 4, 6, 10)
    radius: int = 50
    hidden: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    epochs: int = 20
    batch: int = 32
    crop: int = 2048
    steps: int = 32
    val_fraction: float = 0.15
    patience: int = 5
    hard_negatives: int = 2000
    hard_weight: float = 5.0
    hard_start: int = 5
    hard_interval: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1
    max_pos_weight: float = 1000.0
    clip: float = 5.0
    seed: int = 0
    chunk: int = 100_000


class ModelProtocol(Protocol):
    label_names: list[str]
    thresholds: NDArray[np.float32] | None

    def fit(
        self,
        features: NDArray[np.float32],
        labels: dict[str, NDArray[np.int64]],
        motif_mask: NDArray[np.bool_] | None = None,
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


def average_precision(
    scores: NDArray[np.float32], labels: NDArray[np.bool_]
) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    true_positive = np.cumsum(y)
    precision = true_positive / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


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

    def _normalize(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.mean is not None and self.std is not None
        return (features - self.mean) / self.std

    def _train(
        self,
        features: NDArray[np.float32],
        truth: NDArray[np.bool_],
        rng: np.random.Generator,
        validation: tuple[NDArray[np.float32], NDArray[np.bool_]] | None = None,
        motif_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        config = self.config
        assert self.net is not None and self.mean is not None and self.std is not None
        length = features.shape[1]
        crop = min(config.crop, length)
        normalized = self._normalize(features)
        pos_weight = torch.ones(len(self.label_names), dtype=torch.float32)
        for row in range(len(self.label_names)):
            n_pos = float(truth[row].sum())
            n_neg = length - n_pos
            pos_weight[row] = min(n_neg / max(n_pos, 1.0), config.max_pos_weight)
        hard_weight = np.ones_like(truth, dtype=np.float32)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1)
        )
        best_score = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(config.epochs):
            if (
                motif_mask is not None
                and config.hard_negatives > 0
                and epoch >= config.hard_start
                and (epoch - config.hard_start) % config.hard_interval == 0
            ):
                hard_weight = np.ones_like(truth, dtype=np.float32)
                scores = self._score(features)
                for row in range(len(self.label_names)):
                    candidates = np.flatnonzero(motif_mask[row] & ~truth[row])
                    if not len(candidates):
                        continue
                    count = min(config.hard_negatives, len(candidates))
                    order = np.argpartition(-scores[row][candidates], count - 1)[:count]
                    hard_weight[row, candidates[order]] = config.hard_weight
            self.net.train()
            for _ in range(config.steps):
                starts = rng.integers(0, length - crop + 1, size=config.batch)
                x = torch.as_tensor(
                    np.stack([normalized[:, s : s + crop] for s in starts]),
                    dtype=torch.float32,
                )
                y = torch.as_tensor(
                    np.stack([truth[:, s : s + crop] for s in starts]),
                    dtype=torch.float32,
                )
                w = torch.as_tensor(
                    np.stack([hard_weight[:, s : s + crop] for s in starts]),
                    dtype=torch.float32,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = self.net(x)
                weight = pos_weight[None, :, None] * y + (1.0 - y)
                loss = (
                    F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                    * weight
                    * w
                ).mean()
                loss.backward()
                if config.clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
                optimizer.step()
            scheduler.step()
            if validation is None:
                continue
            score = self._mean_ap(*validation)
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.net.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _mean_ap(
        self, features: NDArray[np.float32], truth: NDArray[np.bool_]
    ) -> float:
        scores = self._score(features)
        values = [
            average_precision(scores[row], truth[row])
            for row in range(len(self.label_names))
            if truth[row].any()
        ]
        return float(np.mean(values)) if values else 0.0

    def _score(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.net is not None and self.mean is not None and self.std is not None
        self.net.eval()
        length = features.shape[1]
        out = np.zeros((len(self.label_names), length), dtype=np.float32)
        margin = max(64, 4 * self.config.radius)
        with torch.no_grad():
            for start in range(0, length, self.config.chunk):
                end = min(start + self.config.chunk, length)
                low = max(0, start - margin)
                high = min(length, end + margin)
                window = self._normalize(features[:, low:high])
                logits = self.net(torch.as_tensor(window[None], dtype=torch.float32))[0]
                out[:, start:end] = torch.sigmoid(
                    logits[:, start - low : end - low]
                ).numpy()
        return out

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
        self,
        features: NDArray[np.float32],
        labels: dict[str, NDArray[np.int64]],
        motif_mask: NDArray[np.bool_] | None = None,
    ) -> SpotModel:
        config = self.config
        rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        length = features.shape[1]
        self.label_names = list(labels)
        truth = np.zeros((len(self.label_names), length), dtype=bool)
        for row, label in enumerate(self.label_names):
            truth[row, labels[label]] = True

        sample = rng.choice(length, size=min(length, 100_000), replace=False)
        windows = self._windows(features, sample)
        self.mean = windows.mean(axis=(0, 2))[:, None]
        self.std = windows.std(axis=(0, 2))[:, None] + 1e-6

        self.net = SpotNet(features.shape[0], len(self.label_names), config)
        if config.val_fraction > 0:
            split = int(length * (1.0 - config.val_fraction))
            train_features, train_truth = features[:, :split], truth[:, :split]
            validation = (features[:, split:], truth[:, split:])
            calibrate_features, calibrate_truth = validation
            train_motif = motif_mask[:, :split] if motif_mask is not None else None
        else:
            train_features, train_truth = features, truth
            validation = None
            calibrate_features, calibrate_truth = features, truth
            train_motif = motif_mask
        self._train(train_features, train_truth, rng, validation, train_motif)
        self.thresholds = self._calibrate(calibrate_features, calibrate_truth)
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
        return cls._from_data(torch.load(path, weights_only=False))

    @classmethod
    def _from_data(cls, data: dict) -> SpotModel:
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
