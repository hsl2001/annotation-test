from __future__ import annotations

import numpy as np
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


def background_mean(channels: NDArray[np.float32], radius: int) -> NDArray[np.float64]:
    width = 2 * radius + 1
    n = channels.shape[1]
    prefix = np.concatenate(
        [np.zeros((channels.shape[0], 1)), np.cumsum(channels, axis=1)], axis=1
    )
    mean = np.zeros((channels.shape[0], width), dtype=np.float64)
    for k in range(width):
        d = k - radius
        low, high = max(0, d), min(n, n + d)
        mean[:, k] = (prefix[:, high] - prefix[:, low]) / max(n, 1)
    return mean.ravel()


def accumulate_gram(
    gram: NDArray[np.float64],
    spectra: list[NDArray[np.complex128]],
    fft_size: int,
    width: int,
) -> None:
    channels = len(spectra)
    max_lag = width - 1
    for c1 in range(channels):
        for c2 in range(c1, channels):
            correlation = np.fft.irfft(
                np.conj(spectra[c1]) * spectra[c2], fft_size
            )
            values = np.concatenate(
                [correlation[fft_size - max_lag :], correlation[: max_lag + 1]]
            )
            for d in range(-max_lag, max_lag + 1):
                value = values[d + max_lag]
                low, high = max(0, -d), min(width, width - d)
                rows = c1 * width + np.arange(low, high)
                columns = c2 * width + np.arange(low, high) + d
                gram[rows, columns] += value
                if c1 != c2:
                    gram[columns, rows] += value


def design_snr_filter(
    gram: NDArray[np.float64],
    positive_windows: NDArray[np.float32],
    background: NDArray[np.float64],
    length: int,
    channels: int,
    width: int,
    regularization: float = 0.01,
) -> NDArray[np.float32]:
    mean_positive = positive_windows.astype(np.float64).mean(axis=0)
    covariance = gram / max(length, 1) - np.outer(background, background)
    dimension = covariance.shape[0]
    shrinkage = np.trace(covariance) / dimension
    weights = np.linalg.solve(
        covariance + regularization * shrinkage * np.eye(dimension),
        mean_positive - background,
    )
    norm = np.linalg.norm(weights)
    return (weights / (norm if norm > 0 else 1.0)).reshape(
        channels, width
    ).astype(np.float32)


def channel_spectra(
    channels: NDArray[np.float32], width: int
) -> tuple[list[NDArray[np.complex128]], int]:
    fft_size = 1 << (channels.shape[1] + width).bit_length()
    return [np.fft.rfft(row, fft_size) for row in channels], fft_size


def convolve(
    spectra: list[NDArray[np.complex128]],
    fft_size: int,
    weights: NDArray[np.float32],
    length: int,
    center: int,
) -> NDArray[np.float32]:
    spectrum = np.zeros(fft_size // 2 + 1, dtype=np.complex128)
    for index, kernel in enumerate(weights):
        spectrum += spectra[index] * np.fft.rfft(kernel[::-1], fft_size)
    return np.fft.irfft(spectrum, fft_size)[center : center + length].astype(
        np.float32
    )


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


class Model:
    def __init__(self, radius: int = 26, regularization: float = 0.01):
        self.radius = radius
        self.width = 2 * radius + 1
        self.regularization = regularization
        self.kernels: NDArray[np.float32] | None = None
        self.thresholds: NDArray[np.float32] | None = None

    def fit(self, X: NDArray[np.float32], Y: NDArray[np.bool_]) -> Model:
        channels, length = X.shape
        sites = Y.shape[0]
        dimension = channels * self.width

        fft_size = 1 << (length + 2 * (self.width - 1)).bit_length()
        spectra = [np.fft.rfft(row, fft_size) for row in X]
        gram = np.zeros((dimension, dimension), dtype=np.float64)
        accumulate_gram(gram, spectra, fft_size, self.width)
        background = background_mean(X, self.radius)

        kernels = np.zeros((sites, channels, self.width), dtype=np.float32)
        thresholds = np.zeros(sites, dtype=np.float32)
        spectra2, fft_size2 = channel_spectra(X, self.width)
        for site in range(sites):
            positives = np.flatnonzero(Y[site]).astype(np.int64)
            if not len(positives):
                continue
            windows = extract_windows(X, positives, self.radius).reshape(
                len(positives), dimension
            )
            kernels[site] = design_snr_filter(
                gram,
                windows,
                background,
                length,
                channels,
                self.width,
                self.regularization,
            )
            response = convolve(
                spectra2, fft_size2, kernels[site], length, self.radius
            )
            positions = np.flatnonzero(local_max(response)).astype(np.int64)
            if len(positions):
                low = np.searchsorted(positives, positions - 2)
                high = np.searchsorted(positives, positions + 2, side="right")
                background_positions = positions[high == low]
                if len(background_positions):
                    thresholds[site] = np.quantile(
                        response[background_positions], 0.90
                    )
        self.kernels = kernels
        self.thresholds = thresholds
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.kernels is None:
            raise RuntimeError("model is not fitted")
        length = X.shape[1]
        spectra, fft_size = channel_spectra(X, self.width)
        out = np.zeros((self.kernels.shape[0], length), dtype=np.float32)
        for site, kernel in enumerate(self.kernels):
            if kernel.any():
                out[site] = convolve(
                    spectra, fft_size, kernel, length, self.radius
                )
        return out
