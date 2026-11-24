import gzip
import math
from itertools import permutations

import numpy as np
from numba import njit
from scipy.fft import fft, ifft, next_fast_len

PERIODS = (2, 3)
MORLET_LIMIT = 8.0
W0 = 6.0

NODES = np.array(
    [
        0.09501250983763744,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732324,
        0.9894009349916499,
    ]
)
WEIGHTS = np.array(
    [
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    ]
)
LUT = np.full(256, 4, np.uint8)
for _code, _value in zip(b"ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)):
    LUT[_code] = _value
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
ROOTS = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
MAPPINGS = np.zeros((5, 6), dtype=np.complex128)
for _column, _permutation in enumerate(permutations((1, 2, 3))):
    MAPPINGS[0, _column] = ROOTS[0]
    MAPPINGS[1, _column] = ROOTS[_permutation[0]]
    MAPPINGS[2, _column] = ROOTS[_permutation[1]]
    MAPPINGS[3, _column] = ROOTS[_permutation[2]]


_PHI = np.array([0, 4, 2, 3, 1, 5])


def magnitude_summary(features):
    w = features.reshape(features.shape[0], 12)
    mag = np.hypot(w[:, :6], w[:, 6:])
    return np.minimum(mag, mag[:, _PHI]).sum(axis=1)


def morlet_integral(lower, upper, w0=W0):
    lower, upper = max(lower, -MORLET_LIMIT), min(upper, MORLET_LIMIT)
    if lower >= upper:
        return 0.0
    squared = w0 * w0
    norm_squared = -math.expm1(-squared) + 2.0 * math.exp(
        -0.75 * squared
    ) * math.expm1(-0.25 * squared)
    norm = 1.0 / math.sqrt(math.sqrt(math.pi) * norm_squared)
    correction = math.exp(-0.5 * squared)
    segments = math.ceil((upper - lower) * 2.0)
    step = (upper - lower) / segments
    integral = 0.0
    for segment in range(segments):
        middle = lower + (segment + 0.5) * step
        left = middle - NODES * step / 2.0
        right = middle + NODES * step / 2.0
        integral += (
            WEIGHTS
            * (step / 2.0)
            * (
                np.exp(-0.5 * left**2) * (np.exp(-1j * w0 * left) - correction)
                + np.exp(-0.5 * right**2) * (np.exp(-1j * w0 * right) - correction)
            )
        ).sum()
    return norm * integral


def build_kernel(period, w0=W0):
    dilation = period * w0 / (2 * math.pi)
    radius = math.ceil(MORLET_LIMIT * dilation + 0.5)
    kernel = np.zeros(2 * radius + 1, dtype=np.complex128)
    for tap in range(len(kernel)):
        displacement = tap - radius
        kernel[tap] = math.sqrt(dilation) * morlet_integral(
            (displacement - 0.5) / dilation, (displacement + 0.5) / dilation, w0
        )
    return kernel


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_fasta(path):
    records, name, chunks = {}, None, []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    records[name] = LUT[np.frombuffer(b"".join(chunks), np.uint8)]
                name, chunks = line[1:].split()[0], []
            elif name is not None:
                chunks.append(line.strip().encode("ascii", errors="replace"))
    if name is not None:
        records[name] = LUT[np.frombuffer(b"".join(chunks), np.uint8)]
    return records


@njit
def _motif_mask(sequence, motifs, lengths):
    n = sequence.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for row in range(motifs.shape[0]):
        k = lengths[row]
        for start in range(n - k + 1):
            hit = True
            for offset in range(k):
                if sequence[start + offset] != motifs[row, offset]:
                    hit = False
                    break
            if hit:
                mask[start] = True
    return mask


class CwtFeatureBank:
    def __init__(self, sequence, periods, w0, chunk=100_000):
        self.length = len(sequence)
        self.codes = sequence
        self.reverse_codes = COMPLEMENT[sequence[::-1]]
        self.chunk = chunk
        kernels = [
            (len(kernel) // 2, kernel)
            for period in periods
            for kernel in [build_kernel(period, w0)]
        ]
        self.max_radius = max(radius for radius, _ in kernels)
        self.channels = 12 * len(kernels)
        self.n_fft = next_fast_len(chunk + 4 * self.max_radius)
        self.kernel_ffts = [
            (radius, np.asarray(fft(kernel, n=self.n_fft), dtype=np.complex128))
            for radius, kernel in kernels
        ]

    def _gather_sequence(self, codes, positions):
        features = np.empty((len(positions), self.channels), dtype=np.float32)
        if not len(positions):
            return features
        order = np.argsort(positions, kind="stable")
        sorted_positions = positions[order]
        length, chunk, max_radius, n_fft = (
            len(codes),
            self.chunk,
            self.max_radius,
            self.n_fft,
        )
        for start in range(int(sorted_positions[0]), int(sorted_positions[-1]) + 1, chunk):
            end = start + chunk
            lo = np.searchsorted(sorted_positions, start, side="left")
            hi = np.searchsorted(sorted_positions, end, side="left")
            if lo == hi:
                continue
            chunk_positions, rows = sorted_positions[lo:hi], order[lo:hi]
            seg_start = start - max_radius
            segment = np.full(end - start + 2 * max_radius, 4, dtype=np.uint8)
            valid_start, valid_end = max(seg_start, 0), min(end + max_radius, length)
            if valid_start < valid_end:
                segment[valid_start - seg_start : valid_end - seg_start] = codes[
                    valid_start:valid_end
                ]
            signal_fft = np.asarray(
                fft(MAPPINGS[segment], n=n_fft, axis=0, workers=-1), dtype=np.complex128
            )
            for kernel_index, (radius, kernel_fft) in enumerate(self.kernel_ffts):
                conv = np.asarray(
                    ifft(signal_fft * kernel_fft[:, None], axis=0, workers=-1),
                    dtype=np.complex128,
                )
                responses = conv[(chunk_positions - seg_start) + radius]
                features[rows, kernel_index * 12 : (kernel_index + 1) * 12] = np.hstack(
                    (responses.real, responses.imag)
                ).astype(np.float32)
        return features

    def gather(self, positions, strands):
        features = np.empty((len(positions), self.channels), dtype=np.float32)
        for value, codes in ((0, self.codes), (1, self.reverse_codes)):
            mask = strands == value
            if mask.any():
                oriented_positions = (
                    positions[mask] if value == 0 else self.length - 1 - positions[mask]
                )
                features[mask] = self._gather_sequence(codes, oriented_positions)
        return features


def cwt_features(bank, positions, strands):
    return bank.gather(positions, strands)
