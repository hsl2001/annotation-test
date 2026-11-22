#!/usr/bin/env python3
import argparse
import gzip
import math
import subprocess
import warnings
from collections import defaultdict
from itertools import pairwise, permutations
from pathlib import Path
from urllib.parse import unquote

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numba import njit
from scipy.fft import fft, ifft, next_fast_len
from sklearn._loss.loss import HalfBinomialLoss
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

KINDS = ("pGS", "mGS", "pIS", "mIS", "pIE", "mIE", "pGE", "mGE")
MOTIFS = {
    "pGS": (b"ATG",),
    "mGS": (b"CAT",),
    "pIS": (b"GT",),
    "mIS": (b"AC",),
    "pIE": (b"AG",),
    "mIE": (b"CT",),
    "pGE": (b"TAA",),
    "mGE": (b"TTA",),
}
SITE_OFFSET = {
    "pGS": 0,
    "mGS": 0,
    "pIS": 0,
    "mIS": 0,
    "pIE": 2,
    "mIE": 2,
    "pGE": 3,
    "mGE": 3,
}
TRAIN = {
    "col": ("data/truth.col.chr5.fasta", "data/truth.col.chr5.gff3"),
    "nip": ("data/truth.nip.chr5.fasta", "data/truth.nip.chr5.gff3"),
    "slm": ("data/truth.slm.chr5.fasta", "data/truth.slm.chr5.gff3"),
}
MODEL = "hints_model.pkl"
HINT_KINDS = ("pGS", "mGS", "pIS", "mIS", "pIE", "mIE")
SPECIES = "arabidopsis"
THRESHOLD = 0.5
F1_FACTOR = 2.0
EPS = np.finfo(float).eps
MAX_POSITIVES = 2000
SEED = 0
PERIODS = (2, 3, 4)
W0S = (8.0,)
RADIUS = 5
MORLET_LIMIT = 8.0
W0 = 6.0
WINDOW = 300
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

TABLES = {}
for _kind, _motifs in MOTIFS.items():
    _table = np.zeros((len(_motifs), max(map(len, _motifs))), dtype=np.uint8)
    _lengths = np.zeros(len(_motifs), dtype=np.int64)
    for _row, _motif in enumerate(_motifs):
        _table[_row, : len(_motif)] = LUT[np.frombuffer(_motif, dtype=np.uint8)]
        _lengths[_row] = len(_motif)
    TABLES[_kind] = (_table, _lengths)

CONFIG = """[SOURCES]
M RM

[SOURCE-PARAMETERS]
T individual_liability

[GENERAL]
      start        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
       stop        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        tss        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        tts        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        ass        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        dss        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
   exonpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
       exon        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
 intronpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
     intron        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
    CDSpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        CDS        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
    UTRpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
        UTR        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
     irpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
nonexonpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1.15
  genicpart        1       1  M    5  0.2 0.4 0.6 0.8 0.25 0.67 1.5 4 9  RM 1 1
"""


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


def read_annotation(path):
    transcripts = defaultdict(lambda: {"seqid": "", "strand": "", "cds": []})
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS":
                continue
            seqid, _, _, a, b, _, strand, _, attributes = fields
            parents = [
                unquote(p)
                for piece in attributes.split(";")
                if piece.startswith("Parent=")
                for p in piece[7:].split(",")
                if p
            ]
            for parent in parents:
                entry = transcripts[parent]
                entry["seqid"], entry["strand"] = seqid, strand
                entry["cds"].append((int(a) - 1, int(b)))
    return [
        (entry["seqid"], entry["strand"], sorted(entry["cds"]))
        for entry in transcripts.values()
        if entry["cds"]
    ]


def oriented_span(span, length, strand):
    a, b = span
    return (length - b, length - a) if strand == "-" else (a, b)


def positive_masks(transcripts, name, length):
    masks = {kind: np.zeros(length, dtype=bool) for kind in KINDS}
    for seqid, strand, spans in transcripts:
        if seqid != name or not spans:
            continue
        cds = sorted(oriented_span(span, length, strand) for span in spans)
        sign = -1 if strand == "+" else 1
        sites = [("GS", cds[0][0]), ("GE", cds[-1][1])]
        for (_, left), (right, _) in pairwise(cds):
            sites += [("IS", left), ("IE", right)]
        for base_kind, position in sites:
            kind = ("p" if strand == "+" else "m") + base_kind
            genomic = (position if strand == "+" else length - 1 - position) + sign * SITE_OFFSET[kind]
            if 0 <= genomic < length:
                masks[kind][genomic] = True
    return masks


@njit(cache=True)
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


def build_candidates(sequence, masks):
    out = [[], [], [], []]
    for kind_index, kind in enumerate(KINDS):
        table, lengths = TABLES[kind]
        found = np.flatnonzero(_motif_mask(sequence, table, lengths))
        if not len(found):
            continue
        if kind.startswith("m"):
            found = found + (max(map(len, MOTIFS[kind])) - 1)
        out[0].append(found.astype(np.int64))
        out[1].append(np.full(len(found), 1 if kind.startswith("m") else 0, dtype=np.int8))
        out[2].append(np.full(len(found), kind_index, dtype=np.int8))
        out[3].append(masks[kind][found].astype(np.int8))
    if not out[0]:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.astype(np.int8), empty.astype(np.int8), empty.astype(np.int8)
    order = np.lexsort((np.concatenate(out[1]), np.concatenate(out[0])))
    return tuple(np.concatenate(x)[order] for x in out)


class CwtFeatureBank:
    def __init__(self, sequence, periods, w0s, chunk=100_000):
        self.length = len(sequence)
        self.codes = sequence
        self.reverse_codes = COMPLEMENT[sequence[::-1]]
        self.chunk = chunk
        kernels = [
            (len(kernel) // 2, kernel)
            for w0 in w0s
            for period in periods
            for kernel in [build_kernel(period, w0)]
        ]
        self.max_radius = max(radius for radius, _ in kernels)
        self.channels = 18 * len(kernels)
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
                features[rows, kernel_index * 18 : (kernel_index + 1) * 18] = np.hstack(
                    (responses.real**2 + responses.imag**2, responses.real, responses.imag)
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


def sample_candidates(kinds, labels, rng):
    selected = []
    for kind_index, kind in enumerate(KINDS):
        positive = np.flatnonzero((kinds == kind_index) & (labels == 1))
        negative = np.flatnonzero((kinds == kind_index) & (labels == 0))
        if not len(positive):
            continue
        n_positive = min(len(positive), MAX_POSITIVES)
        n_negative = min(len(negative), n_positive)
        selected += [
            rng.choice(positive, size=n_positive, replace=False),
            rng.choice(negative, size=n_negative, replace=False),
        ]
    return np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)


@njit(cache=True)
def _stop_arrays(sequence, window):
    length = sequence.shape[0]
    cap = length + window
    next_stop = np.full((3, length + 3), cap, dtype=np.int64)
    prev_stop_end = np.zeros((3, length + 3), dtype=np.int64)
    for frame in range(3):
        for i in range(length - 1, -1, -1):
            if i % 3 == frame:
                is_stop = (
                    i + 2 < length
                    and sequence[i] == 3
                    and sequence[i + 1] == 0
                    and (sequence[i + 2] == 0 or sequence[i + 2] == 2)
                ) or (
                    i + 2 < length
                    and sequence[i] == 3
                    and sequence[i + 1] == 2
                    and sequence[i + 2] == 0
                )
                if is_stop:
                    next_stop[frame, i] = i
                elif i + 3 < length:
                    next_stop[frame, i] = next_stop[frame, i + 3]
            elif i + 1 < length:
                next_stop[frame, i] = next_stop[frame, i + 1]
        for i in range(length):
            if i % 3 == frame:
                is_stop = (
                    i + 2 < length
                    and sequence[i] == 3
                    and sequence[i + 1] == 0
                    and (sequence[i + 2] == 0 or sequence[i + 2] == 2)
                ) or (
                    i + 2 < length
                    and sequence[i] == 3
                    and sequence[i + 1] == 2
                    and sequence[i + 2] == 0
                )
                if is_stop:
                    prev_stop_end[frame, i] = i + 3
                elif i - 3 >= 0:
                    prev_stop_end[frame, i] = prev_stop_end[frame, i - 3]
            elif i - 1 >= 0:
                prev_stop_end[frame, i] = prev_stop_end[frame, i - 1]
    return next_stop, prev_stop_end


def _gc(sequence, positions, width):
    length = len(sequence)
    out = np.zeros(len(positions), dtype=np.float32)
    for i, p in enumerate(positions):
        lo = max(0, p - width // 2)
        hi = min(length, p + width // 2)
        if lo < hi:
            window = sequence[lo:hi]
            valid = window < 4
            if valid.any():
                out[i] = np.isin(window[valid], (1, 2)).mean()
    return out


def _polypyrimidine(sequence, positions, width=20):
    out = np.zeros(len(positions), dtype=np.float32)
    for i, p in enumerate(positions):
        lo = max(0, p - width)
        hi = max(0, p - 4)
        if lo < hi:
            window = sequence[lo:hi]
            valid = window < 4
            if valid.any():
                out[i] = np.isin(window[valid], (1, 3)).mean()
    return out


def coding_features(sequence, positions, strands, window=WINDOW):
    length = len(sequence)
    reverse = COMPLEMENT[sequence[::-1]]
    cap = window / 3.0
    out = np.zeros((len(positions), 14), dtype=np.float32)
    for strand_value, seq in ((0, sequence), (1, reverse)):
        mask = strands == strand_value
        if not mask.any():
            continue
        pos = positions[mask] if strand_value == 0 else length - 1 - positions[mask]
        next_stop, prev_stop_end = _stop_arrays(seq, window)
        up = np.zeros((len(pos), 3), dtype=np.float32)
        down = np.zeros((len(pos), 3), dtype=np.float32)
        up3 = np.zeros((len(pos), 3), dtype=np.float32)
        down3 = np.zeros((len(pos), 3), dtype=np.float32)
        for frame in range(3):
            up[:, frame] = np.minimum(
                np.maximum(pos - prev_stop_end[frame, pos], 0), window
            ) / 3.0
            down[:, frame] = np.minimum(
                np.maximum(next_stop[frame, pos] - pos, 0), window
            ) / 3.0
            shifted_up = np.maximum(pos - 3, 0)
            shifted_down = np.minimum(pos + 3, length - 1)
            up3[:, frame] = np.minimum(
                np.maximum(shifted_up - prev_stop_end[frame, shifted_up], 0), window
            ) / 3.0
            down3[:, frame] = np.minimum(
                np.maximum(next_stop[frame, shifted_down] - shifted_down, 0), window
            ) / 3.0
        rows = np.flatnonzero(mask)
        out[rows, 0] = up.max(axis=1)
        out[rows, 1] = up.min(axis=1)
        out[rows, 2] = up.mean(axis=1)
        out[rows, 3] = down.max(axis=1)
        out[rows, 4] = down.min(axis=1)
        out[rows, 5] = down.mean(axis=1)
        out[rows, 6] = up3.max(axis=1)
        out[rows, 7] = down3.max(axis=1)
        out[rows, 8] = _gc(seq, pos, 100)
        out[rows, 9] = _gc(seq, pos, 20)
        out[rows, 10] = _polypyrimidine(seq, pos)
        out[rows, 11] = out[rows, 0] - out[rows, 3]
        out[rows, 12] = out[rows, 3] - out[rows, 0]
        out[rows, 13] = np.minimum(out[rows, 0], cap) - np.minimum(out[rows, 3], cap)
    return out


class SoftF1Closs:
    def _parts(self, y_true, raw_prediction, sample_weight):
        p = 1.0 / (1.0 + np.exp(-raw_prediction))
        w = np.ones_like(p) if sample_weight is None else sample_weight
        tp = np.sum(w * y_true * p)
        fp = np.sum(w * (1.0 - y_true) * p)
        fn = np.sum(w * y_true * (1.0 - p))
        return p, w, tp, 2.0 * tp + fp + fn

    def loss(self, y_true, raw_prediction, sample_weight=None, loss_out=None, n_threads=1):
        p, _, tp, d = self._parts(y_true, raw_prediction, sample_weight)
        cce = -(y_true * np.log(p + EPS) + (1.0 - y_true) * np.log(1.0 - p + EPS))
        value = cce + F1_FACTOR * (1.0 - 2.0 * tp / (d + EPS))
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        loss_out[:] = value
        return loss_out

    def gradient(self, y_true, raw_prediction, sample_weight=None, gradient_out=None, n_threads=1):
        p, w, tp, d = self._parts(y_true, raw_prediction, sample_weight)
        grad = (p - y_true) + F1_FACTOR * p * (1.0 - p) * (
            -w * (2.0 * y_true * d - 2.0 * tp) / (d**2 + EPS)
        )
        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)
        gradient_out[:] = grad
        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, gradient_out=None, hessian_out=None, n_threads=1):
        p, w, tp, d = self._parts(y_true, raw_prediction, sample_weight)
        grad = (p - y_true) + F1_FACTOR * p * (1.0 - p) * (
            -w * (2.0 * y_true * d - 2.0 * tp) / (d**2 + EPS)
        )
        hess = (1.0 + F1_FACTOR) * p * (1.0 - p)
        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)
        if hessian_out is None:
            hessian_out = np.empty_like(raw_prediction)
        gradient_out[:] = grad
        hessian_out[:] = hess
        return gradient_out, hessian_out

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, loss_out=None, gradient_out=None, n_threads=1):
        self.loss(y_true, raw_prediction, sample_weight, loss_out, n_threads)
        self.gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
        return loss_out, gradient_out


class SoftF1Loss(HalfBinomialLoss):
    def __init__(self):
        super().__init__()
        self.closs = SoftF1Closs()


def cwt_features(bank, positions, strands):
    offsets = np.arange(-RADIUS, RADIUS + 1)
    cwt = bank.gather(
        (positions[:, None] + offsets[None, :]).ravel(),
        np.repeat(strands, len(offsets)),
    ).reshape(len(positions), -1)
    return np.hstack((cwt, coding_features(bank.codes, positions, strands)))


def gather(fasta, gff):
    codes = read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    masks = positive_masks(read_annotation(gff), name, len(sequence))
    positions, strands, kinds, labels = build_candidates(sequence, masks)
    index = sample_candidates(kinds, labels, np.random.default_rng(SEED))
    bank = CwtFeatureBank(sequence, PERIODS, W0S)
    return cwt_features(bank, positions[index], strands[index]), labels[index], kinds[index]


def fit(data, model_path=MODEL):
    model = {}
    for kind in HINT_KINDS:
        kind_index = KINDS.index(kind)
        xs, ys = [], []
        for matrix, labels, kinds in data.values():
            rows = kinds == kind_index
            xs.append(matrix[rows])
            ys.append(labels[rows])
        x, y = np.vstack(xs), np.concatenate(ys)
        clf = HistGradientBoostingClassifier(
            loss=SoftF1Loss(), random_state=SEED
        ).fit(x, y)
        model[kind] = clf
        print(f"{kind}: n={len(x)} acc={clf.score(x, y):.3f}")
    joblib.dump(model, model_path)
    print(f"wrote {model_path}")


def train(genomes=("col", "nip", "slm"), model_path=MODEL):
    fit({genome: gather(*TRAIN[genome]) for genome in genomes}, model_path)


def score_sites(bank, positions, strands, kind, model, chunk=50_000):
    proba = np.empty(len(positions), dtype=np.float32)
    for start in range(0, len(positions), chunk):
        end = min(start + chunk, len(positions))
        features = cwt_features(bank, positions[start:end], strands[start:end])
        proba[start:end] = model[kind].predict_proba(features)[:, 1]
    return proba


def write_hint(handle, name, kind, position, strand, prob):
    if kind == "pGS":
        handle.write(
            f"{name}\tAUGUSTUS\tstart\t{position + 1}\t{position + 3}\t{prob:.6f}\t+\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )
    elif kind == "mGS":
        handle.write(
            f"{name}\tAUGUSTUS\tstart\t{position - 1}\t{position + 1}\t{prob:.6f}\t-\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )
    elif kind == "pIS":
        handle.write(
            f"{name}\tAUGUSTUS\tdss\t{position + 1}\t{position + 1}\t{prob:.6f}\t+\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )
    elif kind == "mIS":
        handle.write(
            f"{name}\tAUGUSTUS\tdss\t{position + 1}\t{position + 1}\t{prob:.6f}\t-\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )
    elif kind == "pIE":
        handle.write(
            f"{name}\tAUGUSTUS\tass\t{position + 2}\t{position + 2}\t{prob:.6f}\t+\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )
    elif kind == "mIE":
        handle.write(
            f"{name}\tAUGUSTUS\tass\t{position}\t{position}\t{prob:.6f}\t-\t.\tgrp={kind}_{position};pri=4;source=M\n"
        )


def predict(fasta, output, model_path=MODEL):
    codes = read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    positions, strands, kinds, _ = build_candidates(
        sequence, {kind: np.zeros(len(sequence), dtype=bool) for kind in KINDS}
    )
    bank = CwtFeatureBank(sequence, PERIODS, W0S)
    model = joblib.load(model_path)
    counts = {}
    with open(output, "w") as handle:
        for kind in HINT_KINDS:
            rows = kinds == KINDS.index(kind)
            if not rows.any():
                continue
            pos, st = positions[rows], strands[rows]
            proba = score_sites(bank, pos, st, kind, model)
            keep = proba >= THRESHOLD
            counts[kind] = int(keep.sum())
            for p, s, prob in zip(pos[keep], st[keep], proba[keep]):
                write_hint(handle, name, kind, int(p), int(s), float(prob))
    print("hints:", " ".join(f"{kind}={counts.get(kind, 0)}" for kind in HINT_KINDS))
    print(f"wrote {output}")


def run(fasta, output, model_path=MODEL, species=SPECIES):
    hints = f"{output}.hints.gff"
    config = f"{output}.extrinsic.cfg"
    gff = f"{output}.gff3"
    predict(fasta, hints, model_path)
    with open(config, "w") as handle:
        handle.write(CONFIG)
    with open(gff, "w") as handle:
        subprocess.run(
            [
                "augustus",
                f"--species={species}",
                "--gff3=on",
                f"--hintsfile={hints}",
                f"--extrinsicCfgFile={config}",
                fasta,
            ],
            stdout=handle,
            check=True,
        )
    print(f"wrote {gff}")


def plot(fasta, truth, output, model_path=MODEL):
    codes = read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    masks = positive_masks(read_annotation(truth), name, len(sequence))
    positions, strands, kinds, labels = build_candidates(sequence, masks)
    bank = CwtFeatureBank(sequence, PERIODS, W0S)
    model = joblib.load(model_path)
    figure, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, kind in zip(axes.ravel(), HINT_KINDS):
        rows = kinds == KINDS.index(kind)
        x = cwt_features(bank, positions[rows], strands[rows])
        y = labels[rows]
        pca = PCA(2, random_state=SEED).fit(x)
        z = pca.transform(x)
        surrogate = LogisticRegression().fit(z, y)
        pad = 0.5
        lo, hi = z.min(axis=0) - pad, z.max(axis=0) + pad
        xx, yy = np.meshgrid(
            np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200)
        )
        proba = surrogate.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(
            xx.shape
        )
        ax.contour(xx, yy, proba, levels=[0.5], colors="k", linewidths=1.2)
        for value, color, marker in ((0, "#1f77b4", "o"), (1, "#d62728", "^")):
            mask = y == value
            if mask.any():
                ax.scatter(z[mask, 0], z[mask, 1], c=color, marker=marker, s=10, alpha=0.7)
        proba_model = model[kind].predict_proba(x)[:, 1]
        pred = (proba_model >= THRESHOLD).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba_model)
        except ValueError:
            auc = float("nan")
        ax.set_title(f"{kind} acc={acc:.3f} AUC={auc:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="negative"),
        Line2D([], [], marker="^", linestyle="", color="#d62728", label="positive"),
        Line2D([], [], color="k", linestyle="-", label="2D logistic approx."),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hints", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--truth")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--species", default=SPECIES)
    args = parser.parse_args()
    if args.hints:
        predict(args.fasta, args.output, args.model)
    elif args.run:
        run(args.fasta, args.output, args.model, args.species)
    else:
        parser.error("choose --hints or --run")
    if args.plot:
        if not args.truth:
            parser.error("--plot requires --truth")
        plot(
            args.fasta,
            args.truth,
            str(Path(args.output).with_suffix(".png")),
            args.model,
        )


if __name__ == "__main__":
    main()
