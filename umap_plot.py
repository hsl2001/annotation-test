#!/usr/bin/env python3
import argparse
import gzip
import math
from collections import defaultdict
from itertools import pairwise, permutations
from pathlib import Path
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from numba import njit
from scipy.fft import fft, ifft, next_fast_len

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
SEED = 0
MAX_POSITIVES = 2000
PERIODS = (2, 3, 4)
RADIUS = 5
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

TABLES = {}
for _kind, _motifs in MOTIFS.items():
    _table = np.zeros((len(_motifs), max(map(len, _motifs))), dtype=np.uint8)
    _lengths = np.zeros(len(_motifs), dtype=np.int64)
    for _row, _motif in enumerate(_motifs):
        _table[_row, : len(_motif)] = LUT[np.frombuffer(_motif, dtype=np.uint8)]
        _lengths[_row] = len(_motif)
    TABLES[_kind] = (_table, _lengths)


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






def cwt_features(bank, positions, strands):
    offsets = np.arange(-RADIUS, RADIUS + 1)
    return bank.gather(
        (positions[:, None] + offsets[None, :]).ravel(),
        np.repeat(strands, len(offsets)),
    ).reshape(len(positions), -1)


def plot(fasta, truth, output):
    codes = read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    masks = positive_masks(read_annotation(truth), name, len(sequence))
    positions, strands, kinds, labels = build_candidates(sequence, masks)
    bank = CwtFeatureBank(sequence, PERIODS, W0)
    index = sample_candidates(kinds, labels, np.random.default_rng(SEED))
    features = cwt_features(bank, positions[index], strands[index])
    labels = labels[index]
    kinds = kinds[index]
    columns = 2
    n_rows = (len(KINDS) + columns - 1) // columns
    figure, axes = plt.subplots(n_rows, columns, figsize=(6 * columns, 6 * n_rows))
    for ax, kind in zip(axes.ravel(), KINDS):
        mask = kinds == KINDS.index(kind)
        if not mask.any():
            ax.axis("off")
            continue
        n = len(features[mask])
        embedding = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, n - 1)),
            random_state=SEED,
            n_jobs=1,
            compatibility_layout=True,
        ).fit_transform(features[mask])
        for value, color, marker, label in (
            (0, "#1f77b4", "o", "negative"),
            (1, "#d62728", "^", "positive"),
        ):
            selected = labels[mask] == value
            if selected.any():
                ax.scatter(
                    embedding[selected, 0],
                    embedding[selected, 1],
                    c=color,
                    marker=marker,
                    s=8,
                    alpha=0.7,
                )
        ax.set_title(kind)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(KINDS) :]:
        ax.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="negative"),
        Line2D([], [], marker="^", linestyle="", color="#d62728", label="positive"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.fasta, args.truth, str(Path(args.output).with_suffix(".png")))


if __name__ == "__main__":
    main()
