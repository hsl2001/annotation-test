#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import math
from collections import defaultdict
from itertools import pairwise, permutations
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numba import njit
from numpy.typing import NDArray
from scipy.fft import fft, ifft, next_fast_len
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STRANDS = ("+", "-")
KINDS = ("GS", "IS", "IE", "GE")
# Forward-strand motifs. The minus strand is searched on the reverse complement,
# so ATG/CAT, GT/AC, AG/CT and stop/revcomp(stop) are all covered.
MOTIFS = {
    "GS": (b"ATG",),
    "IS": (b"GT",),
    "IE": (b"AG",),
    "GE": (b"TAA", b"TAG", b"TGA"),
}
SITE_OFFSET = {"GS": 0, "IS": 0, "IE": 2, "GE": 3}

MORLET_LIMIT = 8.0
W0 = 6.0
NODES = np.array(
    [0.09501250983763744, 0.2816035507792589, 0.4580167776572274,
     0.6178762444026438, 0.7554044083550030, 0.8656312023878318,
     0.9445750230732324, 0.9894009349916499]
)
WEIGHTS = np.array(
    [0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
     0.1495959888165767, 0.1246289712555339, 0.0951585116824928,
     0.0622535239386479, 0.0271524594117541]
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

DATASETS = tuple(
    (name, f"data/truth.{name}.chr5.fasta", f"data/truth.{name}.chr5.gff3")
    for name in ("nip", "col", "slm")
)
POSITIVE_COLOR = "#d62728"
BACKGROUND_COLOR = "#1f77b4"


def morlet_integral(lower: float, upper: float, w0: float = W0) -> complex:
    lower = max(lower, -MORLET_LIMIT)
    upper = min(upper, MORLET_LIMIT)
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


def build_kernel(period: float, w0: float = W0) -> NDArray[np.complex128]:
    """Morlet kernel whose oscillation period is ``period`` base pairs."""
    dilation = period * w0 / (2 * math.pi)
    radius = math.ceil(MORLET_LIMIT * dilation + 0.5)
    kernel = np.zeros(2 * radius + 1, dtype=np.complex128)
    for tap in range(len(kernel)):
        displacement = tap - radius
        kernel[tap] = math.sqrt(dilation) * morlet_integral(
            (displacement - 0.5) / dilation,
            (displacement + 0.5) / dilation,
            w0,
        )
    return kernel


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_fasta(path: str) -> dict[str, NDArray[np.uint8]]:
    records: dict[str, NDArray[np.uint8]] = {}
    name: str | None = None
    chunks: list[bytes] = []
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


def read_annotation(path: str) -> list[tuple[str, str, list[tuple[int, int]]]]:
    transcripts: dict[str, dict] = defaultdict(lambda: {"cds": []})
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                continue
            seqid, _, kind, a, b, _, strand, _, attributes = fields
            if kind != "CDS":
                continue
            parents = [
                unquote(p)
                for piece in attributes.split(";")
                if piece.startswith("Parent=")
                for p in piece[7:].split(",")
                if p
            ]
            span = (int(a) - 1, int(b))
            for parent in parents:
                entry = transcripts[parent]
                entry["seqid"], entry["strand"] = seqid, strand
                entry["cds"].append(span)
    return [
        (entry["seqid"], entry["strand"], sorted(entry["cds"]))
        for entry in transcripts.values()
        if entry["cds"]
    ]


def oriented_span(span: tuple[int, int], length: int, strand: str) -> tuple[int, int]:
    a, b = span
    return (length - b, length - a) if strand == "-" else (a, b)


def oriented(sequence: NDArray[np.uint8], strand: str) -> NDArray[np.uint8]:
    return sequence if strand == "+" else COMPLEMENT[sequence[::-1]]


def to_genomic(
    positions: NDArray[np.int64], length: int, strand: str
) -> NDArray[np.int64]:
    return positions if strand == "+" else length - 1 - positions


def positive_masks(
    transcripts: list[tuple[str, str, list[tuple[int, int]]]], name: str, length: int
) -> dict[tuple[str, str], NDArray[np.bool_]]:
    masks = {(s, k): np.zeros(length, dtype=bool) for s in STRANDS for k in KINDS}
    for seqid, strand, spans in transcripts:
        if seqid != name or not spans:
            continue
        cds = sorted(oriented_span(span, length, strand) for span in spans)
        sign = -1 if strand == "+" else 1
        sites = [("GS", cds[0][0]), ("GE", cds[-1][1])]
        for (_, left), (right, _) in pairwise(cds):
            sites += [("IS", left), ("IE", right)]
        for kind, position in sites:
            genomic = (position if strand == "+" else length - 1 - position) + sign * SITE_OFFSET[kind]
            if 0 <= genomic < length:
                masks[(strand, kind)][genomic] = True
    return masks


@njit(cache=True)
def _motif_mask(
    sequence: NDArray[np.uint8],
    motifs: NDArray[np.uint8],
    lengths: NDArray[np.int64],
) -> NDArray[np.bool_]:
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


def motif_mask(sequence: NDArray[np.uint8], kind: str) -> NDArray[np.bool_]:
    table, lengths = TABLES[kind]
    return _motif_mask(sequence, table, lengths)


def build_candidates(
    sequence: NDArray[np.uint8],
    masks: dict[tuple[str, str], NDArray[np.bool_]],
) -> tuple[
    NDArray[np.int64], NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]
]:
    """Every motif occurrence, labelled 1 if annotated, else 0."""
    length = len(sequence)
    out: list[list[NDArray]] = [[], [], [], []]
    for strand_index, strand in enumerate(STRANDS):
        oriented_sequence = oriented(sequence, strand)
        for kind_index, kind in enumerate(KINDS):
            found = np.flatnonzero(motif_mask(oriented_sequence, kind))
            if not len(found):
                continue
            genomic = to_genomic(found, length, strand)
            out[0].append(genomic)
            out[1].append(np.full(len(found), strand_index, dtype=np.int8))
            out[2].append(np.full(len(found), kind_index, dtype=np.int8))
            out[3].append(masks[(strand, kind)][genomic].astype(np.int8))
    if not out[0]:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.astype(np.int8), empty.astype(np.int8), empty.astype(np.int8)
    order = np.lexsort((np.concatenate(out[1]), np.concatenate(out[0])))
    return (
        np.concatenate(out[0])[order],
        np.concatenate(out[1])[order],
        np.concatenate(out[2])[order],
        np.concatenate(out[3])[order],
    )


class CwtFeatureBank:
    def __init__(
        self,
        sequence: NDArray[np.uint8],
        periods: tuple[float, ...],
        w0s: tuple[float, ...],
        chunk: int = 100_000,
    ):
        self.length = len(sequence)
        self.codes = sequence
        self.reverse_codes = COMPLEMENT[sequence[::-1]]
        self.chunk = chunk
        kernels = []
        for w0 in w0s:
            for period in periods:
                kernel = build_kernel(period, w0)
                kernels.append((len(kernel) // 2, kernel))
        self.max_radius = max(radius for radius, _ in kernels)
        self.channels = 18 * len(kernels)
        self.n_fft = next_fast_len(chunk + 4 * self.max_radius)
        self.kernel_ffts = [
            (radius, np.asarray(fft(kernel, n=self.n_fft), dtype=np.complex128))
            for radius, kernel in kernels
        ]

    def _gather_sequence(
        self, codes: NDArray[np.uint8], positions: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        features = np.empty((len(positions), self.channels), dtype=np.float32)
        if not len(positions):
            return features

        order = np.argsort(positions, kind="stable")
        sorted_positions = positions[order]
        length, chunk, max_radius, n_fft = (
            len(codes), self.chunk, self.max_radius, self.n_fft
        )

        for start in range(
            int(sorted_positions[0]), int(sorted_positions[-1]) + 1, chunk
        ):
            end = start + chunk
            lo = np.searchsorted(sorted_positions, start, side="left")
            hi = np.searchsorted(sorted_positions, end, side="left")
            if lo == hi:
                continue
            chunk_positions = sorted_positions[lo:hi]
            rows = order[lo:hi]

            seg_start = start - max_radius
            segment = np.full(end - start + 2 * max_radius, 4, dtype=np.uint8)
            valid_start = max(seg_start, 0)
            valid_end = min(end + max_radius, length)
            if valid_start < valid_end:
                segment[valid_start - seg_start : valid_end - seg_start] = codes[
                    valid_start:valid_end
                ]

            signal_fft = np.asarray(
                fft(MAPPINGS[segment], n=n_fft, axis=0, workers=-1),
                dtype=np.complex128,
            )
            for kernel_index, (radius, kernel_fft) in enumerate(self.kernel_ffts):
                conv = np.asarray(
                    ifft(signal_fft * kernel_fft[:, None], axis=0, workers=-1),
                    dtype=np.complex128,
                )
                responses = conv[(chunk_positions - seg_start) + radius]
                features[rows, kernel_index * 18 : (kernel_index + 1) * 18] = np.hstack(
                    (
                        responses.real**2 + responses.imag**2,
                        responses.real,
                        responses.imag,
                    )
                ).astype(np.float32)
        return features

    def gather(
        self, positions: NDArray[np.int64], strands: NDArray[np.int8]
    ) -> NDArray[np.float32]:
        features = np.empty((len(positions), self.channels), dtype=np.float32)
        for value, codes in ((0, self.codes), (1, self.reverse_codes)):
            mask = strands == value
            if mask.any():
                oriented_positions = (
                    positions[mask]
                    if value == 0
                    else self.length - 1 - positions[mask]
                )
                features[mask] = self._gather_sequence(codes, oriented_positions)
        return features


def sample_candidates(
    kinds: NDArray[np.int8],
    labels: NDArray[np.int8],
    max_positives: int,
    negatives_per_positive: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Balanced sample: cap positives per kind and match them with background."""
    selected: list[NDArray[np.int64]] = []
    for kind_index, kind in enumerate(KINDS):
        positive = np.flatnonzero((kinds == kind_index) & (labels == 1))
        negative = np.flatnonzero((kinds == kind_index) & (labels == 0))
        if not len(positive):
            print(f"  {kind}: no positives, skipped")
            continue
        n_positive = min(len(positive), max_positives)
        n_negative = min(len(negative), round(n_positive * negatives_per_positive))
        selected += [
            rng.choice(positive, size=n_positive, replace=False),
            rng.choice(negative, size=n_negative, replace=False),
        ]
        print(
            f"  {kind:>4}: positives={n_positive}/{len(positive)} "
            f"background={n_negative}/{len(negative)}"
        )
    return np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)


def _dataset_panels(
    args: argparse.Namespace, fasta: str, annotation: str
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int8]]]:
    codes = read_fasta(fasta)
    name = next(iter(codes))
    length = len(codes[name])
    masks = positive_masks(read_annotation(annotation), name, length)
    positions, strands, kinds, labels = build_candidates(codes[name], masks)
    rng = np.random.default_rng(args.seed)
    index = sample_candidates(
        kinds, labels, args.max_positives, args.negatives_per_positive, rng
    )
    bank = CwtFeatureBank(codes[name], tuple(args.periods), tuple(args.w0), args.chunk)
    offsets = np.arange(
        -args.window_radius, args.window_radius + 1, args.window_step
    )
    matrix = bank.gather(
        (positions[index][:, None] + offsets[None, :]).ravel(),
        np.repeat(strands[index], len(offsets)),
    ).reshape(len(index), -1)
    print(f"  features: {matrix.shape} ({matrix.nbytes / 1e9:.2f} GB)")

    panels = {}
    for kind_index, kind in enumerate(KINDS):
        selected = kinds[index] == kind_index
        if selected.sum() < 20:
            continue
        embedding = PCA(
            n_components=2,
            whiten=True,
            random_state=args.seed,
            svd_solver="randomized",
        ).fit_transform(StandardScaler().fit_transform(matrix[selected]))
        panels[kind] = (embedding, labels[index][selected], strands[index][selected])
    return panels


def run(args: argparse.Namespace) -> int:
    panels = {}
    for dataset, fasta, annotation in DATASETS:
        print(f"=== {dataset} ===")
        panels[dataset] = _dataset_panels(args, fasta, annotation)

    figure, axes = plt.subplots(len(DATASETS), len(KINDS), figsize=(20, 15))
    for row, (dataset, _, _) in enumerate(DATASETS):
        for col, kind in enumerate(KINDS):
            ax = axes[row, col]
            if kind not in panels[dataset]:
                ax.set_title(f"{dataset} {kind} (no data)")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            embedding, truth, strands = panels[dataset][kind]
            for value, color in ((0, BACKGROUND_COLOR), (1, POSITIVE_COLOR)):
                for strand_value, marker in ((0, "o"), (1, "s")):
                    mask = (truth == value) & (strands == strand_value)
                    if mask.any():
                        ax.scatter(
                            embedding[mask, 0],
                            embedding[mask, 1],
                            s=5,
                            alpha=0.6,
                            c=color,
                            marker=marker,
                            linewidths=0.3,
                        )
            ax.set_title(f"{dataset} {kind} (n={len(truth)})")
            ax.set_xticks([])
            ax.set_yticks([])

    handles = [
        Line2D([], [], marker=marker, linestyle="", color=color, label=label)
        for marker, color, label in (
            ("o", BACKGROUND_COLOR, "background (negative)"),
            ("o", POSITIVE_COLOR, "annotated (positive)"),
            ("o", "gray", "+ strand"),
            ("s", "gray", "- strand"),
        )
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cwt_pca.png")
    parser.add_argument(
        "--periods",
        type=float,
        nargs="+",
        default=[2, 3, 4, 6, 7],
        help="Morlet periods in base pairs",
    )
    parser.add_argument("--w0", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    parser.add_argument("--max-positives", type=int, default=2000)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--window-radius", type=int, default=10)
    parser.add_argument(
        "--window-step",
        type=int,
        default=1,
        help="stride between window offsets in bp",
    )
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
