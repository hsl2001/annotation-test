#!/usr/bin/env python3
"""Per-kind PCA of windowed CWT features.

For each motif kind (GS, IS, IE, GE) the script separates annotated sites
(positive) from occurrences of the same motif that are not annotated
(background).  The three chr5 datasets are rendered as a 3 x 4 grid
(rows = datasets, columns = motif kinds); colour encodes positive/background
and marker shape encodes strand (+ circle, - square).

CWT responses are computed with chunked FFT convolution over the sequence,
and only the requested candidate/window positions are kept.  The 2-D view is
the first two PCA components.
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
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
PREFIX = {"+": "p", "-": "m"}
LABELS = (
    "mGS", "pGS", "mIS", "pIS", "mIEm2", "pIEm2",
    "mGEm3TAA", "mGEm3TAG", "mGEm3TGA",
    "pGEm3TAA", "pGEm3TAG", "pGEm3TGA",
)
STOPS = {(3, 0, 0): "TAA", (3, 0, 2): "TAG", (3, 2, 0): "TGA"}
STOP_CODES = np.array([a * 16 + b * 4 + c for (a, b, c) in STOPS])
SITE_OFFSET = {"GS": 0, "IS": 0, "IE": 2, "GE": 3}
MOTIF_OFFSET = {"GS": 0, "IS": 0, "IE": -2, "GE": -3}
MOTIF_WIDTH = {"GS": 3, "IS": 2, "IE": 2, "GE": 3}
GE_LABELS = tuple(label for label in LABELS if "GEm3" in label)
KIND_INDEX = {kind: index for index, kind in enumerate(KINDS)}

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

DATASETS = tuple(
    (name, f"data/truth.{name}.chr5.fasta", f"data/truth.{name}.chr5.gff3")
    for name in ("nip", "col", "slm")
)
POSITIVE_COLOR = "#d62728"
BACKGROUND_COLOR = "#1f77b4"

Panel = tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int8]]


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


def build_kernel(dilation: float, w0: float = W0) -> NDArray[np.complex128]:
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


@dataclass
class Transcript:
    seqid: str
    strand: str
    cds: list[tuple[int, int]]


def read_annotation(path: str) -> list[Transcript]:
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
        Transcript(entry["seqid"], entry["strand"], sorted(entry["cds"]))
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


def site_labels(
    transcripts: list[Transcript], lengths: dict[str, int]
) -> dict[tuple[str, str, str], NDArray[np.int64]]:
    points: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for transcript in transcripts:
        length = lengths.get(transcript.seqid, 0)
        if length == 0 or not transcript.cds:
            continue
        cds = sorted(
            oriented_span(span, length, transcript.strand) for span in transcript.cds
        )
        key = (transcript.seqid, transcript.strand)
        points[key + ("GS",)].add(cds[0][0])
        points[key + ("GE",)].add(cds[-1][1])
        for (_, left), (right, _) in pairwise(cds):
            points[key + ("IS",)].add(left)
            points[key + ("IE",)].add(right)
    return {
        key: np.array(
            [v for v in sorted(values) if 0 <= v < lengths[key[0]]], dtype=np.int64
        )
        for key, values in points.items()
    }


def learn_motifs(
    codes: dict[str, NDArray[np.uint8]],
    labels: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[str, set[bytes]]:
    motifs: dict[str, set[bytes]] = {
        "GS": {bytes((0, 3, 2))},
        "IS": {bytes((2, 3)), bytes((2, 1))},
        "IE": {bytes((0, 2))},
        "GE": {bytes((3, 0, 0)), bytes((3, 0, 2)), bytes((3, 2, 0))},
    }
    for (name, strand, site), positions in labels.items():
        if site not in motifs:
            continue
        sequence = oriented(codes[name], strand)
        offset, width = MOTIF_OFFSET[site], MOTIF_WIDTH[site]
        for position in positions:
            kmer = sequence[position + offset : position + offset + width]
            if len(kmer) and np.all(kmer < 4):
                motifs[site].add(bytes(kmer))
    return motifs


@njit(cache=True)
def _motif_mask(
    sequence: NDArray[np.uint8],
    motifs: NDArray[np.uint8],
    lengths: NDArray[np.int64],
    forward: bool,
) -> NDArray[np.bool_]:
    n = sequence.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for row in range(motifs.shape[0]):
        k = lengths[row]
        if k > n:
            continue
        limit = n - k + 1 if forward else n - k
        for start in range(limit):
            hit = True
            for offset in range(k):
                if sequence[start + offset] != motifs[row, offset]:
                    hit = False
                    break
            if hit:
                mask[start if forward else start + k] = True
    return mask


def motif_mask(
    sequence: NDArray[np.uint8], site: str, motifs: dict[str, set[bytes]]
) -> NDArray[np.bool_]:
    if site not in motifs:
        return np.ones(len(sequence), dtype=bool)
    found = list(motifs[site])
    table = np.zeros((len(found), max(len(motif) for motif in found)), dtype=np.uint8)
    lengths = np.zeros(len(found), dtype=np.int64)
    for row, motif in enumerate(found):
        table[row, : len(motif)] = np.frombuffer(motif, dtype=np.uint8)
        lengths[row] = len(motif)
    return _motif_mask(sequence, table, lengths, site in ("GS", "IS"))


def build_labels(
    codes: dict[str, NDArray[np.uint8]],
    labels: dict[tuple[str, str, str], NDArray[np.int64]],
    name: str,
) -> dict[str, NDArray[np.int64]]:
    length = len(codes[name])
    result: dict[str, list[int]] = {label: [] for label in LABELS}
    for (seqid, strand, site), positions in labels.items():
        if seqid != name or site not in KINDS:
            continue
        prefix = PREFIX[strand]
        genomic = to_genomic(positions, length, strand) + (
            -1 if strand == "+" else 1
        ) * SITE_OFFSET[site]
        if site == "GE":
            sequence = oriented(codes[name], strand)
            for position, target in zip(positions, genomic):
                kmer = sequence[int(position) - 3 : int(position)]
                codon = (
                    STOPS.get((int(kmer[0]), int(kmer[1]), int(kmer[2])), "other")
                    if len(kmer) == 3
                    else "other"
                )
                label = f"{prefix}GEm3{codon}"
                if label in result:
                    result[label].append(int(target))
        elif site == "IE":
            result[f"{prefix}IEm2"].extend(int(x) for x in genomic)
        else:
            result[f"{prefix}{site}"].extend(int(x) for x in genomic)
    return {
        label: np.array(sorted(set(values)), dtype=np.int64)
        for label, values in result.items()
    }


def _shift_mask(mask: NDArray[np.bool_], offset: int) -> NDArray[np.bool_]:
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    if offset < 0:
        out[: n + offset] = mask[-offset:]
    elif offset > 0:
        out[offset:] = mask[: n - offset]
    else:
        out = mask.copy()
    return out


@dataclass
class Candidates:
    positions: NDArray[np.int64]  # genomic, 0-based
    strands: NDArray[np.int8]  # 0 = "+", 1 = "-"
    kinds: NDArray[np.int8]  # index into KINDS
    labels: NDArray[np.int8]  # 1 = annotated site, 0 = background

    def __len__(self) -> int:
        return len(self.positions)


def _mask(values: NDArray[np.int64] | None, length: int) -> NDArray[np.bool_]:
    mask = np.zeros(length, dtype=bool)
    if values is not None:
        mask[values[(values >= 0) & (values < length)]] = True
    return mask


def build_candidates(
    codes: dict[str, NDArray[np.uint8]],
    motifs: dict[str, set[bytes]],
    labels: dict[str, NDArray[np.int64]],
    name: str,
) -> Candidates:
    """Collect every motif-bearing position, labelled by motif kind."""
    sequence = codes[name]
    length = len(sequence)
    annotated = {label: _mask(labels.get(label), length) for label in LABELS}

    positions: list[NDArray[np.int64]] = []
    strands: list[NDArray[np.int8]] = []
    kinds: list[NDArray[np.int8]] = []
    targets: list[NDArray[np.int8]] = []

    for strand_index, strand in enumerate(STRANDS):
        prefix = PREFIX[strand]
        oriented_sequence = oriented(sequence, strand)
        motif_masks = {
            kind: motif_mask(oriented_sequence, kind, motifs) for kind in KINDS
        }
        motif_masks["IE"] = _shift_mask(motif_masks["IE"], -2)
        motif_masks["GE"] = _shift_mask(motif_masks["GE"], -3)
        kind_masks = {
            "GS": annotated[f"{prefix}GS"],
            "IS": annotated[f"{prefix}IS"],
            "IE": annotated[f"{prefix}IEm2"],
            "GE": np.any(
                [annotated[label] for label in GE_LABELS if label.startswith(prefix)],
                axis=0,
            ),
        }

        for kind in KINDS:
            found = np.flatnonzero(motif_masks[kind])
            if kind == "GE":
                found = found[found < length - 2]
                if not len(found):
                    continue
                codons = (
                    oriented_sequence[found].astype(np.int64) * 16
                    + oriented_sequence[found + 1].astype(np.int64) * 4
                    + oriented_sequence[found + 2].astype(np.int64)
                )
                found = found[np.isin(codons, STOP_CODES)]
            if not len(found):
                continue

            genomic = to_genomic(found, length, strand)
            positions.append(genomic)
            strands.append(np.full(len(found), strand_index, dtype=np.int8))
            kinds.append(np.full(len(found), KIND_INDEX[kind], dtype=np.int8))
            targets.append(kind_masks[kind][genomic].astype(np.int8))

    if not positions:
        empty = np.zeros(0, dtype=np.int64)
        return Candidates(
            empty, empty.astype(np.int8), empty.astype(np.int8), empty.astype(np.int8)
        )

    order = np.lexsort((np.concatenate(strands), np.concatenate(positions)))
    return Candidates(
        np.concatenate(positions)[order],
        np.concatenate(strands)[order],
        np.concatenate(kinds)[order],
        np.concatenate(targets)[order],
    )


class CwtFeatureBank:
    """Chunked FFT convolution of the sequence with Morlet wavelets.

    Each mapping contributes magnitude (``re^2 + im^2``), real and imaginary
    parts.  Several ``w0`` values add wavelets with different centre
    frequencies.  The sequence is processed in chunks; only the requested
    positions inside each chunk are kept, so the full response map is never
    materialised.
    """

    def __init__(
        self,
        sequence: NDArray[np.uint8],
        periods: tuple[int, ...],
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
    candidates: Candidates,
    max_positives: int,
    negatives_per_positive: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Balanced sample: cap positives per kind and match them with background."""
    selected: list[NDArray[np.int64]] = []
    for kind_index, kind in enumerate(KINDS):
        positive = np.flatnonzero(
            (candidates.kinds == kind_index) & (candidates.labels == 1)
        )
        negative = np.flatnonzero(
            (candidates.kinds == kind_index) & (candidates.labels == 0)
        )
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


def load_dataset(fasta: str, annotation: str):
    codes = read_fasta(fasta)
    lengths = {name: len(sequence) for name, sequence in codes.items()}
    raw = site_labels(read_annotation(annotation), lengths)
    name = next(iter(codes))
    return codes, name, learn_motifs(codes, raw), build_labels(codes, raw, name)


def run(args: argparse.Namespace) -> int:
    panels: dict[str, dict[str, Panel]] = {}
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


def _dataset_panels(
    args: argparse.Namespace, fasta: str, annotation: str
) -> dict[str, Panel]:
    codes, name, motifs, labels = load_dataset(fasta, annotation)
    candidates = build_candidates(codes, motifs, labels, name)
    rng = np.random.default_rng(args.seed)
    index = sample_candidates(
        candidates, args.max_positives, args.negatives_per_positive, rng
    )
    bank = CwtFeatureBank(codes[name], tuple(args.periods), tuple(args.w0), args.chunk)
    offsets = np.arange(-args.window_radius, args.window_radius + 1)
    positions = (candidates.positions[index][:, None] + offsets[None, :]).ravel()
    strands = np.repeat(candidates.strands[index], len(offsets))
    matrix = bank.gather(positions, strands).reshape(len(index), -1)
    print(f"  features: {matrix.shape} ({matrix.nbytes / 1e9:.2f} GB)")
    truth = candidates.labels[index]
    kinds = candidates.kinds[index]
    strands = candidates.strands[index]

    panels: dict[str, Panel] = {}
    for kind_index, kind in enumerate(KINDS):
        selected = kinds == kind_index
        if selected.sum() < 20:
            continue
        embedding = PCA(
            n_components=2,
            whiten=True,
            random_state=args.seed,
            svd_solver="randomized",
        ).fit_transform(StandardScaler().fit_transform(matrix[selected]))
        panels[kind] = (embedding, truth[selected], strands[selected])
    return panels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cwt_pca.png")
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 3, 4, 6, 10])
    parser.add_argument("--w0", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    parser.add_argument("--max-positives", type=int, default=1000)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--window-radius", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
