from __future__ import annotations

import argparse
import gzip
import itertools
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from model import ModelProtocol

STRANDS = ("+", "-")
KINDS = ("GS", "IS", "IE", "GE")
OFFSET = {"GS": 0, "IS": 0, "IE": 2, "GE": 3}
STOPS = {(3, 0, 0): "TAA", (3, 0, 2): "TAG", (3, 2, 0): "TGA"}
PREFIX = {"+": "p", "-": "m"}
LABELS = (
    "mGS", "pGS",
    "mIS", "pIS",
    "mIEm2", "pIEm2",
    "mGEm3TAA", "mGEm3TAG", "mGEm3TGA",
    "pGEm3TAA", "pGEm3TAG", "pGEm3TGA",
)

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
for _column, _permutation in enumerate(itertools.permutations((1, 2, 3))):
    MAPPINGS[0, _column] = ROOTS[0]
    MAPPINGS[1, _column] = ROOTS[_permutation[0]]
    MAPPINGS[2, _column] = ROOTS[_permutation[1]]
    MAPPINGS[3, _column] = ROOTS[_permutation[2]]


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


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


def average_precision(
    scores: NDArray[np.float32], labels: NDArray[np.bool_]
) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    true_positive = np.cumsum(y)
    precision = true_positive / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


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
        for node, weight in zip(NODES, WEIGHTS):
            left = middle - node * step / 2.0
            right = middle + node * step / 2.0
            integral += weight * (step / 2.0) * (
                math.exp(-0.5 * left**2) * (np.exp(-1j * w0 * left) - correction)
                + math.exp(-0.5 * right**2) * (np.exp(-1j * w0 * right) - correction)
            )
    return norm * integral


def build_kernel(dilation: float, w0: float = W0) -> NDArray[np.complex128]:
    radius = math.ceil(MORLET_LIMIT * dilation + 0.5)
    width = 2 * radius + 1
    kernel = np.zeros(width, dtype=np.complex128)
    for tap in range(width):
        displacement = tap - radius
        lower = (displacement - 0.5) / dilation
        upper = (displacement + 0.5) / dilation
        kernel[tap] = math.sqrt(dilation) * morlet_integral(lower, upper, w0)
    return kernel


@njit(cache=True, parallel=True)
def _features(
    codes: NDArray[np.uint8],
    real: NDArray[np.float64],
    imag: NDArray[np.float64],
    radius: int,
) -> NDArray[np.float32]:
    length = codes.shape[0]
    width = 2 * radius + 1
    channels = real.shape[1]
    out = np.empty((channels, length), dtype=np.float32)
    for position in prange(length):
        for channel in range(channels):
            re = 0.0
            im = 0.0
            for tap in range(width):
                source = position + tap - radius
                if 0 <= source < length:
                    base = codes[source]
                    re += real[base, channel, tap]
                    im += imag[base, channel, tap]
            out[channel, position] = re * re + im * im
    return out


class FilterBank:
    def __init__(self, periods: tuple[int, ...] = (2, 3, 4, 6, 10)):
        self.periods = tuple(periods)
        self.kernels = []
        for period in self.periods:
            kernel = build_kernel(period)
            radius = len(kernel) // 2
            table = MAPPINGS[:, :, None] * kernel[::-1][None, None, :]
            self.kernels.append(
                (
                    radius,
                    np.ascontiguousarray(table.real),
                    np.ascontiguousarray(table.imag),
                )
            )

    def transform(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.concatenate(
            [_features(codes, real, imag, radius) for radius, real, imag in self.kernels],
            axis=0,
        )

    def transform_strands(self, codes: NDArray[np.uint8]) -> NDArray[np.float32]:
        forward = self.transform(codes)
        reverse = self.transform(COMPLEMENT[codes[::-1]])
        return np.concatenate([forward, reverse], axis=0)


def one_hot_strands(codes: NDArray[np.uint8]) -> NDArray[np.float32]:
    """One-hot encode both strands so the network can read motifs directly."""
    table = np.zeros((5, 4), dtype=np.float32)
    table[:4] = np.eye(4, dtype=np.float32)
    forward = table[codes].T
    reverse = table[COMPLEMENT[codes[::-1]]].T
    return np.concatenate([forward, reverse], axis=0)


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
                name = line[1:].split()[0]
                chunks = []
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
    exon: list[tuple[int, int]]
    has_utr: bool


def read_annotation(path: str) -> list[Transcript]:
    transcripts: dict[str, dict] = {}
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
            if kind not in ("exon", "CDS", "five_prime_UTR", "three_prime_UTR"):
                continue
            parents: list[str] = []
            for piece in attributes.split(";"):
                if piece.startswith("Parent="):
                    parents = [unquote(p) for p in piece[7:].split(",") if p]
                    break
            if not parents:
                continue
            span = (int(a) - 1, int(b))
            for parent in parents:
                entry = transcripts.setdefault(
                    parent,
                    {
                        "seqid": seqid,
                        "strand": strand,
                        "cds": [],
                        "exon": [],
                        "utr": False,
                    },
                )
                if kind == "CDS":
                    entry["cds"].append(span)
                elif kind == "exon":
                    entry["exon"].append(span)
                else:
                    entry["utr"] = True

    result: list[Transcript] = []
    for entry in transcripts.values():
        cds = sorted(entry["cds"])
        if not cds:
            continue
        result.append(
            Transcript(
                entry["seqid"],
                entry["strand"],
                cds,
                sorted(entry["exon"]),
                entry["utr"],
            )
        )
    return result


def oriented_span(span: tuple[int, int], length: int, strand: str) -> tuple[int, int]:
    a, b = span
    return (length - b, length - a) if strand == "-" else (a, b)


def oriented(sequence: NDArray[np.uint8], strand: str) -> NDArray[np.uint8]:
    return sequence if strand == "+" else COMPLEMENT[sequence[::-1]]


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
        if transcript.has_utr and transcript.exon:
            exons = sorted(
                oriented_span(span, length, transcript.strand)
                for span in transcript.exon
            )
            if exons[0][0] < cds[0][0]:
                points[key + ("U5",)].add(exons[0][0])
            if exons[-1][1] > cds[-1][1]:
                points[key + ("U3",)].add(exons[-1][1])
    return {
        key: np.array(
            [value for value in sorted(values) if 0 <= value < lengths[key[0]]],
            dtype=np.int64,
        )
        for key, values in points.items()
    }


@dataclass(frozen=True)
class Limits:
    min_exon: int
    min_intron: int
    min_coding: int
    max_exon: int
    max_intron: int
    max_span: int


def derive_limits(transcripts: list[Transcript], lengths: dict[str, int]) -> Limits:
    introns: list[int] = []
    spans: list[int] = []
    exons: list[int] = []
    for transcript in transcripts:
        length = lengths.get(transcript.seqid, 0)
        if length == 0 or not transcript.cds:
            continue
        cds = sorted(
            oriented_span(span, length, transcript.strand) for span in transcript.cds
        )
        spans.append(cds[-1][1] - cds[0][0])
        exons.extend(right - left for left, right in cds)
        introns.extend(
            right - left for (_, left), (right, _) in pairwise(cds) if right > left
        )
    if not introns or not spans or not exons:
        raise SystemExit("annotation lacks the intervals needed to derive limits")
    return Limits(
        min_exon=3,
        min_intron=10,
        min_coding=min(min(spans), 30),
        max_exon=int(max(float(np.percentile(exons, 99.9)) * 3, 10_000)),
        max_intron=int(max(float(np.percentile(introns, 99.9)) * 4, 15_000)),
        max_span=int(max(float(np.percentile(spans, 99.9)) * 3, 20_000)),
    )


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
        for position in positions:
            if site == "GS":
                kmer = sequence[position : position + 3]
            elif site == "IS":
                kmer = sequence[position : position + 2]
            elif site == "IE":
                kmer = sequence[position - 2 : position]
            else:
                kmer = sequence[position - 3 : position]
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
    n = len(sequence)
    if site not in motifs:
        return np.ones(n, dtype=bool)
    found = list(motifs[site])
    table = np.zeros((len(found), max(len(motif) for motif in found)), dtype=np.uint8)
    lengths = np.zeros(len(found), dtype=np.int64)
    for row, motif in enumerate(found):
        table[row, : len(motif)] = np.frombuffer(motif, dtype=np.uint8)
        lengths[row] = len(motif)
    return _motif_mask(sequence, table, lengths, site in ("GS", "IS"))


def motif_mask_matrix(
    codes: NDArray[np.uint8], motifs: dict[str, set[bytes]], label_names: list[str]
) -> NDArray[np.bool_]:
    """Motif-bearing positions for every label, in genomic coordinates."""
    length = len(codes)
    mask = np.zeros((len(label_names), length), dtype=bool)
    index = {label: row for row, label in enumerate(label_names)}
    stop_codes = {"TAA": (3, 0, 0), "TAG": (3, 0, 2), "TGA": (3, 2, 0)}
    for strand in STRANDS:
        oriented_seq = oriented(codes, strand)
        valid = {kind: motif_mask(oriented_seq, kind, motifs) for kind in KINDS}
        valid["IE"] = _shift_mask(valid["IE"], -2)
        valid["GE"] = _shift_mask(valid["GE"], -3)
        prefix = PREFIX[strand]
        for kind in KINDS:
            if kind == "GE":
                positions = np.flatnonzero(valid["GE"])
                positions = positions[positions < length - 2]
                for codon, code in stop_codes.items():
                    label = f"{prefix}GEm3{codon}"
                    if label not in index:
                        continue
                    match = (
                        (oriented_seq[positions] == code[0])
                        & (oriented_seq[positions + 1] == code[1])
                        & (oriented_seq[positions + 2] == code[2])
                    )
                    selected = positions[match]
                    if strand == "-":
                        selected = length - 1 - selected
                    mask[index[label], selected] = True
            else:
                label = f"{prefix}{kind}" if kind in ("GS", "IS") else f"{prefix}IEm2"
                if label not in index:
                    continue
                positions = np.flatnonzero(valid[kind])
                if strand == "-":
                    positions = length - 1 - positions
                mask[index[label], positions] = True
    return mask


@dataclass
class Sites:
    positions: NDArray[np.int64]
    scores: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.positions)


def call_sites(
    score: NDArray[np.float32],
    valid: NDArray[np.bool_] | None = None,
) -> Sites:
    if valid is not None:
        score = np.where(valid, score, -np.inf)
    positions = np.flatnonzero(local_max(score) & (score > 0)).astype(np.int64)
    return Sites(positions, score[positions])


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
        if strand == "+":
            genomic = positions.copy()
            if site == "IE":
                genomic = genomic - 2
            elif site == "GE":
                genomic = genomic - 3
        else:
            genomic = length - 1 - positions
            if site == "IE":
                genomic = genomic + 2
            elif site == "GE":
                genomic = genomic + 3
        if site == "GE":
            sequence = oriented(codes[name], strand)
            for position, target in zip(positions, genomic):
                kmer = sequence[int(position) - 3 : int(position)]
                codon = (
                    STOPS.get((int(kmer[0]), int(kmer[1]), int(kmer[2])), "other")
                    if len(kmer) == 3
                    else "other"
                )
                result[f"{prefix}GEm3{codon}"].append(int(target))
        elif site == "IE":
            result[f"{prefix}IEm2"].extend(int(x) for x in genomic)
        else:
            result[f"{prefix}{site}"].extend(int(x) for x in genomic)
    return {
        label: np.array(sorted(set(values)), dtype=np.int64)
        for label, values in result.items()
    }


def _strand_values(
    values: NDArray[np.float32],
    label_names: list[str],
    strand: str,
    combine: Callable[[list[NDArray[np.float32]]], NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Group per-label values into the four site kinds for one strand."""
    prefix = PREFIX[strand]
    result = {
        kind: values[label_names.index(f"{prefix}{kind}")]
        for kind in ("GS", "IS")
    }
    result["IE"] = values[label_names.index(f"{prefix}IEm2")]
    result["GE"] = combine(
        [
            values[label_names.index(f"{prefix}GEm3{codon}")]
            for codon in ("TAA", "TAG", "TGA")
            if f"{prefix}GEm3{codon}" in label_names
        ]
    )
    return result


def strand_scores(
    scores: NDArray[np.float32], label_names: list[str], strand: str
) -> dict[str, NDArray[np.float32]]:
    return _strand_values(
        scores, label_names, strand, lambda ge: np.stack(ge).max(axis=0)
    )


@dataclass
class Candidate:
    seqid: str
    strand: str
    transcript: tuple[int, int]
    cds: list[tuple[int, int]]
    score: float


STOP_CODONS = (
    np.array([3, 0, 0], dtype=np.uint8),
    np.array([3, 0, 2], dtype=np.uint8),
    np.array([3, 2, 0], dtype=np.uint8),
)


@njit(cache=True)
def _is_stop(a: int, b: int, c: int) -> bool:
    return (a == 3 and b == 0 and (c == 0 or c == 2)) or (a == 3 and b == 2 and c == 0)


@njit(cache=True)
def _valid_orf(
    sequence: NDArray[np.uint8],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> bool:
    total = 0
    for index in range(starts.shape[0]):
        total += ends[index] - starts[index]
    if total < 6 or total % 3:
        return False
    coding = np.empty(total, dtype=np.uint8)
    cursor = 0
    for index in range(starts.shape[0]):
        for position in range(starts[index], ends[index]):
            coding[cursor] = sequence[position]
            cursor += 1
    if coding[0] != 0 or coding[1] != 3 or coding[2] != 2:
        return False
    if not _is_stop(coding[total - 3], coding[total - 2], coding[total - 1]):
        return False
    for codon in range(0, total - 3, 3):
        if _is_stop(coding[codon], coding[codon + 1], coding[codon + 2]):
            return False
    return True


def valid_orf(sequence: NDArray[np.uint8], cds: list[tuple[int, int]]) -> bool:
    starts = np.array([left for left, _ in cds], dtype=np.int64)
    ends = np.array([right for _, right in cds], dtype=np.int64)
    return bool(_valid_orf(sequence, starts, ends))


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


class _Paths:
    def __init__(self):
        self.score: list[float] = []
        self.start: list[int] = []
        self.event: list[int] = []
        self.prev: list[int] = []
        self.frame: list[int] = []

    def add(self, score: float, start: int, event: int, previous: int, frame: int) -> int:
        self.score.append(score)
        self.start.append(start)
        self.event.append(event)
        self.prev.append(previous)
        self.frame.append(frame)
        return len(self.score) - 1


def _gather(
    best_paths: list[list[int]],
    kind_indices: dict[str, list[int]],
    kind: str,
    low: int,
    high: int,
) -> list[int]:
    result: list[int] = []
    for event in kind_indices[kind][low:high]:
        result.extend(best_paths[event])
    return result


def _crosses(positions: list[int], low: int, high: int) -> bool:
    """True when a GE event lies in ``[low, high)``."""
    index = bisect_left(positions, low)
    return index < len(positions) and positions[index] < high


def _has_stop(
    stops_by_frame: list[list[int]],
    start: int,
    end: int,
    frame: int,
    skip_last: bool,
    orf_checks: bool,
) -> bool:
    if not orf_checks:
        return False
    first = start + (3 - frame) % 3
    last = end - 3
    if skip_last:
        last -= 3
    if first > last:
        return False
    candidates = stops_by_frame[first % 3]
    index = bisect_left(candidates, first)
    return bool(index < len(candidates) and candidates[index] <= last)


class Decoder:
    def __init__(
        self,
        limits: Limits,
        motifs: dict[str, set[bytes]],
        max_paths: int = 20,
        max_sites: int = 20_000,
        orf_checks: bool = True,
    ):
        self.limits = limits
        self.motifs = motifs
        self.max_paths = max_paths
        self.max_sites = max_sites
        self.orf_checks = orf_checks

    def _sites(
        self,
        sequence: NDArray[np.uint8],
        scores: dict[str, NDArray[np.float32]],
        thresholds: dict[str, float] | None,
    ) -> dict[str, Sites]:
        valid = {kind: motif_mask(sequence, kind, self.motifs) for kind in KINDS}
        valid["IE"] = _shift_mask(valid["IE"], -2)
        valid["GE"] = _shift_mask(valid["GE"], -3)
        sites: dict[str, Sites] = {}
        for kind in KINDS:
            site = call_sites(scores[kind], valid[kind])
            if thresholds is not None and kind in thresholds:
                keep = site.scores >= thresholds[kind]
                site = Sites(site.positions[keep], site.scores[keep])
            if len(site) > self.max_sites:
                order = np.argsort(-site.scores)[: self.max_sites]
                order.sort()
                site = Sites(site.positions[order], site.scores[order])
            sites[kind] = site
        return sites

    def _events(self, sites: dict[str, Sites]) -> list[tuple[int, str, float]]:
        events: list[tuple[int, str, float]] = []
        for kind in KINDS:
            for position, score in zip(sites[kind].positions, sites[kind].scores):
                events.append((int(position) + OFFSET[kind], kind, float(score)))
        events.sort()
        return events

    def _chain(
        self,
        sequence: NDArray[np.uint8],
        events: list[tuple[int, str, float]],
    ) -> tuple[_Paths, list[list[int]]]:
        limits = self.limits
        length = len(sequence)
        stop_at = np.zeros(length, dtype=bool)
        if length >= 3:
            for stop in STOP_CODONS:
                match = np.ones(length - 2, dtype=bool)
                for offset in range(3):
                    match &= sequence[offset : offset + length - 2] == stop[offset]
                stop_at[: length - 2] |= match
        stop_starts = np.flatnonzero(stop_at)
        stops_by_frame = [
            stop_starts[stop_starts % 3 == frame].tolist() for frame in range(3)
        ]

        best_paths: list[list[int]] = [[] for _ in events]
        paths = _Paths()
        kind_indices: dict[str, list[int]] = {kind: [] for kind in KINDS}
        kind_positions: dict[str, list[int]] = {kind: [] for kind in KINDS}
        ge_positions = [position for position, kind, _ in events if kind == "GE"]

        for event_index, (position, kind, score) in enumerate(events):
            created: list[int] = []
            if kind == "GS":
                created.append(paths.add(score, position, event_index, -1, 0))
            elif kind == "IS":
                for predecessor in ("GS", "IE"):
                    positions_of_kind = kind_positions[predecessor]
                    low = bisect_left(positions_of_kind, position - limits.max_exon)
                    high = bisect_right(positions_of_kind, position - limits.min_exon)
                    for path in _gather(best_paths, kind_indices, predecessor, low, high):
                        if predecessor == "GS":
                            start = paths.start[path]
                            base_frame = 0
                        else:
                            start = events[paths.event[path]][0]
                            base_frame = paths.frame[path]
                        if _crosses(ge_positions, start, position):
                            continue
                        if _has_stop(
                            stops_by_frame,
                            start,
                            position,
                            base_frame,
                            False,
                            self.orf_checks,
                        ):
                            continue
                        frame = (base_frame + position - start) % 3
                        created.append(
                            paths.add(
                                paths.score[path] + score,
                                paths.start[path],
                                event_index,
                                path,
                                frame,
                            )
                        )
            elif kind == "IE":
                positions = kind_positions["IS"]
                low = bisect_left(positions, position - limits.max_intron)
                high = bisect_right(positions, position - limits.min_intron)
                for path in _gather(best_paths, kind_indices, "IS", low, high):
                    start = events[paths.event[path]][0]
                    if _crosses(ge_positions, start, position):
                        continue
                    created.append(
                        paths.add(
                            paths.score[path] + score,
                            paths.start[path],
                            event_index,
                            path,
                            paths.frame[path],
                        )
                    )
            elif kind == "GE":
                for predecessor, minimum, maximum in (
                    ("GS", limits.min_coding, limits.max_span),
                    ("IE", limits.min_exon, min(limits.max_exon, limits.max_span)),
                ):
                    positions_of_kind = kind_positions[predecessor]
                    low = bisect_left(positions_of_kind, position - maximum)
                    high = bisect_right(positions_of_kind, position - minimum)
                    for path in _gather(best_paths, kind_indices, predecessor, low, high):
                        span = position - paths.start[path]
                        if span < limits.min_coding or span > limits.max_span:
                            continue
                        if predecessor == "GS":
                            start = paths.start[path]
                            base_frame = 0
                        else:
                            start = events[paths.event[path]][0]
                            base_frame = paths.frame[path]
                        if _crosses(ge_positions, start, position):
                            continue
                        if (base_frame + position - start) % 3:
                            continue
                        if _has_stop(
                            stops_by_frame,
                            start,
                            position,
                            base_frame,
                            True,
                            self.orf_checks,
                        ):
                            continue
                        created.append(
                            paths.add(
                                paths.score[path] + score,
                                paths.start[path],
                                event_index,
                                path,
                                0,
                            )
                        )
            created.sort(key=lambda path: paths.score[path], reverse=True)
            best_paths[event_index] = created[: self.max_paths]
            kind_indices[kind].append(event_index)
            kind_positions[kind].append(position)
        return paths, best_paths

    def _candidates(
        self,
        seqid: str,
        strand: str,
        sequence: NDArray[np.uint8],
        events: list[tuple[int, str, float]],
        paths: _Paths,
        best_paths: list[list[int]],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for event_index, (position, kind, _) in enumerate(events):
            if kind != "GE":
                continue
            for path in best_paths[event_index]:
                chain: list[int] = []
                current = path
                while current >= 0:
                    chain.append(paths.event[current])
                    current = paths.prev[current]
                chain.reverse()
                if len(chain) < 2 or events[chain[0]][1] != "GS":
                    continue
                gs_pos = events[chain[0]][0]
                ge_pos = position
                introns: list[tuple[int, int]] = []
                for index in range(1, len(chain) - 1, 2):
                    introns.append(
                        (events[chain[index]][0], events[chain[index + 1]][0])
                    )
                cds: list[tuple[int, int]] = []
                cursor = gs_pos
                for is_pos, ie_pos in introns:
                    cds.append((cursor, is_pos))
                    cursor = ie_pos
                cds.append((cursor, ge_pos))
                if self.orf_checks and not valid_orf(sequence, cds):
                    continue
                candidates.append(
                    Candidate(
                        seqid=seqid,
                        strand=strand,
                        transcript=(gs_pos, ge_pos),
                        cds=cds,
                        score=float(paths.score[path]),
                    )
                )
        return candidates

    def decode(
        self,
        seqid: str,
        strand: str,
        sequence: NDArray[np.uint8],
        scores: dict[str, NDArray[np.float32]],
        thresholds: dict[str, float] | None = None,
    ) -> list[Candidate]:
        sites = self._sites(sequence, scores, thresholds)
        events = self._events(sites)
        paths, best_paths = self._chain(sequence, events)
        return self._candidates(seqid, strand, sequence, events, paths, best_paths)


def roc_auc(scores: NDArray[np.float32], labels: NDArray[np.bool_]) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1(scores: NDArray[np.float32], labels: NDArray[np.bool_]) -> tuple[float, float]:
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    fn = y.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    index = int(np.argmax(f1))
    return float(f1[index]), float(scores[order][index])


def slice_labels(
    labels: dict[str, NDArray[np.int64]], low: int, high: int
) -> dict[str, NDArray[np.int64]]:
    return {
        label: positions[(positions >= low) & (positions < high)] - low
        for label, positions in labels.items()
    }


def kind_thresholds(model: ModelProtocol, strand: str) -> dict[str, float]:
    assert model.thresholds is not None
    values = _strand_values(
        model.thresholds,
        model.label_names,
        strand,
        lambda ge: np.array(min(ge), dtype=np.float32),
    )
    return {kind: float(value) for kind, value in values.items()}


def decode_strand(
    decoder: Decoder,
    seqid: str,
    strand: str,
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
    label_names: list[str],
    thresholds: dict[str, float] | None = None,
) -> list[Candidate]:
    sequence = oriented(codes, strand)
    per_kind = strand_scores(scores, label_names, strand)
    if strand == "-":
        per_kind = {kind: values[::-1] for kind, values in per_kind.items()}
    return decoder.decode(seqid, strand, sequence, per_kind, thresholds)


def to_genomic(candidates: list[Candidate], length: int) -> list[Candidate]:
    for candidate in candidates:
        if candidate.strand == "-":
            a, b = candidate.transcript
            candidate.transcript = (length - b, length - a)
            candidate.cds = [(length - y, length - x) for x, y in candidate.cds]
    return candidates


def shift_candidates(candidates: list[Candidate], offset: int) -> list[Candidate]:
    for candidate in candidates:
        a, b = candidate.transcript
        candidate.transcript = (a + offset, b + offset)
        candidate.cds = [(x + offset, y + offset) for x, y in candidate.cds]
    return candidates


def _envelope(cds: list[tuple[int, int]]) -> tuple[int, int]:
    return min(left for left, _ in cds), max(right for _, right in cds)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def select_top(
    candidates: list[Candidate], top: int = 3, min_overlap: float = 0.5
) -> list[Candidate]:
    """Deduplicate candidates and keep the best isoforms per locus.

    The decoder emits many overlapping chains for the same gene. Without this
    step every chain is written out, which floods the output with near-duplicate
    false positives and destroys precision.
    """
    if not candidates:
        return []
    unique: dict[tuple, Candidate] = {}
    for candidate in candidates:
        key = (candidate.seqid, candidate.strand, tuple(sorted(candidate.cds)))
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    pool = sorted(unique.values(), key=lambda candidate: -candidate.score)

    envelopes = {id(candidate): _envelope(candidate.cds) for candidate in pool}
    lengths = {
        id(candidate): sum(b - a for a, b in candidate.cds) for candidate in pool
    }

    by_loc: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in pool:
        by_loc[(candidate.seqid, candidate.strand)].append(candidate)
    starts: dict[tuple[str, str], list[int]] = {}
    max_len: dict[tuple[str, str], int] = {}
    for loc, group in by_loc.items():
        group.sort(key=lambda candidate: envelopes[id(candidate)][0])
        starts[loc] = [envelopes[id(candidate)][0] for candidate in group]
        max_len[loc] = max(
            envelopes[id(candidate)][1] - envelopes[id(candidate)][0]
            for candidate in group
        )

    selected: list[Candidate] = []
    assigned: set[int] = set()
    for primary in pool:
        if id(primary) in assigned:
            continue
        reference = envelopes[id(primary)]
        loc = (primary.seqid, primary.strand)
        group_all = by_loc[loc]
        starts_all = starts[loc]
        low = bisect_left(starts_all, reference[0] - max_len[loc])
        high = bisect_right(starts_all, reference[1])
        group = [
            candidate
            for candidate in group_all[low:high]
            if id(candidate) not in assigned
            and _overlap(envelopes[id(candidate)], reference)
            / max(reference[1] - reference[0], 1)
            >= min_overlap
        ]
        for candidate in group:
            assigned.add(id(candidate))
        if not group:
            continue
        longest = max(group, key=lambda candidate: lengths[id(candidate)])
        longest_span = envelopes[id(longest)]
        isoforms = [
            candidate
            for candidate in group
            if _overlap(envelopes[id(candidate)], longest_span)
            / max(longest_span[1] - longest_span[0], 1)
            >= min_overlap
        ]
        isoforms.sort(
            key=lambda candidate: (-lengths[id(candidate)], -candidate.score)
        )
        selected.extend(isoforms[:top])
    return selected


def write_gff3(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w") as handle:
        handle.write("##gff-version 3\n")
        for index, candidate in enumerate(candidates):
            tid = f"tx{index + 1}"
            a, b = candidate.transcript
            handle.write(
                f"{candidate.seqid}\tspot\tmRNA\t{a + 1}\t{b}\t{candidate.score:.4f}\t"
                f"{candidate.strand}\t.\tID={tid}\n"
            )
            for exon_index, (left, right) in enumerate(candidate.cds):
                handle.write(
                    f"{candidate.seqid}\tspot\texon\t{left + 1}\t{right}\t.\t"
                    f"{candidate.strand}\t.\tID={tid}.exon{exon_index + 1};Parent={tid}\n"
                )
            for cds_index, (left, right) in enumerate(candidate.cds):
                handle.write(
                    f"{candidate.seqid}\tspot\tCDS\t{left + 1}\t{right}\t.\t"
                    f"{candidate.strand}\t.\tID={tid}.cds{cds_index + 1};Parent={tid}\n"
                )


class OracleModel:
    """Perfect per-label scores for the held-out region (decoder test)."""

    def __init__(
        self,
        labels: dict[str, NDArray[np.int64]],
        offset: int,
        length: int,
        label_names: tuple[str, ...] = LABELS,
    ):
        self.labels = labels
        self.offset = offset
        self.length = length
        self.label_names = list(label_names)
        self.thresholds: NDArray[np.float32] | None = np.full(
            len(self.label_names), 0.5, dtype=np.float32
        )

    def fit(
        self,
        features: NDArray[np.float32],
        labels: dict[str, NDArray[np.int64]],
        motif_mask: NDArray[np.bool_] | None = None,
    ) -> OracleModel:
        return self

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        scores = np.zeros(
            (len(self.label_names), features.shape[1]), dtype=np.float32
        )
        for row, label in enumerate(self.label_names):
            positions = self.labels[label]
            positions = positions[
                (positions >= self.offset) & (positions < self.length)
            ]
            scores[row, positions - self.offset] = 1.0
        return scores

    def save(self, path: str) -> None:
        pass


def run(
    args: argparse.Namespace,
    model_factory: Callable[[Config], ModelProtocol] | None = None,
) -> int:
    codes = read_fasta(args.fasta)
    lengths = {name: len(sequence) for name, sequence in codes.items()}
    transcripts = read_annotation(args.annotation)
    raw_labels = site_labels(transcripts, lengths)
    limits = derive_limits(transcripts, lengths)
    motifs = learn_motifs(codes, raw_labels)

    name = next(iter(codes))
    sequence = codes[name]
    length = len(sequence)
    labels = build_labels(codes, raw_labels, name)
    mid = length // 2

    config = Config(
        radius=args.radius,
        epochs=args.epochs,
        seed=args.seed,
        hard_negatives=args.hard_negatives,
        hard_weight=args.hard_weight,
    )
    bank = FilterBank(config.periods)

    if args.load:
        raise SystemExit("--load is not available; inject a model_factory")
    if model_factory is None:
        model = OracleModel(labels, mid, length)
        features = np.zeros((1, length), dtype=np.float32)
    else:
        features = np.concatenate(
            [bank.transform_strands(sequence), one_hot_strands(sequence)], axis=0
        )
        train_motifs = motif_mask_matrix(sequence[:mid], motifs, list(labels))
        model = model_factory(config).fit(
            features[:, :mid], slice_labels(labels, 0, mid), train_motifs
        )
        model.save(args.model)

    scores = model.predict(features[:, mid:])
    print(f"sequence={name} length={length} labels={len(model.label_names)}")
    print(f"{'label':>10} {'n':>5} {'AUC':>6} {'AP':>8} {'peakF1':>7}")
    for row, label in enumerate(model.label_names):
        truth = np.zeros(length - mid, dtype=bool)
        positions = labels[label][(labels[label] >= mid) & (labels[label] < length)] - mid
        truth[positions] = True
        if truth.sum() == 0:
            continue
        f1, _ = best_f1(scores[row], truth)
        print(
            f"{label:>10} {int(truth.sum()):>5} {roc_auc(scores[row], truth):6.3f} "
            f"{average_precision(scores[row], truth):8.5f} {f1:7.3f}"
        )

    decoder = Decoder(limits, motifs)
    candidates: list[Candidate] = []
    for strand in STRANDS:
        thresholds = {
            kind: value * args.threshold_scale
            for kind, value in kind_thresholds(model, strand).items()
        }
        found = decode_strand(
            decoder,
            name,
            strand,
            sequence[mid:],
            scores,
            model.label_names,
            thresholds,
        )
        candidates.extend(shift_candidates(to_genomic(found, length - mid), mid))
    candidates = select_top(candidates, top=args.top)
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    write_gff3(Path(args.gff), candidates)
    print(f"candidates={len(candidates)} wrote {args.gff} and {args.model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/truth.nip.chr5.3m.fasta")
    parser.add_argument("--annotation", default="data/truth.nip.chr5.3m.gff3")
    parser.add_argument("--gff", default="spot.gff3")
    parser.add_argument("--model", default="spot.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--threshold-scale", type=float, default=0.95)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--hard-negatives", type=int, default=2000)
    parser.add_argument("--hard-weight", type=float, default=5.0)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
