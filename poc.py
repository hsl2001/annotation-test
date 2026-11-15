from __future__ import annotations

import argparse
import gzip
import itertools
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from model import Model

SITES = ("U5", "GS", "IS", "IE", "GE", "U3")
STRANDS = ("+", "-")

STRAND_LABELS = {
    "+": tuple(f"p_{site}" for site in SITES),
    "-": tuple(f"m_{site}" for site in SITES),
}

PERIOD = 2.0
W0 = 6.0
MORLET_LIMIT = 8.0

TOP = 5
MIN_OVERLAP = 0.5
BUDGET_FACTOR = 50
MAX_BUDGET = 2_000
MAX_PATHS = 5
INTRON_PENALTY = 4.0
START_CODON = np.array([0, 3, 2], dtype=np.uint8)
STOP_CODONS = (
    np.array([3, 0, 0], dtype=np.uint8),
    np.array([3, 0, 2], dtype=np.uint8),
    np.array([3, 2, 0], dtype=np.uint8),
)

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
            parent = None
            for piece in attributes.split(";"):
                if piece.startswith("Parent="):
                    parent = piece[7:]
                    break
            if parent is None:
                continue
            entry = transcripts.setdefault(
                parent,
                {"seqid": seqid, "strand": strand, "cds": [], "exon": [], "utr": False},
            )
            span = (int(a) - 1, int(b))
            if kind == "CDS":
                entry["cds"].append(span)
            elif kind == "exon":
                entry["exon"].append(span)
            else:
                entry["utr"] = True

    result: list[Transcript] = []
    for entry in transcripts.values():
        cds = sorted(entry["cds"])
        exons = sorted(entry["exon"])
        if not cds:
            cds, exons = exons, []
        result.append(
            Transcript(entry["seqid"], entry["strand"], cds, exons, entry["utr"])
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


def label_matrix(
    labels: dict[tuple[str, str, str], NDArray[np.int64]],
    name: str,
    strand: str,
    length: int,
) -> NDArray[np.bool_]:
    out = np.zeros((len(SITES), length), dtype=np.bool_)
    for row, label in enumerate(STRAND_LABELS[strand]):
        positions = labels.get((name, strand, label[2:]))
        if positions is not None:
            out[row, positions] = True
    return out


@dataclass(frozen=True)
class Limits:
    min_exon: int
    min_intron: int
    min_coding: int
    min_transcript: int
    max_intron: int
    max_span: int
    max_utr: int


def derive_limits(transcripts: list[Transcript], lengths: dict[str, int]) -> Limits:
    introns: list[int] = []
    spans: list[int] = []
    exons: list[int] = []
    utrs: list[int] = []
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
        if transcript.has_utr and transcript.exon:
            exons_oriented = sorted(
                oriented_span(span, length, transcript.strand)
                for span in transcript.exon
            )
            utrs.append(exons_oriented[-1][1] - exons_oriented[0][0])
    if not introns or not spans or not exons:
        raise SystemExit("annotation lacks the intervals needed to derive limits")
    return Limits(
        min_exon=min(min(exons), 3),
        min_intron=min(min(introns), 10),
        min_coding=min(min(spans), 30),
        min_transcript=min(min(utrs) if utrs else min(spans), 30),
        max_intron=max(max(introns) * 3, 10_000),
        max_span=max(max(spans) * 3, 20_000),
        max_utr=max(max(utrs) * 3, 20_000) if utrs else max(max(spans) * 3, 20_000),
    )


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


MORLET = build_kernel(PERIOD)


def features(codes: NDArray[np.uint8]) -> NDArray[np.float32]:
    signal = MAPPINGS[codes]
    length = signal.shape[0]
    radius = len(MORLET) // 2
    fft_size = 1 << (length + len(MORLET) - 1).bit_length()
    kernel_spectrum = np.fft.fft(MORLET, fft_size)
    out = np.empty((signal.shape[1], length), dtype=np.float32)
    for channel in range(signal.shape[1]):
        coefficients = np.fft.ifft(
            np.fft.fft(signal[:, channel], fft_size) * kernel_spectrum
        )[radius : radius + length]
        out[channel] = np.abs(coefficients) ** 2
    return out


def learn_motifs(
    codes: dict[str, NDArray[np.uint8]],
    labels: dict[tuple[str, str, str], NDArray[np.int64]],
) -> dict[str, set[bytes]]:
    motifs: dict[str, set[bytes]] = {
        "GS": {bytes((0, 3, 2))},
        "IS": {bytes((2, 3)), bytes((2, 1))},
        "IE": {bytes((0, 2))},
        "GE": {
            bytes((3, 0, 0)),
            bytes((3, 0, 2)),
            bytes((3, 2, 0)),
        },
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


def motif_mask(
    sequence: NDArray[np.uint8], site: str, motifs: dict[str, set[bytes]]
) -> NDArray[np.bool_]:
    n = len(sequence)
    if site not in motifs:
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for motif in motifs[site]:
        k = len(motif)
        if k > n:
            continue
        match = np.ones(n - k + 1, dtype=bool)
        for offset, value in enumerate(motif):
            match &= sequence[offset : offset + n - k + 1] == value
        if site in ("GS", "IS"):
            mask[: n - k + 1] |= match
        else:
            mask[k:] |= match[: n - k]
    return mask


@dataclass
class Sites:
    positions: NDArray[np.int64]
    scores: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.positions)


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


def call_sites(
    score: NDArray[np.float32],
    budget: int,
    valid: NDArray[np.bool_] | None = None,
) -> Sites:
    if valid is not None:
        score = np.where(valid, score, -np.inf)
    positions = np.flatnonzero(local_max(score)).astype(np.int64)
    if valid is not None:
        positions = positions[valid[positions]]
    if not len(positions):
        return Sites(positions, np.array([], dtype=np.float32))
    values = score[positions]
    if 0 < budget < len(positions):
        order = np.argsort(-values)[:budget]
        order.sort()
        positions, values = positions[order], values[order]
    return Sites(positions, values)


@dataclass
class Candidate:
    seqid: str
    strand: str
    length: int
    transcript: tuple[int, int]
    cds: list[tuple[int, int]]
    exons: list[tuple[int, int]]
    spliced_cds: int
    score: float
    locus: int = 0


def valid_orf(sequence: NDArray[np.uint8], cds: list[tuple[int, int]]) -> bool:
    coding = np.concatenate([sequence[left:right] for left, right in cds])
    if len(coding) < 6 or len(coding) % 3:
        return False
    if not np.array_equal(coding[:3], START_CODON):
        return False
    if not any(np.array_equal(coding[-3:], stop) for stop in STOP_CODONS):
        return False
    codons = coding[:-3].reshape(-1, 3)
    return not any(np.all(codons == stop, axis=1).any() for stop in STOP_CODONS)


def assemble_candidates(
    seqid: str,
    strand: str,
    sequence: NDArray[np.uint8],
    scores: dict[str, NDArray[np.float32]],
    limits: Limits,
    budgets: dict[str, int],
    motifs: dict[str, set[bytes]],
) -> list[Candidate]:
    length = len(sequence)
    valid = {site: motif_mask(sequence, site, motifs) for site in SITES}
    sites = {
        kind: call_sites(scores[kind], budgets[kind], valid[kind])
        for kind in SITES
    }

    events: list[tuple[int, str, float]] = []
    for kind in ("GS", "IS", "IE", "GE"):
        for position, score in zip(sites[kind].positions, sites[kind].scores):
            events.append((int(position), kind, float(score)))
    events.sort()

    n = len(events)
    best_paths: list[list[int]] = [[] for _ in range(n)]
    path_score: list[float] = []
    path_start: list[int] = []
    path_event: list[int] = []
    path_prev: list[int] = []
    kind_indices: dict[str, list[int]] = {kind: [] for kind in ("GS", "IS", "IE", "GE")}
    kind_positions: dict[str, list[int]] = {kind: [] for kind in ("GS", "IS", "IE", "GE")}

    def add_path(score: float, start: int, event: int, previous: int) -> int:
        path_score.append(score)
        path_start.append(start)
        path_event.append(event)
        path_prev.append(previous)
        return len(path_score) - 1

    def gather(kind: str, low: int, high: int) -> list[int]:
        result: list[int] = []
        for event in kind_indices[kind][low:high]:
            result.extend(best_paths[event])
        return result

    for event_index, (position, kind, score) in enumerate(events):
        created: list[int] = []
        if kind == "GS":
            created.append(add_path(score, position, event_index, -1))
        elif kind == "IS":
            for predecessor in ("GS", "IE"):
                high = bisect_right(
                    kind_positions[predecessor], position - limits.min_exon
                )
                for path in gather(predecessor, 0, high):
                    created.append(
                        add_path(
                            path_score[path] + score,
                            path_start[path],
                            event_index,
                            path,
                        )
                    )
        elif kind == "IE":
            positions = kind_positions["IS"]
            low = bisect_left(positions, position - limits.max_intron)
            high = bisect_right(positions, position - limits.min_intron)
            for path in gather("IS", low, high):
                created.append(
                    add_path(
                        path_score[path] + score - INTRON_PENALTY,
                        path_start[path],
                        event_index,
                        path,
                    )
                )
        elif kind == "GE":
            for predecessor, minimum in (
                ("GS", limits.min_coding),
                ("IE", limits.min_exon),
            ):
                high = bisect_right(
                    kind_positions[predecessor], position - minimum
                )
                for path in gather(predecessor, 0, high):
                    span = position - path_start[path]
                    if span < limits.min_coding or span > limits.max_span:
                        continue
                    created.append(
                        add_path(
                            path_score[path] + score,
                            path_start[path],
                            event_index,
                            path,
                        )
                    )
        created.sort(key=lambda path: path_score[path], reverse=True)
        best_paths[event_index] = created[:MAX_PATHS]
        kind_indices[kind].append(event_index)
        kind_positions[kind].append(position)

    candidates: list[Candidate] = []
    for event_index, (position, kind, _) in enumerate(events):
        if kind != "GE":
            continue
        for path in best_paths[event_index]:
            chain: list[int] = []
            current = path
            while current >= 0:
                chain.append(path_event[current])
                current = path_prev[current]
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
            if not valid_orf(sequence, cds):
                continue
            spliced = sum(right - left for left, right in cds)

            transcript = (gs_pos, ge_pos)
            u5_index = bisect_right(sites["U5"].positions, gs_pos) - 1
            u3_index = bisect_left(sites["U3"].positions, ge_pos)
            if u5_index >= 0 and u3_index < len(sites["U3"]):
                u5_pos = int(sites["U5"].positions[u5_index])
                u3_pos = int(sites["U3"].positions[u3_index])
                if limits.min_transcript <= u3_pos - u5_pos <= limits.max_utr:
                    transcript = (u5_pos, u3_pos)

            exons: list[tuple[int, int]] = []
            if transcript[0] < gs_pos:
                exons.append((transcript[0], gs_pos))
            exons.extend(cds)
            if ge_pos < transcript[1]:
                exons.append((ge_pos, transcript[1]))

            candidates.append(
                Candidate(
                    seqid=seqid,
                    strand=strand,
                    length=length,
                    transcript=transcript,
                    cds=cds,
                    exons=exons,
                    spliced_cds=spliced,
                    score=float(path_score[path]),
                )
            )
    return candidates


def cds_envelope(candidate: Candidate) -> tuple[int, int]:
    return candidate.cds[0][0], candidate.cds[-1][1]


def envelope_overlap(
    left: tuple[int, int], right: tuple[int, int], reference: tuple[int, int]
) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    return overlap / max(reference[1] - reference[0], 1)


def select_top(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []

    unique: dict[tuple, Candidate] = {}
    for candidate in candidates:
        key = (candidate.seqid, candidate.strand, tuple(candidate.cds))
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    pool = list(unique.values())
    pool.sort(key=lambda candidate: -candidate.score)

    selected: list[Candidate] = []
    assigned: set[int] = set()
    locus = 0
    for primary in pool:
        if id(primary) in assigned:
            continue
        reference = cds_envelope(primary)
        group = [
            candidate
            for candidate in pool
            if id(candidate) not in assigned
            and candidate.seqid == primary.seqid
            and candidate.strand == primary.strand
            and envelope_overlap(cds_envelope(candidate), reference, reference)
            >= MIN_OVERLAP
        ]
        for candidate in group:
            assigned.add(id(candidate))
        if not group:
            continue
        longest = max(group, key=lambda candidate: candidate.spliced_cds)
        longest_span = cds_envelope(longest)
        isoforms = [
            candidate
            for candidate in group
            if envelope_overlap(cds_envelope(candidate), longest_span, longest_span)
            >= MIN_OVERLAP
        ]
        isoforms.sort(key=lambda candidate: (-candidate.spliced_cds, -candidate.score))
        locus += 1
        for candidate in isoforms[:TOP]:
            candidate.locus = locus
            selected.append(candidate)

    selected.sort(key=lambda candidate: (candidate.seqid, candidate.strand, candidate.transcript))
    return selected


def gff_records(
    candidate: Candidate, gene_number: int, transcript_number: int
) -> list[str]:
    seqid, strand, length = candidate.seqid, candidate.strand, candidate.length

    def convert(position: int) -> int:
        return position if strand == "+" else length - 1 - position

    def interval(start: int, end: int) -> tuple[int, int]:
        first, last = convert(start), convert(end - 1)
        return min(first, last) + 1, max(first, last) + 1

    gene_id = f"{seqid}.g{gene_number}"
    transcript_id = f"{gene_id}.t{transcript_number}"
    a, b = interval(*candidate.transcript)
    out = [f"{seqid}\tpoc\tgene\t{a}\t{b}\t.\t{strand}\t.\tID={gene_id}"]
    out.append(
        f"{seqid}\tpoc\tmRNA\t{a}\t{b}\t.\t{strand}\t.\tID={transcript_id};Parent={gene_id}"
    )
    for index, (left, right) in enumerate(candidate.exons, 1):
        exon_a, exon_b = interval(left, right)
        out.append(
            f"{seqid}\tpoc\texon\t{exon_a}\t{exon_b}\t.\t{strand}\t.\t"
            f"ID={transcript_id}.exon{index};Parent={transcript_id}"
        )
    coded = 0
    for index, (left, right) in enumerate(candidate.cds, 1):
        cds_a, cds_b = interval(left, right)
        phase = (3 - coded % 3) % 3
        out.append(
            f"{seqid}\tpoc\tCDS\t{cds_a}\t{cds_b}\t.\t{strand}\t{phase}\t"
            f"ID={transcript_id}.cds{index};Parent={transcript_id}"
        )
        coded += right - left
    if candidate.transcript[0] < candidate.cds[0][0]:
        utr_a, utr_b = interval(candidate.transcript[0], candidate.cds[0][0])
        out.append(
            f"{seqid}\tpoc\tfive_prime_UTR\t{utr_a}\t{utr_b}\t.\t{strand}\t.\t"
            f"ID={transcript_id}.utr5;Parent={transcript_id}"
        )
    if candidate.transcript[1] > candidate.cds[-1][1]:
        utr_a, utr_b = interval(candidate.cds[-1][1], candidate.transcript[1])
        out.append(
            f"{seqid}\tpoc\tthree_prime_UTR\t{utr_a}\t{utr_b}\t.\t{strand}\t.\t"
            f"ID={transcript_id}.utr3;Parent={transcript_id}"
        )
    return out


def write_gff3(path: str, candidates: list[Candidate]) -> None:
    groups: dict[tuple[str, str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.seqid, candidate.strand, candidate.locus)].append(candidate)
    ordered = sorted(
        groups.values(),
        key=lambda group: (group[0].seqid, group[0].strand, group[0].transcript),
    )
    with open(path, "w") as handle:
        handle.write("##gff-version 3\n")
        for gene_number, group in enumerate(ordered, 1):
            for transcript_number, candidate in enumerate(group, 1):
                handle.writelines(
                    line + "\n"
                    for line in gff_records(
                        candidate, gene_number, transcript_number
                    )
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predict-fasta")
    parser.add_argument("--gff", default="poc.gff3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    codes = read_fasta(args.fasta)
    lengths = {name: len(sequence) for name, sequence in codes.items()}
    transcripts = read_annotation(args.annotation)
    labels = site_labels(transcripts, lengths)
    limits = derive_limits(transcripts, lengths)
    training_length = sum(lengths.values())
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for (_, strand, site), positions in labels.items():
        counts[(strand, site)] += len(positions)

    X_blocks: list[NDArray[np.float32]] = []
    Y_blocks: list[NDArray[np.bool_]] = []
    for name, sequence in codes.items():
        for strand in STRANDS:
            X_blocks.append(features(oriented(sequence, strand)))
            Y_blocks.append(label_matrix(labels, name, strand, len(sequence)))
    X_train = np.concatenate(X_blocks, axis=1)
    Y_train = np.concatenate(Y_blocks, axis=1)

    model = Model().fit(X_train, Y_train)
    assert model.thresholds is not None
    thresholds = model.thresholds
    motifs = learn_motifs(codes, labels)

    predict_codes = (
        read_fasta(args.predict_fasta) if args.predict_fasta else codes
    )
    candidates: list[Candidate] = []
    for name, sequence in predict_codes.items():
        for strand in STRANDS:
            oriented_sequence = oriented(sequence, strand)
            scores = model.predict(features(oriented_sequence))
            score_dict = {
                site: scores[index] - thresholds[index]
                for index, site in enumerate(SITES)
            }
            budgets = {
                site: min(
                    MAX_BUDGET,
                    max(
                        1,
                        round(
                            counts[(strand, site)]
                            * len(sequence)
                            / training_length
                            * BUDGET_FACTOR
                        ),
                    ),
                )
                for site in SITES
            }
            found = assemble_candidates(
                name,
                strand,
                oriented_sequence,
                score_dict,
                limits,
                budgets,
                motifs,
            )
            candidates.extend(found)

    selected = select_top(candidates)
    write_gff3(args.gff, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
