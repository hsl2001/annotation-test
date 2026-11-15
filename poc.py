"""단순화된 유전자 구조 예측 PoC.

파이프라인
  1) Y  GFF3 + FASTA → strand별 6종 경계점 라벨 (p_U5, m_U5, ...)
  2) X  FASTA → 6종 복소수 염기 매핑
  3) X  각 매핑을 period-2 Morlet 커널 하나로 컨볼루션 → (6, L)
  4) 학습  X, Y를 model에 투입 (model은 TODO 자리)
  5) 예측  사이트 호출 → DP 조립 → GFF3
          (spliced CDS가 긴 순 top 5, 가장 긴 CDS 전사물과 50% 이상 겹치는 것만)

사용
  uv run poc.py --fasta genome.fasta --annotation annotation.gff3 --gff out.gff3
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import math
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

# ============================================================== 1. 상수 정의
SITES = ("U5", "GS", "IS", "IE", "GE", "U3")
STRANDS = ("+", "-")

# strand별 Y 행 순서: p_U5, p_GS, p_IS, p_IE, p_GE, p_U3 / m_U5, ...
STRAND_LABELS = {
    "+": tuple(f"p_{site}" for site in SITES),
    "-": tuple(f"m_{site}" for site in SITES),
}

PERIOD = 2.0  # 사용하는 커널은 period-2 Morlet 하나뿐
W0 = 6.0
MORLET_LIMIT = 8.0

TOP = 5
MIN_OVERLAP = 0.5

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

# 6종 복소수 매핑: A=1로 고정하고 (C, G, T)를 (i, -1, -i)의 순열로 할당한다.
# 4개 염기를 4차 단위근에 대응시키는 24개 전단사 중 전역 회전을 제거한 6가지다.
# (기존 BASE 매핑은 이 중 하나를 전역 회전한 것과 같다.)
ROOTS = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
MAPPINGS = np.zeros((5, 6), dtype=np.complex128)
for _column, _permutation in enumerate(itertools.permutations((1, 2, 3))):
    MAPPINGS[0, _column] = ROOTS[0]  # A
    MAPPINGS[1, _column] = ROOTS[_permutation[0]]  # C
    MAPPINGS[2, _column] = ROOTS[_permutation[1]]  # G
    MAPPINGS[3, _column] = ROOTS[_permutation[2]]  # T
# MAPPINGS[4] = 0 for N


# ========================================================== 2. 데이터 준비
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
    """(seqid, strand, site) → 정방향(oriented) 좌표의 경계점들."""
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
    """(6, L) 이진 라벨. 행 순서는 STRAND_LABELS[strand]와 같다."""
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
        min_exon=min(exons),
        min_intron=min(introns),
        min_coding=min(spans),
        min_transcript=min(utrs) if utrs else min(spans),
        max_intron=max(introns),
        max_span=max(spans),
        max_utr=max(utrs) if utrs else max(spans),
    )


# ====================================================== 3. X: 복소수 매핑 + 커널
def build_kernel(dilation: float, w0: float = W0) -> NDArray[np.complex128]:
    """Morlet 웨이블릿을 염기 단위로 적분한 커널."""
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


KERNEL = build_kernel(PERIOD)


def features(codes: NDArray[np.uint8]) -> NDArray[np.float32]:
    """(6, L) 실수 특징. 각 복소수 매핑을 period-2 커널로 컨볼루션한 에너지."""
    n = len(codes)
    width = len(KERNEL)
    fft_size = 1 << (n + width - 1).bit_length()
    kernel_spectrum = np.fft.fft(KERNEL[::-1], fft_size)
    out = np.empty((MAPPINGS.shape[1], n), dtype=np.float32)
    for index in range(MAPPINGS.shape[1]):
        signal = MAPPINGS[codes, index]
        spectrum = np.fft.fft(signal, fft_size) * kernel_spectrum
        coefficients = np.fft.ifft(spectrum)[width // 2 : width // 2 + n]
        values = np.log1p(np.abs(coefficients) ** 2)
        scale = values.std()
        out[index] = (values - values.mean()) / (scale if scale > 1e-6 else 1.0)
    return out


# ================================================================ 4. 모델
class Model:
    """TODO: 실제 모델로 교체할 자리.

    X: (6, L) 특징, Y: (6, L) 이진 라벨 (strand별 모델).
    fit(X, Y) -> self, predict(X) -> (6, L) 점수.
    """

    def fit(self, X: NDArray[np.float32], Y: NDArray[np.bool_]) -> Model:
        # TODO: 실제 모델 구현. 지금은 bias 포함 선형회귀 placeholder.
        gram = np.zeros((X.shape[0] + 1, X.shape[0] + 1), dtype=np.float64)
        cross = np.zeros((X.shape[0] + 1, Y.shape[0]), dtype=np.float64)
        chunk = 1_000_000
        for start in range(0, X.shape[1], chunk):
            x = X[:, start : start + chunk].astype(np.float64)
            y = Y[:, start : start + chunk].astype(np.float64)
            design = np.vstack([x, np.ones((1, x.shape[1]))])
            gram += design @ design.T
            cross += design @ y.T
        self.weights = np.linalg.solve(
            gram + 1e-6 * np.eye(len(gram)), cross
        ).T  # (6, 7)
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        # TODO: 실제 모델 구현.
        out = np.empty((self.weights.shape[0], X.shape[1]), dtype=np.float32)
        chunk = 1_000_000
        for start in range(0, X.shape[1], chunk):
            x = X[:, start : start + chunk].astype(np.float64)
            design = np.vstack([x, np.ones((1, x.shape[1]))])
            out[:, start : start + chunk] = self.weights @ design
        return out


# ============================================================== 5. DP 조립
@dataclass
class Sites:
    positions: NDArray[np.int64]
    scores: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.positions)


def local_max(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    """엄격한 국소최댓값(좌 >, 우 ≥)."""
    if not len(score):
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(score), dtype=bool)
    keep[1:] &= score[1:] > score[:-1]
    keep[:-1] &= score[:-1] >= score[1:]
    return keep


def call_sites(score: NDArray[np.float32], budget: int) -> Sites:
    """국소최댓값 중 상위 budget 개만 채택."""
    positions = np.flatnonzero(local_max(score)).astype(np.int64)
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


def assemble_candidates(
    seqid: str,
    strand: str,
    scores: dict[str, NDArray[np.float32]],
    limits: Limits,
    budgets: dict[str, int],
    length: int,
) -> list[Candidate]:
    """GS → (IS → IE)* → GE 경로를 DP로 찾고, 앞뒤 U5/U3로 전사체 span을 잡는다."""
    sites = {kind: call_sites(scores[kind], budgets[kind]) for kind in SITES}

    events: list[tuple[int, str, float]] = []
    for kind in ("GS", "IS", "IE", "GE"):
        for position, score in zip(sites[kind].positions, sites[kind].scores):
            events.append((int(position), kind, float(score)))
    events.sort()

    n = len(events)
    dp = np.full(n, -np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    start = np.full(n, -1, dtype=np.int64)
    kind_indices: dict[str, list[int]] = {kind: [] for kind in ("GS", "IS", "IE", "GE")}
    kind_positions: dict[str, list[int]] = {kind: [] for kind in ("GS", "IS", "IE", "GE")}

    for event_index, (position, kind, score) in enumerate(events):
        if kind == "GS":
            dp[event_index] = score
            start[event_index] = position
        else:
            best = -np.inf
            best_event = -1
            if kind == "IS":
                # 첫 엑손(GS→IS) 또는 내부 엑손(IE→IS) 뒤 min_exon 이상.
                for predecessor in ("GS", "IE"):
                    positions = kind_positions[predecessor]
                    limit = bisect_right(positions, position - limits.min_exon)
                    if limit:
                        options = kind_indices[predecessor][:limit]
                        values = dp[options]
                        arg = int(np.argmax(values))
                        if values[arg] > best:
                            best = values[arg]
                            best_event = options[arg]
            elif kind == "IE":
                # 인트론 길이가 [min_intron, max_intron]인 IS만.
                positions = kind_positions["IS"]
                low = bisect_left(positions, position - limits.max_intron)
                high = bisect_right(positions, position - limits.min_intron)
                if low < high:
                    options = kind_indices["IS"][low:high]
                    values = dp[options]
                    arg = int(np.argmax(values))
                    if values[arg] > best:
                        best = values[arg]
                        best_event = options[arg]
            elif kind == "GE":
                # 인트론 없음(GS→GE) 또는 마지막 엑손(IE→GE).
                for predecessor, minimum in (
                    ("GS", limits.min_coding),
                    ("IE", limits.min_exon),
                ):
                    positions = kind_positions[predecessor]
                    limit = bisect_right(positions, position - minimum)
                    for candidate_event in kind_indices[predecessor][:limit]:
                        candidate_start = start[candidate_event]
                        if candidate_start < 0:
                            continue
                        span = position - candidate_start
                        if span < limits.min_coding or span > limits.max_span:
                            continue
                        value = dp[candidate_event]
                        if value > best:
                            best = value
                            best_event = candidate_event
            if best_event >= 0 and np.isfinite(best):
                dp[event_index] = score + best
                prev[event_index] = best_event
                start[event_index] = start[best_event]
        kind_indices[kind].append(event_index)
        kind_positions[kind].append(position)

    candidates: list[Candidate] = []
    for event_index, (position, kind, _) in enumerate(events):
        if kind != "GE" or not np.isfinite(dp[event_index]):
            continue
        path: list[int] = []
        current = event_index
        while current >= 0:
            path.append(current)
            current = int(prev[current])
        path.reverse()
        if len(path) < 2 or events[path[0]][1] != "GS":
            continue

        gs_pos = events[path[0]][0]
        ge_pos = position
        introns: list[tuple[int, int]] = []
        for index in range(1, len(path) - 1, 2):
            introns.append((events[path[index]][0], events[path[index + 1]][0]))

        cds: list[tuple[int, int]] = []
        cursor = gs_pos
        for is_pos, ie_pos in introns:
            cds.append((cursor, is_pos))
            cursor = ie_pos
        cds.append((cursor, ge_pos))
        spliced = sum(right - left for left, right in cds)

        # CDS 앞의 가장 가까운 U5, 뒤의 가장 가까운 U3로 transcript span을 잡는다.
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
                score=float(dp[event_index]),
            )
        )
    return candidates


def select_top(candidates: list[Candidate]) -> list[Candidate]:
    """가장 긴 spliced CDS 전사물과 50% 이상 겹치는 것 중 top 5."""
    if not candidates:
        return []
    longest = max(candidates, key=lambda candidate: candidate.spliced_cds)
    span = longest.transcript
    span_length = max(span[1] - span[0], 1)

    def overlap(candidate: Candidate) -> int:
        return max(
            0,
            min(candidate.transcript[1], span[1])
            - max(candidate.transcript[0], span[0]),
        )

    selected = [
        candidate
        for candidate in candidates
        if candidate.seqid == longest.seqid
        and candidate.strand == longest.strand
        and overlap(candidate) / span_length >= MIN_OVERLAP
    ]
    selected.sort(key=lambda candidate: -candidate.spliced_cds)
    return selected[:TOP]


# ================================================================ 6. 출력
def gff_records(candidate: Candidate, number: int) -> list[str]:
    seqid, strand, length = candidate.seqid, candidate.strand, candidate.length

    def convert(position: int) -> int:
        return position if strand == "+" else length - 1 - position

    def interval(start: int, end: int) -> tuple[int, int]:
        first, last = convert(start), convert(end - 1)
        return min(first, last) + 1, max(first, last) + 1

    gene_id = f"{seqid}.g{number}"
    transcript_id = f"{gene_id}.t1"
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
    with open(path, "w") as handle:
        handle.write("##gff-version 3\n")
        for number, candidate in enumerate(candidates, 1):
            handle.writelines(line + "\n" for line in gff_records(candidate, number))


# ================================================================== 7. main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="단순화된 CWT 특징 기반 유전자 구조 예측 PoC"
    )
    parser.add_argument("--fasta", required=True, help="게놈 FASTA (.gz 가능)")
    parser.add_argument("--annotation", required=True, help="학습용 GFF3 (.gz 가능)")
    parser.add_argument("--gff", default="poc.gff3", help="출력 GFF3 경로")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    # --- 1) Y: GFF3 + FASTA → strand별 6종 라벨 ---------------------
    codes = read_fasta(args.fasta)
    lengths = {name: len(sequence) for name, sequence in codes.items()}
    transcripts = read_annotation(args.annotation)
    labels = site_labels(transcripts, lengths)
    limits = derive_limits(transcripts, lengths)
    training_length = sum(lengths.values())
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for (_, strand, site), positions in labels.items():
        counts[(strand, site)] += len(positions)
    print(f"data: {len(codes)} contig(s), {training_length:,} bp")
    print(f"limits: {limits}")

    # --- 2-3) X: 6종 복소수 매핑 → period-2 커널 컨볼루션 ----------
    X_blocks: dict[str, list[NDArray[np.float32]]] = {strand: [] for strand in STRANDS}
    Y_blocks: dict[str, list[NDArray[np.bool_]]] = {strand: [] for strand in STRANDS}
    for name, sequence in codes.items():
        for strand in STRANDS:
            X_blocks[strand].append(features(oriented(sequence, strand)))
            Y_blocks[strand].append(
                label_matrix(labels, name, strand, len(sequence))
            )

    # --- 4) 학습: X, Y → model (TODO) ------------------------------
    models: dict[str, Model] = {}
    for strand in STRANDS:
        X = np.concatenate(X_blocks[strand], axis=1)
        Y = np.concatenate(Y_blocks[strand], axis=1)
        print(f"X[{strand}]: {X.shape}, Y[{strand}]: {Y.shape}")
        models[strand] = Model().fit(X, Y)
    print(f"models fitted (TODO: replace placeholder) ({time.time() - started:.1f}s)")

    # --- 5) 예측 → DP 조립 → GFF3 ---------------------------------
    candidates: list[Candidate] = []
    for name, sequence in codes.items():
        for strand in STRANDS:
            scores = models[strand].predict(features(oriented(sequence, strand)))
            score_dict = {
                site: scores[index] for index, site in enumerate(SITES)
            }
            budgets = {
                site: max(
                    1,
                    round(
                        counts[(strand, site)]
                        * len(sequence)
                        / training_length
                    ),
                )
                for site in SITES
            }
            found = assemble_candidates(
                name, strand, score_dict, limits, budgets, len(sequence)
            )
            candidates.extend(found)
            print(
                f"  {name}/{strand}: {len(found)} candidate(s) "
                f"({time.time() - started:.1f}s)"
            )

    selected = select_top(candidates)
    write_gff3(args.gff, selected)
    print(
        f"wrote {args.gff}: {len(selected)} transcript(s) "
        f"({time.time() - started:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
