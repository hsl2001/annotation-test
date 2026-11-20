"""Rule-based gene annotation from CWT/PCA site scores.

The annotator has three stages:

1. Calibrate a linear discriminant for GS, IS and IE from the CWT features
   of sampled motif occurrences.  The discriminant is the difference between
   the positive and negative feature means.
2. Fold that linear discriminant into a single convolution filter and compute
   a genome-wide score map for each kind and strand.
3. Walk from every GS, splice at positive IS/IE pairs, and stop at the first
   in-frame canonical stop codon (GE).  The resulting CDS chains are written
   as GFF3.

GE is not classified: it is determined by the ORF traversal itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve
from sklearn.metrics import precision_recall_curve

from cwt_pca import (
    KINDS,
    MAPPINGS,
    STRANDS,
    CwtFeatureBank,
    _shift_mask,
    build_candidates,
    build_kernel,
    build_labels,
    motif_mask,
    oriented,
    read_annotation,
    read_fasta,
    sample_candidates,
    site_labels,
)

CANONICAL = {
    "GS": {bytes((0, 3, 2))},
    "IS": {bytes((2, 3)), bytes((2, 1))},
    "IE": {bytes((0, 2))},
    "GE": {bytes((3, 0, 0)), bytes((3, 0, 2)), bytes((3, 2, 0))},
}


@dataclass(slots=True)
class Limits:
    min_exon: int = 30
    max_exon: int = 10_000
    min_intron: int = 30
    max_intron: int = 5_000
    min_coding: int = 300
    max_span: int = 15_000


@dataclass(slots=True)
class Gene:
    start: int  # oriented, 0-based inclusive
    end: int  # oriented, 0-based inclusive
    exons: list[tuple[int, int]]  # oriented
    score: float


@dataclass(slots=True)
class KindModel:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: dict[str, NDArray[np.float64]]
    thresholds: dict[str, float]
    intercept: float = 0.0


def gather_features(
    bank: CwtFeatureBank,
    positions: NDArray[np.int64],
    strands: NDArray[np.int8],
    offsets: NDArray[np.int64],
) -> NDArray[np.float32]:
    flat_positions = (positions[:, None] + offsets[None, :]).ravel()
    flat_strands = np.repeat(strands, len(offsets))
    features = bank.gather(flat_positions, flat_strands)
    return features.reshape(len(positions), len(offsets) * bank.channels)


def real_imag_columns(n_kernels: int, n_offsets: int) -> NDArray[np.int64]:
    channels = 18 * n_kernels
    per_offset = np.concatenate(
        [
            np.arange(kernel * 18 + 6, kernel * 18 + 18)
            for kernel in range(n_kernels)
        ]
    )
    return np.concatenate(
        [per_offset + index * channels for index in range(n_offsets)]
    )


def best_threshold(
    scores: NDArray[np.float64], truth: NDArray[np.int8]
) -> tuple[float, float]:
    if len(np.unique(truth)) < 2:
        return float("-inf"), 0.0
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1))
    threshold = float(thresholds[min(index, len(thresholds) - 1)])
    return threshold, float(f1[index])


def calibrate(
    codes: dict[str, NDArray[np.uint8]],
    name: str,
    labels: dict[str, NDArray[np.int64]],
    args: argparse.Namespace,
) -> tuple[
    dict[str, KindModel],
    CwtFeatureBank,
    NDArray[np.int64],
    list[NDArray[np.complex128]],
]:
    sequence = codes[name]
    candidates = build_candidates(codes, CANONICAL, labels, name)
    rng = np.random.default_rng(args.seed)
    index = sample_candidates(
        candidates, args.max_positives, args.negatives_per_positive, rng
    )
    bank = CwtFeatureBank(
        sequence, tuple(args.periods), tuple(args.w0), args.chunk
    )
    offsets = np.arange(
        -args.window_radius, args.window_radius + 1, args.window_step
    )
    kernels = [
        build_kernel(period, w0) for w0 in args.w0 for period in args.periods
    ]
    columns = real_imag_columns(len(kernels), len(offsets))

    models: dict[str, KindModel] = {}
    for kind_index, kind in enumerate(KINDS):
        if kind == "GE":
            continue
        selected = index[candidates.kinds[index] == kind_index]
        truth = candidates.labels[selected]
        strands = candidates.strands[selected]
        if len(selected) < 50 or len(np.unique(truth)) < 2:
            print(f"  {kind}: insufficient calibration data, skipped")
            continue

        features = gather_features(
            bank, candidates.positions[selected], strands, offsets
        )
        values = features[:, columns].astype(np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (values - mean) / scale

        weights: dict[str, NDArray[np.float64]] = {}
        thresholds: dict[str, float] = {}
        for value, strand in enumerate(STRANDS):
            positive = (truth == 1) & (strands == value)
            negative = (truth == 0) & (strands == value)
            if positive.sum() < 20 or negative.sum() < 20:
                continue
            weight = (
                standardized[positive].mean(axis=0)
                - standardized[negative].mean(axis=0)
            )
            scores = standardized @ weight
            mask = strands == value
            threshold, f1 = best_threshold(scores[mask], truth[mask])
            weights[strand] = weight
            thresholds[strand] = threshold
            print(
                f"  {kind} {strand}: n={int(positive.sum() + negative.sum())} "
                f"pos={int(positive.sum())} F1={f1:.3f} threshold={threshold:.3f}"
            )
        models[kind] = KindModel(mean, scale, weights, thresholds)
    return models, bank, offsets, kernels


def combined_filter(
    model: KindModel,
    strand: str,
    offsets: NDArray[np.int64],
    kernels: list[NDArray[np.complex128]],
    window_radius: int,
) -> tuple[NDArray[np.complex128], float]:
    weight = model.weights[strand]
    scaled = weight / model.scale
    bias = (
        -float(np.dot(weight, model.mean / model.scale)) + model.intercept
    )
    n_kernels = len(kernels)
    max_radius = max(len(kernel) // 2 for kernel in kernels)
    filter_radius = window_radius + max_radius
    combined = np.zeros((5, 2 * filter_radius + 1), dtype=np.complex128)

    for offset_index, offset in enumerate(offsets):
        for kernel_index, kernel in enumerate(kernels):
            radius = len(kernel) // 2
            base = offset_index * (n_kernels * 12) + kernel_index * 12
            for mapping in range(6):
                real = scaled[base + mapping]
                imag = scaled[base + 6 + mapping]
                coefficient = real - 1j * imag
                for displacement in range(-radius, radius + 1):
                    tap = radius - displacement
                    combined[:, offset + displacement + filter_radius] += (
                        coefficient * MAPPINGS[:, mapping] * kernel[tap]
                    )
    return combined, bias


def score_map(
    sequence: NDArray[np.uint8],
    combined: NDArray[np.complex128],
    bias: float,
    filter_radius: int,
) -> NDArray[np.float64]:
    scores = np.zeros(len(sequence), dtype=np.float64)
    for code in range(4):
        indicator = np.pad(
            (sequence == code).astype(np.float32),
            filter_radius,
            mode="constant",
        )
        convolution = fftconvolve(
            indicator, combined[code, ::-1], mode="valid"
        )
        scores += convolution.real
    scores += bias
    return scores


def annotate_strand(
    sequence: NDArray[np.uint8],
    models: dict[str, KindModel],
    scores: dict[str, NDArray[np.float64]],
    strand: str,
    limits: Limits,
) -> list[Gene]:
    gs_mask = motif_mask(sequence, "GS", CANONICAL)
    is_mask = motif_mask(sequence, "IS", CANONICAL)
    ie_mask = _shift_mask(motif_mask(sequence, "IE", CANONICAL), -2)
    ge_mask = _shift_mask(motif_mask(sequence, "GE", CANONICAL), -3)

    gs_positive = gs_mask & (scores["GS"] >= models["GS"].thresholds[strand])
    is_positive = is_mask & (scores["IS"] >= models["IS"].thresholds[strand])
    ie_positive = ie_mask & (scores["IE"] >= models["IE"].thresholds[strand])

    is_positions = np.flatnonzero(is_positive)
    is_scores = scores["IS"][is_positions]
    ie_positions = np.flatnonzero(ie_positive)
    ie_scores = scores["IE"][ie_positions]
    ge_positions = np.flatnonzero(ge_mask)
    stop_by_frame = [
        ge_positions[ge_positions % 3 == frame] for frame in range(3)
    ]

    genes: list[Gene] = []
    for gs in np.flatnonzero(gs_positive):
        gene = traverse(
            int(gs),
            sequence,
            float(scores["GS"][gs]),
            is_positions,
            is_scores,
            ie_positions,
            ie_scores,
            stop_by_frame,
            limits,
        )
        if gene is not None:
            genes.append(gene)
    return genes


def traverse(
    gs: int,
    sequence: NDArray[np.uint8],
    gs_score: float,
    is_positions: NDArray[np.int64],
    is_scores: NDArray[np.float64],
    ie_positions: NDArray[np.int64],
    ie_scores: NDArray[np.float64],
    stop_by_frame: list[NDArray[np.int64]],
    limits: Limits,
) -> Gene | None:
    start = gs
    cursor = gs + 3
    coding = 3
    exon_start = gs
    exons: list[tuple[int, int]] = []
    score = gs_score
    limit = min(start + limits.max_span, len(sequence) - 2)

    while cursor < limit:
        target_frame = (cursor - coding) % 3
        stops = stop_by_frame[target_frame]
        stop_index = int(np.searchsorted(stops, cursor))
        next_stop = int(stops[stop_index]) if stop_index < len(stops) else None

        is_index = int(np.searchsorted(is_positions, cursor))
        next_is = (
            int(is_positions[is_index]) if is_index < len(is_positions) else None
        )
        if next_stop is None and next_is is None:
            break

        if next_is is not None and (next_stop is None or next_is < next_stop):
            exon_length = next_is - exon_start
            if exon_length >= limits.min_exon:
                low = int(
                    np.searchsorted(
                        ie_positions,
                        next_is + limits.min_intron - 2,
                        side="left",
                    )
                )
                high = int(
                    np.searchsorted(
                        ie_positions,
                        next_is + limits.max_intron - 2,
                        side="right",
                    )
                )
                if low < high:
                    best = low + int(np.argmax(ie_scores[low:high]))
                    ie = int(ie_positions[best])
                    exons.append((exon_start, next_is - 1))
                    score += float(is_scores[is_index]) + float(ie_scores[best])
                    coding += next_is - cursor
                    cursor = ie + 2
                    exon_start = cursor
                    continue
            coding += next_is - cursor + 1
            cursor = next_is + 1
            continue

        if next_stop is None:
            break
        coding += next_stop - cursor
        exon_end = next_stop + 2
        exons.append((exon_start, exon_end))
        span = exon_end - start + 1
        if (
            coding + 3 >= limits.min_coding
            and span <= limits.max_span
            and all(end - begin + 1 >= limits.min_exon for begin, end in exons)
        ):
            return Gene(start, exon_end, exons, score)
        return None
    return None


def genomic_span(
    span: tuple[int, int], length: int, strand: str
) -> tuple[int, int]:
    begin, end = span
    return (
        (begin, end)
        if strand == "+"
        else (length - 1 - end, length - 1 - begin)
    )


def to_genomic(
    genes: list[Gene], length: int, strand: str
) -> list[tuple[int, int, list[tuple[int, int]], float]]:
    result = []
    for gene in genes:
        exons = sorted(
            genomic_span(exon, length, strand) for exon in gene.exons
        )
        result.append(
            (
                min(begin for begin, _ in exons),
                max(end for _, end in exons),
                exons,
                gene.score,
            )
        )
    return result


def write_gff3(
    path: Path,
    seqid: str,
    genes: list[tuple[str, int, int, list[tuple[int, int]], float]],
) -> None:
    with path.open("w") as handle:
        handle.write("##gff-version 3\n")
        for index, (strand, start, end, exons, score) in enumerate(genes):
            gene_id = f"gene{index + 1}"
            transcript_id = f"tx{index + 1}"
            handle.write(
                f"{seqid}\tspot\tgene\t{start + 1}\t{end + 1}\t{score:.4f}\t"
                f"{strand}\t.\tID={gene_id}\n"
            )
            handle.write(
                f"{seqid}\tspot\tmRNA\t{start + 1}\t{end + 1}\t{score:.4f}\t"
                f"{strand}\t.\tID={transcript_id};Parent={gene_id}\n"
            )
            for exon_index, (begin, finish) in enumerate(exons):
                handle.write(
                    f"{seqid}\tspot\texon\t{begin + 1}\t{finish + 1}\t.\t"
                    f"{strand}\t.\tID={transcript_id}.exon{exon_index + 1};"
                    f"Parent={transcript_id}\n"
                )
                handle.write(
                    f"{seqid}\tspot\tCDS\t{begin + 1}\t{finish + 1}\t.\t"
                    f"{strand}\t.\tID={transcript_id}.cds{exon_index + 1};"
                    f"Parent={transcript_id}\n"
                )


def run(args: argparse.Namespace) -> int:
    codes = read_fasta(args.fasta)
    name = next(iter(codes))
    lengths = {name: len(codes[name])}
    transcripts = read_annotation(args.annotation)
    raw_labels = site_labels(transcripts, lengths)
    labels = build_labels(codes, raw_labels, name)

    print(f"calibrating on {args.fasta}")
    models, _, offsets, kernels = calibrate(codes, name, labels, args)
    missing = [kind for kind in ("GS", "IS", "IE") if kind not in models]
    if missing:
        raise SystemExit(f"could not calibrate: {', '.join(missing)}")

    sequence = codes[name]
    candidates = build_candidates(codes, CANONICAL, labels, name)
    combined = {
        (kind, strand): combined_filter(
            models[kind],
            strand,
            offsets,
            kernels,
            args.window_radius,
        )
        for kind in ("GS", "IS", "IE")
        for strand in STRANDS
        if strand in models[kind].weights
    }

    genes: list[tuple[str, int, int, list[tuple[int, int]], float]] = []
    for strand in STRANDS:
        if not all(
            strand in models[kind].weights for kind in ("GS", "IS", "IE")
        ):
            print(f"{strand}: incomplete calibration, skipped")
            continue
        oriented_sequence = oriented(sequence, strand)
        scores = {}
        for kind in ("GS", "IS", "IE"):
            if (kind, strand) not in combined:
                continue
            filter_array, bias = combined[kind, strand]
            filter_radius = filter_array.shape[1] // 2
            scores[kind] = score_map(
                oriented_sequence, filter_array, bias, filter_radius
            )
        strand_index = STRANDS.index(strand)
        for kind in ("GS", "IS", "IE"):
            kind_index = KINDS.index(kind)
            selected = (candidates.kinds == kind_index) & (
                candidates.strands == strand_index
            )
            if not selected.any():
                continue
            positions = candidates.positions[selected]
            oriented_positions = (
                positions
                if strand_index == 0
                else len(sequence) - 1 - positions
            )
            kind_scores = scores[kind][oriented_positions]
            if args.threshold_mode == "genome":
                threshold, f1 = best_threshold(
                    kind_scores, candidates.labels[selected]
                )
                print(
                    f"  {kind} {strand}: genome F1={f1:.3f} "
                    f"threshold={threshold:.3f}"
                )
            elif args.threshold_mode == "quantile":
                background = kind_scores[candidates.labels[selected] == 0]
                threshold = float(
                    np.quantile(background, args.threshold_quantile)
                )
                print(
                    f"  {kind} {strand}: q={args.threshold_quantile} "
                    f"threshold={threshold:.3f}"
                )
            else:
                threshold = models[kind].thresholds[strand]
                print(
                    f"  {kind} {strand}: balanced "
                    f"threshold={threshold:.3f}"
                )
            models[kind].thresholds[strand] = threshold
        found = annotate_strand(
            oriented_sequence,
            models,
            scores,
            strand,
            Limits(
                min_coding=args.min_coding,
                max_span=args.max_span,
            ),
        )
        for start, end, exons, score in to_genomic(found, len(sequence), strand):
            genes.append((strand, start, end, exons, score))
        print(f"{strand}: {len(found)} genes")

    unique: dict[tuple, tuple] = {}
    for gene in genes:
        key = (gene[0], gene[1], gene[2], tuple(gene[3]))
        if key not in unique or gene[4] > unique[key][4]:
            unique[key] = gene
    result = sorted(unique.values(), key=lambda gene: (gene[1], gene[0]))

    write_gff3(Path(args.gff), name, result)
    print(f"wrote {args.gff} ({len(result)} genes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/truth.nip.chr5.fasta")
    parser.add_argument("--annotation", default="data/truth.nip.chr5.gff3")
    parser.add_argument("--gff", default="spot.gff3")
    parser.add_argument("--periods", type=float, nargs="+", default=[2, 3, 4, 6, 7])
    parser.add_argument("--w0", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    parser.add_argument("--window-radius", type=int, default=10)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-positives", type=int, default=2000)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument(
        "--threshold-mode",
        choices=("balanced", "genome", "quantile"),
        default="genome",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--min-coding", type=int, default=300)
    parser.add_argument("--max-span", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
