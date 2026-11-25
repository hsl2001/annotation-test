from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from accuracy import read_gff3
from cwt_features import LUT, PERIODS, W0, CwtFeatureBank, _motif_mask, cwt_features

GROUPS = ("DSS", "ASS", "RC_DSS", "RC_ASS")
GROUP_CONFIG = {
    "DSS": ("+", "GT", 0, 0),
    "ASS": ("+", "AG", 0, 0),
    "RC_DSS": ("-", "AC", 1, 1),
    "RC_ASS": ("-", "CT", 1, 1),
}
FEATURE_CHANNELS = 12 * len(PERIODS)
MAX_TEST_POSITIVES = 2000
GENES_PER_DATASET = 100
MATCH_THRESHOLD = 0.025
TEMPLATE_PATH = "splice_templates_train_100genes.npz"
PLOT_PATH = "splice_sn_pr_human_vp.png"
DATASETS = {
    "arabidopsis": ("data/truth.col.chr5.fasta.gz", "data/truth.col.chr5.gff3.gz"),
    "rice": ("data/truth.nip.chr5.fasta.gz", "data/truth.nip.chr5.gff3.gz"),
    "tomato": ("data/truth.slm.chr5.fasta.gz", "data/truth.slm.chr5.gff3.gz"),
    "human": ("data/truth.human.chr21.fasta.gz", "data/truth.human.chr21.gff3.gz"),
    "mouse": ("data/truth.mouse.chr19.fasta.gz", "data/truth.mouse.chr19.gff3.gz"),
    "zebrafish": (
        "data/truth.zebrafish.chr25.fasta.gz",
        "data/truth.zebrafish.chr25.gff3.gz",
    ),
    "celegans": ("data/truth.elegans.chrI.fasta.gz", "data/truth.elegans.chrI.gff3.gz"),
    "vp": ("data/truth.vp.chr12.fasta.gz", "data/truth.vp.chr12.gff3.gz"),
}
TRAIN_DATASETS = ("arabidopsis", "rice", "tomato", "mouse", "zebrafish", "celegans")
TEST_DATASETS = ("human", "vp")
MOTIF_TABLES = {}
for _group in GROUPS:
    _motif = GROUP_CONFIG[_group][1]
    _codes = LUT[np.frombuffer(_motif.encode("ascii"), dtype=np.uint8)]
    MOTIF_TABLES[_group] = (_codes.reshape(1, 2), np.array([2], dtype=np.int64))


def read_single_fasta(path: str | Path) -> tuple[str, np.ndarray]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    name = None
    sequence = bytearray()
    with opener(path, "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    raise ValueError(f"{path}: expected one FASTA record")
                name = line[1:].split()[0]
            elif name is not None:
                sequence.extend(line.strip().upper().encode("ascii"))
    if name is None or not sequence:
        raise ValueError(f"{path}: missing or empty FASTA record")
    return name, LUT[np.frombuffer(sequence, dtype=np.uint8)]


def genomic_sites(transcripts, seqid: str):
    sites_by_gene = defaultdict(lambda: {group: set() for group in GROUPS})
    gene_bounds: dict[str, tuple[int, int]] = {}

    for transcript in transcripts:
        if transcript.seqid != seqid:
            continue
        spans = transcript.spans.get("CDS")
        if not spans:
            continue

        lo = min(start for start, _ in spans) - 1
        hi = max(end for _, end in spans)
        previous = gene_bounds.get(transcript.gid)
        gene_bounds[transcript.gid] = (
            min(lo, previous[0]) if previous else lo,
            max(hi, previous[1]) if previous else hi,
        )

        for (_, left_end), (right_start, _) in pairwise(spans):
            intron_start = left_end + 1
            intron_end = right_start - 1
            if intron_start > intron_end:
                continue

            if transcript.strand == "+":
                donor_start = intron_start - 1
                acceptor_start = intron_end - 2
                donor_group, acceptor_group = "DSS", "ASS"
            else:
                donor_start = intron_end - 2
                acceptor_start = intron_start - 1
                donor_group, acceptor_group = "RC_DSS", "RC_ASS"
            sites_by_gene[transcript.gid][donor_group].add(donor_start)
            sites_by_gene[transcript.gid][acceptor_group].add(acceptor_start)

    return sites_by_gene, gene_bounds


def feature_coordinates(starts, group: str):
    _, _, strand_code, position_shift = GROUP_CONFIG[group]
    positions = np.asarray(starts, dtype=np.int64) + position_shift
    strands = np.full(len(positions), strand_code, dtype=np.int8)
    return positions, strands


def canonical_group_sites(sequence, starts, group: str):
    motif = MOTIF_TABLES[group][0][0]
    return sorted(
        {
            start
            for start in starts
            if 0 <= start
            and start + 2 <= len(sequence)
            and sequence[start] == motif[0]
            and sequence[start + 1] == motif[1]
        }
    )


def canonical_sites(sequence, sites_by_gene, group: str):
    return sorted(
        {
            start
            for groups in sites_by_gene.values()
            for start in canonical_group_sites(sequence, groups[group], group)
        }
    )


def collect_features(bank, positions_by_group):
    features = {}
    for strand_code in (0, 1):
        groups = [group for group in GROUPS if GROUP_CONFIG[group][2] == strand_code]
        combined_starts = []
        lengths = {}
        for group in groups:
            lengths[group] = len(positions_by_group[group])
            combined_starts.extend(positions_by_group[group])
        if not combined_starts:
            for group in groups:
                features[group] = np.empty((0, FEATURE_CHANNELS), dtype=np.float32)
            continue
        positions, strands = feature_coordinates(combined_starts, groups[0])
        combined_features = cwt_features(bank, positions, strands)
        offset = 0
        for group in groups:
            end = offset + lengths[group]
            features[group] = combined_features[offset:end]
            offset = end
    return features


def extract_dataset_templates(name: str, genes_per_dataset: int, rng):
    fasta, truth = DATASETS[name]
    seqid, sequence = read_single_fasta(fasta)
    transcripts, _ = read_gff3(truth)
    sites_by_gene, _ = genomic_sites(transcripts, seqid)
    gene_positions = {}
    for gene, groups in sites_by_gene.items():
        positions = {
            group: canonical_group_sites(sequence, groups[group], group)
            for group in GROUPS
        }
        if any(positions[group] for group in GROUPS):
            gene_positions[gene] = positions

    eligible_genes = sorted(gene_positions)
    if not eligible_genes:
        raise ValueError(f"{name}: no genes with canonical splice sites")
    selected_genes = rng.choice(
        eligible_genes,
        size=min(genes_per_dataset, len(eligible_genes)),
        replace=False,
    ).tolist()
    bank = CwtFeatureBank(sequence, PERIODS, W0)
    chunks = {group: [] for group in GROUPS}
    for gene in selected_genes:
        features = collect_features(bank, gene_positions[gene])
        for group in GROUPS:
            if len(features[group]):
                chunks[group].append(features[group])

    templates = {
        group: np.concatenate(chunks[group], axis=0).astype(np.float32)
        if chunks[group]
        else np.empty((0, FEATURE_CHANNELS), dtype=np.float32)
        for group in GROUPS
    }
    return templates, len(selected_genes)


def write_template_file(path, templates):
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            DSS=np.asarray(templates["DSS"], dtype=np.float32),
            ASS=np.asarray(templates["ASS"], dtype=np.float32),
            RC_DSS=np.asarray(templates["RC_DSS"], dtype=np.float32),
            RC_ASS=np.asarray(templates["RC_ASS"], dtype=np.float32),
        )


def load_template_file(path):
    with np.load(path, allow_pickle=False) as archive:
        templates = {
            group: np.asarray(archive[group], dtype=np.float32) for group in GROUPS
        }
    for group, matrix in templates.items():
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_CHANNELS or not len(matrix):
            raise ValueError(f"{path}: invalid or empty feature templates for {group}")
    return templates


def scan_candidates(sequence, group: str):
    motifs, lengths = MOTIF_TABLES[group]
    return np.flatnonzero(_motif_mask(sequence, motifs, lengths))


def sample_candidates(starts, truth, rng, max_positives=MAX_TEST_POSITIVES):
    positive = np.asarray(sorted(truth), dtype=np.int64)
    n_positive = min(len(positive), max_positives)
    n_negative = min(len(starts) - len(positive), n_positive)
    if not n_positive:
        return [], 0, 0
    selected_positive = rng.choice(positive, size=n_positive, replace=False)
    selected_negative = []
    seen_negative = set()
    while len(selected_negative) < n_negative:
        remaining = n_negative - len(selected_negative)
        indices = rng.integers(0, len(starts), size=max(32, 2 * remaining))
        for index in indices:
            start = int(starts[index])
            if start not in truth and start not in seen_negative:
                selected_negative.append(start)
                seen_negative.add(start)
                if len(selected_negative) == n_negative:
                    break
    selected = np.concatenate(
        (selected_positive, np.asarray(selected_negative, dtype=np.int64))
    )
    selected.sort()
    return selected.tolist(), n_positive, n_negative


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(
        features,
        norms,
        out=np.zeros_like(features, dtype=np.float32),
        where=norms > 1e-12,
    )


@njit(cache=True)
def nearest_cosine(candidates, templates):
    distances = np.empty(candidates.shape[0], dtype=np.float32)
    for candidate_index in range(candidates.shape[0]):
        best_similarity = -np.inf
        for template_index in range(templates.shape[0]):
            similarity = 0.0
            for channel in range(candidates.shape[1]):
                similarity += (
                    candidates[candidate_index, channel]
                    * templates[template_index, channel]
                )
            best_similarity = max(best_similarity, similarity)
        if best_similarity > 1.0:
            best_similarity = 1.0
        elif best_similarity < -1.0:
            best_similarity = -1.0
        distances[candidate_index] = 1.0 - best_similarity
    return distances


def counts(predicted: set[int], truth: set[int]):
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def result_at_threshold(starts, distances, truth, threshold):
    predicted = {
        start
        for start, distance in zip(starts, distances)
        if float(distance) <= threshold
    }
    return counts(predicted, truth)


def aggregate(results):
    tp = sum(result["tp"] for result in results)
    fp = sum(result["fp"] for result in results)
    fn = sum(result["fn"] for result in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_result(result):
    return (
        f"F1={result['f1']:.3f}  Pr={result['precision']:.3f}  "
        f"Sn={result['recall']:.3f}  "
        f"TP={result['tp']} FP={result['fp']} FN={result['fn']}"
    )


def sn_pr_curve(starts, distances, truth):
    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_starts = np.asarray(starts, dtype=np.int64)[order]
    positive_count = len(set(starts) & truth)
    thresholds = [float("-inf")]
    sensitivities = [0.0]
    precisions = [1.0]
    tp = fp = 0
    index = 0

    while index < len(sorted_distances):
        threshold = float(sorted_distances[index])
        next_index = index
        while (
            next_index < len(sorted_distances)
            and sorted_distances[next_index] == threshold
        ):
            if int(sorted_starts[next_index]) in truth:
                tp += 1
            else:
                fp += 1
            next_index += 1
        thresholds.append(threshold)
        sensitivities.append(tp / positive_count if positive_count else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 1.0)
        index = next_index

    return np.asarray(thresholds), np.asarray(sensitivities), np.asarray(precisions)


def plot_sn_pr(test_results, threshold, output):
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for row, dataset in enumerate(TEST_DATASETS):
        for column, group in enumerate(GROUPS):
            axis = axes[row, column]
            starts, distances, truth = test_results[dataset][group]
            _, sensitivities, precisions = sn_pr_curve(starts, distances, truth)
            axis.plot(precisions, sensitivities, linewidth=1.5)
            predicted = {
                start
                for start, distance in zip(starts, distances)
                if float(distance) <= threshold
            }
            result = counts(predicted, truth)
            axis.scatter(
                [result["precision"]],
                [result["recall"]],
                marker="x",
                s=55,
                label=f"threshold={threshold:.3f}",
            )
            axis.set_title(f"{dataset} {group}")
            axis.set_xlabel("Precision (Pr)")
            axis.set_ylabel("Sensitivity (Sn)")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=200, metadata={})
    plt.close(figure)


def train_templates():
    chunks = {group: [] for group in GROUPS}
    rng = np.random.default_rng(0)
    for dataset in TRAIN_DATASETS:
        print(f"Sampling {GENES_PER_DATASET} genes from {dataset}", flush=True)
        dataset_templates, selected_count = extract_dataset_templates(
            dataset, GENES_PER_DATASET, rng
        )
        print(f"  selected {selected_count} genes", flush=True)
        for group in GROUPS:
            if len(dataset_templates[group]):
                chunks[group].append(dataset_templates[group])
        del dataset_templates

    templates = {
        group: np.concatenate(chunks[group], axis=0).astype(np.float32)
        if chunks[group]
        else np.empty((0, FEATURE_CHANNELS), dtype=np.float32)
        for group in GROUPS
    }
    missing = [group for group in GROUPS if not len(templates[group])]
    if missing:
        raise ValueError("no templates for " + ", ".join(missing))
    write_template_file(TEMPLATE_PATH, templates)
    print("Training template counts:")
    for group in GROUPS:
        print(f"  {group:7s} {len(templates[group]):,}")
    print(f"Template bank written: {TEMPLATE_PATH}")


def test_templates():
    templates = load_template_file(TEMPLATE_PATH)
    template_units = {group: normalize_features(templates[group]) for group in GROUPS}
    print("Training template counts:")
    for group in GROUPS:
        print(f"  {group:7s} {len(templates[group]):,}")

    rng = np.random.default_rng(0)
    test_results = {}
    for dataset in TEST_DATASETS:
        fasta, truth_path = DATASETS[dataset]
        seqid, sequence = read_single_fasta(fasta)
        transcripts, _ = read_gff3(truth_path)
        sites_by_gene, _ = genomic_sites(transcripts, seqid)
        truth_by_group = {
            group: set(canonical_sites(sequence, sites_by_gene, group))
            for group in GROUPS
        }
        sampled_starts = {}
        sampled_truth = {}
        pool_counts = {}
        for group in GROUPS:
            starts = scan_candidates(sequence, group)
            positive_pool = len(truth_by_group[group])
            selected, _, _ = sample_candidates(
                starts, truth_by_group[group], rng, MAX_TEST_POSITIVES
            )
            sampled_starts[group] = selected
            sampled_truth[group] = set(selected) & truth_by_group[group]
            pool_counts[group] = (
                len(starts),
                positive_pool,
                len(starts) - positive_pool,
            )
            del starts

        bank = CwtFeatureBank(sequence, PERIODS, W0)
        candidate_features = collect_features(bank, sampled_starts)
        test_results[dataset] = {}
        class_results = []
        print(
            f"\n{dataset} ({seqid}, {len(sequence):,} bp), threshold={MATCH_THRESHOLD:.3f}"
        )
        for group in GROUPS:
            starts = sampled_starts[group]
            truth = sampled_truth[group]
            distances = nearest_cosine(
                normalize_features(candidate_features[group]),
                template_units[group],
            )
            result = result_at_threshold(starts, distances, truth, MATCH_THRESHOLD)
            baseline = counts(set(starts), truth)
            full_candidates, full_positive, full_negative = pool_counts[group]
            print(
                f"  {group:7s} pool={full_candidates:,} "
                f"sample={len(starts):,} ({len(truth):,}+{len(starts) - len(truth):,}) "
                f"annotated_pool={full_positive:,} negatives_pool={full_negative:,}"
            )
            print(f"    motif-only: {format_result(baseline)}")
            print(f"    CWT match : {format_result(result)}")
            test_results[dataset][group] = (starts, distances, truth)
            class_results.append(result)

        print(f"  overall:    {format_result(aggregate(class_results))}")
        del candidate_features, bank, transcripts, sites_by_gene, sequence

    plot_sn_pr(test_results, MATCH_THRESHOLD, PLOT_PATH)
    print(f"Sn-Pr curve saved: {PLOT_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.train:
            train_templates()
        else:
            test_templates()
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
