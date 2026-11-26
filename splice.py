from __future__ import annotations

import argparse
import gzip
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from accuracy import read_gff3
from cwt_features import (
    LUT,
    NUC_CHANNELS,
    PERIODS,
    W0,
    CwtFeatureBank,
    _motif_mask,
    cwt_features,
)

GROUPS = ("DSS", "ASS", "RC_DSS", "RC_ASS")
GROUP_CONFIG = {
    "DSS": ("+", "GT", 0, 0),
    "ASS": ("+", "AG", 0, 0),
    "RC_DSS": ("-", "AC", 1, 1),
    "RC_ASS": ("-", "CT", 1, 1),
}
FEATURE_CHANNELS = 12 * len(PERIODS) + NUC_CHANNELS
GENES_PER_DATASET = 200
MATCH_THRESHOLD = 0.3
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

MEAN_TEMPLATES = {
    "DSS": np.array([
        -0.02015858, 0.20301543, 0.10183430, 0.03576760, 0.23699326, -0.05224745,
        0.17255813, -0.11668257, 0.26057319, -0.25184153, 0.09332536, -0.12984864,
        0.02205124, -0.01772014, 0.20856216, -0.27800688, 0.19093773, -0.25585992,
        0.20649622, -0.24030143, 0.18434927, -0.22267700, -0.07593747, -0.03616608,
        0.00000000, 0.00000000, 0.52250534, 0.00000000,
    ], dtype=np.float32),
    "ASS": np.array([
        0.25847275, 0.21973603, 0.24298485, 0.26404051, 0.18648045, 0.24627284,
        -0.06075848, -0.00096609, -0.04299080, 0.05553830, 0.00131368, 0.04005040,
        0.32667070, 0.32417963, 0.06422902, 0.32646119, -0.08949851, 0.17522473,
        -0.20593880, 0.05878445, -0.05470234, 0.21251198, -0.05242078, -0.04992970,
        0.43590076, 0.00000000, 0.00000000, 0.00000000,
    ], dtype=np.float32),
    "RC_DSS": np.array([
        -0.01638368, 0.20514241, 0.10666255, 0.03964332, 0.23888384, -0.04966148,
        0.16966113, -0.11888419, 0.25896594, -0.25110548, 0.09346684, -0.12805925,
        0.01489289, -0.02312330, 0.19958161, -0.28314254, 0.19011473, -0.25459323,
        0.20408075, -0.24062721, 0.17553144, -0.23116033, -0.08448780, -0.04647160,
        0.00000000, 0.00000000, 0.52171076, 0.00000000,
    ], dtype=np.float32),
    "RC_ASS": np.array([
        0.26128365, 0.21938377, 0.24132898, 0.26230556, 0.18641439, 0.24929084,
        -0.05884708, 0.00402938, -0.04583236, 0.05894397, -0.00291057, 0.03898931,
        0.32618959, 0.32842114, 0.06629172, 0.32643693, -0.08258182, 0.17533184,
        -0.19629761, 0.06161605, -0.04519251, 0.21048959, -0.04717672, -0.04940827,
        0.43926931, 0.00000000, 0.00000000, 0.00000000,
    ], dtype=np.float32),
}


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
    for transcript in transcripts:
        if transcript.seqid != seqid:
            continue
        spans = transcript.spans.get("CDS")
        if not spans:
            continue
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

    return sites_by_gene


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
    sites_by_gene = genomic_sites(transcripts, seqid)
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



def scan_candidates(sequence, group: str):
    motifs, lengths = MOTIF_TABLES[group]
    return np.flatnonzero(_motif_mask(sequence, motifs, lengths))


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(
        features,
        norms,
        out=np.zeros_like(features, dtype=np.float32),
        where=norms > 1e-12,
    )


def mean_template_distance(features, group):
    distances = 1.0 - features @ MEAN_TEMPLATES[group]
    np.clip(distances, 0.0, 2.0, out=distances)
    return distances


def score_candidates(bank, starts, group, chunk_size=100_000):
    starts = np.asarray(starts, dtype=np.int64)
    distances = np.empty(len(starts), dtype=np.float32)
    _, _, strand_code, position_shift = GROUP_CONFIG[group]
    strands = np.full(min(chunk_size, len(starts)), strand_code, dtype=np.int8)
    for offset in range(0, len(starts), chunk_size):
        end = min(offset + chunk_size, len(starts))
        positions = starts[offset:end] + position_shift
        features = bank.gather(positions, strands[: end - offset])
        distances[offset:end] = mean_template_distance(
            normalize_features(features), group
        )
    return distances


def _metrics(tp, fp, fn):
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
    starts = np.asarray(starts, dtype=np.int64)
    truth_positions = np.fromiter(truth, dtype=np.int64, count=len(truth))
    is_truth = np.isin(starts, truth_positions, assume_unique=True)
    predicted = distances <= threshold
    tp = int(np.count_nonzero(predicted & is_truth))
    fp = int(np.count_nonzero(predicted)) - tp
    return _metrics(tp, fp, len(truth) - tp)


def aggregate(results):
    return _metrics(
        sum(result["tp"] for result in results),
        sum(result["fp"] for result in results),
        sum(result["fn"] for result in results),
    )


def format_result(result):
    return (
        f"F1={result['f1']:.3f}  Pr={result['precision']:.3f}  "
        f"Sn={result['recall']:.3f}  "
        f"TP={result['tp']} FP={result['fp']} FN={result['fn']}"
    )


def sn_pr_curve(starts, distances, truth):
    if not len(distances):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_starts = np.asarray(starts, dtype=np.int64)[order]
    truth_positions = np.fromiter(truth, dtype=np.int64, count=len(truth))
    sorted_truth = np.isin(sorted_starts, truth_positions, assume_unique=True)
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)
    group_ends = np.r_[
        np.flatnonzero(sorted_distances[1:] != sorted_distances[:-1]),
        len(sorted_distances) - 1,
    ]
    tp = cumulative_tp[group_ends]
    predicted = group_ends + 1
    thresholds = np.r_[float("-inf"), sorted_distances[group_ends]]
    sensitivities = np.r_[0.0, tp / len(truth) if truth else np.zeros(len(tp))]
    precisions = np.r_[1.0, tp / predicted]
    return thresholds, sensitivities, precisions


def plot_sn_pr(test_results, threshold, output):
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for row, dataset in enumerate(TEST_DATASETS):
        for column, group in enumerate(GROUPS):
            axis = axes[row, column]
            starts, distances, truth = test_results[dataset][group]
            _, sensitivities, precisions = sn_pr_curve(starts, distances, truth)
            axis.plot(precisions, sensitivities, linewidth=1.5)
            result = result_at_threshold(starts, distances, truth, threshold)
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


def _format_mean_templates(means):
    lines = ["MEAN_TEMPLATES = {"]
    for group in GROUPS:
        vector = means[group]
        lines.append(f'    "{group}": np.array([')
        for start in range(0, len(vector), 6):
            chunk = ", ".join(f"{value:.8f}" for value in vector[start : start + 6])
            lines.append(f"        {chunk},")
        lines.append("    ], dtype=np.float32),")
    lines.append("}")
    return "\n".join(lines)


def update_mean_templates(templates):
    means = {}
    for group in GROUPS:
        mean = templates[group].astype(np.float64).mean(axis=0)
        mean /= np.linalg.norm(mean)
        means[group] = mean

    source_path = Path(__file__)
    text = source_path.read_text(encoding="utf-8")
    pattern = re.compile(r"MEAN_TEMPLATES = \{.*?\n\}", re.DOTALL)
    if not pattern.search(text):
        raise ValueError("could not locate MEAN_TEMPLATES block to update")
    text = pattern.sub(lambda _match: _format_mean_templates(means), text, count=1)
    source_path.write_text(text, encoding="utf-8")

    for group in GROUPS:
        MEAN_TEMPLATES[group] = means[group].astype(np.float32)
    return means


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

    means = update_mean_templates(templates)
    print("Updated MEAN_TEMPLATES in splice.py:")
    for group in GROUPS:
        print(f"  {group:7s} |mean|={np.linalg.norm(means[group]):.4f}")

def test_templates():
    print("Using hardcoded mean templates:")
    for group in GROUPS:
        print(f"  {group:7s} dim={len(MEAN_TEMPLATES[group])}")

    test_results = {}
    for dataset in TEST_DATASETS:
        fasta, truth_path = DATASETS[dataset]
        seqid, sequence = read_single_fasta(fasta)
        transcripts, _ = read_gff3(truth_path)
        sites_by_gene = genomic_sites(transcripts, seqid)
        truth_by_group = {
            group: set(canonical_sites(sequence, sites_by_gene, group))
            for group in GROUPS
        }
        bank = CwtFeatureBank(sequence, PERIODS, W0)
        test_results[dataset] = {}
        class_results = []
        print(
            f"\n{dataset} ({seqid}, {len(sequence):,} bp), "
            f"all motif candidates, mean template, threshold={MATCH_THRESHOLD:.3f}",
            flush=True,
        )
        for group in GROUPS:
            starts = scan_candidates(sequence, group)
            truth = truth_by_group[group]
            distances = score_candidates(bank, starts, group)
            result = result_at_threshold(starts, distances, truth, MATCH_THRESHOLD)
            predicted_count = result["tp"] + result["fp"]
            print(
                f"  {group:7s} candidates={len(starts):,} "
                f"annotated={len(truth):,} CWT-predicted={predicted_count:,}",
                flush=True,
            )
            print(f"    CWT match : {format_result(result)}")
            test_results[dataset][group] = (starts, distances, truth)
            class_results.append(result)
            del starts, distances

        print(f"  overall:    {format_result(aggregate(class_results))}", flush=True)
        del bank, transcripts, sites_by_gene, sequence

    plot_sn_pr(test_results, MATCH_THRESHOLD, PLOT_PATH)
    print(f"Full-candidate Sn-Pr curve saved: {PLOT_PATH}")


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
