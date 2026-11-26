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
from cwt_features import LUT, PERIODS, W0, CwtFeatureBank, _motif_mask, cwt_features

GROUPS = ("DSS", "ASS", "RC_DSS", "RC_ASS")
GROUP_CONFIG = {
    "DSS": ("+", "GT", 0, 0),
    "ASS": ("+", "AG", 0, 0),
    "RC_DSS": ("-", "AC", 1, 1),
    "RC_ASS": ("-", "CT", 1, 1),
}
FEATURE_CHANNELS = 12 * len(PERIODS)
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
        -0.02364266, 0.23810327, 0.11943467, 0.04194944, 0.27795361, -0.06127755,
        0.20238194, -0.13684922, 0.30560894, -0.29536816, 0.10945510, -0.15229083,
        0.02586243, -0.02078278, 0.24460867, -0.32605575, 0.22393815, -0.30008106,
        0.24218566, -0.28183355, 0.21621098, -0.26116303, -0.08906200, -0.04241679,
    ], dtype=np.float32),
    "ASS": np.array([
        0.28719362, 0.24415257, 0.26998474, 0.29338006, 0.20720171, 0.27363809,
        -0.06750981, -0.00107344, -0.04776784, 0.06170959, 0.00145965, 0.04450070,
        0.36296957, 0.36020169, 0.07136600, 0.36273678, -0.09944337, 0.19469529,
        -0.22882222, 0.06531644, -0.06078073, 0.23612580, -0.05824565, -0.05547777,
    ], dtype=np.float32),
    "RC_DSS": np.array([
        -0.01920437, 0.24046069, 0.12502608, 0.04646850, 0.28001120, -0.05821144,
        0.19887079, -0.13935185, 0.30355073, -0.29433698, 0.10955854, -0.15010653,
        0.01745692, -0.02710432, 0.23394252, -0.33188969, 0.22284578, -0.29842519,
        0.23921626, -0.28205472, 0.20575176, -0.27095797, -0.09903361, -0.05447237,
    ], dtype=np.float32),
    "RC_ASS": np.array([
        0.29084663, 0.24420598, 0.26863419, 0.29198416, 0.20750628, 0.27749690,
        -0.06550534, 0.00448528, -0.05101807, 0.06561320, -0.00323989, 0.04340076,
        0.36309636, 0.36558041, 0.07379231, 0.36337169, -0.09192556, 0.19516978,
        -0.21850773, 0.06858761, -0.05030582, 0.23430547, -0.05251454, -0.05499858,
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
    starts = np.asarray(starts, dtype=np.int64)
    truth_positions = np.fromiter(truth, dtype=np.int64, count=len(truth))
    is_truth = np.isin(starts, truth_positions, assume_unique=True)
    predicted = distances <= threshold
    tp = int(np.count_nonzero(predicted & is_truth))
    fp = int(np.count_nonzero(predicted)) - tp
    fn = len(truth) - tp
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
        sites_by_gene, _ = genomic_sites(transcripts, seqid)
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
