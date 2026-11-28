#!/usr/bin/env python3
import argparse
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score

from scrim.cwt_features import (
    COMPLEMENT as COMPLEMENT,
    LUT,
    MAPPINGS as MAPPINGS,
    MORLET_LIMIT as MORLET_LIMIT,
    NODES as NODES,
    PERIODS,
    ROOTS as ROOTS,
    W0,
    WEIGHTS as WEIGHTS,
    CwtFeatureBank,
    _motif_mask,
    build_kernel as build_kernel,
    cwt_features,
    morlet_integral as morlet_integral,
    open_text,
    read_fasta,
)

KINDS = ("GT", "AC", "AG", "CT")
MOTIFS = {kind: (kind.encode("ascii"),) for kind in KINDS}
SEED = 0
MAX_POSITIVES = 2000

TABLES = {}
for _kind, _motifs in MOTIFS.items():
    _table = np.zeros((len(_motifs), max(map(len, _motifs))), dtype=np.uint8)
    _lengths = np.zeros(len(_motifs), dtype=np.int64)
    for _row, _motif in enumerate(_motifs):
        _table[_row, : len(_motif)] = LUT[np.frombuffer(_motif, dtype=np.uint8)]
        _lengths[_row] = len(_motif)
    TABLES[_kind] = (_table, _lengths)

DATASETS = (
    (
        "arabidopsis",
        "data/truth.arabidopsis.chr5.fasta.gz",
        "data/truth.arabidopsis.chr5.gff3.gz",
    ),
    ("rice", "data/truth.rice.chr5.fasta.gz", "data/truth.rice.chr5.gff3.gz"),
    ("tomato", "data/truth.tomato.chr5.fasta.gz", "data/truth.tomato.chr5.gff3.gz"),
    ("human", "data/truth.human.chr21.fasta.gz", "data/truth.human.chr21.gff3.gz"),
    ("mouse", "data/truth.mouse.chr19.fasta.gz", "data/truth.mouse.chr19.gff3.gz"),
    ("zebrafish", "data/truth.zebrafish.chr25.fasta.gz", "data/truth.zebrafish.chr25.gff3.gz"),
    ("celegans", "data/truth.elegans.chrI.fasta.gz", "data/truth.elegans.chrI.gff3.gz"),
)



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


def splice_sites(transcripts, name, length):
    masks = {kind: np.zeros(length, dtype=bool) for kind in KINDS}
    for seqid, strand, spans in transcripts:
        if seqid != name:
            continue
        for (_, left), (right, _) in pairwise(spans):
            if left >= right:
                continue
            donor, acceptor = ("GT", "AG") if strand == "+" else ("AC", "CT")
            donor_position = left if strand == "+" else right - 2
            acceptor_position = right - 2 if strand == "+" else left
            if 0 <= donor_position < length:
                masks[donor][donor_position] = True
            if 0 <= acceptor_position < length:
                masks[acceptor][acceptor_position] = True
    return masks



def build_candidates(sequence, masks=None):
    out = [[], [], [], []]
    for kind_index, kind in enumerate(KINDS):
        table, lengths = TABLES[kind]
        found = np.flatnonzero(_motif_mask(sequence, table, lengths))
        if not len(found):
            continue
        labels = (
            masks[kind][found].astype(np.int8)
            if masks is not None
            else np.zeros(len(found), dtype=np.int8)
        )
        strand = 0 if kind in ("GT", "AG") else 1
        if strand:
            found = found + (max(map(len, MOTIFS[kind])) - 1)
        out[0].append(found.astype(np.int64))
        out[1].append(np.full(len(found), strand, dtype=np.int8))
        out[2].append(np.full(len(found), kind_index, dtype=np.int8))
        out[3].append(labels)
    if not out[0]:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.astype(np.int8), empty.astype(np.int8), empty.astype(np.int8)
    order = np.lexsort((np.concatenate(out[1]), np.concatenate(out[0])))
    return tuple(np.concatenate(x)[order] for x in out)


def sample_candidates(kinds, labels, rng, max_positives):
    selected = []
    for kind_index, kind in enumerate(KINDS):
        positive = np.flatnonzero((kinds == kind_index) & (labels == 1))
        negative = np.flatnonzero((kinds == kind_index) & (labels == 0))
        if not len(positive):
            continue
        n_positive = min(len(positive), max_positives)
        n_negative = min(len(negative), n_positive)
        selected += [
            rng.choice(positive, size=n_positive, replace=False),
            rng.choice(negative, size=n_negative, replace=False),
        ]
    return np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)



def load(fasta, truth, max_positives):
    codes = read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    masks = splice_sites(read_annotation(truth), name, len(sequence))
    positions, strands, kinds, labels = build_candidates(sequence, masks)
    bank = CwtFeatureBank(sequence, PERIODS, W0)
    index = sample_candidates(
        kinds, labels, np.random.default_rng(SEED), max_positives
    )
    features = cwt_features(bank, positions[index], strands[index])
    return features, labels[index], kinds[index]


def plot(output, max_positives=MAX_POSITIVES):
    figure, axes = plt.subplots(
        len(DATASETS), len(KINDS), figsize=(6 * len(KINDS), 6 * len(DATASETS))
    )
    for row, (label, fasta, truth) in enumerate(DATASETS):
        features, labels, kinds = load(fasta, truth, max_positives)
        for column, kind in enumerate(KINDS):
            ax = axes[row, column]
            mask = kinds == column
            y = labels[mask]
            if mask.sum() < 3 or len(np.unique(y)) < 2:
                ax.axis("off")
                continue
            embedding = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, int(mask.sum()) - 1)),
                random_state=SEED,
                n_jobs=1,
                compatibility_layout=True,
            ).fit_transform(features[mask])
            score = float(silhouette_score(embedding, y))
            for value, color, marker in ((0, "#1f77b4", "o"), (1, "#d62728", "^")):
                selected = y == value
                if selected.any():
                    ax.scatter(
                        embedding[selected, 0],
                        embedding[selected, 1],
                        c=color,
                        marker=marker,
                        s=8,
                        alpha=0.7,
                    )
            ax.set_title(f"{label} {kind}  sil={score:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="negative"),
        Line2D([], [], marker="^", linestyle="", color="#d62728", label="positive"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.03, 1, 0.99))
    figure.savefig(str(Path(output).with_suffix(".png")), dpi=330)
    plt.close(figure)
    print(f"wrote {Path(output).with_suffix('.png')}")


def export_csv(path, max_positives=MAX_POSITIVES):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("dataset\tkind\tx\ty\tlabel\n")
        for label, fasta, truth in DATASETS:
            features, labels, kinds = load(fasta, truth, max_positives)
            if len(features) < 3:
                continue
            embedding = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, len(features) - 1)),
                random_state=SEED,
                n_jobs=1,
                compatibility_layout=True,
            ).fit_transform(features)
            for (x, y), kind, value in zip(embedding, kinds, labels):
                handle.write(f"{label}\t{KINDS[kind]}\t{x:.6g}\t{y:.6g}\t{int(value)}\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="umap_chr5")
    parser.add_argument("--max-positives", type=int, default=MAX_POSITIVES)
    parser.add_argument("--csv", default=None, help="export the joint UMAP embedding instead of plotting")
    args = parser.parse_args()
    if args.csv is not None:
        export_csv(args.csv, args.max_positives)
    else:
        plot(args.output, args.max_positives)


if __name__ == "__main__":
    main()
