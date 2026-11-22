#!/usr/bin/env python3
import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score

import umap_plot as u

KINDS = ("GT", "AC", "AG", "CT")
MOTIFS = {kind: (kind.encode("ascii"),) for kind in KINDS}
TABLES = {}
for _kind, _motifs in MOTIFS.items():
    _table = np.zeros((len(_motifs), max(map(len, _motifs))), dtype=np.uint8)
    _lengths = np.zeros(len(_motifs), dtype=np.int64)
    for _row, _motif in enumerate(_motifs):
        _table[_row, : len(_motif)] = u.LUT[np.frombuffer(_motif, dtype=np.uint8)]
        _lengths[_row] = len(_motif)
    TABLES[_kind] = (_table, _lengths)


def splice_sites(transcripts, name, length):
    masks = {kind: np.zeros(length, dtype=bool) for kind in KINDS}
    for seqid, strand, spans in transcripts:
        if seqid != name:
            continue
        # Spans are genomic; a minus-strand donor reads AC on the forward strand,
        # so its donor site sits at the intron's right end instead of the left.
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


def build_candidates(sequence, masks):
    out = [[], [], [], []]
    for kind_index, kind in enumerate(KINDS):
        table, lengths = TABLES[kind]
        found = np.flatnonzero(u._motif_mask(sequence, table, lengths))
        if not len(found):
            continue
        labels = masks[kind][found].astype(np.int8)
        out[0].append(found.astype(np.int64))
        out[1].append(np.zeros(len(found), dtype=np.int8))
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


def plot(fasta, truth, output, max_positives=2000):
    codes = u.read_fasta(fasta)
    name = next(iter(codes))
    sequence = codes[name]
    masks = splice_sites(u.read_annotation(truth), name, len(sequence))
    positions, strands, kinds, labels = build_candidates(sequence, masks)
    bank = u.CwtFeatureBank(sequence, u.PERIODS, u.W0)
    index = sample_candidates(
        kinds, labels, np.random.default_rng(u.SEED), max_positives
    )
    features = u.cwt_features(bank, positions[index], strands[index])
    labels = labels[index]
    kinds = kinds[index]
    columns = 4
    n_rows = (len(KINDS) + columns - 1) // columns
    figure, axes = plt.subplots(n_rows, columns, figsize=(6 * columns, 6 * n_rows))
    scores = []
    for ax, kind in zip(axes.ravel(), KINDS):
        mask = kinds == KINDS.index(kind)
        if not mask.any():
            ax.axis("off")
            continue
        n = len(features[mask])
        y = labels[mask]
        if n < 3:
            ax.axis("off")
            scores.append(
                {
                    "kind": kind,
                    "n": n,
                    "positive": int(y.sum()),
                    "silhouette": float("nan"),
                }
            )
            continue
        embedding = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, n - 1)),
            random_state=u.SEED,
            n_jobs=1,
            compatibility_layout=True,
        ).fit_transform(features[mask])
        score = float("nan")
        if len(np.unique(y)) > 1:
            score = float(silhouette_score(embedding, y))
        scores.append(
            {
                "kind": kind,
                "n": n,
                "positive": int(y.sum()),
                "silhouette": score,
            }
        )
        for value, color, marker, label in (
            (0, "#1f77b4", "o", "negative"),
            (1, "#d62728", "^", "positive"),
        ):
            selected = labels[mask] == value
            if selected.any():
                ax.scatter(
                    embedding[selected, 0],
                    embedding[selected, 1],
                    c=color,
                    marker=marker,
                    s=8,
                    alpha=0.7,
                )
        ax.set_title(kind)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(KINDS) :]:
        ax.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="negative"),
        Line2D([], [], marker="^", linestyle="", color="#d62728", label="positive"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"wrote {output}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-positives", type=int, default=2000)
    args = parser.parse_args()
    plot(
        args.fasta,
        args.truth,
        str(Path(args.output).with_suffix(".png")),
        args.max_positives,
    )


if __name__ == "__main__":
    main()
