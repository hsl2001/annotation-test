from __future__ import annotations

import argparse

import numpy as np

import main
from model import Model


def load(fasta: str, annotation: str):
    codes = main.read_fasta(fasta)
    lengths = {name: len(sequence) for name, sequence in codes.items()}
    transcripts = main.read_annotation(annotation)
    labels = main.site_labels(transcripts, lengths)
    return codes, labels


def build(codes, labels):
    X_blocks = []
    Y_blocks = []
    for name, sequence in codes.items():
        for strand in main.STRANDS:
            X_blocks.append(main.features(main.oriented(sequence, strand)))
            Y_blocks.append(main.label_matrix(labels, name, strand, len(sequence)))
    return np.concatenate(X_blocks, axis=1), np.concatenate(Y_blocks, axis=1)


def call(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.flatnonzero(main.local_max(score) & (score > threshold))


def match(predicted: np.ndarray, truth: np.ndarray, tolerance: int):
    used = np.zeros(len(predicted), dtype=bool)
    true_positive = 0
    for position in truth:
        low = np.searchsorted(predicted, position - tolerance)
        high = np.searchsorted(predicted, position + tolerance, side="right")
        for index in range(low, high):
            if not used[index]:
                used[index] = True
                true_positive += 1
                break
    return true_positive, len(predicted) - true_positive, len(truth) - true_positive


def evaluate(model: Model, codes, labels, tolerance: int):
    assert model.thresholds is not None
    totals = {site: [0, 0, 0] for site in main.SITES}
    for name, sequence in codes.items():
        for strand in main.STRANDS:
            scores = model.predict(main.features(main.oriented(sequence, strand)))
            for index, site in enumerate(main.SITES):
                predicted = call(scores[index], model.thresholds[index])
                truth = labels.get((name, strand, site), np.array([], dtype=np.int64))
                tp, fp, fn = match(predicted, truth, tolerance)
                totals[site][0] += tp
                totals[site][1] += fp
                totals[site][2] += fn
    return totals


def report(totals) -> None:
    print(
        f"{'site':6s} {'precision':>10s} {'recall':>10s} {'f1':>10s} "
        f"{'tp':>7s} {'fp':>7s} {'fn':>7s}"
    )
    macro = [0.0, 0.0, 0.0]
    for site in main.SITES:
        tp, fp, fn = totals[site]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        macro[0] += precision
        macro[1] += recall
        macro[2] += f1
        print(
            f"{site:6s} {precision:10.4f} {recall:10.4f} {f1:10.4f} "
            f"{tp:7d} {fp:7d} {fn:7d}"
        )
    count = len(main.SITES)
    print(
        f"{'macro':6s} {macro[0] / count:10.4f} {macro[1] / count:10.4f} "
        f"{macro[2] / count:10.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predict-fasta")
    parser.add_argument("--predict-annotation")
    parser.add_argument("--tolerance", type=int, default=2)
    return parser


def main_entry(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    codes, labels = load(args.fasta, args.annotation)
    X, Y = build(codes, labels)
    model = Model().fit(X, Y)

    if args.predict_fasta and args.predict_annotation:
        test_codes, test_labels = load(args.predict_fasta, args.predict_annotation)
    else:
        test_codes, test_labels = codes, labels

    totals = evaluate(model, test_codes, test_labels, args.tolerance)
    report(totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_entry())
