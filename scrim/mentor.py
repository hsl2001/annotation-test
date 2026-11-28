from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import replace
from math import sqrt
from urllib.parse import unquote

import numpy as np
from numba import njit

from .critic import ACCEPTOR, DONOR, SiteExample, fit_reviewer
from .student import (
    ENTRY_COST,
    N_CONTEXTS,
    ORDER,
    Gene,
    Genome,
    Model,
    advance_context,
    junctions,
    oriented_cds,
    predict,
    reverse_spans,
)

UPDATES = 5
ANCHOR_WEIGHT = 0.5


@njit(cache=True)
def _count(sequence, runs, phases):
    result = np.zeros((phases, N_CONTEXTS, 4), dtype=np.float64)
    for start, end, phase, weight in runs:
        phase = int(phase)
        context = known = 0
        for p in range(int(start), int(end)):
            base = sequence[p]
            if base < 4:
                if known >= ORDER:
                    result[phase, context, base] += weight
                context, known = advance_context(context, known, base)
            else:
                context = known = 0
            phase = (phase + 1) % phases
    return result


def counts(strand, runs, phases=1):
    result = _count(
        strand.bases, np.asarray(runs, dtype=np.float64).reshape(-1, 4), phases
    )
    return result if phases > 1 else result[0]


def _probabilities(count, anchor=None):
    if anchor is not None:
        count = count + 4 * anchor
    shape = count.shape[:-2]
    probs = np.full((*shape, 1, 4), 0.25)
    for order in range(ORDER + 1):
        contexts = 4**order
        pooled = count.reshape(*shape, -1, contexts, 4).sum(axis=-3)
        backoff = np.tile(probs, (*([1] * len(shape)), contexts // probs.shape[-2], 1))
        pooled += 4 * backoff
        probs = pooled / pooled.sum(axis=-1, keepdims=True)
    return (
        probs
        if anchor is None
        else ANCHOR_WEIGHT * anchor + (1 - ANCHOR_WEIGHT) * probs
    )



def seed_predictions(genome: Genome, path: str) -> dict[str, list[Gene]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    qualities: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9 or fields[2].lower() != "cds":
                continue
            seqid, strand = fields[0], fields[6]
            if strand not in ("+", "-"):
                raise ValueError(f"{path}: invalid strand {strand!r}")
            first, last = int(fields[3]), int(fields[4])
            if not 1 <= first <= last:
                raise ValueError(f"{path}: invalid interval {first}-{last}")
            attributes = {}
            for part in fields[8].split(";"):
                name, separator, value = part.strip().partition("=")
                if separator and name == "Parent":
                    attributes[name] = unquote(value)
            parents = [p for p in attributes.get("Parent", "").split(",") if p]
            if not parents:
                raise ValueError(f"{path}: CDS lacks Parent=")
            for parent in parents:
                key = (seqid, strand, parent)
                groups[key].append((first, last))
                if fields[5] not in (".", ""):
                    qualities[key].append(float(fields[5]))
    result: dict[str, list[Gene]] = {name: [] for name in genome}
    for key, spans in groups.items():
        seqid, strand, _ = key
        if seqid not in result:
            continue
        spans.sort()
        cds = tuple((first - 1, last) for first, last in spans)
        values = qualities[key]
        score = ENTRY_COST * sum(values) / len(values) if values else ENTRY_COST
        result[seqid].append(Gene(cds, strand, score))
    return result


def from_seed(genome: Genome, predictions: dict[str, list[Gene]]) -> Model:
    background = np.zeros((N_CONTEXTS, 4))
    coding = np.zeros((3, N_CONTEXTS, 4))
    lengths: list[int] = []
    for name, strands in genome.items():
        length = len(strands[0].bases)
        gaps = list(_gaps(predictions[name], length))
        reverse_gaps = [
            (a, b, phase, weight)
            for (a, b), (_, _, phase, weight) in zip(
                reverse_spans([(a, b) for a, b, _, _ in gaps], length), reversed(gaps)
            )
        ]
        for strand, record in enumerate((strands[0], strands[1])):
            background += counts(record, gaps if strand == 0 else reverse_gaps)
            runs = []
            for gene in predictions[name]:
                if (gene.strand == "-") != bool(strand):
                    continue
                cds = oriented_cds(gene, length)
                phase = 0
                for index, (a, b) in enumerate(cds):
                    end = b if index < len(cds) - 1 else b - 3
                    runs.append((a, end, phase, 1.0))
                    if index:
                        lengths.append(a - cds[index - 1][1])
                    phase = (phase + end - a) % 3
            coding += counts(record, runs, 3)
    background = _probabilities(background)
    prior = np.broadcast_to(background, coding.shape)
    coding = _probabilities(coding, prior) if coding.any() else prior
    anchor = Model(background, coding)
    mean, phase_prob = _duration(lengths, np.ones(len(lengths)), anchor)
    return Model(background, coding, mean, phase_prob)


def _gaps(predictions, length):
    cursor = 0
    for gene in predictions:
        start, end = gene.cds[0][0], gene.cds[-1][1]
        if cursor < start:
            yield cursor, start, 0, 1.0
        yield start, end, 0, ENTRY_COST / (gene.score + ENTRY_COST)
        cursor = end
    if cursor < length:
        yield cursor, length, 0, 1.0


def _duration(lengths, confidence, anchor):
    if not lengths:
        return anchor.intron_mean, anchor.phases
    lengths = np.asarray(lengths, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    strength = sqrt(len(lengths))
    mean = (confidence * lengths).sum() + strength * anchor.intron_mean[0]
    mean /= confidence.sum() + strength
    return np.array([mean]), anchor.phases


def _training_batch(genome, predictions):
    sequences = {
        name: (forward.bases, reverse.bases)
        for name, (forward, reverse) in genome.items()
    }
    sites, genes = [], []
    for name, predictions_on_contig in predictions.items():
        length = len(sequences[name][0])
        for gene in predictions_on_contig:
            strand = int(gene.strand == "-")
            cds = oriented_cds(gene, length)
            first = len(sites)
            for donor, acceptor in junctions(cds):
                sites.extend(
                    (
                        SiteExample(name, strand, DONOR, donor),
                        SiteExample(name, strand, ACCEPTOR, acceptor - 2),
                    )
                )
            genes.append((name, strand, cds, gene.score, slice(first, len(sites))))
    return sequences, sites, genes


def refit(genome, predictions, anchor):
    _, _, genes = _training_batch(genome, predictions)
    exon_runs = {(name, strand): [] for name in genome for strand in range(2)}
    lengths, confidence = [], []
    for name, strand, cds, score, selection in genes:
        weights = np.ones((selection.stop - selection.start) // 2, dtype=np.float64)
        trust = score / (score + ENTRY_COST)
        phase = 0
        for i, (a, b) in enumerate(cds):
            left = weights[i - 1] if i else 1.0
            right = weights[i] if i < len(weights) else 1.0
            end = b if i < len(weights) else b - 3
            exon_runs[name, strand].append((a, end, phase, trust * min(left, right)))
            if i:
                donor = cds[i - 1][1]
                lengths.append(a - donor)
                confidence.append(trust * left)
            phase = (phase + b - a) % 3

    background = np.zeros_like(anchor.background)
    coding = np.zeros_like(anchor.coding)
    for name, (forward, reverse) in genome.items():
        length = len(forward.bases)
        gaps = list(_gaps(predictions[name], length))
        reverse_gaps = [
            (a, b, phase, weight)
            for (a, b), (_, _, phase, weight) in zip(
                reverse_spans([(a, b) for a, b, _, _ in gaps], length), reversed(gaps)
            )
        ]
        for strand, record in enumerate((forward, reverse)):
            background += counts(record, gaps if strand == 0 else reverse_gaps)
            coding += counts(record, exon_runs[name, strand], 3)
    background = _probabilities(background, anchor.background)
    coding = _probabilities(coding, anchor.coding) if coding.any() else anchor.coding
    mean, phase_prob = _duration(lengths, confidence, anchor)
    return Model(background, coding, mean, phase_prob)


def _fit_models(genome: Genome, initial: dict[str, list[Gene]]):
    anchor = model = from_seed(genome, initial)
    blind = initial
    for round_index in range(UPDATES + 1):
        sequences, sites, _ = _training_batch(genome, blind)
        reviewer = fit_reviewer(sequences, sites)
        if round_index:
            model = refit(genome, blind, anchor)
        model = replace(model, critic=reviewer)
        blind = predict(genome, model)
        yield model, blind


def fit(genome: Genome, seed_path: str):
    initial = seed_predictions(genome, seed_path)
    rounds = _fit_models(genome, initial)
    model, predictions = next(rounds)
    for model, predictions in rounds:
        pass
    return model, predictions
