from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np
from numba import njit
from scipy.special import softmax

from .critic import ACCEPTOR, DONOR, SiteExample, fit_reviewer
from .student import (
    EMPTY_MASK,
    ENTRY_COST,
    INTRON_DELAY,
    N_CONTEXTS,
    ORDER,
    Genome,
    Model,
    advance_context,
    intron_parameters,
    is_start,
    is_stop,
    junctions,
    oriented_cds,
    predict,
    reverse_spans,
)

UPDATES = 5


@njit(cache=True)
def _count(sequence, ignored, runs, phases):
    result = np.zeros((phases, N_CONTEXTS, 4), dtype=np.float64)
    for start, end, phase, weight in runs:
        phase = int(phase)
        context = known = 0
        for p in range(int(start), int(end)):
            base = sequence[p]
            if base < 4 and not (len(ignored) and ignored[p]):
                if known >= ORDER:
                    result[phase, context, base] += weight
                context, known = advance_context(context, known, base)
            else:
                context = known = 0
            phase = (phase + 1) % phases
    return result


def counts(strand, runs, phases=1, ignore_mask=False):
    ignored = strand.mask if ignore_mask else EMPTY_MASK
    result = _count(
        strand.bases, ignored, np.asarray(runs, dtype=np.float64).reshape(-1, 4), phases
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
    return probs if anchor is None else (anchor + probs) / 2


@njit(cache=True)
def _seeds(sequence):
    for frame in range(3):
        start = -1
        for p in range(frame, len(sequence) - 2, 3):
            a, b, c = sequence[p], sequence[p + 1], sequence[p + 2]
            if a == 4 or b == 4 or c == 4:
                start = -1
            elif is_stop(a, b, c):
                if start >= 0 and p + 3 - start >= 330:
                    yield start, p
                start = -1
            elif start < 0 and is_start(a, b, c):
                start = p


def bootstrap(genome: Genome) -> Model:
    background = np.zeros((N_CONTEXTS, 4))
    coding = np.zeros((3, N_CONTEXTS, 4))
    for strands in genome.values():
        length = len(strands[0].bases)
        occupied = np.zeros(length, dtype=bool)
        candidates = []
        runs = ([], [])
        for strand, record in enumerate(strands):
            background += counts(record, [(0, length, 0, 1)])
            for start, end in _seeds(record.bases):
                if len(record.mask) and record.mask[start:end].mean() > 0.5:
                    continue
                lo, hi = (
                    (start, end + 3)
                    if strand == 0
                    else (length - end - 3, length - start)
                )
                candidates.append((end - start, lo, hi, strand, start, end))
        for _, lo, hi, strand, start, end in sorted(candidates, reverse=True):
            if not occupied[lo:hi].any():
                occupied[lo:hi] = True
                runs[strand].append((start, end, 0, 1))
        for record, selected in zip(strands, runs):
            coding += counts(record, selected, 3, ignore_mask=True)
    background = _probabilities(background)
    prior = np.broadcast_to(background, coding.shape)
    return Model(background, _probabilities(coding, prior) if coding.any() else prior)


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


def _duration(lengths, phases, confidence, anchor):
    if not lengths:
        return anchor.intron_mean, anchor.intron_mix, anchor.phases
    lengths = np.asarray(lengths)[:, None]
    confidence = np.asarray(confidence)
    opening, extension = intron_parameters(anchor)
    delta = lengths - INTRON_DELAY
    log_mass = np.where(delta >= 0, opening + delta * extension, -np.inf)
    responsibility = softmax(log_mass, axis=1) * confidence[:, None]
    strength = sqrt(len(lengths))
    pseudo = anchor.intron_mix * strength
    mass = responsibility.sum(axis=0) + pseudo
    means = (
        (responsibility * lengths).sum(axis=0) + pseudo * anchor.intron_mean
    ) / mass
    mixture = mass / mass.sum()
    phase_prob = (
        np.bincount(phases, weights=confidence, minlength=3) + anchor.phases * strength
    )
    return means, mixture, phase_prob / phase_prob.sum()


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


def refit(genome, predictions, anchor, reviewer):
    sequences, sites, genes = _training_batch(genome, predictions)
    reliability = reviewer.evaluate(sequences, sites)
    exon_runs = {(name, strand): [] for name in genome for strand in range(2)}
    lengths, phases, confidence = [], [], []
    for name, strand, cds, score, selection in genes:
        weights = reliability[selection].reshape(-1, 2).min(axis=1)
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
                phases.append(phase)
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
            coding += counts(record, exon_runs[name, strand], 3, ignore_mask=True)
    background = _probabilities(background, anchor.background)
    coding = _probabilities(coding, anchor.coding) if coding.any() else anchor.coding
    mean, mix, phase_prob = _duration(lengths, phases, confidence, anchor)
    return Model(background, coding, mean, mix, phase_prob)


def _fit_models(genome: Genome):
    anchor = model = bootstrap(genome)
    blind = predict(genome, model)
    yield model, blind
    sequences, sites, _ = _training_batch(genome, blind)
    reviewer = fit_reviewer(sequences, sites)
    for _ in range(UPDATES):
        if any(blind.values()):
            model = refit(genome, blind, anchor, reviewer)
            blind = predict(genome, model)
        if reviewer.critic is None:
            yield model, blind
        else:
            yield replace(model, critic=reviewer.critic), None


def fit(genome: Genome):
    rounds = _fit_models(genome)
    model, predictions = next(rounds)
    for model, predictions in rounds:
        pass
    return model, predictions if predictions is not None else predict(genome, model)
