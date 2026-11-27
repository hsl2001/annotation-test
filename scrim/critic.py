from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from operator import index

import numpy as np
from numba import njit
from numba.extending import register_jitable
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import expit, logit

from .cwt_features import (
    PERIODS,
    W0,
    CwtFeatureBank,
    build_kernel,
    nucleotide_codes,
)

DONOR = 0
ACCEPTOR = 1


_BLOCK_SIZE = 4096
_MIN_BLOCKS = 3
_MIN_PAIRS = 24
_MAX_PAIRS = 8192
_MAX_FOLDS = 5
_PSEUDO_NOISE = 0.01


_BASE_OFFSETS = np.r_[-6:0, 2:6]
_CWT_CHANNELS = 12 * len(PERIODS)
_CWT_FLANK = max(len(build_kernel(period)) // 2 for period in PERIODS)


@dataclass(frozen=True, slots=True)
class SiteExample:
    seqid: str
    strand: int
    kind: int
    position: int


def _check_kind(kind):
    if kind not in (DONOR, ACCEPTOR):
        raise ValueError(f"unknown splice-site kind: {kind!r}")


@register_jitable(inline="always")
def _is_core(first, second, kind):
    if kind == DONOR:
        return (first == 2) & ((second == 3) | (second == 1))
    return (first == 0) & (second == 2)


@njit
def _find_motifs(sequence, kind):
    count = 0
    for p in range(len(sequence) - 1):
        count += _is_core(sequence[p], sequence[p + 1], kind)
    sites = np.empty(count, dtype=np.intp)
    i = 0
    for p in range(len(sequence) - 1):
        if _is_core(sequence[p], sequence[p + 1], kind):
            sites[i] = p
            i += 1
    return sites


def motif_sites(sequence: np.ndarray, kind: int) -> np.ndarray:
    _check_kind(kind)
    return _find_motifs(nucleotide_codes(sequence), kind)


@njit(inline="always")
def _base_at(sequence, position):
    return sequence[position] if 0 <= position < len(sequence) else 4


@njit
def _onehot_batch(sequence, positions):
    features = np.zeros((len(positions), 4 * len(_BASE_OFFSETS)), dtype=np.float32)
    for i, p in enumerate(positions):
        for j, offset in enumerate(_BASE_OFFSETS):
            base = _base_at(sequence, p + offset)
            if base < 4:
                features[i, j * 4 + base] = 1.0
    return features


_CWT_CACHE = {}


def _cwt_sites(sequence):
    key = id(sequence)
    entry = _CWT_CACHE.get(key)
    if entry is None or entry[0] is not sequence:
        sites = np.sort(
            np.concatenate(
                (motif_sites(sequence, DONOR), motif_sites(sequence, ACCEPTOR))
            )
        )
        bank = CwtFeatureBank(sequence, PERIODS, W0)
        cwt = bank.gather(sites, np.zeros(len(sites), dtype=np.int8))[:, :_CWT_CHANNELS]
        entry = (sequence, sites, cwt)
        _CWT_CACHE[key] = entry
    return entry[1], entry[2]


def _features(sequence, positions):
    positions = np.asarray(positions, dtype=np.intp)
    sites, cwt = _cwt_sites(sequence)
    return np.hstack(
        (_onehot_batch(sequence, positions), cwt[np.searchsorted(sites, positions)])
    )


def _linear(features, model):
    return np.einsum("ij,j->i", features, model[:-1]) + model[-1]


def _learned_evidence(prior, features, model):
    raw = prior + _linear(features, model)
    probability = _PSEUDO_NOISE + (1 - 2 * _PSEUDO_NOISE) * expit(raw)
    return np.where(np.any(features, axis=1), logit(probability), prior)


def _score_valid(sequence, positions, kind, model):
    prior = np.zeros(len(positions), dtype=np.float64)
    if model is None:
        return prior
    return _learned_evidence(prior, _features(sequence, positions), model)


def _genomic_left(length, strand, position):
    return position if strand == 0 else length - 2 - position


def _block_bounds(length, block):
    start = block * _BLOCK_SIZE
    return start, min(start + _BLOCK_SIZE, length)


def _match_block(sequences, sites, kind, block, flank, occupied, rng):
    length = len(sequences[0])
    lo, hi = _block_bounds(length, block)
    genomic = rng.permutation(
        np.fromiter(
            (p for p in range(lo + flank, hi - flank - 1) if p not in occupied),
            dtype=np.intp,
        )
    )
    if not len(genomic):
        return {}
    strands, positions = np.asarray(sites, dtype=np.intp).T
    eligible = np.zeros((len(sites), len(genomic)), dtype=bool)
    for strand, sequence in enumerate(sequences):
        rows = np.flatnonzero(strands == strand)
        if not len(rows):
            continue
        decoys = _genomic_left(length, strand, genomic)
        second = sequence[decoys + 1]
        columns = np.flatnonzero(_is_core(sequence[decoys], second, kind))
        positives = positions[rows, None]
        eligible[np.ix_(rows, columns)] = (
            sequence[positives + 1] == second[columns]
        ) & (np.abs(positives - decoys[columns]) >= 2 * flank + 2)

    assigned = maximum_bipartite_matching(csr_matrix(eligible), perm_type="column")
    matched = {}
    for strand in (0, 1):
        keep = (strands == strand) & (assigned >= 0)
        if keep.any():
            negatives = _genomic_left(length, strand, genomic[assigned[keep]])
            matched[strand] = np.column_stack((positions[keep], negatives))
    return matched


def _matched_groups(sequences, groups, occupied, kind, flank, rng):
    candidates = [
        (block, strand, p) for block, entries in groups.items() for strand, p in entries
    ]
    if len(candidates) < _MIN_PAIRS:
        return {}
    sampled = defaultdict(list)
    for i in rng.choice(
        len(candidates), min(len(candidates), _MAX_PAIRS), replace=False
    ):
        block, strand, position = candidates[i]
        sampled[block].append((strand, position))
    matched = {}
    for key, sites in sampled.items():
        seqid, block = key
        strands = _match_block(
            sequences[seqid], sites, kind, block, flank, occupied[seqid], rng
        )
        if strands:
            matched[key] = strands
    return matched


def _fit(features, prior, pair_weight):
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    offset = prior.ravel()
    weight = np.repeat(pair_weight * 0.5, 2)
    label = np.tile((1.0, 0.0), len(prior))
    sign = 1.0 - 2.0 * label

    def objective(model):
        coef = model[:-1]
        scores = offset + _linear(x, model)
        loss = weight @ np.logaddexp(0, sign * scores) + 0.5 * (coef @ coef)
        error = weight * (expit(scores) - label)
        gradient = np.einsum("ij,i->j", x, error) + coef
        return loss, np.r_[gradient, error.sum()]

    initial = np.zeros(x.shape[1] + 1)
    initial[-1] = -np.average(offset, weights=weight)
    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True)
    return fitted.x if fitted.success else None


def _pair_loss(evidence):
    return np.logaddexp(0, evidence * (-1, 1)).mean(axis=1)


def _improves(loss, baseline, blocks, counts):
    gain = np.bincount(blocks, weights=baseline - loss) / counts
    return gain.mean() > max(np.finfo(float).eps, gain.std(ddof=1) / np.sqrt(len(gain)))


@dataclass(frozen=True, slots=True)
class _KindFit:
    full: np.ndarray
    folds: tuple[np.ndarray, ...]
    fold_by_block: dict[tuple[str, int], int]


def _fit_kind(sequences, groups, occupied, kind, flank, rng):
    if len(groups) < _MIN_BLOCKS:
        return None
    matched = _matched_groups(sequences, groups, occupied, kind, flank, rng)
    counts = np.array([sum(map(len, strands.values())) for strands in matched.values()])
    if len(counts) < _MIN_BLOCKS or counts.sum() < _MIN_PAIRS:
        return None
    loads = np.zeros(min(_MAX_FOLDS, len(counts)), dtype=np.intp)
    fold_for = np.empty(len(counts), dtype=np.intp)
    for block in np.argsort(-counts, kind="stable"):
        fold = loads.argmin()
        fold_for[block] = fold
        loads[fold] += counts[block]
    if counts.sum() - loads.max() < _MIN_PAIRS // 2:
        return None

    features, priors = [], []
    for (seqid, _), strands in matched.items():
        for strand, pairs in strands.items():
            sequence = sequences[seqid][strand]
            positions = pairs.ravel()
            data = _features(sequence, positions)
            features.append(data.reshape(len(pairs), 2, -1))
            priors.append(np.zeros(pairs.shape, dtype=np.float64))
    features, prior = np.concatenate(features), np.concatenate(priors)
    blocks = np.repeat(np.arange(len(counts)), counts)
    folds = fold_for[blocks]
    pair_weight = 1 / counts[blocks]
    oof = np.empty_like(prior)
    fits = []
    for fold in range(len(loads)):
        test = folds == fold
        model = _fit(features[~test], prior[~test], pair_weight[~test])
        if model is None:
            return None
        fits.append(model)
        oof[test] = _learned_evidence(
            prior[test].ravel(),
            features[test].reshape(-1, features.shape[-1]),
            model,
        ).reshape(-1, 2)

    loss = _pair_loss(oof)
    if not all(
        _improves(loss, baseline, blocks, counts)
        for baseline in (_pair_loss(prior), np.log(2))
    ):
        return None
    fitted = _fit(features, prior, pair_weight)
    if fitted is None:
        return None

    return _KindFit(fitted, tuple(fits), dict(zip(matched, fold_for.tolist())))


def _example_groups(sequences, examples, flank, occupied=None):
    groups = defaultdict(dict), defaultdict(dict)
    for i, example in enumerate(examples):
        seqid = example.seqid
        if seqid not in sequences:
            continue
        try:
            strand, kind, p = (
                index(example.strand),
                index(example.kind),
                index(example.position),
            )
        except TypeError:
            continue
        if strand not in (0, 1) or kind not in (DONOR, ACCEPTOR):
            continue
        oriented = sequences[seqid]
        length = len(oriented[0])
        if len(oriented[1]) != length or not 0 <= p < length - 1:
            continue
        sequence = oriented[strand]
        if not _is_core(sequence[p], sequence[p + 1], kind):
            continue
        genomic = _genomic_left(length, strand, p)
        if occupied is not None:
            occupied[seqid].add(genomic)
        block = genomic // _BLOCK_SIZE
        lo, hi = _block_bounds(length, block)
        if lo + flank <= genomic < hi - flank - 1:
            groups[kind][seqid, block].setdefault((strand, p), []).append(i)
    return groups


@dataclass(frozen=True, slots=True)
class Reviewer:
    _fits: tuple[_KindFit | None, _KindFit | None]

    def evaluate(
        self,
        sequences: dict[str, tuple[np.ndarray, np.ndarray]],
        examples: list[SiteExample],
    ) -> np.ndarray:
        weights = np.full(len(examples), 0.5)
        if all(fit is None for fit in self._fits):
            return weights
        groups = _example_groups(sequences, examples, _CWT_FLANK)
        for kind, fit in enumerate(self._fits):
            if fit is None:
                continue
            for block, entries in groups[kind].items():
                fold = fit.fold_by_block.get(block)
                model = fit.full if fold is None else fit.folds[fold]
                for strand in (0, 1):
                    positions = np.array(
                        [p for s, p in entries if s == strand], dtype=np.intp
                    )
                    if len(positions):
                        evidence = _score_valid(
                            sequences[block[0]][strand], positions, kind, model
                        )
                        for p, probability in zip(positions, expit(evidence)):
                            weights[entries[strand, p]] = probability
        return weights


def fit_reviewer(
    sequences: dict[str, tuple[np.ndarray, np.ndarray]],
    examples: list[SiteExample],
) -> Reviewer:
    if len(examples) < _MIN_PAIRS:
        return Reviewer((None, None))
    flank = _CWT_FLANK
    occupied = defaultdict(set)
    groups = _example_groups(sequences, examples, flank, occupied)
    donor, acceptor = (
        _fit_kind(
            sequences, groups[kind], occupied, kind, flank, np.random.default_rng(kind)
        )
        for kind in (DONOR, ACCEPTOR)
    )
    if donor is None and acceptor is None:
        return Reviewer((None, None))
    return Reviewer((donor, acceptor))
