from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from operator import index

import numpy as np
from numba import njit
from numba.extending import register_jitable
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import expit, logit

from .cwt_features import PERIODS, build_kernel, coordinates, nucleotide_codes

DONOR = 0
ACCEPTOR = 1

_BATCH = 8192
_BLOCK_SIZE = 4096
_MIN_BLOCKS = 3
_MIN_PAIRS = 24
_MAX_PAIRS = 8192
_MAX_FOLDS = 5
_PSEUDO_NOISE = 0.01


def _ratios(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.concatenate(
        (4 * probabilities, np.ones((*probabilities.shape[:-1], 1))), axis=-1
    )


_DONOR_OFFSETS = np.array([-2, -1, 2, 3, 4, 5])
_DONOR_INTRON = (
    (0.48, 0.09, 0.34, 0.09),
    (0.60, 0.11, 0.17, 0.12),
    (0.13, 0.10, 0.65, 0.12),
    (0.18, 0.16, 0.20, 0.46),
)
_DONOR_LOG_RATIOS = np.log(
    _ratios(
        (
            ((0.55, 0.15, 0.18, 0.12), (0.10, 0.08, 0.75, 0.07), *_DONOR_INTRON),
            ((0.65, 0.12, 0.14, 0.09), (0.08, 0.06, 0.81, 0.05), *_DONOR_INTRON),
        )
    )
)
_ACCEPTOR_OFFSETS = np.array([-1, 2])
_ACCEPTOR_LOG_RATIOS = np.log(
    _ratios(((0.10, 0.40, 0.10, 0.40), (0.31, 0.19, 0.31, 0.19)))
)
_TRACT_OFFSETS = np.arange(-18, -3)
_TRACT_LOG_RATIOS = np.log(_ratios((0.20, 0.29, 0.20, 0.31)))
_BRANCH_CENTERS = np.arange(-37, -19)
_BRANCH_OFFSETS = np.array([-3, -2, 0, 1])
_BRANCH_RATIOS = _ratios(
    (
        (0.19, 0.31, 0.19, 0.31),
        (0.15, 0.15, 0.15, 0.55),
        (0.65, 0.12, 0.11, 0.12),
        (0.19, 0.31, 0.19, 0.31),
    )
)

_BASE_OFFSETS = np.r_[-6:0, 2:6]
_ANCHORS = (-14, 15)
_CONTRASTS = np.array(
    [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1], [0, 0, 0]],
    dtype=np.float32,
)


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


@njit(inline="always")
def _positional_evidence(sequence, position, offsets, log_ratios):
    evidence = 0.0
    for i, offset in enumerate(offsets):
        evidence += log_ratios[i, _base_at(sequence, position + offset)]
    return evidence


@njit
def _prior_batch(sequence, positions, kind):
    scores = np.empty(len(positions), dtype=np.float64)
    for i, p in enumerate(positions):
        if kind == DONOR:
            context = _DONOR_LOG_RATIOS[int(sequence[p + 1] == 1)]
            scores[i] = _positional_evidence(sequence, p, _DONOR_OFFSETS, context)
        else:
            evidence = _positional_evidence(
                sequence, p, _ACCEPTOR_OFFSETS, _ACCEPTOR_LOG_RATIOS
            )
            for offset in _TRACT_OFFSETS:
                evidence += _TRACT_LOG_RATIOS[_base_at(sequence, p + offset)]
            branch = 0.0
            for center in _BRANCH_CENTERS:
                ratio = 1.0
                for j, offset in enumerate(_BRANCH_OFFSETS):
                    ratio *= _BRANCH_RATIOS[j, _base_at(sequence, p + center + offset)]
                branch += ratio
            scores[i] = evidence + np.log(branch / len(_BRANCH_CENTERS))
    return scores


@cache
def _wavelets():
    waves = [
        (anchor - len(kernel) // 2, kernel)
        for kernel in map(build_kernel, PERIODS)
        for anchor in _ANCHORS
    ]
    starts = np.array([start for start, _ in waves])
    sizes = np.array([len(kernel) for _, kernel in waves])
    taps = np.zeros((len(waves), sizes.max(), *_CONTRASTS.shape), dtype=np.complex64)
    for i, (start, kernel) in enumerate(waves):
        offsets = start + np.arange(len(kernel))
        weights = kernel[::-1].copy()
        weights[(offsets == 0) | (offsets == 1)] = 0
        weights /= np.linalg.norm(weights)
        taps[i, : len(kernel)] = weights[:, None, None] * _CONTRASTS
    flank = max(
        -int(_BRANCH_CENTERS[0] + _BRANCH_OFFSETS[0]),
        -int(starts.min()),
        int((starts + sizes).max()) - 2,
    )
    return (taps, starts, sizes), flank


@njit
def _context_batch(sequence, positions, taps, starts, sizes):
    nuc_width = 4 * len(_BASE_OFFSETS)
    features = np.zeros(
        (len(positions), nuc_width + len(starts) * taps.shape[-1]), dtype=np.float32
    )
    for i, p in enumerate(positions):
        for j, offset in enumerate(_BASE_OFFSETS):
            base = _base_at(sequence, p + offset)
            if base < 4:
                features[i, j * 4 + base] = 1.0
        col = nuc_width
        for wave, offset in enumerate(starts):
            start = p + offset
            lo, hi = max(0, -start), min(sizes[wave], len(sequence) - start)
            purine = at = amino = np.complex64(0)
            for tap in range(lo, hi):
                base = sequence[start + tap]
                purine += taps[wave, tap, base, 0]
                at += taps[wave, tap, base, 1]
                amino += taps[wave, tap, base, 2]
            for value in (purine, at, amino):
                features[i, col] = np.log1p(value.real**2 + value.imag**2)
                col += 1
    return features


def _features(sequence, positions):
    return _context_batch(sequence, positions, *_wavelets()[0])


def _linear(features, model):
    return np.einsum("ij,j->i", features, model[:-1]) + model[-1]


def _learned_evidence(prior, features, model):
    raw = prior + _linear(features, model)
    probability = _PSEUDO_NOISE + (1 - 2 * _PSEUDO_NOISE) * expit(raw)
    return np.where(np.any(features, axis=1), logit(probability), prior)


def _score_valid(sequence, positions, kind, model):
    prior = _prior_batch(sequence, positions, kind)
    if model is None:
        return prior
    return _learned_evidence(prior, _features(sequence, positions), model)


def _score(sequence, sites, kind, model):
    positions = coordinates(sites)
    sequence = nucleotide_codes(sequence)
    scores = np.zeros(len(positions), dtype=np.float64)
    for start in range(0, len(positions), _BATCH):
        chunk = positions[start : start + _BATCH]
        valid = np.flatnonzero((chunk >= 0) & (chunk < len(sequence) - 1))
        p = chunk[valid].astype(np.intp, copy=False)
        matches = _is_core(sequence[p], sequence[p + 1], kind)
        if matches.any():
            scores[start + valid[matches]] = _score_valid(
                sequence, p[matches], kind, model
            )
    return scores


def signal_prior(sequence: np.ndarray, sites: np.ndarray, kind: int) -> np.ndarray:
    _check_kind(kind)
    return _score(sequence, sites, kind, None)


@dataclass(frozen=True, slots=True)
class Critic:
    _models: tuple[np.ndarray | None, np.ndarray | None]

    def score(self, sequence: np.ndarray, sites: np.ndarray, kind: int) -> np.ndarray:
        _check_kind(kind)
        return _score(sequence, sites, kind, self._models[kind])


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
            priors.append(_prior_batch(sequence, positions, kind).reshape(pairs.shape))
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
    critic: Critic | None
    _fits: tuple[_KindFit | None, _KindFit | None]

    def evaluate(
        self,
        sequences: dict[str, tuple[np.ndarray, np.ndarray]],
        examples: list[SiteExample],
    ) -> np.ndarray:
        weights = np.full(len(examples), 0.5)
        if self.critic is None:
            return weights
        groups = _example_groups(sequences, examples, _wavelets()[1])
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
        return Reviewer(None, (None, None))
    _, flank = _wavelets()
    occupied = defaultdict(set)
    groups = _example_groups(sequences, examples, flank, occupied)
    donor, acceptor = (
        _fit_kind(
            sequences, groups[kind], occupied, kind, flank, np.random.default_rng(kind)
        )
        for kind in (DONOR, ACCEPTOR)
    )
    if donor is None and acceptor is None:
        return Reviewer(None, (None, None))
    models = (
        donor.full if donor is not None else None,
        acceptor.full if acceptor is not None else None,
    )
    return Reviewer(Critic(models), (donor, acceptor))
