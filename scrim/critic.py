from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numba.extending import register_jitable
from scipy.optimize import minimize
from scipy.special import expit, logit

from .cwt_features import PERIODS, W0, CwtFeatureBank, nucleotide_codes

DONOR = 0
ACCEPTOR = 1

_MIN_PAIRS = 24
_PSEUDO_NOISE = 0.005

_BASE_OFFSETS = np.r_[-6:0, 2:6]
_CWT_CHANNELS = 12 * len(PERIODS)


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


@dataclass(frozen=True, slots=True)
class Reviewer:
    _models: tuple[np.ndarray | None, np.ndarray | None]

    def score(self, sequence: np.ndarray, positions, kind: int) -> np.ndarray:
        """Log-odds evidence for oriented splice-site positions."""
        _check_kind(kind)
        positions = np.asarray(positions, dtype=np.intp)
        model = self._models[kind]
        if model is None or not len(positions):
            return np.zeros(len(positions), dtype=np.float64)
        valid = (
            (positions >= 0)
            & (positions < len(sequence) - 1)
            & _is_core(
                sequence[np.clip(positions, 0, len(sequence) - 2)],
                sequence[np.clip(positions + 1, 0, len(sequence) - 1)],
                kind,
            )
        )
        evidence = np.zeros(len(positions), dtype=np.float64)
        if valid.any():
            evidence[valid] = _score_valid(sequence, positions[valid], kind, model)
        return evidence


def _fit(features, labels):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    def objective(model):
        scores = x @ model[:-1] + model[-1]
        loss = np.logaddexp(0, scores - 2 * y * scores).mean()
        loss += 0.5 * (model[:-1] @ model[:-1]) / len(x)
        error = (expit(scores) - y) / len(x)
        gradient = np.r_[x.T @ error + model[:-1] / len(x), error.sum()]
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B")
    return result.x if result.success else None


def fit_reviewer(
    sequences: dict[str, tuple[np.ndarray, np.ndarray]],
    examples: list[SiteExample],
) -> Reviewer:
    if len(examples) < _MIN_PAIRS:
        return Reviewer((None, None))
    rng = np.random.default_rng(0)
    models = []
    for kind in (DONOR, ACCEPTOR):
        positives = []
        for example in examples:
            if example.kind != kind or example.seqid not in sequences:
                continue
            sequence = sequences[example.seqid][example.strand]
            p = example.position
            if 0 <= p < len(sequence) - 1 and _is_core(
                sequence[p], sequence[p + 1], kind
            ):
                positives.append((example.seqid, example.strand, p))
        if len(positives) < _MIN_PAIRS:
            models.append(None)
            continue
        negatives = []
        for name, strands in sequences.items():
            for strand, sequence in enumerate(strands):
                sites = motif_sites(sequence, kind)
                if len(sites):
                    chosen = rng.choice(
                        len(sites), size=min(len(sites), len(positives)), replace=False
                    )
                    negatives.extend((name, strand, int(sites[i])) for i in chosen)
        positive_set = set(positives)
        rows, labels = [], []
        for name, strand, p in positives + negatives:
            rows.append(_features(sequences[name][strand], np.array([p]))[0])
            labels.append(1.0 if (name, strand, p) in positive_set else 0.0)
        models.append(_fit(rows, labels))
    return Reviewer(tuple(models))
