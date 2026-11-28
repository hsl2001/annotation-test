from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .cwt_features import PERIODS, W0, CwtFeatureBank, nucleotide_codes

DONOR = 0
ACCEPTOR = 1

_BASE_OFFSETS = np.r_[-6:0, 2:6]
_CWT_CHANNELS = 12 * len(PERIODS)


@dataclass(frozen=True, slots=True)
class Site:
    seqid: str
    strand: int
    kind: int
    position: int


@dataclass(frozen=True, slots=True)
class Reviewer:
    weights: np.ndarray | None

    def score(self, bank: CwtFeatureBank, sites: list[Site]) -> np.ndarray:
        if self.weights is None or not sites:
            return np.zeros(len(sites), np.float64)
        design = _features_batch(bank, sites)
        design = np.hstack((design, np.ones((len(design), 1), np.float64)))
        return design @ self.weights


def _onehot(sequence, position):
    result = np.zeros(4 * len(_BASE_OFFSETS), np.float32)
    for index, offset in enumerate(_BASE_OFFSETS):
        p = position + offset
        if 0 <= p < len(sequence) and sequence[p] < 4:
            result[index * 4 + sequence[p]] = 1.0
    return result


def _features_batch(bank: CwtFeatureBank, sites: list[Site]) -> np.ndarray:
    rows = []
    for strand in (0, 1):
        selected = [s for s in sites if s.strand == strand]
        if not selected:
            continue
        sequence = bank.codes if strand == 0 else bank.reverse_codes
        positions = np.array(
            [s.position if strand == 0 else len(sequence) - 2 - s.position
             for s in selected],
            np.int64,
        )
        onehot = np.vstack([_onehot(sequence, p) for p in positions])
        cwt = bank.gather(positions, np.zeros(len(positions), np.int8))[:, :_CWT_CHANNELS]
        rows.append(np.hstack((onehot, cwt)))
    return np.vstack(rows)


def _features(bank: CwtFeatureBank, site: Site) -> np.ndarray:
    return _features_batch(bank, [site])[0]


def sites_from_annotation(transcripts, seqid: str, length: int) -> list[Site]:
    sites = []
    for transcript in transcripts:
        if transcript.seqid != seqid:
            continue
        strand = int(transcript.strand == "-")
        spans = transcript.spans.get("CDS") or transcript.spans["exon"]
        for (a, b), (c, d) in zip(spans, spans[1:]):
            if strand == 0:
                donor = b
                acceptor = c - 2
            else:
                donor = length - c + 1
                acceptor = length - b - 2
            sites.append(Site(seqid, strand, DONOR, donor))
            sites.append(Site(seqid, strand, ACCEPTOR, acceptor))
    return sites


def candidate_sites(sequence: np.ndarray, kind: int) -> np.ndarray:
    codes = nucleotide_codes(sequence)
    if kind == DONOR:
        return np.arange(1, len(codes) - 2, dtype=np.int64)
    return np.arange(1, len(codes) - 2, dtype=np.int64)


def fit(banks: dict[str, CwtFeatureBank], positives: list[Site], negatives: list[Site]) -> Reviewer:
    if not positives or not negatives:
        return Reviewer(None)
    rows = []
    labels = []
    for seqid, bank in banks.items():
        for strand in (0, 1):
            selected = [s for s in positives + negatives if s.seqid == seqid and s.strand == strand]
            if not selected:
                continue
            rows.append(_features_batch(bank, selected))
            labels.append(np.array([1.0 if s in positives else 0.0 for s in selected]))
    if not rows:
        return Reviewer(None)
    x = np.vstack(rows).astype(np.float64)
    y = np.concatenate(labels)
    x = np.hstack((x, np.ones((len(x), 1), np.float64)))
    penalty = np.ones(x.shape[1], np.float64)
    penalty[-1] = 0.0

    def objective(weights):
        logits = x @ weights
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * np.dot(penalty * weights, weights) / len(y)
        gradient = x.T @ (expit(logits) - y) / len(y)
        gradient += penalty * weights / len(y)
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return Reviewer(result.x if result.success else None)


def train(banks: dict[str, CwtFeatureBank], transcripts, rng=None) -> Reviewer:
    rng = np.random.default_rng(0) if rng is None else rng
    positives = []
    negatives = []
    for seqid, bank in banks.items():
        length = len(bank.codes)
        sites = sites_from_annotation(transcripts, seqid, length)
        positives.extend(sites)
        for kind in (DONOR, ACCEPTOR):
            candidates = candidate_sites(bank.codes, kind)
            count = sum(1 for s in sites if s.kind == kind)
            if not count or not len(candidates):
                continue
            chosen = rng.choice(len(candidates), size=min(len(candidates), count), replace=False)
            negatives.extend(Site(seqid, 0, kind, int(candidates[i])) for i in chosen)
    return fit(banks, positives, negatives)


def select(reviewer: Reviewer, bank: CwtFeatureBank, sites: list[Site]) -> list[Site]:
    if not sites or reviewer.weights is None:
        return sites
    scores = reviewer.score(bank, sites)
    return [site for site, score in zip(sites, scores) if score > 0]
