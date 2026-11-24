import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from urllib.parse import unquote

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import cwt_features as cwt

SITES = ("dss", "ass")


@dataclass(frozen=True)
class Region:
    seqid: str
    offset: int
    sequence: np.ndarray
    sites: dict


@dataclass(frozen=True)
class Detector:
    filters: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class Model:
    detector: Detector
    thresholds: tuple


def read_annotation(path):
    transcripts = {}
    with cwt.open_text(str(path)) as handle:
        for line_no, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 tab-separated fields")
            seqid, _, kind, a, b, _, strand, _, attributes = fields
            if kind.lower() != "cds":
                continue
            try:
                a, b = int(a), int(b)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid CDS coordinates") from exc
            if a < 1 or b < a or strand not in ("+", "-"):
                raise ValueError(f"{path}:{line_no}: invalid CDS interval or strand")
            attrs = dict(piece.strip().split("=", 1) for piece in attributes.split(";") if "=" in piece)
            parents = [unquote(p) for p in attrs.get("Parent", "").split(",") if p]
            if not parents:
                raise ValueError(f"{path}:{line_no}: CDS lacks Parent=")
            loc = seqid, strand
            for tid in parents:
                previous, spans = transcripts.setdefault(tid, (loc, []))
                if previous != loc:
                    raise ValueError(f"{path}:{line_no}: transcript {tid!r} spans different contigs or strands")
                spans.append((a, b))

    sites = defaultdict(set)
    for (seqid, strand), spans in transcripts.values():
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = merged[-1][0], max(merged[-1][1], b)
            else:
                merged.append((a, b))
        for (_, b), (a, _) in pairwise(merged):
            donor, acceptor = (b + 1, a - 1) if strand == "+" else (a - 1, b + 1)
            sites[seqid, "dss", strand].add(donor)
            sites[seqid, "ass", strand].add(acceptor)
    return {key: np.array(sorted(positions)) for key, positions in sites.items()}


def regions(fasta, annotation=None):
    records = cwt.read_fasta(str(fasta))
    if not records or not any(len(sequence) for sequence in records.values()):
        raise ValueError(f"{fasta}: no FASTA sequence")
    sites = read_annotation(annotation) if annotation is not None else {}
    return [Region(seqid, 0, sequence, sites) for seqid, sequence in records.items() if len(sequence)]


def source_splits(sources):
    stages = ([], [], [])
    for fasta, annotation in sources:
        for region in regions(fasta, annotation):
            n = len(region.sequence)
            for i, stage in enumerate(stages):
                a, b = n * i // len(stages), n * (i + 1) // len(stages)
                if b > a:
                    stage.append(Region(region.seqid, a, region.sequence[a:b], region.sites))
    return stages


def labeled_positions(region, strand, role):
    known = region.sites.get((region.seqid, SITES[role], strand), ())
    lo, hi = np.searchsorted(known, (region.offset, region.offset + len(region.sequence)), side="right")
    relative = np.asarray(known[lo:hi], dtype=np.int64) - region.offset
    return len(region.sequence) - relative[::-1] if strand == "-" else relative - 1


def reference_by_strand(region):
    return {strand: [labeled_positions(region, strand, role) for role in range(len(SITES))]
            for strand in ("+", "-")}


def ref_count(regions, role):
    return sum(len(labeled_positions(region, strand, role)) for region in regions for strand in ("+", "-"))


def block_labels(positions, begin, size):
    lo, hi = np.searchsorted(positions, (begin, begin + size))
    result = np.zeros(size, dtype=bool)
    result[positions[lo:hi] - begin] = True
    return result


def wave_frames(bank, strand):
    for begin in range(0, bank.length, bank.chunk):
        positions = np.arange(begin, min(begin + bank.chunk, bank.length))
        if strand == "-":
            positions = bank.length - 1 - positions
        yield cwt.cwt_features(bank, positions, np.full(len(positions), strand == "-", dtype=np.int8))


def wave_blocks(bank):
    radius = bank.max_radius
    pad = np.zeros((radius, bank.channels), dtype=np.float32)
    for strand in ("+", "-"):
        frames = wave_frames(bank, strand)
        past = pad
        active = next(frames, None)
        begin = 0
        while active is not None:
            following = next(frames, None)
            future = pad if following is None else following[:radius]
            if len(future) < radius:
                future = np.concatenate((future, pad[:radius - len(future)]))
            wave = np.concatenate((past, active, future))
            yield strand, begin, len(active), wave
            past = wave[len(active):len(active) + radius]
            begin += len(active)
            active = following


def patches(wave, positions, width):
    return wave[positions[:, None] + np.arange(width)].reshape(len(positions), width * wave.shape[1])


def scan_wave(detector, wave, size, kernels):
    width = len(detector.filters)
    n = next_fast_len(len(wave) + width - 1, real=True)
    if n not in kernels:
        kernels[n] = rfft(detector.filters[::-1], n=n, axis=0)
    signal = rfft(wave.astype(np.float64), n=n, axis=0, workers=-1)
    response = irfft(np.einsum("fc,fcr->fr", signal, kernels[n]), n=n, axis=0)
    return (response[width - 1:width - 1 + size] + detector.bias).astype(np.float32)


def scores(detector, region):
    bank = cwt.CwtFeatureBank(region.sequence, cwt.PERIODS, cwt.W0)
    kernels = {}
    for strand, begin, size, wave in wave_blocks(bank):
        yield strand, begin, scan_wave(detector, wave, size, kernels)


def top_indices(scores, count):
    if len(scores) <= count:
        return np.arange(len(scores))
    return np.argpartition(scores, -count)[-count:]


def sampled_waveforms(samples, mining=None):
    counts = [ref_count(samples, role) for role in range(len(SITES))]
    total = 2 * sum(len(region.sequence) for region in samples)
    if not all(0 < n < total for n in counts):
        raise ValueError("training split requires both labels and background for each site")
    budget = [min(n, total - n) for n in counts]
    indices = [(2 * np.arange(k) + 1) * (total - n) // (2 * k) for n, k in zip(counts, budget)]
    seen = [0] * len(SITES)
    positive, ordinary = ([[] for _ in SITES] for _ in range(2))
    hard = [None] * len(SITES)
    hard_scores = [np.empty(0, dtype=np.float32) for _ in SITES]
    for region in samples:
        bank = cwt.CwtFeatureBank(region.sequence, cwt.PERIODS, cwt.W0)
        width = 2 * bank.max_radius + 1
        known = reference_by_strand(region)
        kernels = {}
        for strand, begin, size, wave in wave_blocks(bank):
            response = scan_wave(mining, wave, size, kernels) if mining is not None else None
            for role, positions in enumerate(known[strand]):
                y = block_labels(positions, begin, size)
                if y.any():
                    positive[role].append(patches(wave, np.flatnonzero(y), width))
                negative = np.flatnonzero(~y)
                a, b = np.searchsorted(indices[role], (seen[role], seen[role] + len(negative)))
                if b > a:
                    ordinary[role].append(patches(wave, negative[indices[role][a:b] - seen[role]], width))
                seen[role] += len(negative)
                if response is not None:
                    z = response[negative, role]
                    prior = hard_scores[role]
                    pick = top_indices(np.concatenate((prior, z)), budget[role])
                    take_old = pick < len(prior)
                    emerging = patches(wave, negative[pick[~take_old] - len(prior)], width)
                    hard[role] = np.concatenate((hard[role][pick[take_old]], emerging)) if len(prior) else emerging
                    hard_scores[role] = np.r_[prior[pick[take_old]], z[pick[~take_old] - len(prior)]]
    result = []
    for role, n in enumerate(counts):
        extra = (hard[role],) if hard[role] is not None else ()
        x = np.concatenate((*positive[role], *ordinary[role], *extra))
        result.append((x, np.arange(len(x)) < n))
    return result, width, bank.channels


def fit_detector(samples, mining=None):
    training, width, channels = sampled_waveforms(samples, mining)
    filters, bias = [], []
    for waveforms, hit in training:
        x = waveforms.astype(np.float64)
        mean = x.mean(axis=0)
        x -= mean
        cov = x.T @ x / len(x)
        variance, axes = np.linalg.eigh(cov)
        keep = variance > variance[-1] * len(variance) * np.finfo(np.float64).eps
        if not keep.any():
            raise ValueError("no temporal CWT variation in the training split")
        whitening = axes[:, keep] / np.sqrt(variance[keep])
        fitted = LogisticRegression(C=np.inf, solver="newton-cholesky").fit(x @ whitening, hit)
        coef = whitening @ fitted.coef_[0]
        filters.append(coef.reshape(width, channels))
        bias.append(fitted.intercept_[0] - mean @ coef)
    return Detector(np.stack(filters, axis=2), np.asarray(bias))


def f1_threshold(values, truth):
    if not len(values) or not truth.any():
        raise ValueError("cutoff split requires labeled splice sites")
    order = np.argsort(-values, kind="stable")
    z = values[order]
    ends = np.r_[np.flatnonzero(z[:-1] != z[1:]), len(z) - 1]
    f1 = 2 * truth[order].cumsum()[ends] / (truth.sum() + ends + 1)
    best = f1.argmax()
    return float(z[ends[best]]), float(f1[best])


def fit(sources):
    fitting, mining, cutoff = source_splits(sources)
    base = fit_detector(fitting)
    hard = fit_detector(mining, base)
    joined = Detector(np.concatenate((base.filters, hard.filters), axis=2), np.r_[base.bias, hard.bias])
    response, truth = [], [[] for _ in SITES]
    for region in cutoff:
        known = reference_by_strand(region)
        for strand, begin, z in scores(joined, region):
            response.append(z)
            for role, positions in enumerate(known[strand]):
                truth[role].append(block_labels(positions, begin, len(z)))
    if not response:
        raise ValueError("empty cutoff split")
    response = np.concatenate(response)
    picked, thresholds = [], []
    for role, parts in enumerate(truth):
        labels = np.concatenate(parts)
        options = (role, role + len(SITES))
        index, threshold, _ = max(((i, *f1_threshold(response[:, i], labels)) for i in options),
                                  key=lambda option: option[2])
        picked.append(index)
        thresholds.append(threshold)
    detector = Detector(joined.filters[:, :, picked], joined.bias[picked])
    return Model(detector, tuple(thresholds))


def metrics_for(hit, z, n_ref, threshold):
    chosen = z >= threshold
    tp = int(hit[chosen].sum())
    n_pred = int(chosen.sum())
    return {"candidate_count": len(hit), "reference": n_ref, "prediction": n_pred,
            "tp": tp, "fp": n_pred - tp, "fn": n_ref - tp,
            "coverage": int(hit.sum()) / n_ref if n_ref else None,
            "ap": average_precision_score(hit, z) if n_ref else None,
            "precision": tp / n_pred if n_pred else None, "recall": tp / n_ref if n_ref else None,
            "f1": 2 * tp / (n_pred + n_ref) if n_pred + n_ref else None}


def predict(model, fasta, output, reference=None):
    target = regions(fasta, reference)
    observations = ([], [])
    output.write("##gff-version 3\n")
    for region in target:
        known = reference_by_strand(region) if reference is not None else None
        rows = []
        for strand, begin, scores_for_sites in scores(model.detector, region):
            for role, z in enumerate(scores_for_sites.T):
                if known is not None:
                    observations[role].append((block_labels(known[strand][role], begin, len(z)), z))
                selected = np.flatnonzero(z >= model.thresholds[role])
                pos = begin + selected
                coords = len(region.sequence) - pos if strand == "-" else pos + 1
                rows.extend((int(coord + region.offset), strand, SITES[role], float(value))
                            for coord, value in zip(coords, z[selected]))
        for coord, strand, site, score in sorted(rows):
            output.write(f"{region.seqid}\tkws\t{site}\t{coord}\t{coord}\t{score:.9g}\t{strand}\t.\tsrc=E\n")
    if reference is None:
        return None
    summary = {}
    for role, parts in enumerate(observations):
        y, z = (np.concatenate(x) for x in zip(*parts))
        summary[SITES[role]] = metrics_for(y, z, ref_count(target, role), model.thresholds[role])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect coding splice sites from the 24-channel CWT signal")
    parser.add_argument("--train-fasta", action="append", required=True)
    parser.add_argument("--train-gff3", action="append", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="evaluate the target without fitting or calibrating on it")
    args = parser.parse_args(argv)
    if len(args.train_fasta) != len(args.train_gff3):
        parser.error("provide one --train-gff3 per --train-fasta")
    try:
        model = fit(list(zip(args.train_fasta, args.train_gff3)))
        with open(args.output, "x", encoding="utf-8") as output:
            summary = predict(model, args.fasta, output, args.reference)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if summary is not None:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
