from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

import numpy as np
from numba import njit

from .critic import ACCEPTOR, DONOR, Critic, motif_sites, signal_prior
from .cwt_features import COMPLEMENT, LUT, nucleotide_codes
from .cwt_features import read_fasta as read_records

ORDER = 5
N_CONTEXTS = 4**ORDER
CONTEXT_SIZES = 4 ** np.arange(ORDER + 1)
CONTEXT_OFFSETS = (CONTEXT_SIZES - 1) // 3
MIN_INTRON = 32
INTRON_DELAY = MIN_INTRON * np.array([1])
INTRON_MEAN = np.array([270.0])
INTRON_MIX = np.array([1.0])
SPLICE_PHASES = np.array([0.5, 0.25, 0.25])
EXON_EXIT = 1 / 160
SPLICE_FRACTION = 0.8
ENTRY_COST = np.log(5000)
STOP_COST = -np.log(EXON_EXIT * (1 - SPLICE_FRACTION) / 3)
DONOR_COST = -np.log(EXON_EXIT * SPLICE_FRACTION)
EXON_CONTINUE = np.log1p(-EXON_EXIT)
MASK_COST = np.log(2) / (ORDER + 1)
GC_DONOR = 0.02
EMPTY_MASK = np.empty(0, dtype=bool)
PHASE = np.array([0, 1, 1, 2, 2, 2], dtype=np.int8)
NEXT_STATE = np.array(
    [
        [1, 1, 1, 2],
        [3, 3, 3, 3],
        [4, 3, 5, 3],
        [0, 0, 0, 0],
        [-1, 0, -1, 0],
        [-1, 0, 0, 0],
    ],
    dtype=np.int8,
)
N_STATES = len(PHASE)
Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Strand:
    bases: np.ndarray
    mask: np.ndarray = field(default_factory=lambda: EMPTY_MASK)

    def __post_init__(self):
        bases, mask = nucleotide_codes(self.bases), np.asarray(self.mask)
        if mask.ndim != 1 or mask.dtype != np.bool_ or mask.size not in (0, bases.size):
            raise ValueError("mask must be empty or have one boolean per base")
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "mask", mask)


Genome = dict[str, tuple[Strand, Strand]]


@dataclass(frozen=True, slots=True)
class Model:
    background: np.ndarray
    coding: np.ndarray
    intron_mean: np.ndarray = field(default_factory=lambda: INTRON_MEAN.copy())
    intron_mix: np.ndarray = field(default_factory=lambda: INTRON_MIX.copy())
    phases: np.ndarray = field(default_factory=lambda: SPLICE_PHASES.copy())
    critic: Critic | None = None

    def __post_init__(self):
        shapes = {
            "background": (N_CONTEXTS, 4),
            "coding": (3, N_CONTEXTS, 4),
            "intron_mix": INTRON_DELAY.shape,
            "phases": (3,),
            "intron_mean": INTRON_DELAY.shape,
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            array = np.asarray(value)
            if array.shape != shape or array.dtype.kind not in "fiu":
                raise ValueError(f"invalid {name} parameters")
            array = array.astype(np.float64, copy=False)
            if not np.all(np.isfinite(array) & (array > 0)):
                raise ValueError(f"invalid {name} parameters")
            if name == "intron_mean":
                if np.any(array <= INTRON_DELAY):
                    raise ValueError("intron means must exceed their minimum lengths")
            elif not np.allclose(array.sum(axis=-1), 1.0):
                raise ValueError(f"{name} probabilities must sum to 1")
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class Gene:
    cds: tuple[Span, ...]
    strand: str
    score: float


def _encode(raw):
    bases = LUT[raw]
    lower = (bases < 4) & (raw >= ord("a"))
    mask = lower if lower.any() and np.any((bases < 4) & ~lower) else EMPTY_MASK
    return Strand(bases, mask)


def read_fasta(path: str | Path) -> dict[str, Strand]:
    return read_records(path, _encode)


def orient(records: dict[str, Strand]) -> Genome:
    return {
        name: (s, Strand(COMPLEMENT[s.bases[::-1]], s.mask[::-1]))
        for name, s in records.items()
    }


def reverse_spans(spans, length):
    return tuple((length - end, length - start) for start, end in reversed(spans))


def oriented_cds(gene: Gene, length: int) -> tuple[Span, ...]:
    return gene.cds if gene.strand == "+" else reverse_spans(gene.cds, length)


def junctions(cds):
    return ((left[1], right[0]) for left, right in pairwise(cds))


def intron_parameters(model: Model) -> tuple[np.ndarray, np.ndarray]:
    hazard = 1 / (model.intron_mean - INTRON_DELAY + 1)
    return np.log(model.intron_mix) + np.log(hazard), np.log1p(-hazard)


@njit(cache=True, inline="always")
def is_start(a, b, c):
    return a == 0 and b == 3 and c == 2


@njit(cache=True, inline="always")
def is_stop(a, b, c):
    return a == 3 and ((b == 0 and (c == 0 or c == 2)) or (b == 2 and c == 0))


@njit(cache=True)
def _coding_signals(sequence):
    starts, stops = [], []
    for p in range(len(sequence) - 2):
        a, b, c = sequence[p], sequence[p + 1], sequence[p + 2]
        if is_start(a, b, c):
            starts.append(p)
        elif is_stop(a, b, c):
            stops.append(p)
    return np.asarray(starts, dtype=np.int32), np.asarray(stops, dtype=np.int32)


@njit(cache=True, inline="always")
def advance_context(context, depth, base):
    if base == 4:
        return 0, 0
    return (4 * context + base) % N_CONTEXTS, min(depth + 1, ORDER)


@njit(cache=True, inline="always")
def _mask_cost(mask, position):
    return MASK_COST if len(mask) and mask[position] else 0.0


@njit(cache=True)
def _motif_odds(sequence, positions, length, background, mask):
    scores = np.empty(len(positions), dtype=np.float64)
    for i, p in enumerate(positions):
        context = depth = 0
        for j in range(max(0, p - ORDER), p):
            context, depth = advance_context(context, depth, sequence[j])
        score = 0.0
        for j in range(p, p + length):
            base = sequence[j]
            score -= background[CONTEXT_OFFSETS[depth] + context, base] + _mask_cost(
                mask, j
            )
            context, depth = advance_context(context, depth, base)
        scores[i] = score
    return scores


def _splice_signal(sequence, kind, critic, background):
    sites = motif_sites(sequence, kind)
    score = signal_prior if critic is None else critic.score
    scores = score(sequence, sites, kind) + _motif_odds(
        sequence, sites, 2, background, EMPTY_MASK
    )
    if kind == DONOR:
        scores += np.where(
            sequence[sites + 1] == 1, np.log(GC_DONOR), np.log1p(-GC_DONOR)
        )
    else:
        sites += 2
    return sites.astype(np.int32), scores.astype(np.float32)


def _context_logits(probabilities):
    levels = [probabilities]
    shape = probabilities.shape[:-2]
    for size in CONTEXT_SIZES[-2::-1]:
        levels.append(levels[-1].reshape(*shape, 4, size, 4).mean(axis=-3))
    return np.log(np.concatenate(levels[::-1], axis=-2))


@njit(cache=True)
def _decode(
    sequence,
    masked,
    coding,
    background,
    starts,
    start_scores,
    stops,
    stop_scores,
    donors,
    donor_scores,
    acceptors,
    acceptor_scores,
    intron_open,
    intron_extend,
    phase_scores,
):
    n = len(sequence)
    start_prev = np.empty(len(starts), dtype=np.int32)
    end_key = np.empty(len(stops), dtype=np.int32)
    end_total = np.empty(len(stops), dtype=np.float64)
    trace_prev = np.empty((len(acceptors), N_STATES), dtype=np.int32)
    trace_donor = np.empty_like(trace_prev)

    exon = np.full((ORDER + 1, N_STATES), -np.inf)
    next_exon = np.empty_like(exon)
    keys = np.zeros(exon.shape, dtype=np.int32)
    next_keys = np.zeros_like(keys)
    active = np.zeros(ORDER + 1, dtype=np.bool_)
    next_active = np.empty_like(active)
    intron = np.full((len(intron_open), N_STATES), -np.inf)
    intron_keys = np.zeros(intron.shape, dtype=np.int32)
    intron_donors = np.zeros_like(intron_keys)
    ring_size = INTRON_DELAY[-1]
    donor_ring = np.empty((ring_size, N_STATES), dtype=np.float64)
    key_ring = np.empty(donor_ring.shape, dtype=np.int32)
    pos_ring = np.full(ring_size, -1, dtype=np.int32)
    start_ring = np.full(3, -np.inf)
    stop_ring = np.full(3, -np.inf)
    start_ids = np.zeros(3, dtype=np.int32)
    stop_ids = np.zeros(3, dtype=np.int32)

    intergenic = 0.0
    last_gene = -1
    si = ti = di = ai = context = depth = block_start = 0
    start_age = min(3, ORDER)
    for p in range(n + 1):
        slot = p % 3
        if stop_ring[slot] > intergenic:
            intergenic, last_gene = stop_ring[slot], stop_ids[slot]
        stop_ring[slot] = -np.inf
        if start_ring[slot] > exon[start_age, 0]:
            exon[start_age, 0], keys[start_age, 0] = (
                start_ring[slot],
                -(start_ids[slot] + 1),
            )
            active[start_age] = True
        start_ring[slot] = -np.inf

        for m in range(len(intron_open)):
            d = p - INTRON_DELAY[m]
            if d >= block_start:
                ring = d % ring_size
                if pos_ring[ring] == d:
                    offset = intron_open[m] - p * intron_extend[m]
                    for state in range(N_STATES):
                        candidate = donor_ring[ring, state] + offset
                        if candidate > intron[m, state]:
                            intron[m, state] = candidate
                            intron_keys[m, state] = key_ring[ring, state]
                            intron_donors[m, state] = d

        if di < len(donors) and donors[di] == p:
            ring = p % ring_size
            donor_ring[ring, :] = -np.inf
            splice = donor_scores[di]
            for age in range(1, ORDER + 1):
                if active[age]:
                    for state in range(N_STATES):
                        candidate = (
                            exon[age, state] + splice + phase_scores[PHASE[state]]
                        )
                        if candidate > donor_ring[ring, state]:
                            donor_ring[ring, state], key_ring[ring, state] = (
                                candidate,
                                keys[age, state],
                            )
            pos_ring[ring] = p
            di += 1

        if ai < len(acceptors) and acceptors[ai] == p:
            for m in range(len(intron_open)):
                offset = acceptor_scores[ai] + p * intron_extend[m]
                for state in range(N_STATES):
                    candidate = intron[m, state] + offset
                    if candidate > exon[0, state]:
                        exon[0, state] = candidate
                        keys[0, state] = ai * N_STATES + state
                        active[0] = True
                        trace_prev[ai, state] = intron_keys[m, state]
                        trace_donor[ai, state] = intron_donors[m, state]
            ai += 1

        if si < len(starts) and starts[si] == p:
            kozak = 0.0
            if p >= 3 and sequence[p - 3] < 4:
                kozak += np.log(1.4 if sequence[p - 3] % 2 == 0 else 0.6)
            if p + 3 < n and sequence[p + 3] < 4:
                kozak += np.log(1.6 if sequence[p + 3] == 2 else 0.8)
            start_ring[slot] = intergenic + start_scores[si] - ENTRY_COST + kozak
            start_ids[slot], start_prev[si] = si, last_gene
            si += 1

        if ti < len(stops) and stops[ti] == p:
            age = exon[:, 0].argmax()
            terminal = exon[age, 0] + stop_scores[ti] - STOP_COST
            stop_ring[slot], stop_ids[slot] = terminal, ti
            end_total[ti], end_key[ti] = terminal, keys[age, 0]
            ti += 1

        if p == n:
            break
        base = sequence[p]
        if base == 4:
            active[:] = False
            exon[:] = -np.inf
            intron[:, :] = -np.inf
            block_start, context, depth = p + 1, 0, 0
            continue
        position_context = CONTEXT_OFFSETS[depth] + context
        penalty = (
            background[position_context, base] - EXON_CONTINUE + _mask_cost(masked, p)
        )
        next_exon[:] = -np.inf
        next_active[:] = False
        for age in range(ORDER + 1):
            if not active[age]:
                continue
            k = CONTEXT_OFFSETS[age] + context % CONTEXT_SIZES[age]
            next_age = min(age + 1, ORDER)
            for state in range(N_STATES):
                target = NEXT_STATE[state, base]
                if target >= 0:
                    candidate = (
                        exon[age, state] + coding[PHASE[state], k, base] - penalty
                    )
                    if candidate > next_exon[next_age, target]:
                        next_exon[next_age, target], next_keys[next_age, target] = (
                            candidate,
                            keys[age, state],
                        )
                        next_active[next_age] = True
        exon, next_exon = next_exon, exon
        keys, next_keys = next_keys, keys
        active, next_active = next_active, active
        context, depth = advance_context(context, depth, base)
    return last_gene, start_prev, end_key, end_total, trace_prev, trace_donor


def _predict_strand(strand: Strand, model: Model, direction: str) -> list[Gene]:
    sequence = strand.bases
    if len(sequence) < 6:
        return []
    limit = np.iinfo(np.int32).max
    if len(sequence) >= limit:
        raise ValueError("FASTA record too long for 32-bit coordinates")
    starts, stops = _coding_signals(sequence)
    if not (len(starts) and len(stops)):
        return []
    background = _context_logits(model.background)
    donors, ds = _splice_signal(sequence, DONOR, model.critic, background)
    acceptors, ac = _splice_signal(sequence, ACCEPTOR, model.critic, background)
    if len(acceptors) * N_STATES > limit:
        raise ValueError("too many acceptor candidates for 32-bit traceback")
    opening, extension = intron_parameters(model)
    index, start_prev, end_key, end_total, trace_prev, trace_donor = _decode(
        sequence,
        strand.mask,
        _context_logits(model.coding),
        background,
        starts,
        _motif_odds(sequence, starts, 3, background, strand.mask),
        stops,
        _motif_odds(sequence, stops, 3, background, strand.mask),
        donors,
        ds,
        acceptors,
        ac,
        opening - DONOR_COST,
        extension,
        np.log(3 * model.phases),
    )
    genes = []
    while index >= 0:
        end, key = int(stops[index]) + 3, int(end_key[index])
        parts = []
        while key >= 0:
            a, state = divmod(key, N_STATES)
            parts.append((int(acceptors[a]), end))
            end, key = int(trace_donor[a, state]), int(trace_prev[a, state])
        start = -key - 1
        parts.append((int(starts[start]), end))
        parts = tuple(reversed(parts))
        cds = parts if direction == "+" else reverse_spans(parts, len(sequence))
        previous = start_prev[start]
        score = end_total[index] - (end_total[previous] if previous >= 0 else 0.0)
        genes.append(Gene(cds, direction, float(score)))
        index = previous
    return genes


def _resolve(genes):
    genes.sort(key=lambda g: (g.cds[-1][1], g.cds[0][0], g.strand))
    ends = [g.cds[-1][1] for g in genes]
    scores = [0.0]
    predecessors = []
    for i, gene in enumerate(genes):
        previous = bisect_right(ends, gene.cds[0][0], hi=i)
        split = scores[previous] + gene.score
        take = split > scores[-1]
        predecessors.append(previous if take else -1)
        scores.append(split if take else scores[-1])
    chosen = []
    i = len(genes)
    while i:
        previous = predecessors[i - 1]
        if previous < 0:
            i -= 1
        else:
            chosen.append(genes[i - 1])
            i = previous
    return chosen[::-1]


def predict(genome: Genome, model: Model) -> dict[str, list[Gene]]:
    return {
        name: _resolve(
            _predict_strand(forward, model, "+") + _predict_strand(reverse, model, "-")
        )
        for name, (forward, reverse) in genome.items()
    }


def _emit(handle, escaped, strand, kind, a, b, phase, attributes):
    handle.write(
        f"{escaped}\tGHMM\t{kind}\t{a}\t{b}\t.\t{strand}\t{phase}\t{attributes}\n"
    )


def write_gff3(
    path: str | Path, genome: Genome, predictions: dict[str, list[Gene]]
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##gff-version 3\n")
        index = 0
        for seqid, (forward, _) in genome.items():
            escaped = quote(seqid, safe=":^*$@!+_?-|")
            handle.write(f"##sequence-region {escaped} 1 {len(forward.bases)}\n")
            for gene in predictions[seqid]:
                index += 1
                gid, tid = f"ghmm{index}", f"ghmm{index}.t1"
                start, end = gene.cds[0][0] + 1, gene.cds[-1][1]

                emit = partial(_emit, handle, escaped, gene.strand)

                emit("gene", start, end, ".", f"ID={gid}")
                emit("mRNA", start, end, ".", f"ID={tid};Parent={gid}")
                emitted = 0
                for a, b in gene.cds if gene.strand == "+" else reversed(gene.cds):
                    emit("exon", a + 1, b, ".", f"Parent={tid}")
                    emit("CDS", a + 1, b, -emitted % 3, f"Parent={tid}")
                    emitted += b - a
