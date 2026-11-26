from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from numba import njit

NSYM = 4
ORDER = 3
CTX = NSYM**ORDER
MIN_INTRON = 60
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _code, _base in enumerate(b"ACGT"):
    BASE_LUT[_base] = _code
    BASE_LUT[_base + 32] = _code

STATES = ["I", "U5", "U3"]
STATES += [f"E{frame}" for frame in range(3)]
STATES += [f"S{frame}" for frame in range(3)]
STATES += [f"T{frame}" for frame in range(3)]
STATES += [f"D{frame}_{part}" for frame in range(3) for part in range(2)]
STATES += [f"N{phase}" for phase in range(3)]
STATES += [f"A{phase}_{part}" for phase in range(3) for part in range(2)]
SIDX = {state: index for index, state in enumerate(STATES)}
NSTATE = len(STATES)

ALLOWED = defaultdict(set)
for _source, _target in [
    ("I", "I"), ("I", "U5"), ("I", "S0"), ("U5", "U5"), ("U5", "S0"),
    ("S0", "S1"), ("S1", "S2"), ("S2", "E0"), ("T0", "T1"), ("T1", "T2"),
    ("T2", "U3"), ("T2", "I"), ("U3", "U3"), ("U3", "I"),
]:
    ALLOWED[_source].add(_target)
for _frame in range(3):
    ALLOWED[f"E{_frame}"].update({f"E{(_frame + 1) % 3}", f"D{_frame}_0", "T0"})
    ALLOWED[f"D{_frame}_0"].add(f"D{_frame}_1")
    ALLOWED[f"D{_frame}_1"].add(f"N{(_frame + 1) % 3}")
for _phase in range(3):
    ALLOWED[f"N{_phase}"].update({f"N{_phase}", f"A{_phase}_0"})
    ALLOWED[f"A{_phase}_0"].add(f"A{_phase}_1")
    ALLOWED[f"A{_phase}_1"].add(f"E{_phase}")

CODING_IDX = np.array([SIDX[state] for state in STATES if state[0] in "SET"], dtype=np.int64)
EXONIC_IDX = np.array([SIDX[state] for state in STATES if state[0] in "SETU"], dtype=np.int64)
DONOR_IDX = np.array([SIDX[f"D{frame}_0"] for frame in range(3)], dtype=np.int64)
ACCEPTOR_IDX = np.array([SIDX[f"A{phase}_0"] for phase in range(3)], dtype=np.int64)
MASKS = {
    SIDX["S0"]: (0,), SIDX["S1"]: (3,), SIDX["S2"]: (2,),
    SIDX["T0"]: (3,), SIDX["T1"]: (0, 2), SIDX["T2"]: (0, 2),
}
END_STATES = np.array([SIDX["I"], SIDX["T2"], SIDX["U3"]], dtype=np.int64)
MAX_PREDECESSORS = max(
    sum(1 for targets in ALLOWED.values() if target in targets) for target in STATES
)
PREDECESSOR_IDS = np.full((NSTATE, MAX_PREDECESSORS), -1, dtype=np.int64)
PREDECESSOR_COUNTS = np.zeros(NSTATE, dtype=np.int64)
for _target_index, _target in enumerate(STATES):
    _items = [SIDX[_source] for _source, _targets in ALLOWED.items() if _target in _targets]
    PREDECESSOR_COUNTS[_target_index] = len(_items)
    PREDECESSOR_IDS[_target_index, : len(_items)] = _items

DATASETS = {
    "arabidopsis": ("data/truth.col.chr5.fasta.gz", "data/truth.col.chr5.gff3.gz"),
    "rice": ("data/truth.nip.chr5.fasta.gz", "data/truth.nip.chr5.gff3.gz"),
    "tomato": ("data/truth.slm.chr5.fasta.gz", "data/truth.slm.chr5.gff3.gz"),
    "human": ("data/truth.human.chr21.fasta.gz", "data/truth.human.chr21.gff3.gz"),
    "mouse": ("data/truth.mouse.chr19.fasta.gz", "data/truth.mouse.chr19.gff3.gz"),
    "zebrafish": ("data/truth.zebrafish.chr25.fasta.gz", "data/truth.zebrafish.chr25.gff3.gz"),
    "celegans": ("data/truth.elegans.chrI.fasta.gz", "data/truth.elegans.chrI.gff3.gz"),
    "vp": ("data/truth.vp.chr12.fasta.gz", "data/truth.vp.chr12.gff3.gz"),
}
SOURCE_DATASETS = ("arabidopsis", "rice", "tomato", "mouse", "zebrafish", "celegans")
TARGET_DATASETS = ("human", "vp")


def read_fasta(path: str | Path) -> tuple[str, np.ndarray]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    name = None
    bases = bytearray()
    with opener(path, "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    raise ValueError(f"{path}: expected exactly one FASTA record")
                name = line[1:].split()[0]
            elif name is not None:
                bases.extend(line.strip().upper().encode("ascii"))
    if name is None or not bases:
        raise ValueError(f"{path}: missing or empty FASTA record")
    return name, BASE_LUT[np.frombuffer(bases, dtype=np.uint8)]


def _attributes(text: str) -> dict[str, str]:
    return {
        key: unquote(value)
        for part in text.split(";")
        if "=" in part
        for key, value in [part.split("=", 1)]
    }


def _merge_spans(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def read_transcripts(path: str | Path):
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    definitions = {}
    exons = defaultdict(list)
    cds_spans = defaultdict(list)
    transcript_types = {"mrna", "transcript", "primary_transcript", "ncrna", "lnc_rna"}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                continue
            seqid, _, feature, start, end, _, strand, _, attr_text = fields
            feature = feature.lower()
            attrs = _attributes(attr_text)
            if feature in transcript_types:
                tid = attrs.get("ID")
                if tid:
                    definitions[tid] = (
                        seqid,
                        strand,
                        attrs.get("Parent", tid),
                    )
            elif feature in ("exon", "cds"):
                span = (int(start) - 1, int(end))
                for parent in attrs.get("Parent", "").split(","):
                    if parent:
                        if parent not in definitions:
                            definitions[parent] = (seqid, strand, parent)
                        (cds_spans if feature == "cds" else exons)[parent].append(span)
    transcripts = []
    for tid, (seqid, strand, gene) in definitions.items():
        merged_cds = _merge_spans(cds_spans.get(tid, []))
        if not merged_cds:
            continue
        merged_exons = _merge_spans(exons.get(tid, [])) or merged_cds
        transcripts.append(
            {"tid": tid, "gene": gene, "seqid": seqid, "strand": strand,
             "exons": merged_exons, "cds": merged_cds}
        )
    return transcripts


def _select_transcripts(transcripts, seqid: str):
    best = {}
    for transcript in transcripts:
        if transcript["seqid"] != seqid:
            continue
        coding_length = sum(end - start for start, end in transcript["cds"])
        gene = transcript["gene"]
        if gene not in best or coding_length > best[gene][0]:
            best[gene] = (coding_length, transcript)
    return [transcript for _, transcript in best.values()]


def _coding_state(position: int, coding_length: int) -> str:
    if position < 3:
        return f"S{position}"
    if position >= coding_length - 3:
        return f"T{position - (coding_length - 3)}"
    return f"E{position % 3}"


def _intron_states(length: int, coding_position: int, coding_length: int):
    if 0 < coding_position < coding_length and length >= 4:
        donor_phase = (coding_position - 1) % 3
        acceptor_phase = coding_position % 3
        return ([f"D{donor_phase}_0", f"D{donor_phase}_1"]
                + [f"N{acceptor_phase}"] * (length - 4)
                + [f"A{acceptor_phase}_0", f"A{acceptor_phase}_1"])
    return ["N0"] * length


def label_transcript(transcript):
    lo = min(start for start, _ in transcript["exons"])
    hi = max(end for _, end in transcript["exons"])
    positions = list(range(hi - 1, lo - 1, -1)) if transcript["strand"] == "-" else list(range(lo, hi))
    exon_mask = np.zeros(hi - lo, dtype=np.bool_)
    cds_mask = np.zeros(hi - lo, dtype=np.bool_)
    for start, end in transcript["exons"]:
        exon_mask[start - lo : end - lo] = True
    for start, end in transcript["cds"]:
        cds_mask[start - lo : end - lo] = True
    coding_length = int(cds_mask.sum())
    labeled_positions = []
    labels = []
    coding_position = 0
    index = 0
    while index < len(positions):
        position = positions[index]
        if exon_mask[position - lo]:
            if cds_mask[position - lo]:
                state = _coding_state(coding_position, coding_length)
                coding_position += 1
            else:
                state = "U5" if coding_position == 0 else "U3"
            labeled_positions.append(position)
            labels.append(SIDX[state])
            index += 1
        else:
            end_index = index
            while end_index < len(positions) and not exon_mask[positions[end_index] - lo]:
                end_index += 1
            for position, state in zip(
                positions[index:end_index],
                _intron_states(end_index - index, coding_position, coding_length),
            ):
                labeled_positions.append(position)
                labels.append(SIDX[state])
            index = end_index
    return np.asarray(labeled_positions, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def context_features(sequence, positions, strand: str):
    positions = np.asarray(positions, dtype=np.int64)
    length = len(sequence)
    if strand == "+":
        raw_bases = sequence[positions]
    else:
        raw_bases = COMP[sequence[positions]]
    bases = np.where(raw_bases < NSYM, raw_bases, 0).astype(np.int64)
    valid = raw_bases < NSYM
    contexts = np.zeros(len(positions), dtype=np.int64)

    for offset in range(ORDER):
        source_positions = (
            positions - ORDER + offset
            if strand == "+"
            else positions + ORDER - offset
        )
        in_bounds = (source_positions >= 0) & (source_positions < length)
        source_bases = np.zeros(len(positions), dtype=np.uint8)
        if in_bounds.any():
            source = sequence[source_positions[in_bounds]]
            if strand == "-":
                source = COMP[source]
            source_bases[in_bounds] = source
            valid[in_bounds] &= source < NSYM
        contexts = contexts * NSYM + np.where(source_bases < NSYM, source_bases, 0)

    return bases, contexts, valid


def _accumulate_emissions(counts, sequence, positions, states, strand, weights=None):
    bases, contexts, valid = context_features(sequence, positions, strand)
    if not valid.any():
        return
    flat = (states[valid] * CTX + contexts[valid]) * NSYM + bases[valid]
    if weights is None:
        counts += np.bincount(flat, minlength=NSTATE * CTX * NSYM).reshape(
            NSTATE, CTX, NSYM
        )
    else:
        counts += np.bincount(
            flat,
            weights=weights[valid],
            minlength=NSTATE * CTX * NSYM,
        ).reshape(NSTATE, CTX, NSYM)


def _sample_uncovered(covered, count, rng):
    available = int(len(covered) - covered.sum())
    count = min(count, available)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == available:
        return np.flatnonzero(~covered).astype(np.int64)
    selected = np.empty(count, dtype=np.int64)
    filled = 0
    while filled < count:
        remaining = count - filled
        candidates = rng.integers(0, len(covered), size=max(1024, remaining * 2))
        candidates = candidates[~covered[candidates]]
        take = min(remaining, len(candidates))
        if take:
            selected[filled : filled + take] = candidates[:take]
            filled += take
    return selected


def fit_counts(sequence, transcripts, seqid: str, max_genes=None, background_sample=100_000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    all_transcripts = _select_transcripts(transcripts, seqid)
    if max_genes is not None and len(all_transcripts) > max_genes:
        selected = rng.choice(len(all_transcripts), size=max_genes, replace=False)
        selected_transcripts = [all_transcripts[int(index)] for index in selected]
    else:
        selected_transcripts = all_transcripts

    covered = np.zeros(len(sequence), dtype=np.bool_)
    for transcript in all_transcripts:
        start = min(a for a, _ in transcript["exons"])
        end = max(b for _, b in transcript["exons"])
        covered[start:end] = True

    emission_counts = np.zeros((NSTATE, CTX, NSYM), dtype=np.float64)
    transition_counts = np.zeros((NSTATE, NSTATE), dtype=np.float64)
    intergenic = SIDX["I"]
    for transcript in selected_transcripts:
        positions, states = label_transcript(transcript)
        _accumulate_emissions(
            emission_counts, sequence, positions, states, transcript["strand"]
        )
        if len(states):
            transition_counts += np.bincount(
                states[:-1] * NSTATE + states[1:], minlength=NSTATE * NSTATE
            ).reshape(NSTATE, NSTATE)
            transition_counts[intergenic, states[0]] += 1.0
            transition_counts[states[-1], intergenic] += 1.0

    uncovered_count = int(len(sequence) - covered.sum())
    background_positions = _sample_uncovered(covered, background_sample, rng)
    if len(background_positions):
        states = np.full(len(background_positions), intergenic, dtype=np.int64)
        weights = np.full(
            len(background_positions), uncovered_count / len(background_positions)
        )
        _accumulate_emissions(
            emission_counts, sequence, background_positions, states, "+", weights
        )
    transition_counts[intergenic, intergenic] += uncovered_count

    return {
        "emission_counts": emission_counts,
        "transition_counts": transition_counts,
        "genes": len(selected_transcripts),
    }


def normalize_counts(counts, emission_alpha=1.0, transition_alpha=0.5):
    emission_counts = np.asarray(counts["emission_counts"], dtype=np.float64) + emission_alpha
    emission = emission_counts / emission_counts.sum(axis=2, keepdims=True)
    log_emission = np.log(emission).astype(np.float32)

    transition_counts = np.asarray(counts["transition_counts"], dtype=np.float64).copy()
    log_transition = np.full((NSTATE, NSTATE), -np.inf, dtype=np.float32)
    for source, targets in ALLOWED.items():
        target_indices = np.asarray([SIDX[target] for target in targets], dtype=np.int64)
        row = transition_counts[SIDX[source], target_indices] + transition_alpha
        row /= row.sum()
        log_transition[SIDX[source], target_indices] = np.log(row).astype(np.float32)
    return {"log_emission": log_emission, "log_transition": log_transition}


def save_counts(path, counts):
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            emission_counts=np.asarray(counts["emission_counts"], dtype=np.float64),
            transition_counts=np.asarray(counts["transition_counts"], dtype=np.float64),
        )


def load_counts(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            "emission_counts": np.asarray(archive["emission_counts"], dtype=np.float64),
            "transition_counts": np.asarray(archive["transition_counts"], dtype=np.float64),
        }


def save_model(path, model):
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            log_emission=np.asarray(model["log_emission"], dtype=np.float32),
            log_transition=np.asarray(model["log_transition"], dtype=np.float32),
        )


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if {"log_emission", "log_transition"} <= keys:
            return {
                "log_emission": np.asarray(archive["log_emission"], dtype=np.float32),
                "log_transition": np.asarray(archive["log_transition"], dtype=np.float32),
            }
        if {"emission_counts", "transition_counts"} <= keys:
            counts = {
                "emission_counts": np.asarray(archive["emission_counts"], dtype=np.float64),
                "transition_counts": np.asarray(archive["transition_counts"], dtype=np.float64),
            }
        else:
            raise ValueError(f"{path}: expected GHMM parameter arrays")
    return normalize_counts(counts)


def emission(sequence, positions, strand, model, donor_bonus=None, acceptor_bonus=None):
    bases, contexts, valid = context_features(sequence, positions, strand)
    log_emission = model["log_emission"][:, contexts, bases].copy()
    for state, required in MASKS.items():
        disallowed = valid & ~np.isin(bases, required)
        log_emission[state, disallowed] = -1e9
    log_emission[:, ~valid] = 0.0
    if donor_bonus is not None:
        log_emission[DONOR_IDX, :] += donor_bonus[positions]
    if acceptor_bonus is not None:
        log_emission[ACCEPTOR_IDX, :] += acceptor_bonus[positions]
    return log_emission


@njit(cache=True)
def _viterbi(log_emission, log_transition, predecessor_ids, predecessor_counts, start_state, end_states):
    nstate, length = log_emission.shape
    negative = -1e30
    scores = np.full((length, nstate), negative, dtype=np.float64)
    back = np.zeros((length, nstate), dtype=np.int64)
    scores[0, start_state] = log_emission[start_state, 0]
    for position in range(1, length):
        for target in range(nstate):
            best_score = negative
            best_source = 0
            for index in range(predecessor_counts[target]):
                source = predecessor_ids[target, index]
                score = scores[position - 1, source] + log_transition[source, target]
                if score > best_score:
                    best_score = score
                    best_source = source
            scores[position, target] = best_score + log_emission[target, position]
            back[position, target] = best_source
    final_state = end_states[0]
    end_score = scores[length - 1, final_state]
    for index in range(1, len(end_states)):
        candidate = end_states[index]
        if scores[length - 1, candidate] > end_score:
            final_state = candidate
            end_score = scores[length - 1, candidate]
    states = np.empty(length, dtype=np.uint8)
    states[length - 1] = final_state
    for position in range(length - 1, 0, -1):
        states[position - 1] = back[position, states[position]]
    return states


def decode_full(sequence, model, block_size=200_000, overlap=100,
                donor_bonus=None, acceptor_bonus=None):
    length = len(sequence)
    states = np.empty(length, dtype=np.uint8)
    for core_start in range(0, length, block_size):
        core_end = min(core_start + block_size, length)
        start = max(0, core_start - overlap)
        end = min(length, core_end + overlap)
        positions = np.arange(start, end, dtype=np.int64)
        log_emission = emission(
            sequence, positions, "+", model, donor_bonus, acceptor_bonus
        )
        block_states = _viterbi(
            log_emission,
            model["log_transition"],
            PREDECESSOR_IDS,
            PREDECESSOR_COUNTS,
            SIDX["I"],
            END_STATES,
        )
        states[core_start:core_end] = block_states[core_start - start : core_end - start]
    return states


def _merge(spans, min_gap=1):
    result = []
    for start, end in sorted(spans):
        if result and start - result[-1][1] < min_gap:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def extract(states, length, strand):
    active = states != SIDX["I"]
    padded = np.pad(active, 1)
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    genes = []
    for start, end in zip(starts, ends):
        segment = states[start:end]
        if SIDX["S0"] not in segment or SIDX["T2"] not in segment:
            continue
        coding = np.flatnonzero(np.isin(segment, CODING_IDX)) + start
        exonic = np.flatnonzero(np.isin(segment, EXONIC_IDX)) + start
        cds = _merge([(int(pos), int(pos) + 1) for pos in coding], MIN_INTRON)
        exons = _merge([(int(pos), int(pos) + 1) for pos in exonic], MIN_INTRON)
        if strand == "-":
            cds = _merge(sorted((length - end, length - start) for start, end in cds), MIN_INTRON)
            exons = _merge(sorted((length - end, length - start) for start, end in exons), MIN_INTRON)
        if cds and exons:
            genes.append((cds, exons, strand))
    return genes


def predict(sequence, model, block_size=200_000, splice_bonus=None):
    forward_donor = forward_acceptor = reverse_donor = reverse_acceptor = None
    if splice_bonus is not None:
        forward_donor, forward_acceptor, reverse_donor, reverse_acceptor = splice_bonus
    forward_states = decode_full(
        sequence, model, block_size, donor_bonus=forward_donor, acceptor_bonus=forward_acceptor
    )
    forward_genes = extract(forward_states, len(sequence), "+")
    del forward_states
    reverse_sequence = COMP[sequence[::-1]]
    reverse_states = decode_full(
        reverse_sequence, model, block_size,
        donor_bonus=reverse_donor, acceptor_bonus=reverse_acceptor,
    )
    reverse_genes = extract(reverse_states, len(sequence), "-")
    del reverse_states, reverse_sequence
    return forward_genes + reverse_genes


def genes_to_transcripts(genes, seqid: str):
    return [
        {"tid": f"pseudo{index}", "gene": f"pseudo{index}", "seqid": seqid,
         "strand": strand, "exons": list(exons), "cds": list(cds)}
        for index, (cds, exons, strand) in enumerate(genes)
    ]


def write_gff3(path, seqid: str, genes):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##gff-version 3\n")
        for index, (cds, exons, strand) in enumerate(genes):
            tid = f"pred{index}"
            start = min(a for a, _ in exons)
            end = max(b for _, b in exons)
            handle.write(
                f"{seqid}\tghmm\tmRNA\t{start + 1}\t{end}\t.\t{strand}\t.\tID={tid}\n"
            )
            for start, end in exons:
                handle.write(
                    f"{seqid}\tghmm\texon\t{start + 1}\t{end}\t.\t{strand}\t.\tParent={tid}\n"
                )
            for start, end in cds:
                handle.write(
                    f"{seqid}\tghmm\tCDS\t{start + 1}\t{end}\t.\t{strand}\t0\tParent={tid}\n"
                )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=200_000)
    args = parser.parse_args(argv)
    try:
        seqid, sequence = read_fasta(args.fasta)
        model = load_model(args.model)
        genes = predict(sequence, model, args.block_size)
        write_gff3(args.output, seqid, genes)
        print(f"wrote {args.output} ({len(genes)} genes)")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
