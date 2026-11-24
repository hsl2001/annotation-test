import argparse
import gzip
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from numba import njit

import umap_ss_plot as u

Q2 = 28.0
NSYM = 8
ORDER = 3
CTX = NSYM**ORDER
MIN_INTRON = 60
PHI = np.array([0, 4, 2, 3, 1, 5])
N_OFFSET = 2 * u.RADIUS + 1
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

STATES = ["I", "U5", "U3"]
STATES += [f"E{f}" for f in range(3)]
STATES += [f"S{f}" for f in range(3)]
STATES += [f"T{f}" for f in range(3)]
STATES += [f"D{f}_{k}" for f in range(3) for k in range(2)]
STATES += [f"N{p}" for p in range(3)]
STATES += [f"A{p}_{k}" for p in range(3) for k in range(2)]
SIDX = {s: i for i, s in enumerate(STATES)}
NSTATE = len(STATES)

ALLOWED = defaultdict(set)
for a, b in [
    ("I", "I"), ("I", "U5"), ("I", "S0"), ("U5", "U5"), ("U5", "S0"),
    ("S0", "S1"), ("S1", "S2"), ("S2", "E0"), ("T0", "T1"), ("T1", "T2"),
    ("T2", "U3"), ("T2", "I"), ("U3", "U3"), ("U3", "I"),
]:
    ALLOWED[a].add(b)
for f in range(3):
    ALLOWED[f"E{f}"].update({f"E{(f + 1) % 3}", f"D{f}_0", "T0"})
    ALLOWED[f"D{f}_0"].add(f"D{f}_1")
    ALLOWED[f"D{f}_1"].add(f"N{(f + 1) % 3}")
for p in range(3):
    ALLOWED[f"N{p}"].update({f"N{p}", f"A{p}_0"})
    ALLOWED[f"A{p}_0"].add(f"A{p}_1")
    ALLOWED[f"A{p}_1"].add(f"E{p}")

CODING_IDX = np.array([SIDX[s] for s in STATES if s[0] in "SET"])
EXONIC_IDX = np.array([SIDX[s] for s in STATES if s[0] in "SETU"])

MASKS = {
    SIDX["S0"]: [0], SIDX["S1"]: [3], SIDX["S2"]: [2],
    SIDX["T0"]: [3], SIDX["T1"]: [0, 2], SIDX["T2"]: [0, 2],
}
ENDS = np.array([SIDX["I"], SIDX["T2"], SIDX["U3"]], dtype=np.int64)

DATASETS = {
    "arabidopsis": ("data/truth.col.chr5.100k.fasta.gz", "data/truth.col.chr5.100k.gff3.gz"),
    "arabidopsis_3m": ("data/truth.col.chr5.3m.fasta.gz", "data/truth.col.chr5.3m.gff3.gz"),
    "arabidopsis_chr5": ("data/truth.col.chr5.fasta.gz", "data/truth.col.chr5.gff3.gz"),
    "rice": ("data/truth.nip.chr5.100k.fasta.gz", "data/truth.nip.chr5.100k.gff3.gz"),
    "rice_3m": ("data/truth.nip.chr5.3m.fasta.gz", "data/truth.nip.chr5.3m.gff3.gz"),
    "rice_chr5": ("data/truth.nip.chr5.fasta.gz", "data/truth.nip.chr5.gff3.gz"),
    "tomato": ("data/truth.slm.chr5.100k.fasta.gz", "data/truth.slm.chr5.100k.gff3.gz"),
    "tomato_3m": ("data/truth.slm.chr5.3m.fasta.gz", "data/truth.slm.chr5.3m.gff3.gz"),
    "tomato_chr5": ("data/truth.slm.chr5.fasta.gz", "data/truth.slm.chr5.gff3.gz"),
    "human": ("data/truth.human.chr21.fasta.gz", "data/truth.human.chr21.gff3.gz"),
    "mouse": ("data/truth.mouse.chr19.fasta.gz", "data/truth.mouse.chr19.gff3.gz"),
    "zebrafish": ("data/truth.zebrafish.chr25.fasta.gz", "data/truth.zebrafish.chr25.gff3.gz"),
    "celegans": ("data/truth.elegans.chrI.fasta.gz", "data/truth.elegans.chrI.gff3.gz"),
    "vp": ("data/truth.vp.chr12.fasta.gz", "data/truth.vp.chr12.gff3.gz"),
    "vp_3m": ("data/truth.vp.chr12.3m.fasta", "data/truth.vp.chr12.3m.gff3"),
}


def min_sum(features):
    n = features.shape[0]
    w = features.reshape(n, N_OFFSET, 12)
    mag = np.hypot(w[..., :6], w[..., 6:])
    return np.minimum(mag, mag[..., PHI]).sum(axis=(1, 2))


def _block_scalar(job):
    seq, a, b = job
    bank = u.CwtFeatureBank(seq, (3,), u.W0)
    pos = np.arange(a, b)
    return a, min_sum(u.cwt_features(bank, pos, np.zeros(len(pos), dtype=np.int8)))


def scalar_track(seq, block=500_000, workers=None):
    jobs = [(seq, a, min(a + block, len(seq))) for a in range(0, len(seq), block)]
    out = np.empty(len(seq))
    with Pool(workers) as pool:
        for a, v in pool.imap_unordered(_block_scalar, jobs):
            out[a : a + len(v)] = v
    return out


def encode(bases, s):
    b = np.clip(bases, 0, 3)
    return (b * 2 + (s >= Q2)).astype(np.int64)


def context_idx(codes_arr):
    c = codes_arr.astype(np.int64)
    padded = np.concatenate([np.zeros(ORDER, np.int64), c])
    idx = np.zeros(len(c), np.int64)
    for k in range(ORDER):
        idx = idx * NSYM + padded[k : k + len(c)]
    return idx


def _attrs(text):
    return dict(p.split("=", 1) for p in text.split(";") if "=" in p)


def _merge(spans, min_gap=1):
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] < min_gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _overlaps(tx, region):
    return region is not None and any(a < region[1] and region[0] < b for a, b in tx["exons"])


def read_transcripts(path):
    opener = gzip.open if path.endswith(".gz") else open
    info, exons, cds = {}, defaultdict(list), defaultdict(list)
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9:
                continue
            kind, attrs = f[2].lower(), _attrs(f[8])
            if kind == "mrna":
                if attrs.get("ID"):
                    info[attrs["ID"]] = (f[0], f[6], attrs.get("Parent", attrs["ID"]))
            elif kind in ("exon", "cds"):
                span = (int(f[3]) - 1, int(f[4]))
                for p in attrs.get("Parent", "").split(","):
                    if p:
                        (exons if kind == "exon" else cds)[p].append(span)
    return [{"seqid": seqid, "strand": strand, "gene": gene,
             "exons": _merge(exons.get(tid, cds[tid])), "cds": _merge(cds[tid])}
            for tid, (seqid, strand, gene) in info.items() if cds.get(tid)]


def _coding_state(k, n_cds):
    if k < 3:
        return f"S{k}"
    if k >= n_cds - 3:
        return f"T{k - (n_cds - 3)}"
    return f"E{k % 3}"


def _intron_states(m, k, n_cds):
    if 0 < k < n_cds and m >= 4:
        f, p = (k - 1) % 3, k % 3
        return [f"D{f}_0", f"D{f}_1"] + [f"N{p}"] * (m - 4) + [f"A{p}_0", f"A{p}_1"]
    return ["N0"] * m


def label_transcript(tx):
    a0 = min(a for a, _ in tx["exons"])
    b0 = max(b for _, b in tx["exons"])
    order = range(b0 - 1, a0 - 1, -1) if tx["strand"] == "-" else range(a0, b0)
    exon = np.zeros(b0 - a0, bool)
    cds = np.zeros(b0 - a0, bool)
    for a, b in tx["exons"]:
        exon[a - a0 : b - a0] = True
    for a, b in tx["cds"]:
        cds[a - a0 : b - a0] = True
    n_cds = int(cds.sum())
    pos = list(order)
    P, S = [], []
    k, i = 0, 0
    while i < len(pos):
        p = pos[i]
        if exon[p - a0]:
            if cds[p - a0]:
                st = _coding_state(k, n_cds)
                k += 1
            else:
                st = "U5" if k == 0 else "U3"
            P.append(p)
            S.append(SIDX[st])
            i += 1
        else:
            j = i
            while j < len(pos) and not exon[pos[j] - a0]:
                j += 1
            run = pos[i:j]
            for q, st in zip(run, _intron_states(len(run), k, n_cds)):
                P.append(q)
                S.append(SIDX[st])
            i = j
    return np.array(P, np.int64), np.array(S, np.int64)


def _select_transcripts(transcripts, seqid, exclude):
    best = {}
    for tx in transcripts:
        if tx["seqid"] != seqid or _overlaps(tx, exclude):
            continue
        ncds = sum(b - a for a, b in tx["cds"])
        if tx["gene"] not in best or ncds > best[tx["gene"]][0]:
            best[tx["gene"]] = (ncds, tx)
    return [tx for _, tx in best.values()]


def _smooth_transitions(trans):
    for a, succ in ALLOWED.items():
        for c2 in succ:
            if trans[SIDX[a], SIDX[c2]] == 0:
                trans[SIDX[a], SIDX[c2]] = 0.5


def _log_transitions(trans):
    logT = np.full((NSTATE, NSTATE), -np.inf)
    for a, succ in ALLOWED.items():
        total = trans[SIDX[a], [SIDX[c2] for c2 in succ]].sum()
        for c2 in succ:
            logT[SIDX[a], SIDX[c2]] = np.log(trans[SIDX[a], SIDX[c2]] / total)
    return logT


def train(seq, s, transcripts, seqid, exclude=None):
    codes_fwd = encode(seq, s)
    ctx_fwd = context_idx(codes_fwd)
    codes_rev = encode(COMP[seq], s)
    ctx_rev = context_idx(codes_rev[::-1])[::-1]
    transcripts = _select_transcripts(transcripts, seqid, exclude)
    idx_parts = []
    trans = np.zeros((NSTATE, NSTATE))
    covered = np.zeros(len(seq), bool)
    I = SIDX["I"]
    for tx in transcripts:
        covered[min(a for a, _ in tx["exons"]) : max(b for _, b in tx["exons"])] = True
        pos, sts = label_transcript(tx)
        if tx["strand"] == "-":
            c, cx = codes_rev[pos], ctx_rev[pos]
        else:
            c, cx = codes_fwd[pos], ctx_fwd[pos]
        idx_parts.append((sts * CTX + cx) * NSYM + c)
        trans += np.bincount(sts[:-1] * NSTATE + sts[1:], minlength=NSTATE**2).reshape(NSTATE, NSTATE)
        trans[I, sts[0]] += 1
        trans[sts[-1], I] += 1
    if exclude is not None:
        covered[exclude[0] : exclude[1]] = True
    ig = np.flatnonzero(~covered)
    c, cx = codes_fwd[ig], ctx_fwd[ig]
    idx_parts.append((I * CTX + cx) * NSYM + c)
    trans[I, I] = max(1, len(ig))
    _smooth_transitions(trans)
    markov = np.bincount(np.concatenate(idx_parts), minlength=NSTATE * CTX * NSYM).reshape(NSTATE, CTX, NSYM) + 1.0
    logmarkov = np.log(markov / markov.sum(2, keepdims=True))
    return logmarkov.astype(np.float32), _log_transitions(trans), len(transcripts)


def emission(codes_arr, logmarkov):
    ctx = context_idx(codes_arr)
    logE = logmarkov[:, ctx, codes_arr]
    b = codes_arr // 2
    for st, req in MASKS.items():
        logE[st, ~np.isin(b, req)] = -1e9
    return logE


@njit(cache=True)
def _viterbi(logE, logT, start, ends):
    n, length = logE.shape
    neg = -1e18
    dp = np.full((length, n), neg)
    ptr = np.zeros((length, n), np.int64)
    dp[0, start] = logE[start, 0]
    for t in range(1, length):
        for s in range(n):
            best, ba = neg, 0
            for a in range(n):
                v = dp[t - 1, a] + logT[a, s]
                if v > best:
                    best, ba = v, a
            dp[t, s] = best + logE[s, t]
            ptr[t, s] = ba
    be = ends[np.argmax(dp[length - 1, ends])]
    st = np.empty(length, np.int64)
    st[length - 1] = be
    for t in range(length - 1, 0, -1):
        st[t - 1] = ptr[t, st[t]]
    return st


def viterbi(logE, logT):
    return _viterbi(np.ascontiguousarray(logE, dtype=np.float64),
                    np.ascontiguousarray(logT), SIDX["I"], ENDS)


_DECODE = {}


def _init_decode(seq, s, lm, lt):
    _DECODE.update(seq=seq, s=s, lm=lm, lt=lt)


def _decode_block(job):
    a, b, ca, cb = job
    codes = encode(_DECODE["seq"][a:b], _DECODE["s"][a:b])
    st = viterbi(emission(codes, _DECODE["lm"]), _DECODE["lt"])
    return ca, st[ca - a : cb - a]


def decode_full(seq, s, lm, lt, workers, block=1_000_000, ov=200):
    L = len(seq)
    jobs = [(max(0, c - ov), min(L, c + block + ov), c, min(c + block, L))
            for c in range(0, L, block)]
    states = np.zeros(L, np.int64)
    with Pool(workers, initializer=_init_decode, initargs=(seq, s, lm, lt)) as pool:
        for ca, st in pool.imap_unordered(_decode_block, jobs):
            states[ca : ca + len(st)] = st
    return states


def extract(states, L, strand):
    out = []
    active = states != SIDX["I"]
    padded = np.pad(active, 1)
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for t, j in zip(starts, ends):
        seg = states[t:j]
        if SIDX["S0"] not in seg or SIDX["T2"] not in seg:
            continue
        coding = np.flatnonzero(np.isin(seg, CODING_IDX)) + t
        exonic = np.flatnonzero(np.isin(seg, EXONIC_IDX)) + t
        cds = _merge([(int(x), int(x) + 1) for x in coding], MIN_INTRON)
        exons = _merge([(int(x), int(x) + 1) for x in exonic], MIN_INTRON)
        if strand == "-":
            cds = _merge(sorted((L - y, L - x) for x, y in cds), MIN_INTRON)
            exons = _merge(sorted((L - y, L - x) for x, y in exons), MIN_INTRON)
        if cds:
            out.append((cds, exons, strand))
    return out


def _parse_region(text):
    if not text:
        return None
    start, end = map(int, text.split(":"))
    return start, end


def _load_dataset(key, workers):
    recs = u.read_fasta(DATASETS[key][0])
    name = next(iter(recs))
    seq = recs[name]
    return name, seq, scalar_track(seq, workers=workers).astype(np.float32)


def _gff3_lines(name, genes):
    lines = []
    for n, (cds, exons, strand) in enumerate(genes):
        tid = f"pred{n}"
        lo = min(x for x, _ in exons)
        hi = max(y for _, y in exons)
        lines.append(f"{name}\tghmm\tmRNA\t{lo + 1}\t{hi}\t.\t{strand}\t.\tID={tid}\n")
        lines.extend(f"{name}\tghmm\texon\t{x + 1}\t{y}\t.\t{strand}\t.\tParent={tid}\n" for x, y in exons)
        lines.extend(f"{name}\tghmm\tCDS\t{x + 1}\t{y}\t.\t{strand}\t0\tParent={tid}\n" for x, y in cds)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="arabidopsis_chr5", choices=DATASETS)
    parser.add_argument("--decode", default=None, choices=DATASETS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--exclude", default=None,
                        help="START:END (0-based, half-open) excluded from training")
    args = parser.parse_args()
    decode_key = args.decode or args.train
    exclude = _parse_region(args.exclude)

    tname, tseq, ts = _load_dataset(args.train, args.workers)
    ttrans = read_transcripts(DATASETS[args.train][1])
    lm, lt, n_train = train(tseq, ts, ttrans, tname, exclude)
    print(f"trained on {args.train}: {n_train} transcripts")

    if decode_key == args.train:
        name, seq, s = tname, tseq, ts
    else:
        name, seq, s = _load_dataset(decode_key, args.workers)
    L = len(seq)
    rseq = COMP[seq[::-1]]

    fwd = decode_full(seq, s, lm, lt, args.workers)
    rev = decode_full(rseq, s[::-1], lm, lt, args.workers)
    genes = extract(fwd, L, "+") + extract(rev, L, "-")

    out = f"ghmm.{decode_key}.gff3"
    with open(out, "w") as fh:
        fh.writelines(_gff3_lines(name, genes))
    print(f"wrote {out} ({len(genes)} genes)")


if __name__ == "__main__":
    main()
