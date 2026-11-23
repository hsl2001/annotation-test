#!/usr/bin/env python3
import argparse
from collections import defaultdict
from itertools import pairwise
from multiprocessing import Pool

import numpy as np
from numba import njit

import umap_ss_plot as u

Q2 = np.array([25.916, 27.273, 27.955, 26.239])
NSYM = 8
ORDER = 5
CTX = NSYM**ORDER
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

CODING = {s for s in STATES if s[0] in "SET"}
EXONIC = CODING | {"U5", "U3"}
CODING_IDX = np.array([SIDX[s] for s in CODING])
EXONIC_IDX = np.array([SIDX[s] for s in EXONIC])

MASKS = {
    SIDX["S0"]: [0], SIDX["S1"]: [3], SIDX["S2"]: [2],
    SIDX["T0"]: [3], SIDX["T1"]: [0, 2], SIDX["T2"]: [0, 2],
}

DATASETS = {
    "col": ("data/truth.col.chr5.100k.fasta.gz", "data/truth.col.chr5.100k.gff3.gz"),
    "nip": ("data/truth.nip.chr5.100k.fasta.gz", "data/truth.nip.chr5.100k.gff3.gz"),
    "slm": ("data/truth.slm.chr5.100k.fasta.gz", "data/truth.slm.chr5.100k.gff3.gz"),
    "col3m": ("data/truth.col.chr5.3m.fasta.gz", "data/truth.col.chr5.3m.gff3.gz"),
    "nip3m": ("data/truth.nip.chr5.3m.fasta.gz", "data/truth.nip.chr5.3m.gff3.gz"),
    "slm3m": ("data/truth.slm.chr5.3m.fasta.gz", "data/truth.slm.chr5.3m.gff3.gz"),
    "colfull": ("data/truth.col.chr5.fasta.gz", "data/truth.col.chr5.gff3.gz"),
    "nipfull": ("data/truth.nip.chr5.fasta.gz", "data/truth.nip.chr5.gff3.gz"),
    "slmfull": ("data/truth.slm.chr5.fasta.gz", "data/truth.slm.chr5.gff3.gz"),
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
    return (b * 2 + (s >= Q2[b])).astype(np.int64)


def context_idx(codes_arr):
    c = codes_arr.astype(np.int64)
    padded = np.concatenate([np.zeros(ORDER, np.int64), c])
    idx = np.zeros(len(c), np.int64)
    for k in range(ORDER):
        idx = idx * NSYM + padded[k : k + len(c)]
    return idx


def _attrs(text):
    return dict(p.split("=", 1) for p in text.split(";") if "=" in p)


def _merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def read_transcripts(path):
    import gzip

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
    out = []
    for tid, (seqid, strand, gene) in info.items():
        if cds.get(tid):
            out.append(dict(seqid=seqid, strand=strand, gene=gene,
                            exons=_merge(exons.get(tid, cds[tid])), cds=_merge(cds[tid])))
    return out


def label_transcript(seq, s, tx):
    a0 = min(a for a, _ in tx["exons"])
    b0 = max(b for _, b in tx["exons"])
    minus = tx["strand"] == "-"
    order = range(b0 - 1, a0 - 1, -1) if minus else range(a0, b0)
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
                st = f"S{k}" if k < 3 else (f"T{k - (n_cds - 3)}" if k >= n_cds - 3 else f"E{k % 3}")
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
            m = len(run)
            if 0 < k < n_cds and m >= 4:
                f, p2 = (k - 1) % 3, k % 3
                for r, q in enumerate(run):
                    st = (f"D{f}_0" if r == 0 else f"D{f}_1" if r == 1
                          else f"A{p2}_0" if r == m - 2 else f"A{p2}_1" if r == m - 1
                          else f"N{p2}")
                    P.append(q)
                    S.append(SIDX[st])
            else:
                for q in run:
                    P.append(q)
                    S.append(SIDX["N0"])
            i = j
    return np.array(P, np.int64), np.array(S, np.int64)


def train(seq, transcripts, seqid, codes_fwd, ctx_fwd, codes_rev, ctx_rev):
    idx_parts, c0_parts = [], []
    trans = np.zeros((NSTATE, NSTATE))
    covered = np.zeros(len(seq), bool)

    best = {}
    for tx in transcripts:
        if tx["seqid"] != seqid:
            continue
        ncds = sum(b - a for a, b in tx["cds"])
        key = (tx["seqid"], tx["gene"])
        if key not in best or ncds > best[key][0]:
            best[key] = (ncds, tx)
    transcripts = [tx for _, tx in best.values()]

    for tx in transcripts:
        if tx["seqid"] != seqid:
            continue
        covered[min(a for a, _ in tx["exons"]) : max(b for _, b in tx["exons"])] = True
        pos, sts = label_transcript(seq, None, tx)
        if not len(sts):
            continue
        if tx["strand"] == "-":
            c, cx = codes_rev[pos], ctx_rev[pos]
        else:
            c, cx = codes_fwd[pos], ctx_fwd[pos]
        idx_parts.append((sts * CTX + cx) * NSYM + c)
        c0_parts.append(sts * NSYM + c)
        trans += np.bincount(sts[:-1] * NSTATE + sts[1:], minlength=NSTATE**2).reshape(NSTATE, NSTATE)

    I = SIDX["I"]
    ig = np.flatnonzero(~covered)
    c, cx = codes_fwd[ig], ctx_fwd[ig]
    idx_parts.append((I * CTX + cx) * NSYM + c)
    c0_parts.append(I * NSYM + c)
    trans[I, I] = max(1, len(ig))
    for a, succ in ALLOWED.items():
        for c2 in succ:
            if trans[SIDX[a], SIDX[c2]] == 0:
                trans[SIDX[a], SIDX[c2]] = 0.5

    markov = np.bincount(np.concatenate(idx_parts), minlength=NSTATE * CTX * NSYM).reshape(NSTATE, CTX, NSYM) + 1.0
    c0 = np.bincount(np.concatenate(c0_parts), minlength=NSTATE * NSYM).reshape(NSTATE, NSYM) + 1.0
    p0 = c0 / c0.sum(1, keepdims=True)
    alpha = 5.0
    logmarkov = np.log((markov + alpha * p0[:, None, :]) / (markov.sum(2, keepdims=True) + alpha))

    logT = np.full((NSTATE, NSTATE), -np.inf)
    for a, succ in ALLOWED.items():
        tot = trans[SIDX[a], [SIDX[c2] for c2 in succ]].sum()
        for c2 in succ:
            logT[SIDX[a], SIDX[c2]] = np.log(trans[SIDX[a], SIDX[c2]] / tot)
    return logmarkov.astype(np.float32), logT


def emission(codes_arr, logmarkov):
    ctx = context_idx(codes_arr)
    L = len(codes_arr)
    logE = np.empty((NSTATE, L), np.float32)
    for st in range(NSTATE):
        logE[st] = logmarkov[st][ctx, codes_arr]
    b = codes_arr // 2
    for st, req in MASKS.items():
        invalid = ~np.isin(b, req)
        if invalid.any():
            logE[st, invalid] = -1e9
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
    be, bv = ends[0], dp[length - 1, ends[0]]
    for e in ends[1:]:
        if dp[length - 1, e] > bv:
            bv, be = dp[length - 1, e], e
    st = np.empty(length, np.int64)
    st[length - 1] = be
    for t in range(length - 1, 0, -1):
        st[t - 1] = ptr[t, st[t]]
    return st


def viterbi(logE, logT):
    ends = np.array([SIDX["I"], SIDX["T2"], SIDX["U3"]], dtype=np.int64)
    return _viterbi(np.ascontiguousarray(logE, dtype=np.float64),
                    np.ascontiguousarray(logT), SIDX["I"], ends)


_G = {}


def _init(seq, s, lm, lt):
    _G["seq"], _G["s"], _G["lm"], _G["lt"] = seq, s, lm, lt


def _decode_job(job):
    a, b, ca, cb = job
    codes = encode(_G["seq"][a:b], _G["s"][a:b])
    st = viterbi(emission(codes, _G["lm"]), _G["lt"])
    return ca, st[ca - a : cb - a]


def decode_full(seq, s, lm, lt, workers, block=1_000_000, ov=200):
    L = len(seq)
    jobs = [(max(0, c - ov), min(L, c + block + ov), c, min(c + block, L))
            for c in range(0, L, block)]
    states = np.zeros(L, np.int64)
    with Pool(workers, initializer=_init, initargs=(seq, s, lm, lt)) as pool:
        for ca, st in pool.imap_unordered(_decode_job, jobs):
            states[ca : ca + len(st)] = st
    return states


def extract(states, L, strand):
    out, t, n = [], 0, len(states)
    while t < n:
        if states[t] == SIDX["I"]:
            t += 1
            continue
        j = t
        while j < n and states[j] != SIDX["I"]:
            j += 1
        seg = states[t:j]
        if SIDX["S0"] in seg and SIDX["T2"] in seg:
            coding = np.flatnonzero(np.isin(seg, CODING_IDX)) + t
            exonic = np.flatnonzero(np.isin(seg, EXONIC_IDX)) + t
            cds = _merge([(int(x), int(x) + 1) for x in coding])
            exons = _merge([(int(x), int(x) + 1) for x in exonic])
            if strand == "-":
                cds = _merge(sorted((L - y, L - x) for x, y in cds))
                exons = _merge(sorted((L - y, L - x) for x, y in exons))
            if cds:
                out.append((cds, exons, strand))
        t = j
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="colfull", choices=DATASETS)
    parser.add_argument("--decode", default=None, choices=DATASETS)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    decode_key = args.decode or args.train

    trecs = u.read_fasta(DATASETS[args.train][0])
    tname = next(iter(trecs))
    tseq = trecs[tname]
    ttrans = read_transcripts(DATASETS[args.train][1])
    ts = scalar_track(tseq, workers=args.workers).astype(np.float32)

    codes_fwd = encode(tseq, ts)
    ctx_fwd = context_idx(codes_fwd)
    tcodes_rev = encode(COMP[tseq], ts)
    ctx_rev = context_idx(tcodes_rev[::-1])[::-1]
    lm, lt = train(tseq, ttrans, tname, codes_fwd, ctx_fwd, tcodes_rev, ctx_rev)
    print(f"trained on {args.train}: {len(ttrans)} transcripts")
    del codes_fwd, ctx_fwd, tcodes_rev, ctx_rev

    if decode_key == args.train:
        name, seq, s = tname, tseq, ts
    else:
        recs = u.read_fasta(DATASETS[decode_key][0])
        name = next(iter(recs))
        seq = recs[name]
        s = scalar_track(seq, workers=args.workers).astype(np.float32)
    L = len(seq)

    fwd = decode_full(seq, s, lm, lt, args.workers)
    rev = decode_full(COMP[seq[::-1]], s[::-1], lm, lt, args.workers)
    genes = extract(fwd, L, "+") + extract(rev, L, "-")

    out = f"ghmm.{decode_key}.gff3"
    with open(out, "w") as fh:
        for n, (cds, exons, strand) in enumerate(genes):
            tid = f"pred{n}"
            lo = min(x for x, _ in exons)
            hi = max(y for _, y in exons)
            fh.write(f"{name}\tghmm\tmRNA\t{lo + 1}\t{hi}\t.\t{strand}\t.\tID={tid}\n")
            for x, y in exons:
                fh.write(f"{name}\tghmm\texon\t{x + 1}\t{y}\t.\t{strand}\t.\tParent={tid}\n")
            for x, y in cds:
                fh.write(f"{name}\tghmm\tCDS\t{x + 1}\t{y}\t.\t{strand}\t0\tParent={tid}\n")
    print(f"wrote {out} ({len(genes)} genes)")


if __name__ == "__main__":
    main()