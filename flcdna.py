#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flcdna_eval_bam.py
==================
Evaluation of a GFF3/GTF annotation (-b) against full-length cDNA
long-read alignments supplied as BAM (-a).

Outputs
-------
<prefix>.transcript_metrics.tsv.gz     per-transcript support metrics
<prefix>.gene_metrics.tsv.gz           per-gene rollup
<prefix>.introns_unsupported.bed       annotated introns with NO FL junction support
<prefix>.novel_junctions.bed           FL junctions absent from the annotation
<prefix>.unannotated_covered.bed       covered, non-exonic runs (candidate novel loci)
<prefix>.summary.json                  machine-readable summary
<prefix>.report.txt                    human report
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysam

LOG = logging.getLogger("flcdna")

RE_GTF_TID = re.compile(r'transcript_id[ =]"?([^\";]+)"?')
RE_GTF_GID = re.compile(r'gene_id[ =]"?([^\";]+)"?')
RE_PARENT = re.compile(r'(?:^|;)\s*Parent=([^;]+)')
RE_ID = re.compile(r'(?:^|;)\s*ID=([^;]+)')
RE_GENE_NAME = re.compile(r'(?:^|;)\s*(?:gene_name|Name|gene)[ =]"?([^\";]+)"?')

BIG = 1 << 33          # sentinel scale for segmented prefix-max (int64-safe)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def _runs(mask):
    """[(start, end)) runs of True in a 1-D bool array."""
    if not mask.any():
        return []
    d = np.diff(mask.view(np.int8), prepend=np.int8(0), append=np.int8(0))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def merge_intervals(iv):
    """iv: (n,2) int array of [start, end).  Returns list of merged (s, e)."""
    if iv.size == 0:
        return []
    o = np.argsort(iv[:, 0], kind="stable")
    s = iv[o, 0]
    e = iv[o, 1]
    cs = np.maximum.accumulate(e)
    prev = np.empty_like(cs)
    prev[0] = -1
    prev[1:] = cs[:-1]
    idx = np.flatnonzero(s > prev)
    starts = s[idx]
    ends = np.maximum.reduceat(e, idx)
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------

def load_annotation(path):
    t0 = time.time()
    tx_names, tx_map, tx_gene = [], {}, []
    tx_chrom, tx_strand = array("i"), array("b")
    chrom_codes, chrom_names = {}, []

    ex_tx, ex_s, ex_e = array("q"), array("q"), array("q")
    cds_tx, cds_s, cds_e = array("q"), array("q"), array("q")
    tx_parent = {}

    def ccode(c):
        v = chrom_codes.get(c)
        if v is None:
            v = len(chrom_names)
            chrom_codes[c] = v
            chrom_names.append(c)
        return v

    def txidx(tid, chrom, strand, gid=None):
        i = tx_map.get(tid)
        if i is None:
            i = len(tx_names)
            tx_map[tid] = i
            tx_names.append(tid)
            tx_chrom.append(ccode(chrom))
            tx_strand.append(strand)
            tx_gene.append(gid if gid is not None else "")
        elif gid and not tx_gene[i]:
            tx_gene[i] = gid
        return i

    fmt = None
    n = 0
    with _open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat = f[2]
            if feat not in ("exon", "CDS", "transcript", "mRNA", "gene"):
                continue
            n += 1
            attr = f[8]
            if fmt is None:
                fmt = "gtf" if "transcript_id" in attr else "gff3"
            strand = 1 if f[6] == "+" else (-1 if f[6] == "-" else 0)

            if feat == "gene":
                continue
            if feat in ("transcript", "mRNA"):
                mid = RE_ID.search(attr)
                mpa = RE_PARENT.search(attr)
                if mid:
                    tx_parent[mid.group(1)] = mpa.group(1).split(",")[0] if mpa else ""
                continue

            if fmt == "gtf":
                mt = RE_GTF_TID.search(attr)
                if not mt:
                    continue
                mg = RE_GTF_GID.search(attr)
                tid = mt.group(1)
                gid = mg.group(1) if mg else None
            else:
                mpa = RE_PARENT.search(attr)
                if not mpa:
                    continue
                tid = mpa.group(1).split(",")[0]
                gid = None

            i = txidx(tid, f[0], strand, gid)
            if feat == "exon":
                ex_tx.append(i); ex_s.append(int(f[3])); ex_e.append(int(f[4]))
            else:
                cds_tx.append(i); cds_s.append(int(f[3])); cds_e.append(int(f[4]))

    if fmt == "gff3":
        for tid, gene in tx_parent.items():
            i = tx_map.get(tid)
            if i is not None and gene:
                tx_gene[i] = gene
    for i, g in enumerate(tx_gene):
        if not g:
            tx_gene[i] = tx_names[i]

    LOG.info("parsed %s: %d lines, %d transcripts, format=%s (%.1fs)",
             os.path.basename(path), n, len(tx_names), fmt, time.time() - t0)

    return dict(tx_names=tx_names, tx_gene=tx_gene, tx_chrom=tx_chrom,
                tx_strand=tx_strand, chrom_names=chrom_names,
                ex_tx=ex_tx, ex_s=ex_s, ex_e=ex_e,
                cds_tx=cds_tx, cds_s=cds_s, cds_e=cds_e)


def finalize_annotation(raw, dedupe=True):
    """Merge exons per transcript, derive introns/TSS/TES.  Coords 0-based half-open."""
    ntx = len(raw["tx_names"])
    tx_name = np.asarray(raw["tx_names"], dtype=object)
    tx_gene = np.asarray(raw["tx_gene"], dtype=object)
    tx_chrom = np.asarray(raw["tx_chrom"], dtype=np.int32)
    tx_strand = np.asarray(raw["tx_strand"], dtype=np.int8)

    ex_tx = np.asarray(raw["ex_tx"], dtype=np.int64)
    ex_s = np.asarray(raw["ex_s"], dtype=np.int64)
    ex_e = np.asarray(raw["ex_e"], dtype=np.int64)

    cds_tx = np.asarray(raw["cds_tx"], dtype=np.int64)
    if ex_tx.size == 0:
        if cds_tx.size == 0:
            raise SystemExit("[fatal] no exon/CDS features in annotation")
        LOG.warning("no exon features; using CDS as exon structure")
        ex_tx, ex_s, ex_e = cds_tx, np.asarray(raw["cds_s"], np.int64), np.asarray(raw["cds_e"], np.int64)
    elif cds_tx.size:
        has = np.zeros(ntx, dtype=bool)
        has[ex_tx] = True
        add = ~has[cds_tx]
        if add.any():
            ex_tx = np.concatenate([ex_tx, cds_tx[add]])
            ex_s = np.concatenate([ex_s, np.asarray(raw["cds_s"], np.int64)[add]])
            ex_e = np.concatenate([ex_e, np.asarray(raw["cds_e"], np.int64)[add]])
            LOG.info("added %d CDS-only transcripts", int(add.sum()))

    # 1-based inclusive -> 0-based half-open
    ex_s = ex_s - 1

    # ---- merge overlapping/adjacent exons per transcript (vectorised) ----
    order = np.lexsort((ex_e, ex_s, ex_tx))
    et, S, E = ex_tx[order], ex_s[order], ex_e[order]
    n = et.size
    gstart = np.empty(n, dtype=bool)
    gstart[0] = True
    gstart[1:] = et[1:] != et[:-1]
    gid = np.cumsum(gstart) - 1
    acc = np.maximum.accumulate(E + gid * BIG) - gid * BIG     # segmented prefix max
    prev = np.empty(n, dtype=np.int64)
    prev[0] = -1
    prev[1:] = acc[:-1]
    newrun = gstart | (S > prev)

    ridx = np.flatnonzero(newrun)
    mex_s = S[ridx]
    mex_e = np.maximum.reduceat(E, ridx)
    mex_tx = et[ridx]

    counts = np.bincount(mex_tx, minlength=ntx).astype(np.int64)
    tx_off = np.zeros(ntx + 1, dtype=np.int64)
    np.cumsum(counts, out=tx_off[1:])
    keep_tx = counts > 0
    if not keep_tx.all():
        LOG.warning("dropping %d transcripts without exon structure", int((~keep_tx).sum()))

    # ---- exact duplicate structures ----
    n_dup = 0
    if dedupe:
        seen = set()
        drop = np.zeros(ntx, dtype=bool)
        for i in np.flatnonzero(keep_tx):
            a, b = tx_off[i], tx_off[i + 1]
            key = (int(tx_chrom[i]), int(tx_strand[i]),
                   mex_s[a:b].tobytes(), mex_e[a:b].tobytes())
            if key in seen:
                drop[i] = True
            else:
                seen.add(key)
        n_dup = int(drop.sum())
        keep_tx &= ~drop
        if n_dup:
            LOG.info("flagged %d duplicate transcript structures", n_dup)

    # ---- introns between consecutive merged exons ----
    same = np.zeros(mex_tx.size, dtype=bool)
    same[:-1] = mex_tx[1:] == mex_tx[:-1]
    ipos = np.flatnonzero(same)
    in_s = mex_e[ipos]
    in_e = mex_s[ipos + 1]
    ok = in_e > in_s
    in_s, in_e, in_tx = in_s[ok], in_e[ok], mex_tx[ipos][ok]

    # ---- TSS / TES ----
    first = np.where(keep_tx, tx_off[:-1], 0)
    last = np.where(keep_tx, np.maximum(tx_off[1:] - 1, 0), 0)
    plus = tx_strand >= 0
    tss = np.where(plus, mex_s[first], mex_e[last])
    tes = np.where(plus, mex_e[last], mex_s[first])

    return dict(
        tx_name=tx_name, tx_gene=tx_gene, tx_chrom=tx_chrom, tx_strand=tx_strand,
        chrom_names=raw["chrom_names"], n_tx=ntx, keep_tx=keep_tx, n_dup=n_dup,
        tx_off=tx_off, mex_s=mex_s, mex_e=mex_e, mex_tx=mex_tx,
        in_s=in_s, in_e=in_e, in_tx=in_tx, tss=tss, tes=tes,
        tx_ex_len=np.bincount(mex_tx, weights=(mex_e - mex_s).astype(float), minlength=ntx),
        tx_in_len=np.bincount(in_tx, weights=(in_e - in_s).astype(float), minlength=ntx),
        tx_n_ex=counts,
    )


# ---------------------------------------------------------------------------
# per-chromosome payload
# ---------------------------------------------------------------------------

def make_payload(ann, chrom_code, chrom_name, size, end_win):
    keep = ann["keep_tx"] & (ann["tx_chrom"] == chrom_code)
    tx_local = np.flatnonzero(keep)
    if tx_local.size == 0:
        return None

    tx_off = ann["tx_off"]
    mex_s, mex_e = ann["mex_s"], ann["mex_e"]
    tx_strand = ann["tx_strand"]

    payload = {"chrom": chrom_name, "size": int(size), "tx_local": tx_local,
               "ex": {}, "intr": {}, "tss": {}, "tes": {}}
    half = end_win // 2

    def exon_indices(sel):
        cnt = np.diff(tx_off)[sel]
        tot = int(cnt.sum())
        if tot == 0:
            return np.empty(0, np.int64)
        starts = np.repeat(tx_off[sel], cnt)
        off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return starts + off

    for strand in (1, -1):
        sel = tx_local[tx_strand[tx_local] == strand]
        if sel.size == 0:
            continue

        ei = exon_indices(sel)
        s0 = mex_s[ei]
        e0 = mex_e[ei]
        o = np.argsort(s0, kind="stable")
        payload["ex"][strand] = dict(s0=s0[o], e0=e0[o], out=ei[o],
                                     maxlen=int((e0 - s0).max()))

        in_mask = tx_strand[ann["in_tx"]] == strand
        ipos = np.flatnonzero(in_mask & keep[ann["in_tx"]])
        if ipos.size:
            is0 = ann["in_s"][ipos]
            ie0 = ann["in_e"][ipos]
            oi = np.argsort(is0, kind="stable")
            payload["intr"][strand] = dict(s0=is0[oi], e0=ie0[oi], out=ipos[oi])

        for key, coords in (("tss", ann["tss"][sel]), ("tes", ann["tes"][sel])):
            ws0 = np.clip(coords - half, 0, None)
            we0 = np.minimum(coords + (end_win - half), size)
            w = np.maximum(we0, ws0 + 1)
            oo = np.argsort(ws0, kind="stable")
            payload[key][strand] = dict(s0=ws0[oo], e0=w[oo], out=sel[oo])

    return payload


def build_blocks(payload, size, mode, block_size):
    if mode == "genome":
        return [(bs, min(size, bs + block_size)) for bs in range(0, size, block_size)]
    parts = []
    for key in ("ex", "intr", "tss", "tes"):
        for d in payload[key].values():
            if d["s0"].size:
                parts.append(np.column_stack([d["s0"], np.minimum(d["e0"], size)]))
    if not parts:
        return []
    merged = merge_intervals(np.vstack(parts))
    blocks = []
    for a, b in merged:
        a = max(0, a)
        b = min(size, b)
        while a < b:
            c = min(b, a + block_size)
            blocks.append((a, c))
            a = c
    return blocks


# ---------------------------------------------------------------------------
# coverage primitives
# ---------------------------------------------------------------------------

def make_cb(strand, min_mapq, ex_flags):
    def cb(read):
        if ex_flags and (read.flag & ex_flags):
            return False
        if min_mapq and read.mapping_quality < min_mapq:
            return False
        if strand is not None and bool(read.is_reverse) != (strand < 0):
            return False
        return True
    return cb


def coverage(bams, chrom, start, end, min_bq, cb):
    L = end - start
    if L <= 0:
        return np.zeros(0, dtype=np.int32)
    acc = np.zeros(L, dtype=np.int32)
    for bam in bams:
        try:
            arrs = bam.count_coverage(chrom, start, end,
                                      quality_threshold=min_bq, read_callback=cb)
        except (ValueError, RuntimeError):
            continue
        for a in arrs:
            acc += a
    return acc


def _mask(s0, e0, maxlen, bs, be):
    L = be - bs
    if s0.size == 0:
        return None
    lo = int(np.searchsorted(s0, bs - maxlen, "left"))
    hi = int(np.searchsorted(s0, be, "left"))
    if hi <= lo:
        return None
    ss = np.maximum(s0[lo:hi], bs) - bs
    ee = np.minimum(e0[lo:hi], be) - bs
    k = ee > ss
    if not k.any():
        return None
    ss, ee = ss[k], ee[k]
    d = np.bincount(np.concatenate([ss, ee]),
                    weights=np.concatenate([np.ones(ss.size), -np.ones(ee.size)]),
                    minlength=L + 1)
    return np.cumsum(d[:L]) > 0.5


# ---------------------------------------------------------------------------
# junctions from CIGAR
# ---------------------------------------------------------------------------

def junctions_from_bam(bams, chrom, min_mapq, ex_flags):
    counts = {}
    for bam in bams:
        try:
            it = bam.fetch(chrom)
        except (ValueError, RuntimeError):
            continue
        for read in it:
            if ex_flags and (read.flag & ex_flags):
                continue
            if min_mapq and read.mapping_quality < min_mapq:
                continue
            ct = read.cigartuples
            if not ct or read.reference_start is None:
                continue
            pos = read.reference_start
            strand = -1 if read.is_reverse else 1
            for op, ln in ct:
                if op == 3:                                   # N -> intron
                    if ln > 0:
                        k = (strand, pos, pos + ln)
                        counts[k] = counts.get(k, 0) + 1
                    pos += ln
                elif op in (0, 2, 7, 8):                      # M/D/=/X
                    pos += ln
    return counts


def match_junctions(jcounts, payload, min_reads, slack, max_novel):
    """Returns (global intron indices hit, bool array, novel junction list)."""
    sup = {k for k, v in jcounts.items() if v >= min_reads}
    if slack:
        sup = {k for k in sup}  # slack handled below
    positions, hits = [], []
    ann_keys = set()
    for strand, d in payload["intr"].items():
        s0, e0, out = d["s0"], d["e0"], d["out"]
        for i in range(s0.size):
            a, b = int(s0[i]), int(e0[i])
            ann_keys.add((strand, a, b))
            if slack:
                found = False
                for da in range(-slack, slack + 1):
                    for db in range(-slack, slack + 1):
                        if (strand, a + da, b + db) in sup:
                            found = True
                            break
                    if found:
                        break
                positions.append(int(out[i]))
                hits.append(found)
            else:
                positions.append(int(out[i]))
                hits.append((strand, a, b) in sup)

    novel = [k for k in sup if k not in ann_keys]
    novel.sort(key=lambda x: (x[0], x[1], x[2]))
    if len(novel) > max_novel:
        novel = novel[:max_novel]
    return np.asarray(positions, np.int64), np.asarray(hits, bool), novel


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def process_chrom(task):
    chrom = task["chrom"]
    size = task["size"]
    payload = task["payload"]
    P = task["params"]

    bams = [pysam.AlignmentFile(p, "rb") for p in P["bam_paths"]]
    try:
        thresholds = [float(x) for x in P["thresholds"]]
        T = len(thresholds)
        t0 = thresholds[0]
        stranded = P["stranded"]
        want_part = P["partition"] and P["scan"] == "genome"
        blocks = build_blocks(payload, size, P["scan"], P["block"])

        active = sorted(set(payload["ex"]) | set(payload["tss"]))
        need_strands = (1, -1) if (stranded and want_part) else tuple(active)

        cb_fast = {s: None for s in (1, -1, None)}
        for s in need_strands:
            cb_fast[s] = make_cb(s if stranded else None, P["min_mapq"], P["exclude_flags"])
        cb_fast[None] = make_cb(None, P["min_mapq"], P["exclude_flags"])

        res = {}
        for key in ("ex", "intr", "tss", "tes"):
            res[key] = {}
            for s, d in payload[key].items():
                n = d["out"].size
                res[key][s] = dict(out=d["out"], sum=np.zeros(n),
                                   cnt=[np.zeros(n, np.int32) for _ in range(T)])

        part = dict(cov_total=0, exon_same=0, exon_any=0, nonexon=0)
        novel = []
        novel_trunc = False

        for bs, be in blocks:
            L = be - bs
            if L <= 0:
                continue
            cov = {}
            for s in need_strands:
                cov[s] = coverage(bams, chrom, bs, be, P["min_baseq"], cb_fast[s])
            if stranded:
                tot = np.zeros(L, np.int32)
                for s in (1, -1):
                    if s in cov:
                        tot = tot + cov[s]
            else:
                tot = cov.get(active[0]) if active else np.zeros(L, np.int32)
                for s in active:
                    cov[s] = tot

            for s in active:
                v = cov[s]
                Pv = np.zeros(L + 1, dtype=np.int64)
                Pv[1:] = np.cumsum(v, dtype=np.int64)
                Cs = []
                for t in thresholds:
                    c = np.zeros(L + 1, dtype=np.int64)
                    c[1:] = np.cumsum(v >= t, dtype=np.int32)
                    Cs.append(c)

                for key in ("ex", "intr", "tss", "tes"):
                    d = payload[key].get(s)
                    if d is None:
                        continue
                    lo = int(np.searchsorted(d["s0"], bs, "left"))
                    hi = int(np.searchsorted(d["s0"], be, "left"))
                    if hi <= lo:
                        continue
                    ss = d["s0"][lo:hi]
                    ee = d["e0"][lo:hi]
                    inside = ee <= be
                    j = np.flatnonzero(inside)
                    if j.size:
                        rs = (ss[j] - bs)
                        re = (ee[j] - bs)
                        res[key][s]["sum"][lo + j] = (Pv[re] - Pv[rs])
                        for k in range(T):
                            res[key][s]["cnt"][k][lo + j] = (Cs[k][re] - Cs[k][rs])
                    for jj in np.flatnonzero(~inside):
                        a = max(0, int(ss[jj]))
                        b = min(size, int(ee[jj]))
                        arr = coverage(bams, chrom, a, b, P["min_baseq"], cb_fast[s])
                        res[key][s]["sum"][lo + jj] = float(arr.sum(dtype=np.int64))
                        for k, t in enumerate(thresholds):
                            res[key][s]["cnt"][k][lo + jj] = int((arr >= t).sum())

            if want_part:
                masks = {}
                for s, d in payload["ex"].items():
                    m = _mask(d["s0"], d["e0"], d["maxlen"], bs, be)
                    if m is not None:
                        masks[s] = m
                union = None
                for m in masks.values():
                    union = m if union is None else (union | m)
                if stranded:
                    covmask = (tot >= t0)
                    nb = int(covmask.sum())
                    if nb:
                        part["cov_total"] += nb
                        if union is not None:
                            part["exon_any"] += int((covmask & union).sum())
                            for s, m in masks.items():
                                part["exon_same"] += int(((cov[s] >= t0) & m).sum())
                            part["nonexon"] += nb - int((covmask & union).sum())
                        else:
                            part["nonexon"] += nb
                        rest = covmask & ~union if union is not None else covmask
                else:
                    covmask = (tot >= t0)
                    nb = int(covmask.sum())
                    if nb:
                        part["cov_total"] += nb
                        if union is not None:
                            a = int((covmask & union).sum())
                            part["exon_any"] += a
                            part["exon_same"] += a
                            part["nonexon"] += nb - a
                        else:
                            part["nonexon"] += nb
                        rest = covmask & ~union if union is not None else covmask

                if rest.any():
                    Pt = np.zeros(L + 1, dtype=np.int64)
                    Pt[1:] = np.cumsum(tot, dtype=np.int64)
                    for a0, b0 in _runs(rest):
                        if b0 - a0 < P["min_novel_len"]:
                            continue
                        if len(novel) >= P["max_novel"]:
                            novel_trunc = True
                            break
                        novel.append((bs + a0, bs + b0, float(Pt[b0] - Pt[a0])))

        merged = []
        for a, b, sm in novel:
            if merged and a <= merged[-1][1]:
                if b > merged[-1][1]:
                    merged[-1][1] = b
                merged[-1][2] += sm
            else:
                merged.append([a, b, sm])
        merged = [m for m in merged
                  if (m[1] - m[0]) >= P["min_novel_len"]
                  and m[2] / max(1, m[1] - m[0]) >= P["min_novel_depth"]]

        # ---- junctions ----
        intr_pos = np.empty(0, np.int64)
        intr_hit = np.empty(0, bool)
        novel_j = []
        if P["junctions"] == "bam":
            jc = junctions_from_bam(bams, chrom, P["min_mapq"], P["exclude_flags"])
            intr_pos, intr_hit, novel_j = match_junctions(
                jc, payload, P["min_junction_reads"], P["junction_slack"], P["max_novel_junctions"])

        return dict(chrom=chrom, skip=False, res=res, part=part, novel=merged,
                    novel_trunc=novel_trunc, intr_pos=intr_pos, intr_hit=intr_hit,
                    novel_j=novel_j)
    finally:
        for b in bams:
            try:
                b.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Evaluate a GFF3/GTF annotation against FL-cDNA BAM alignments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-a", "--bam", nargs="+", required=True,
                    help="FL-cDNA BAM(s); multiple files are summed (replicates)")
    ap.add_argument("-b", "--gff", required=True, help="annotation GFF3/GTF (.gz ok)")
    ap.add_argument("-o", "--out", default="flcdna_eval", help="output prefix")

    ap.add_argument("-t", "--thresholds", default="1,3,10",
                    help="coverage depth thresholds, comma separated")
    ap.add_argument("--stranded", dest="stranded", action="store_true", default=True,
                    help="split coverage by read strand (default)")
    ap.add_argument("--unstranded", dest="stranded", action="store_false",
                    help="single coverage track (2x faster, no antisense split)")
    ap.add_argument("--scan", choices=["genome", "features"], default="genome",
                    help="genome = whole-chromosome blocks (enables partition); "
                         "features = only annotated intervals (much faster)")
    ap.add_argument("--block", type=int, default=1_000_000, help="block size (bp)")
    ap.add_argument("--end-window", type=int, default=60,
                    help="window width (bp) centred on TSS/TES for end support")
    ap.add_argument("--min-baseq", type=int, default=0, help="minimum base quality")
    ap.add_argument("--min-mapq", type=int, default=0, help="minimum mapping quality")
    ap.add_argument("--exclude-flags", type=lambda s: int(s, 0), default=0x904,
                    help="BAM flags to exclude (0x904 = unmapped|secondary|supplementary)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep exact-duplicate transcript structures")

    ap.add_argument("--junctions", choices=["bam", "none"], default="bam",
                    help="bam = extract splice junctions from CIGAR N operators")
    ap.add_argument("--min-junction-reads", type=int, default=2)
    ap.add_argument("--junction-slack", type=int, default=0,
                    help="bp tolerance when matching junctions")
    ap.add_argument("--max-novel-junctions", type=int, default=200_000)

    ap.add_argument("--partition", dest="partition", action="store_true", default=True)
    ap.add_argument("--no-partition", dest="partition", action="store_false")
    ap.add_argument("--min-novel-len", type=int, default=150)
    ap.add_argument("--min-novel-depth", type=float, default=1.0)
    ap.add_argument("--max-novel", type=int, default=200_000)

    ap.add_argument("--min-tx-exon-len", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel chromosomes")
    ap.add_argument("--chroms", default=None, help="comma-separated subset (debug)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    if a.jobs == 0:
        a.jobs = os.cpu_count() or 1

    outdir = os.path.dirname(os.path.abspath(a.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    thresholds = [float(x) for x in a.thresholds.split(",")]

    # ---- BAM sanity ----
    ref_len = {}
    for p in a.bam:
        with pysam.AlignmentFile(p, "rb") as b:
            if not b.has_index():
                raise SystemExit(f"[fatal] BAM has no index: {p}")
            for name, ln in zip(b.references, b.lengths):
                ref_len.setdefault(name, ln)
    LOG.info("BAM reference set: %d sequences", len(ref_len))

    # ---- annotation ----
    ann = finalize_annotation(load_annotation(a.gff), dedupe=not a.keep_duplicates)
    keep = ann["keep_tx"] & (ann["tx_ex_len"] >= a.min_tx_exon_len)
    if not keep.all():
        LOG.info("dropping %d transcripts shorter than %d bp",
                 int((~keep).sum()), a.min_tx_exon_len)
        ann["keep_tx"] = keep
    LOG.info("%d transcripts under evaluation", int(keep.sum()))

    # ---- chromosome name reconciliation ----
    ann_chroms = list(ann["chrom_names"])
    missing = [c for c in ann_chroms if c not in ref_len]
    if missing:
        alt = {}
        for c in ref_len:
            alt.setdefault(c[3:] if c.startswith("chr") else "chr" + c, c)
        fix = {m: alt[m] for m in missing if m in alt}
        if fix:
            LOG.warning("renaming %d seqids to match BAM (e.g. %s -> %s)",
                        len(fix), *list(fix.items())[0])
            name_of = {i: n for i, n in enumerate(ann_chroms)}
            new_names = [fix.get(n, n) for n in ann_chroms]
            idx = {n: i for i, n in enumerate(new_names)}
            tc = ann["tx_chrom"]
            chg = np.flatnonzero(np.array([n in fix for n in ann_chroms], bool))
            remap = np.zeros(max(len(new_names), 1), dtype=np.int32)
            for old_i, n in name_of.items():
                remap[old_i] = idx[fix.get(n, n)]
            ann["tx_chrom"] = remap[tc]
            ann["chrom_names"] = new_names
        still = [c for c in ann["chrom_names"] if c not in ref_len]
        if still:
            LOG.warning("%d seqids absent from BAM: %s%s",
                        len(still), ", ".join(still[:5]), " ..." if len(still) > 5 else "")

    # ---- tasks ----
    want = set(a.chroms.split(",")) if a.chroms else None
    params_common = dict(
        bam_paths=[os.path.abspath(p) for p in a.bam],
        thresholds=thresholds, stranded=a.stranded, block=a.block,
        end_window=a.end_window, min_baseq=a.min_baseq, min_mapq=a.min_mapq,
        exclude_flags=a.exclude_flags, partition=a.partition, scan=a.scan,
        min_novel_len=a.min_novel_len, min_novel_depth=a.min_novel_depth,
        max_novel=a.max_novel, junctions=a.junctions,
        min_junction_reads=a.min_junction_reads, junction_slack=a.junction_slack,
        max_novel_junctions=a.max_novel_junctions,
    )

    tasks = []
    for code, cname in enumerate(ann["chrom_names"]):
        if cname not in ref_len:
            continue
        if want and cname not in want:
            continue
        pl = make_payload(ann, code, cname, ref_len[cname], a.end_window)
        if pl is None:
            continue
        tasks.append(dict(chrom=cname, size=ref_len[cname], payload=pl,
                          params=params_common))
    if not tasks:
        raise SystemExit("[fatal] no chromosome shared by annotation and BAM")

    n_intr = ann["in_s"].size
    n_tx = ann["n_tx"]

    ex_sum = np.zeros(ann["mex_s"].size)
    in_sum = np.zeros(n_intr)
    ex_cnt = [np.zeros(ann["mex_s"].size, np.int32) for _ in thresholds]
    in_cnt = [np.zeros(n_intr, np.int32) for _ in thresholds]
    tss_sum = np.zeros(n_tx)
    tes_sum = np.zeros(n_tx)
    intr_hit = np.zeros(n_intr, dtype=bool)
    intra_seen = np.zeros(n_intr, dtype=bool)
    novel_junctions = []
    part = dict(cov_total=0, exon_same=0, exon_any=0, nonexon=0)
    novel_all = []
    novel_trunc = False

    t0 = time.time()
    results = []
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(process_chrom, t): t["chrom"] for t in tasks}
            for f in as_completed(futs):
                results.append(f.result())
                LOG.info("finished %s", futs[f])
    else:
        for t in tasks:
            LOG.info("processing %s (%s bp)", t["chrom"], f"{t['size']:,}")
            results.append(process_chrom(t))
    LOG.info("coverage/junction scan done in %.1f s", time.time() - t0)

    # ---- gather ----
    for r in results:
        if r.get("skip"):
            LOG.warning("skipped %s", r["chrom"])
            continue
        for k in part:
            part[k] += r["part"][k]
        novel_all.extend((r["chrom"], a0, b0, sm) for a0, b0, sm in r["novel"])
        novel_trunc |= r["novel_trunc"]
        for key, ssum, scnt in (("ex", ex_sum, ex_cnt), ("intr", in_sum, in_cnt)):
            for s, d in r["res"][key].items():
                ssum[d["out"]] = d["sum"]
                for i in range(len(thresholds)):
                    scnt[i][d["out"]] = d["cnt"][i]
        for s, d in r["res"]["tss"].items():
            tss_sum[d["out"]] = d["sum"]
        for s, d in r["res"]["tes"].items():
            tes_sum[d["out"]] = d["sum"]
        if r["intr_pos"].size:
            intr_hit[r["intr_pos"]] = r["intr_hit"]
            intra_seen[r["intr_pos"]] = True
        novel_junctions.extend((r["chrom"], k[0], k[1], k[2]) for k in r["novel_j"])

    # ---- per-transcript aggregation ----
    mex_tx, in_tx = ann["mex_tx"], ann["in_tx"]
    tx_ex_sum = np.bincount(mex_tx, weights=ex_sum, minlength=n_tx)
    tx_in_sum = np.bincount(in_tx, weights=in_sum, minlength=n_tx)
    tx_ex_cov = [np.bincount(mex_tx, weights=ex_cnt[i], minlength=n_tx) for i in range(len(thresholds))]
    tx_in_cov = [np.bincount(in_tx, weights=in_cnt[i], minlength=n_tx) for i in range(len(thresholds))]

    ex_len = np.maximum(ann["tx_ex_len"], 1.0)
    frac = [np.clip(tx_ex_cov[i] / ex_len, 0, 1) for i in range(len(thresholds))]
    with np.errstate(invalid="ignore", divide="ignore"):
        ifrac = [np.where(ann["tx_in_len"] > 0, tx_in_cov[i] / np.maximum(ann["tx_in_len"], 1.0), np.nan)
                 for i in range(len(thresholds))]
        ex_mean = tx_ex_sum / ex_len
        in_mean = np.where(ann["tx_in_len"] > 0, tx_in_sum / np.maximum(ann["tx_in_len"], 1.0), np.nan)

    end_w = max(1, a.end_window)
    tss_mean = tss_sum / end_w
    tes_mean = tes_sum / end_w

    n_intr_tx = np.bincount(in_tx, minlength=n_tx)
    n_intr_hit = np.bincount(in_tx[in_tx >= 0], weights=intr_hit.astype(float), minlength=n_tx) \
        if n_intr else np.zeros(n_tx)
    with np.errstate(invalid="ignore", divide="ignore"):
        jfrac = np.where(n_intr_tx > 0, n_intr_hit / np.maximum(n_intr_tx, 1), np.nan)

    # ---- support class ----
    f0 = frac[0]
    cls = np.full(n_tx, "unsupported", dtype=object)
    cls[f0 > 0] = "weak"
    cls[f0 >= 0.5] = "partial"
    cls[f0 >= 0.8] = "good"
    ok_in = np.where(ann["tx_in_len"] > 0, np.nan_to_num(ifrac[0], nan=0.0) <= 0.15, True)
    cls[(f0 >= 0.9) & ok_in] = "strong"

    keep = ann["keep_tx"]
    idx = np.flatnonzero(keep)

    # ---- transcript table ----
    fn = f"{a.out}.transcript_metrics.tsv.gz"
    cols = ["tx_id", "gene_id", "chrom", "strand", "n_exons", "exon_len", "intron_len",
            "exon_mean_depth", "intron_mean_depth"]
    cols += [f"frac_exon_cov_t{t:g}" for t in thresholds]
    cols += ["tss_win_mean", "tes_win_mean", "frac_introns_supported", "support_class"]
    with gzip.open(fn, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for i in idx:
            row = [ann["tx_name"][i], ann["tx_gene"][i],
                   ann["chrom_names"][ann["tx_chrom"][i]],
                   "+" if ann["tx_strand"][i] > 0 else "-",
                   str(int(ann["tx_n_ex"][i])),
                   f"{ann['tx_ex_len'][i]:.0f}", f"{ann['tx_in_len'][i]:.0f}",
                   f"{ex_mean[i]:.3f}",
                   "NA" if np.isnan(in_mean[i]) else f"{in_mean[i]:.3f}"]
            row += [f"{frac[k][i]:.4f}" for k in range(len(thresholds))]
            row += [f"{tss_mean[i]:.3f}", f"{tes_mean[i]:.3f}",
                    "NA" if np.isnan(jfrac[i]) else f"{jfrac[i]:.3f}", cls[i]]
            fh.write("\t".join(row) + "\n")
    LOG.info("wrote %s", fn)

    # ---- gene table ----
    fn = f"{a.out}.gene_metrics.tsv.gz"
    gnames = np.asarray(ann["tx_gene"], dtype=object)[idx]
    order = np.argsort(gnames, kind="stable")
    f0s = f0[idx][order]
    gk = gnames[order]
    n_gene = 0
    with gzip.open(fn, "wt") as fh:
        fh.write("gene_id\tn_tx\tn_supported\tn_good\tmax_frac_exon_cov\tmean_frac_exon_cov\n")
        if gk.size:
            bounds = np.r_[0, np.flatnonzero(gk[1:] != gk[:-1]) + 1, gk.size]
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                seg = f0s[b0:b1]
                fh.write(f"{gk[b0]}\t{b1-b0}\t{int((seg>0).sum())}\t{int((seg>=0.8).sum())}"
                         f"\t{seg.max():.4f}\t{seg.mean():.4f}\n")
                n_gene += 1
    LOG.info("wrote %s (%d genes)", fn, n_gene)

    # ---- introns without junction support ----
    n_unsup_intr = 0
    if a.junctions == "bam" and n_intr:
        fn = f"{a.out}.introns_unsupported.bed"
        cn = ann["chrom_names"]
        with open(fn, "w") as fh:
            for i in range(n_intr):
                if intra_seen[i] and not intr_hit[i]:
                    t = ann["in_tx"][i]
                    fh.write(f"{cn[ann['tx_chrom'][t]]}\t{ann['in_s'][i]}\t{ann['in_e'][i]}\t"
                             f"{ann['tx_name'][t]}\t0\t{'+' if ann['tx_strand'][t] > 0 else '-'}\n")
                    n_unsup_intr += 1
        LOG.info("wrote %s (%d introns, %s)", fn, n_unsup_intr,
                 "TRUNCATED" if False else "no FL junction support")

    # ---- novel junctions ----
    if novel_junctions:
        fn = f"{a.out}.novel_junctions.bed"
        with open(fn, "w") as fh:
            for chrom, strand, s0, e0 in novel_junctions[: a.max_novel_junctions]:
                fh.write(f"{chrom}\t{s0}\t{e0}\tFL_junction\t0\t{'+' if strand > 0 else '-'}\n")
        LOG.info("wrote %s (%d junctions, %s)", fn,
                 min(len(novel_junctions), a.max_novel_junctions),
                 "TRUNCATED" if len(novel_junctions) > a.max_novel_junctions else "complete")

    # ---- unannotated covered regions ----
    if novel_all and a.partition:
        fn = f"{a.out}.unannotated_covered.bed"
        n_w = 0
        with open(fn, "w") as fh:
            for chrom, s0, e0, sm in novel_all:
                L = e0 - s0
                d = sm / max(1, L)
                if L < a.min_novel_len or d < a.min_novel_depth:
                    continue
                fh.write(f"{chrom}\t{s0}\t{e0}\tnovel_cov\t{int(round(d))}\t.\n")
                n_w += 1
        LOG.info("wrote %s (%d regions%s)", fn, n_w, ", TRUNCATED" if novel_trunc else "")

    # ---- summary ----
    n_kept = int(keep.sum())
    class_counts = {c: int(((cls == c) & keep).sum())
                    for c in ("strong", "good", "partial", "weak", "unsupported")}

    summ = dict(
        annotation=os.path.abspath(a.gff),
        bam=[os.path.abspath(p) for p in a.bam],
        stranded=a.stranded, scan=a.scan, thresholds=thresholds,
        block_size=a.block, end_window=a.end_window,
        min_mapq=a.min_mapq, min_baseq=a.min_baseq, exclude_flags=hex(a.exclude_flags),
        transcripts_evaluated=n_kept,
        duplicates_flagged=int(ann.get("n_dup", 0)),
        transcripts_by_class=class_counts,
        frac_exon_cov_median={f"t{t:g}": float(np.median(frac[k][keep]))
                              for k, t in enumerate(thresholds)},
        frac_exon_cov_median_multi_exon={
            f"t{t:g}": (float(np.median(frac[k][keep & (ann["tx_n_ex"] > 1)]))
                        if (keep & (ann["tx_n_ex"] > 1)).any() else None)
            for k, t in enumerate(thresholds)},
        frac_exon_cov_median_single_exon={
            f"t{t:g}": (float(np.median(frac[k][keep & (ann["tx_n_ex"] <= 1)]))
                        if (keep & (ann["tx_n_ex"] <= 1)).any() else None)
            for k, t in enumerate(thresholds)},
        base_partition=dict(
            cov_total=int(part["cov_total"]),
            exon_same_strand=int(part["exon_same"]),
            exon_any_strand=int(part["exon_any"]),
            non_exonic=int(part["nonexon"]),
            annotation_explains_frac=(part["exon_any"] / part["cov_total"]
                                      if part["cov_total"] else None)),
        junctions=dict(
            annotation_introns=int(n_intr),
            supported=int(intr_hit.sum()) if n_intr else 0,
            supported_frac=(float(intr_hit.sum() / max(1, n_intr)) if n_intr else None),
            unsupported=int(n_unsup_intr),
            novel_in_annotation=len(novel_junctions)),
    )
    with open(f"{a.out}.summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)

    # ---- report ----
    L = []
    A = L.append
    A("=" * 78)
    A("FL-cDNA BAM vs annotation  —  support report")
    A("=" * 78)
    A(f"BAM           : {', '.join(os.path.basename(p) for p in a.bam)}")
    A(f"annotation    : {os.path.basename(a.gff)}")
    A(f"mode          : {'strand-aware' if a.stranded else 'unstranded'}, "
      f"scan={a.scan}, block={a.block:,} bp")
    A(f"thresholds    : {', '.join(str(t) for t in thresholds)}   "
      f"MAPQ>={a.min_mapq}  flags!={hex(a.exclude_flags)}")
    A(f"transcripts   : {n_kept:,} evaluated  "
      f"({ann.get('n_dup', 0):,} duplicate structures flagged)")
    A("")
    if a.partition and part["cov_total"]:
        ct = part["cov_total"]
        A("-- covered-base partition (t = %g) " % thresholds[0] + "-" * 36)
        A(f"  total covered bases     : {ct:>15,}")
        A(f"  in exons (same strand)  : {part['exon_same']:>15,}  "
          f"({_pct(part['exon_same'], ct):5.2f} %)")
        A(f"  in exons (any strand)   : {part['exon_any']:>15,}  "
          f"({_pct(part['exon_any'], ct):5.2f} %)")
        A(f"  non-exonic              : {part['nonexon']:>15,}  "
          f"({_pct(part['nonexon'], ct):5.2f} %)")
        A(f"  -> annotation explains  : {_pct(part['exon_any'], ct):5.2f} % of covered bases")
        A("")
    A("-- transcript support class (fraction of exon bases covered) " + "-" * 17)
    A(f"{'class':<14}{'n':>12}{'%':>10}")
    for c in ("strong", "good", "partial", "weak", "unsupported"):
        n = class_counts[c]
        A(f"{c:<14}{n:>12,}{_pct(n, max(1, n_kept)):>9.2f}%")
    A(f"{'median frac':<14}{np.median(frac[0][keep]):>12.4f}" if n_kept else "")
    A("")
    A("-- by exon count " + "-" * 59)
    for label, sel in (("single-exon", keep & (ann["tx_n_ex"] <= 1)),
                       ("multi-exon ", keep & (ann["tx_n_ex"] > 1))):
        n = int(sel.sum())
        if n:
            A(f"  {label}: n={n:>8,}  median frac_cov={np.median(frac[0][sel]):.4f}  "
              f"supported(>=0.5)={_pct(int((frac[0][sel] >= .5).sum()), n):5.2f} %")
    if n_intr:
        A("")
        A("-- splice junctions (CIGAR N, >=%d reads) " % a.min_junction_reads + "-" * 35)
        A(f"  annotated introns              : {n_intr:,}")
        A(f"  with FL junction support       : {int(intr_hit.sum()):,}  "
          f"({_pct(int(intr_hit.sum()), n_intr):.2f} %)")
        A(f"  without support                : {n_unsup_intr:,}   -> introns_unsupported.bed")
        A(f"  FL junctions absent from annot.: {len(novel_junctions):,}   -> novel_junctions.bed")
    A("")
    A("-- interpretation notes " + "-" * 52)
    A("  * bigWig is gone: coverage AND splice structure both come from the same BAM,")
    A("    so coverage and junction evidence are mutually consistent by construction.")
    A("  * 'unsupported' models are NOT proven false positives: low depth, 3' bias or")
    A("    an unannotated isoform produce the same signal. Report as FL-unsupported.")
    A("  * Sweep -t, --end-window and --min-junction-reads before concluding.")
    A("=" * 78)
    report = "\n".join(L)
    print(report)
    with open(f"{a.out}.report.txt", "w") as fh:
        fh.write(report + "\n")

    LOG.info("done -> %s.{transcript_metrics,gene_metrics}.tsv.gz, .summary.json, .report.txt", a.out)


if __name__ == "__main__":
    main()
