#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from scrim.gff3 import TRANSCRIPT_TYPES as TRANSCRIPT_TYPES, Span, Transcript, merge, parse_gff3

Loc = tuple[str, str]
IntervalKey = tuple[str, str, int, int]
Chain = tuple[str, str, tuple[Span, ...]]
ORDER = (
    "nt", "exon-segment", "intron-segment",
    "donor-site", "acceptor-site", "start-site", "stop-site",
    "transcript-CDS-chain", "alternative-isoform",
)


@dataclass(frozen=True, slots=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def reference(self) -> int:
        return self.tp + self.fn

    @property
    def prediction(self) -> int:
        return self.tp + self.fp

    def as_dict(self) -> dict:
        total = self.reference + self.prediction
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "reference": self.reference, "prediction": self.prediction,
            "precision": self.tp / self.prediction if self.prediction else (0.0 if total else None),
            "recall": self.tp / self.reference if self.reference else None,
            "f1": 2 * self.tp / total if total else None,
        }


def read_gff3(path: str) -> tuple[list[Transcript], dict]:
    transcripts = parse_gff3(path).transcripts
    return transcripts, {
        "transcripts_scored": len(transcripts),
        "transcripts_with_CDS": sum(bool(t.spans["CDS"]) for t in transcripts),
        "unique_seqids": len({t.seqid for t in transcripts}),
    }


def chain_keys(transcripts: list[Transcript]) -> set[Chain]:
    return {(t.seqid, t.strand, tuple(t.spans.get("CDS") or t.spans["exon"])) for t in transcripts
            if t.spans.get("CDS") or t.spans.get("exon")}


def exact_counts(ref: set, pred: set) -> Counts:
    tp = len(ref & pred)
    return Counts(tp, len(pred) - tp, len(ref) - tp)


def coverage(chains: set[Chain], envelope: bool = False) -> dict[Loc, list[Span]]:
    by_loc = defaultdict(list)
    for seqid, strand, spans in chains:
        by_loc[seqid, strand].extend([(spans[0][0], spans[-1][1])] if envelope else spans)
    return {loc: merge(spans) for loc, spans in by_loc.items()}


def base_counts(ref: set[Chain], pred: set[Chain]) -> Counts:
    a, b = coverage(ref), coverage(pred)
    nref = sum(y - x + 1 for spans in a.values() for x, y in spans)
    npred = sum(y - x + 1 for spans in b.values() for x, y in spans)
    overlap = 0
    for loc, xspans in a.items():
        yspans = b.get(loc, [])
        i = j = 0
        while i < len(xspans) and j < len(yspans):
            x0, x1 = xspans[i]
            y0, y1 = yspans[j]
            overlap += max(0, min(x1, y1) - max(x0, y0) + 1)
            if x1 <= y1:
                i += 1
            else:
                j += 1
    return Counts(overlap, npred - overlap, nref - overlap)


def intron_keys(chains: set[Chain]) -> set[IntervalKey]:
    return {(seqid, strand, b + 1, c - 1) for seqid, strand, spans in chains
            for (_, b), (c, _) in pairwise(spans) if b + 1 < c}


def junction_sites(introns: set[IntervalKey], site: str) -> set[tuple[str, str, int]]:
    return {(seqid, strand, a if (strand == "+") == (site == "donor") else b)
            for seqid, strand, a, b in introns}


def coding_sites(chains: set[Chain], site: str) -> set[tuple[str, str, int]]:
    return {(s, d, spans[0][0] if (d == "+") == (site == "start") else spans[-1][1])
            for s, d, spans in chains}


def alternative_chains(transcripts: list[Transcript]) -> set[Chain]:
    groups = defaultdict(set)
    for t in transcripts:
        spans = t.spans.get("CDS") or t.spans.get("exon")
        if spans:
            groups[t.gid, t.seqid, t.strand].add(tuple(spans))
    return {(seqid, strand, c) for (_, seqid, strand), forms in groups.items()
            if len(forms) > 1 for c in forms}


def overlapping(chains: set[Chain], regions: dict[Loc, list[Span]]) -> set[Chain]:
    result = set()
    for seqid, strand, spans in chains:
        intervals = regions.get((seqid, strand), [])
        i = bisect_left(intervals, (spans[-1][1] + 1,)) - 1
        if i >= 0 and intervals[i][1] >= spans[0][0]:
            result.add((seqid, strand, spans))
    return result



def calculate(ref: list[Transcript], pred: list[Transcript]) -> dict[str, Counts]:
    rchains, pchains = chain_keys(ref), chain_keys(pred)
    ri, pi = intron_keys(rchains), intron_keys(pchains)
    regions = coverage(alternative_chains(ref) | alternative_chains(pred), True)
    result = {
        "nt": base_counts(rchains, pchains),
        "exon-segment": exact_counts(
            {(s, d, a, b) for s, d, spans in rchains for a, b in spans},
            {(s, d, a, b) for s, d, spans in pchains for a, b in spans}),
        "intron-segment": exact_counts(ri, pi),
        "transcript-CDS-chain": exact_counts(rchains, pchains),
        "alternative-isoform": exact_counts(overlapping(rchains, regions), overlapping(pchains, regions)),

    }
    for site in ("donor", "acceptor"):
        result[f"{site}-site"] = exact_counts(junction_sites(ri, site), junction_sites(pi, site))
    for site in ("start", "stop"):
        result[f"{site}-site"] = exact_counts(coding_sites(rchains, site), coding_sites(pchains, site))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("--protein-coding-only", action="store_true")

    parser.add_argument("--format", choices=("text", "tsv", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        ref, ref_meta = read_gff3(args.reference)
        pred, pred_meta = read_gff3(args.query)
        if args.protein_coding_only:
            ref = [t for t in ref if t.spans["CDS"]]
            pred = [t for t in pred if t.spans["CDS"]]
        result = calculate(ref, pred)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps({"settings": {"protein_coding_only": args.protein_coding_only},
                          "reference": {**ref_meta, "transcripts_scored": len(ref)},
                          "query": {**pred_meta, "transcripts_scored": len(pred)},
                          "metrics": {k: result[k].as_dict() for k in ORDER}}, indent=2))
    elif args.format == "tsv":
        keys = ("f1", "precision", "recall", "tp", "fp", "fn", "reference", "prediction")
        print("\t".join(("metric", *keys)))
        for name in ORDER:
            d = result[name].as_dict()
            print("\t".join((name, *("NA" if d[k] is None else f"{d[k]:.8f}" if k in keys[:3]
                                      else str(d[k]) for k in keys))))
    else:
        rows: list[tuple[str, ...]] = [("metric", "F1", "precision", "recall", "TP", "FP", "FN")]
        for name in ORDER:
            d = result[name].as_dict()
            rows.append((name, *("NA" if d[k] is None else f"{d[k]:.2%}"
                                 for k in ("f1", "precision", "recall")),
                         *(str(d[k]) for k in ("tp", "fp", "fn"))))
        widths = [max(map(len, column)) for column in zip(*rows)]
        for row in rows:
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
