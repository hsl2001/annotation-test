from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from accuracy import calculate

from .gff3 import Span, Transcript, contains, parse_gff3

PRIMARY = "transcript-CDS-chain"
METRICS = (PRIMARY, "nt", "exon-segment", "intron-segment", "donor-site", "acceptor-site")
LABELS = {"reference": "Reference", "query": "GHMM", "augustus": "AUGUSTUS"}


@dataclass(frozen=True, slots=True)
class Region:
    seqid: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.seqid}:{self.start}-{self.end}"


def parse_region(value: str) -> Region:
    match = re.fullmatch(r"(\S+):([0-9]+)-([0-9]+)", value)
    if not match or not 1 <= int(match[2]) <= int(match[3]):
        raise ValueError("expected SEQID:START-END with 1 <= START <= END")
    return Region(match[1], int(match[2]), int(match[3]))


@dataclass(slots=True)
class _Input:
    transcripts: list[Transcript]
    info: dict
    declared: dict[str, Span]
    observed: dict[str, Span]


def _load_input(path: str, region: Region | None, offset: int) -> _Input:
    path = str(path)
    window = (region.seqid, (region.start - offset, region.end - offset)) if region is not None else None
    parsed = parse_gff3(path, track_scope=True, window=window)
    if window is not None:
        seqid, span = window
        if seqid in parsed.declared and not contains(parsed.declared[seqid], span):
            raise ValueError(f"{path}: declared sequence span after +{offset} offset does not cover {region}")

    selected = []
    with_cds = 0
    for t in parsed.transcripts:
        if not (cds := t.spans["CDS"]):
            continue
        with_cds += 1
        if region is not None and (t.seqid != region.seqid or t.tid in parsed.outside):
            continue
        if offset:
            cds = [(a + offset, b + offset) for a, b in cds]
            t.spans = {"exon": cds, "CDS": cds}
        selected.append(t)
    return _Input(selected, {"path": path, "seqids": sorted(parsed.declared.keys() | parsed.observed.keys()),
                             "transcripts_with_CDS": with_cds, "transcripts_scored": len(selected)},
                  parsed.declared, parsed.observed)


def load_input(path: str, region: Region | None, offset: int) -> tuple[list[Transcript], dict]:
    loaded = _load_input(path, region, offset)
    return loaded.transcripts, loaded.info


def check_seqids(inputs: dict[str, dict], region: Region | None) -> None:
    known = [info for info in inputs.values() if info["seqids"]]
    for info in known:
        if region is not None:
            if region.seqid not in info["seqids"]:
                raise ValueError(f"{info['path']}: seqid {region.seqid!r} not found; found {info['seqids']}")
        elif info["seqids"] != known[0]["seqids"]:
            raise ValueError(f"seqid mismatch: {known[0]['path']} has {known[0]['seqids']}, "
                             f"{info['path']} has {info['seqids']}")


def _check_full_scope(inputs: Collection[_Input]) -> None:
    known: dict[str, tuple[Span, str]] = {}
    for loaded in inputs:
        path = loaded.info["path"]
        for seqid, span in loaded.declared.items():
            other_span, other_path = known.setdefault(seqid, (span, path))
            if span != other_span:
                raise ValueError(f"sequence span mismatch: {other_path} has {Region(seqid, *other_span)}, "
                                 f"{path} has {Region(seqid, *span)}; use --region to compare a common interval")
    for loaded in inputs:
        for seqid, span in loaded.observed.items():
            if seqid in known:
                declared, path = known[seqid]
                if not contains(declared, span):
                    raise ValueError(f"{loaded.info['path']} has feature coordinates {Region(seqid, *span)} outside "
                                     f"{path}'s span {Region(seqid, *declared)}; use --region to compare a common interval")


def make_report(paths: dict[str, str], region: Region | None, query_offset: int) -> dict:
    if query_offset < 0:
        raise ValueError("--query-offset must be nonnegative")
    if query_offset and region is None:
        raise ValueError("a nonzero --query-offset requires --region")
    if region is not None:
        if not 1 <= region.start <= region.end:
            raise ValueError("--region must have 1 <= START <= END")
        if region.start <= query_offset:
            raise ValueError("--region starts before the query slice begins (1 + --query-offset)")

    loaded = {role: _load_input(paths[role], region, query_offset if role == "query" else 0) for role in LABELS}
    inputs = {role: data.info for role, data in loaded.items()}
    check_seqids(inputs, region)
    if region is None:
        _check_full_scope(loaded.values())

    results = {role: calculate(loaded["reference"].transcripts, loaded[role].transcripts)
               for role in ("query", "augustus")}
    metrics = {}
    for metric in METRICS:
        scores = {role: result[metric].as_dict() for role, result in results.items()}
        query_f1, augustus_f1 = scores["query"]["f1"], scores["augustus"]["f1"]
        delta = query_f1 - augustus_f1 if query_f1 is not None and augustus_f1 is not None else None
        metrics[metric] = {**scores, "delta_f1": {"query_minus_augustus": delta}}

    primary = metrics[PRIMARY]
    delta = primary["delta_f1"]["query_minus_augustus"]
    outcome = "win" if primary["query"]["reference"] and delta > 0 else "no-win"

    return {
        "settings": {"protein_coding_only": True, "region": str(region) if region else None,
                     "query_offset": query_offset, "region_policy": "fully contained transcripts"},
        "inputs": inputs,
        "primary": {"metric": PRIMARY, "score": "f1", "outcome": outcome},
        "metrics": metrics,
    }


def print_table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(map(len, column)) for column in zip(*rows)]
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def print_text(report: dict) -> None:
    rows = [("metric", "GHMM F1", "AUGUSTUS F1", "delta (pp)")]
    for metric, result in report["metrics"].items():
        query, augustus = (result[role]["f1"] for role in ("query", "augustus"))
        delta = result["delta_f1"]["query_minus_augustus"]
        rows.append((metric, "NA" if query is None else f"{query:.2%}",
                     "NA" if augustus is None else f"{augustus:.2%}",
                     "NA" if delta is None else f"{100 * delta:+.2f}"))
    print_table(rows)
    print(f"{PRIMARY}: {report['primary']['outcome']}")
