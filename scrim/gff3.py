from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.parse import unquote

Span = tuple[int, int]
TRANSCRIPT_TYPES = {
    "mrna",
    "transcript",
    "primary_transcript",
    "ncrna",
    "lnc_rna",
    "trna",
    "rrna",
    "mirna",
    "snrna",
    "snorna",
    "pseudogenic_transcript",
}


@dataclass(slots=True)
class Transcript:
    tid: str
    gid: str
    seqid: str
    strand: str
    spans: dict[str, list[Span]] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedGFF3:
    transcripts: list[Transcript]
    declared: dict[str, Span]
    observed: dict[str, Span]
    outside: set[str]


def merge(spans: list[Span]) -> list[Span]:
    result: list[Span] = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def contains(outer: Span, inner: Span) -> bool:
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


def _identifier(value: str, where: str, *, strict: bool = True) -> str:
    if not strict:
        return unquote(value)
    if value and "%" not in value:
        return value
    if not value or re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError(f"{where}: invalid GFF3 identifier {value!r}")
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{where}: invalid GFF3 identifier {value!r}") from exc


def _span(
    first: str, last: str, where: str, kind: str, interval_error: str | None = None
) -> Span:
    try:
        start, end = int(first), int(last)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid {kind} coordinates") from exc
    if not 1 <= start <= end:
        raise ValueError(f"{where}: {interval_error or f'invalid {kind} coordinates'}")
    return start, end


def _attributes(text: str, where: str, *, strict: bool) -> dict[str, str]:
    attrs = {}
    for part in text.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name in ("ID", "Parent"):
            if strict and name in attrs:
                raise ValueError(f"{where}: duplicate {name} attribute")
            attrs[name] = value
    return attrs


def _fasta_lengths(lines: Iterator[tuple[int, str]], path: str) -> dict[str, int]:
    lengths: dict[str, int] = {}
    seqid = None
    for line_no, line in lines:
        if line.startswith(">"):
            name = line[1:].split(maxsplit=1)
            if not name:
                raise ValueError(f"{path}:{line_no}: missing FASTA ID")
            seqid = _identifier(name[0], f"{path}:{line_no}")
            if seqid in lengths:
                raise ValueError(f"{path}:{line_no}: duplicate FASTA ID {seqid!r}")
            lengths[seqid] = 0
        elif line.strip():
            if seqid is None:
                raise ValueError(f"{path}:{line_no}: expected a FASTA header")
            lengths[seqid] += sum(map(len, line.split()))
    for seqid, length in lengths.items():
        if not length:
            raise ValueError(f"{path}: empty FASTA record {seqid!r}")
    return lengths


def parse_gff3(
    path: str, *, track_scope: bool = False, window: tuple[str, Span] | None = None
) -> ParsedGFF3:
    path = str(path)
    declared: dict[str, Span] = {}
    observed: dict[str, Span] = {}
    definitions: dict[str, tuple[str, str, str]] = {}
    models: dict[str, Transcript] = {}
    outside: set[str] = set()
    fasta = {}
    interval_error = None if track_scope else "invalid interval or strand"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        lines = enumerate(handle, 1)
        for line_no, line in lines:
            where = f"{path}:{line_no}"
            if line.startswith("##"):
                if not track_scope:
                    if line.startswith("##FASTA"):
                        break
                    continue
                directive = line.split()
                if directive[0] == "##FASTA":
                    if len(directive) != 1:
                        raise ValueError(f"{where}: invalid ##FASTA directive")
                    fasta = _fasta_lengths(lines, path)
                    break
                if directive[0] == "##sequence-region":
                    if len(directive) != 4:
                        raise ValueError(
                            f"{where}: invalid ##sequence-region directive"
                        )
                    _, name, first, last = directive
                    seqid = _identifier(name, where)
                    span = _span(first, last, where, "##sequence-region")
                    if declared.setdefault(seqid, span) != span:
                        raise ValueError(
                            f"{where}: conflicting ##sequence-region for {seqid!r}"
                        )
                continue
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{where}: expected 9 tab-separated fields")
            name, _, kind, first, last, _, strand, _, attributes = fields
            kind = kind.lower()
            is_transcript = kind in TRANSCRIPT_TYPES
            is_segment = kind in ("cds", "exon")
            if not (track_scope or is_transcript or is_segment):
                continue
            seqid = _identifier(name, where) if track_scope else name
            span = _span(first, last, where, "feature", interval_error)
            if track_scope:
                start, end = observed.get(seqid, span)
                observed[seqid] = min(start, span[0]), max(end, span[1])
            crosses = (
                track_scope
                and window is not None
                and seqid == window[0]
                and not contains(window[1], span)
            )
            if not (is_transcript or is_segment or crosses):
                continue
            if (is_transcript or is_segment) and strand not in ("+", "-"):
                raise ValueError(f"{where}: {interval_error or 'invalid strand'}")
            attrs = _attributes(attributes, where, strict=track_scope)
            parents = (
                [
                    _identifier(p, where, strict=track_scope)
                    for p in attrs["Parent"].split(",")
                    if track_scope or p
                ]
                if "Parent" in attrs
                else []
            )
            if is_transcript:
                if value := attrs.get("ID"):
                    tid = _identifier(value, where, strict=track_scope)
                    definition = (parents[0] if parents else tid, seqid, strand)
                    if definitions.setdefault(tid, definition) != definition:
                        raise ValueError(f"{where}: conflicting transcript ID {tid!r}")
                    if crosses:
                        outside.add(tid)
                continue
            if crosses:
                outside.update(parents)
            if not is_segment:
                continue
            if not parents:
                raise ValueError(f"{where}: {kind} lacks Parent=")
            for tid in parents:
                if tid not in models:
                    models[tid] = Transcript(tid, tid, seqid, strand)
                t = models[tid]
                if (t.seqid, t.strand) != (seqid, strand):
                    raise ValueError(
                        f"{where}: transcript {tid!r} spans different contigs or strands"
                    )
                if kind == "cds":
                    t.spans.pop("exon", None)
                    t.spans.setdefault("CDS", []).append(span)
                elif not t.spans.get("CDS"):
                    t.spans.setdefault("exon", []).append(span)

    for seqid, length in fasta.items():
        start, end = declared.setdefault(seqid, (1, length))
        if length < end and length != end - start + 1:
            raise ValueError(
                f"{path}: FASTA length for {seqid!r} conflicts with declared sequence span {start}-{end}"
            )
    for seqid, span in observed.items():
        if seqid in declared and not contains(declared[seqid], span):
            start, end = declared[seqid]
            raise ValueError(
                f"{path}: feature coordinates {seqid}:{span[0]}-{span[1]} outside declared sequence span "
                f"{seqid}:{start}-{end}"
            )
    for tid, t in models.items():
        t.gid, seqid, strand = definitions.get(tid, (tid, t.seqid, t.strand))
        if (seqid, strand) != (t.seqid, t.strand):
            raise ValueError(
                f"{path}: {tid!r} location/strand conflicts with parent transcript"
            )
        cds = merge(t.spans.get("CDS", []))
        t.spans = {"exon": cds or merge(t.spans.get("exon", [])), "CDS": cds}
    return ParsedGFF3(list(models.values()), declared, observed, outside)
