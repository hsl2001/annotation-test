from __future__ import annotations

import gzip
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


def merge(spans: list[Span]) -> list[Span]:
    result: list[Span] = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def _span(first: str, last: str, where: str) -> Span:
    try:
        start, end = int(first), int(last)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid interval or strand") from exc
    if not 1 <= start <= end:
        raise ValueError(f"{where}: invalid interval or strand")
    return start, end


def _attributes(text: str) -> dict[str, str]:
    attrs = {}
    for part in text.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name in ("ID", "Parent"):
            attrs[name] = value
    return attrs


def parse_gff3(path: str) -> list[Transcript]:
    path = str(path)
    definitions: dict[str, tuple[str, str, str]] = {}
    models: dict[str, Transcript] = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            where = f"{path}:{line_no}"
            if line.startswith("##"):
                if line.startswith("##FASTA"):
                    break
                continue
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{where}: expected 9 tab-separated fields")
            name, _, kind, first, last, _, strand, _, attributes = fields
            kind = kind.lower()
            if kind not in TRANSCRIPT_TYPES and kind not in ("cds", "exon"):
                continue
            if strand not in ("+", "-"):
                raise ValueError(f"{where}: invalid interval or strand")
            span = _span(first, last, where)
            attrs = _attributes(attributes)
            parents = [unquote(p) for p in attrs.get("Parent", "").split(",") if p]
            if kind in TRANSCRIPT_TYPES:
                if value := attrs.get("ID"):
                    tid = unquote(value)
                    definition = (parents[0] if parents else tid, name, strand)
                    if definitions.setdefault(tid, definition) != definition:
                        raise ValueError(f"{where}: conflicting transcript ID {tid!r}")
                continue
            if not parents:
                raise ValueError(f"{where}: {kind} lacks Parent=")
            for tid in parents:
                t = models.setdefault(tid, Transcript(tid, tid, name, strand))
                if (t.seqid, t.strand) != (name, strand):
                    raise ValueError(
                        f"{where}: transcript {tid!r} spans different contigs or strands"
                    )
                if kind == "cds":
                    t.spans.pop("exon", None)
                    t.spans.setdefault("CDS", []).append(span)
                elif not t.spans.get("CDS"):
                    t.spans.setdefault("exon", []).append(span)

    for tid, t in models.items():
        gid, seqid, strand = definitions.get(tid, (tid, t.seqid, t.strand))
        if (seqid, strand) != (t.seqid, t.strand):
            raise ValueError(
                f"{path}: {tid!r} location/strand conflicts with parent transcript"
            )
        t.gid = gid
        cds = merge(t.spans.get("CDS", []))
        t.spans = {"exon": cds or merge(t.spans.get("exon", [])), "CDS": cds}
    return list(models.values())


def write_gff3(path: str, transcripts: list[Transcript]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##gff-version 3\n")
        for index, transcript in enumerate(transcripts, 1):
            gid = f"gene{index}"
            tid = f"{gid}.t1"
            seqid = transcript.seqid
            strand = transcript.strand
            spans = transcript.spans.get("CDS") or transcript.spans["exon"]
            start, end = spans[0][0], spans[-1][1]
            handle.write(
                f"{seqid}\tgff3\tgene\t{start}\t{end}\t.\t{strand}\t.\tID={gid}\n"
            )
            handle.write(
                f"{seqid}\tgff3\tmRNA\t{start}\t{end}\t.\t{strand}\t.\t"
                f"ID={tid};Parent={gid}\n"
            )
            phase = 0
            ordered = spans if strand == "+" else tuple(reversed(spans))
            for a, b in ordered:
                handle.write(
                    f"{seqid}\tgff3\texon\t{a}\t{b}\t.\t{strand}\t.\tParent={tid}\n"
                )
                handle.write(
                    f"{seqid}\tgff3\tCDS\t{a}\t{b}\t.\t{strand}\t{phase}\t"
                    f"Parent={tid}\n"
                )
                phase = (phase - (b - a + 1)) % 3
