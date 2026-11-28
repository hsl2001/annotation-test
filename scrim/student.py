from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cwt_features import COMPLEMENT, LUT, nucleotide_codes
from .cwt_features import read_fasta as read_records

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Strand:
    bases: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "bases", nucleotide_codes(self.bases))


Genome = dict[str, tuple[Strand, Strand]]


@dataclass(frozen=True, slots=True)
class Gene:
    cds: tuple[Span, ...]
    strand: str
    score: float


def read_fasta(path) -> dict[str, Strand]:
    return read_records(path, lambda raw: Strand(LUT[raw]))


def orient(records: dict[str, Strand]) -> Genome:
    return {
        name: (record, Strand(COMPLEMENT[record.bases[::-1]]))
        for name, record in records.items()
    }


def reverse_spans(spans, length: int) -> tuple[Span, ...]:
    return tuple((length - end, length - start) for start, end in reversed(spans))


def decode(seed_transcripts) -> dict[str, list[Gene]]:
    result: dict[str, list[Gene]] = {}
    for transcript in seed_transcripts:
        spans = transcript.spans.get("CDS") or transcript.spans["exon"]
        result.setdefault(transcript.seqid, []).append(
            Gene(tuple(spans), transcript.strand, 0.0)
        )
    return result


def write_gff3(path, genome: Genome, predictions: dict[str, list[Gene]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##gff-version 3\n")
        index = 0
        for seqid, (forward, _) in genome.items():
            handle.write(f"##sequence-region {seqid} 1 {len(forward.bases)}\n")
            for gene in predictions.get(seqid, []):
                index += 1
                gid = f"gene{index}"
                tid = f"{gid}.t1"
                start, end = gene.cds[0][0], gene.cds[-1][1]
                emit = lambda kind, a, b, phase, attrs: handle.write(
                    f"{seqid}\tstudent\t{kind}\t{a}\t{b}\t.\t{gene.strand}\t{phase}\t{attrs}\n"
                )
                emit("gene", start, end, ".", f"ID={gid}")
                emit("mRNA", start, end, ".", f"ID={tid};Parent={gid}")
                phase = 0
                ordered = gene.cds if gene.strand == "+" else tuple(reversed(gene.cds))
                for a, b in ordered:
                    emit("exon", a, b, ".", f"Parent={tid}")
                    emit("CDS", a, b, phase, f"Parent={tid}")
                    phase = (phase - (b - a + 1)) % 3
