import argparse
import zlib
from pathlib import Path


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if any(
        _same_file(output, Path(path))
        for path in (args.fasta, args.target)
        if path is not None
    ):
        raise ValueError("--output must be distinct from --fasta and --target")
    if output.name.endswith(".gz"):
        raise ValueError("--output must be uncompressed (use a path not ending in .gz)")

    from . import mentor, student

    genome = student.orient(student.read_fasta(args.fasta))
    model, predictions = mentor.fit(genome, args.seed_gff3)
    if args.target is not None:
        genome = student.orient(student.read_fasta(args.target))
        predictions = student.predict(genome, model)
    student.write_gff3(args.output, genome, predictions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--seed-gff3", required=True)
    parser.add_argument("--target")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, EOFError, zlib.error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
