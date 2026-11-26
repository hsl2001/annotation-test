import argparse
import json
import zlib
from pathlib import Path


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def _predict(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if any(_same_file(output, Path(path)) for path in (args.fasta, args.target) if path is not None):
        raise ValueError("--output must be distinct from --fasta and --target")
    if output.name.endswith(".gz"):
        raise ValueError("--output must be uncompressed (use a path not ending in .gz)")

    from . import mentor, student

    genome = student.orient(student.read_fasta(args.fasta))
    model, predictions = mentor.fit(genome)
    if args.target is not None:
        genome = student.orient(student.read_fasta(args.target))
        predictions = student.predict(genome, model)
    student.write_gff3(args.output, genome, predictions)


def _score(args: argparse.Namespace) -> None:
    from .benchmark import make_report, parse_region, print_text

    paths = {"reference": args.reference, "query": args.query, "augustus": args.augustus}
    region = parse_region(args.region) if args.region is not None else None
    report = make_report(paths, region, args.query_offset)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", add_help=False, allow_abbrev=False)
    predict.add_argument("--fasta", required=True)
    predict.add_argument("--target")
    predict.add_argument("--output", required=True)

    score = commands.add_parser("score", add_help=False, allow_abbrev=False)
    score.add_argument("--reference", required=True)
    score.add_argument("--query", required=True)
    score.add_argument("--augustus", required=True)
    score.add_argument("--region")
    score.add_argument("--query-offset", type=int, default=0)
    score.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            _predict(args)
        else:
            _score(args)
    except (OSError, ValueError, EOFError, zlib.error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
