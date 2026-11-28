import argparse
import zlib
from pathlib import Path

def run(args: argparse.Namespace) -> None:
    # TODO
    return None

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
