"""Generate the development-only R06 refinement audit artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .r06_refinement_evaluator import write_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()
    write_reports(args.root, args.output_dir)


if __name__ == "__main__":
    main()
