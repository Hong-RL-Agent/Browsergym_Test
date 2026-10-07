"""CLI for the evaluation-only R06 semantics diagnostic."""

from pathlib import Path

from .r06_semantics_diagnostic import write_reports


if __name__ == "__main__":
    print(write_reports(Path(__file__).resolve().parents[2]))
