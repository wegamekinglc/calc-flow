"""Attribution-only worker for the fixed DAL-392 group_by workload."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.groupby_diagnostic.dal392 import worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("worker",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.output, phases=True)


if __name__ == "__main__":
    main()
