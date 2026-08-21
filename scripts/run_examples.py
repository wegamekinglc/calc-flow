"""Run every user-facing Calc Flow example with one maintained command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXAMPLES = tuple(
    str(path.relative_to(REPOSITORY_ROOT))
    for path in sorted((REPOSITORY_ROOT / "examples").glob("[0-9][0-9]_*.py"))
)
RUST_EXAMPLES = (
    "expression_pipeline",
    "sql_join",
    "continuous_runtime",
    "windowed_streaming",
)


def _commands(surface: str) -> tuple[list[str], ...]:
    commands: list[list[str]] = []
    if surface in {"all", "python"}:
        commands.extend([sys.executable, path] for path in PYTHON_EXAMPLES)
    if surface in {"all", "rust"}:
        commands.extend(
            ["cargo", "run", "-p", "calc-flow", "--example", name]
            for name in RUST_EXAMPLES
        )
    return tuple(commands)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        choices=("all", "python", "rust"),
        default="all",
        help="select which examples to run (default: all)",
    )
    arguments = parser.parse_args(argv)
    environment = os.environ.copy()
    environment.setdefault("JAX_PLATFORMS", "cpu")

    for command in _commands(arguments.surface):
        print(f"+ {' '.join(command)}", flush=True)
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
