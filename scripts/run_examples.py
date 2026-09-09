"""Run Calc Flow examples, optionally including external-service connectors."""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXAMPLES = tuple(
    str(path.relative_to(REPOSITORY_ROOT))
    for path in sorted((REPOSITORY_ROOT / "examples").glob("[0-9][0-9]_*.py"))
) + ("examples/symbolic_event_window.py",)
SERVICE_PYTHON_EXAMPLES = (
    "examples/16_kafka_source.py",
    "examples/17_postgresql_source.py",
    "examples/18_mysql_source.py",
    "examples/19_clickhouse_source.py",
    "examples/20_http_source.py",
    "examples/21_websocket_source.py",
    "examples/22_kafka_sink.py",
    "examples/23_postgresql_sink.py",
    "examples/24_mysql_sink.py",
    "examples/25_clickhouse_sink.py",
)
RUST_EXAMPLES = (
    "expression_pipeline",
    "sql_join",
    "continuous_runtime",
    "windowed_streaming",
)


def _commands(surface: str, *, include_services: bool = False) -> tuple[list[str], ...]:
    commands: list[list[str]] = []
    if surface in {"all", "python"}:
        commands.extend(
            [sys.executable, path]
            for path in PYTHON_EXAMPLES
            if include_services or path not in SERVICE_PYTHON_EXAMPLES
        )
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
    parser.add_argument(
        "--include-services",
        action="store_true",
        help="also run connector examples requiring prepared external services",
    )
    arguments = parser.parse_args(argv)
    environment = os.environ.copy()
    environment.setdefault("JAX_PLATFORMS", "cpu")

    if arguments.surface in {"all", "python"} and not arguments.include_services:
        print("Skipping external-service examples (enable with --include-services):")
        for path in SERVICE_PYTHON_EXAMPLES:
            print(f"  {path}")

    for command in _commands(
        arguments.surface, include_services=arguments.include_services
    ):
        print(f"+ {' '.join(command)}", flush=True)
        # Every executable and argument comes from the fixed inventories above;
        # argparse accepts only declared choices and a boolean flag. Never use
        # a shell here, so example names cannot become command syntax.
        try:
            subprocess.run(  # nosec B603  # nosemgrep
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
