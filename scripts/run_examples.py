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
)
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
    "examples/27_kafka_protobuf_source.py",
    "examples/28_kafka_custom_decoder.py",
)
RUST_CONNECTOR_FEATURES = {
    "15_file_source": "file",
    "16_kafka_source": "file,kafka",
    "17_postgresql_source": "file,postgresql",
    "18_mysql_source": "file,mysql",
    "19_clickhouse_source": "file,clickhouse",
    "20_http_source": "file,http",
    "21_websocket_source": "file,websocket",
    "22_kafka_sink": "file,kafka",
    "23_postgresql_sink": "file,postgresql",
    "24_mysql_sink": "file,mysql",
    "25_clickhouse_sink": "file,clickhouse",
    "27_kafka_protobuf_source": "file,kafka",
    "28_kafka_custom_decoder": "file,kafka",
    "late_output_recovery": "file",
}


def _rust_name(path: str) -> str:
    return Path(path).stem


def _example_arguments(name: str) -> list[str]:
    if name not in {
        "30_streaming_stock_positions",
        "31_streaming_account_market_values",
    }:
        return []
    arguments = [
        "--rows",
        "30",
        "--accounts",
        "2",
        "--stocks",
        "3",
        "--batch-size",
        "4",
    ]
    if name == "31_streaming_account_market_values":
        arguments.extend(["--quote-rows", "9"])
    return arguments


RUST_EXAMPLES = tuple(
    dict.fromkeys(
        ("expression_pipeline", "windowed_streaming")
        + tuple(_rust_name(path) for path in PYTHON_EXAMPLES)
        + ("late_output_recovery",)
    )
)
SERVICE_RUST_EXAMPLES = tuple(_rust_name(path) for path in SERVICE_PYTHON_EXAMPLES)


def _commands(surface: str, *, include_services: bool = False) -> tuple[list[str], ...]:
    commands: list[list[str]] = []
    if surface in {"all", "python"}:
        commands.extend(
            [sys.executable, path, *_example_arguments(Path(path).stem)]
            for path in PYTHON_EXAMPLES
            if include_services or path not in SERVICE_PYTHON_EXAMPLES
        )
    if surface in {"all", "rust"}:
        for name in RUST_EXAMPLES:
            if not include_services and name in SERVICE_RUST_EXAMPLES:
                continue
            features = RUST_CONNECTOR_FEATURES.get(name)
            package = "calc-flow-connectors" if features else "calc-flow"
            command = ["cargo", "run", "-p", package, "--example", name]
            if features:
                command.extend(["--features", features])
            if name in {"27_kafka_protobuf_source", "28_kafka_custom_decoder"}:
                command.extend(["--", "--consume"])
            arguments = _example_arguments(name)
            if arguments:
                command.extend(["--", *arguments])
            commands.append(command)
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
