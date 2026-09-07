"""Read a bounded ClickHouse snapshot with an explicit schema and cursor.

Prepare the two sample orders described in examples/README.md before running.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as parquet

from calc_flow import PipelineBuilder, ProjectDocument, Runtime, StreamingRunner


def build_project(directory: Path) -> ProjectDocument:
    graph = PipelineBuilder("clickhouse-source").expression(
        "calculate", "total = quantity * price"
    )
    return ProjectDocument.model_validate(
        {
            **graph.project,
            "data_sources": [],
            "runtime": {"mode": "stream", "options": {}},
            "sources": [
                {
                    "binding": "input",
                    "connector": {
                        "provider": "calc-flow-connectors",
                        "name": "clickhouse",
                        "version": "2.0.0",
                    },
                    "secrets": {
                        "url": {"resolver": "environment", "key": "CALC_FLOW_CH_URL"}
                    },
                    "options": {
                        "table": "calc_flow_example_orders",
                        "mode": "snapshot",
                        "cursor_column": "sequence",
                        "tie_breaker_column": "id",
                        "tie_breaker_unique": True,
                        "columns": [
                            {
                                "name": "sequence",
                                "data_type": "int64",
                                "nullable": False,
                            },
                            {"name": "id", "data_type": "int64", "nullable": False},
                            {
                                "name": "quantity",
                                "data_type": "int64",
                                "nullable": False,
                            },
                            {
                                "name": "price",
                                "data_type": "float64",
                                "nullable": False,
                            },
                        ],
                        "max_batch_rows": 1,
                        "max_batch_bytes": 8388608,
                    },
                    "watermark": {"policy": "disabled"},
                }
            ],
            "sinks": [
                {
                    "binding": "output",
                    "connector": {
                        "provider": "calc-flow-connectors",
                        "name": "file",
                        "version": "2.0.0",
                    },
                    "format": {"name": "parquet", "version": "1"},
                    "options": {"path": str(directory), "output": "results"},
                    "delivery": "at_least_once",
                }
            ],
            "state": {"root": str(directory / "state"), "retention": 2},
        }
    )


def read_totals(directory: Path) -> list[float]:
    parts = sorted((directory / "results").rglob("*.parquet"))
    if not parts:
        raise RuntimeError("unexpected missing Parquet output")
    table = pa.concat_tables([parquet.read_table(part) for part in parts])
    return table.sort_by("id")["total"].to_pylist()


async def run(directory: Path, *, timeout: float = 60) -> None:
    plan = Runtime().compile_stream_project(build_project(directory).canonical_json())
    job = None
    try:
        async with asyncio.timeout(timeout):
            job = await StreamingRunner(plan).start_async()
            outcome = await job.wait_async()
        if outcome.state != "completed":
            raise RuntimeError(f"unexpected clickhouse job outcome: {outcome}")
        totals = await asyncio.to_thread(read_totals, directory)
        if totals != [20.0, 60.0]:
            raise RuntimeError(f"unexpected clickhouse totals: {totals}")
        print("clickhouse -> Parquet:", totals)
        print("delivery:", job.status()["delivery"]["output"])
    finally:
        if job is not None:
            await job.cancel_async()


def main() -> None:
    available = {connector.name for connector in Runtime().capabilities().connectors}
    if not {"file", "clickhouse"} <= available:
        raise SystemExit(
            "Build a wheel with connector-clickhouse and connector-file; "
            "see examples/README.md."
        )
    if not os.environ.get("CALC_FLOW_CH_URL"):
        raise SystemExit(
            "Set CALC_FLOW_CH_URL after preparing the sample service; "
            "see examples/README.md."
        )
    with TemporaryDirectory(prefix="calc-flow-clickhouse-source-") as directory:
        asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    main()
