"""Read CSV, JSON Lines, and Parquet with the native file source connector."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as parquet

from calc_flow import PipelineBuilder, ProjectDocument, Runtime, StreamingRunner


def write_input(directory: Path, format_name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": [1, 2], "quantity": [2, 3], "price": [10.0, 20.0]})
    path = directory / f"orders.{format_name}"
    if format_name == "csv":
        csv.write_csv(table, path)
    elif format_name == "json":
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in table.to_pylist()),
            encoding="utf-8",
        )
    else:
        parquet.write_table(table, path)
    return path


def build_project(directory: Path, format_name: str) -> ProjectDocument:
    connector = {
        "provider": "calc-flow-connectors",
        "name": "file",
        "version": "2.0.0",
    }
    graph = PipelineBuilder(f"file-{format_name}").expression(
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
                    "connector": connector,
                    "format": {"name": format_name, "version": "1"},
                    "options": {
                        "path": str(directory / f"orders.{format_name}"),
                        "format": format_name,
                        "header": True,
                        "max_batch_rows": 8192,
                    },
                    "watermark": {"policy": "disabled"},
                }
            ],
            "sinks": [
                {
                    "binding": "output",
                    "connector": connector,
                    "format": {"name": "parquet", "version": "1"},
                    "options": {"path": str(directory), "output": "results"},
                    "delivery": "exactly_once",
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


async def run(directory: Path, format_name: str) -> None:
    await asyncio.to_thread(write_input, directory, format_name)
    project = build_project(directory, format_name)
    plan = Runtime().compile_stream_project(project.canonical_json())
    job = None
    try:
        async with asyncio.timeout(60):
            job = await StreamingRunner(plan).start_async()
            outcome = await job.wait_async()
        if outcome.state != "completed":
            raise RuntimeError(f"unexpected file job outcome: {outcome}")
        totals = await asyncio.to_thread(read_totals, directory)
        if totals != [20.0, 60.0]:
            raise RuntimeError(f"unexpected {format_name} totals: {totals}")
        if job.status()["delivery"]["output"]["effective"] != "exactly_once":
            raise RuntimeError("unexpected file delivery guarantee")
        print(f"{format_name} -> Parquet: {totals} (exactly_once)")
    finally:
        if job is not None:
            await job.cancel_async()


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-file-source-") as directory:
        for format_name in ("csv", "json", "parquet"):
            asyncio.run(run(Path(directory) / format_name, format_name))


if __name__ == "__main__":
    main()
