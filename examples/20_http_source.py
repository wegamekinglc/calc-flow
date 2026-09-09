"""Poll a JSON Lines HTTP endpoint with the native source connector.

Prepare the two sample orders described in examples/README.md before running.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as parquet

import calc_flow as cf
from calc_flow import ProjectDocument, Runtime, StreamingRunner


def order_totals(orders: cf.TableExpr) -> cf.TableExpr:
    totals = orders.with_columns(
        total=cf.row.cast(orders["quantity"], "float64") * orders["price"]
    )
    return totals.filter(totals["quantity"] > 0).select("id", "total")


def build_project(directory: Path) -> ProjectDocument:
    # The JSON decoder infers object fields in name order.
    schema = [
        cf.Field("id", "int64"),
        cf.Field("price", "float64"),
        cf.Field("quantity", "int64"),
    ]
    orders = cf.table_input("orders", schema=schema)
    graph = cf.Program("http-source", outputs={"totals": orders.pipe(order_totals)})
    return ProjectDocument.model_validate(
        {
            **graph.to_project(mode="stream").model_dump(),
            "data_sources": [],
            "runtime": {"mode": "stream", "options": {}},
            "sources": [
                {
                    "binding": "input",
                    "connector": {
                        "provider": "calc-flow-connectors",
                        "name": "http",
                        "version": "2.0.0",
                    },
                    "format": {"name": "json", "version": "1"},
                    "secrets": {
                        "url": {"resolver": "environment", "key": "CALC_FLOW_HTTP_URL"}
                    },
                    "options": {
                        "conditional": True,
                        "poll_interval_ms": 1000,
                        "timeout_seconds": 10,
                        "max_retries": 2,
                        "max_response_bytes": 8388608,
                        "max_batch_rows": 8192,
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
                    "delivery": "best_effort",
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
            # A continuous source does not end after the sample. Wait for sink
            # delivery before draining; observing source reads alone is too early.
            while job.status()["state"] == "running":
                delivered = sum(
                    int(sink["delivered_rows"])
                    for sink in job.status()["sinks"].values()
                )
                if delivered >= 2:
                    break
                await asyncio.sleep(0.05)
            outcome = await job.shutdown_async()
        if outcome.state != "completed":
            raise RuntimeError(f"unexpected http job outcome: {outcome}")
        totals = await asyncio.to_thread(read_totals, directory)
        if totals != [20.0, 60.0]:
            raise RuntimeError(f"unexpected http totals: {totals}")
        print("http -> Parquet:", totals)
        print("delivery:", job.status()["delivery"]["output"])
    finally:
        if job is not None:
            await job.cancel_async()


def main() -> None:
    available = {connector.name for connector in Runtime().capabilities().connectors}
    if not {"file", "http"} <= available:
        raise SystemExit(
            "Build a wheel with connector-http and connector-file; "
            "see examples/README.md."
        )
    if not os.environ.get("CALC_FLOW_HTTP_URL"):
        raise SystemExit(
            "Set CALC_FLOW_HTTP_URL after preparing the sample service; "
            "see examples/README.md."
        )
    with TemporaryDirectory(prefix="calc-flow-http-source-") as directory:
        asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    main()
