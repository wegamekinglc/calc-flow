"""Write calculated order totals through the native postgresql sink.

Prepare the empty demo destination in docs/connectors/postgresql.md first.
The destination retains its rows; only local inputs/checkpoints are temporary.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as parquet

import calc_flow as cf


def order_totals(orders: cf.TableExpr) -> cf.TableExpr:
    totals = orders.with_columns(
        total=cf.row.cast(orders["quantity"], "float64") * orders["price"]
    )
    return totals.filter(totals["quantity"] > 0).select("id", "total")


def sample_orders() -> pa.Table:
    return pa.table(
        {"id": [1, 2, 3], "quantity": [2, 3, 0], "price": [10.0, 20.0, 99.0]}
    )


def build_project(directory: Path) -> cf.ProjectDocument:
    orders = cf.table_input("orders", schema=sample_orders().schema)
    program = cf.Program(
        "postgresql-sink", outputs={"totals": orders.pipe(order_totals)}
    )
    sink = {
        "binding": "output",
        "connector": {
            "provider": "calc-flow-connectors",
            "name": "postgresql",
            "version": "2.0.0",
        },
        "secrets": {"url": {"resolver": "environment", "key": "CALC_FLOW_PG_URL"}},
        "options": {
            "table": "calc_flow_example_totals",
            "mode": "append",
            "pipeline": directory.name,
            "output": "totals",
        },
        "delivery": "at_least_once",
    }
    return cf.ProjectDocument.model_validate(
        {
            **program.to_project(mode="stream").model_dump(),
            "data_sources": [],
            "runtime": {"mode": "stream", "options": {}},
            "sources": [
                {
                    "binding": "input",
                    "connector": {
                        "provider": "calc-flow-connectors",
                        "name": "file",
                        "version": "2.0.0",
                    },
                    "format": {"name": "parquet", "version": "1"},
                    "options": {
                        "path": str(directory / "orders.parquet"),
                        "format": "parquet",
                    },
                    "watermark": {"policy": "disabled"},
                }
            ],
            "sinks": [sink],
            "state": {"root": str(directory / "state"), "retention": 2},
        }
    )


async def run(directory: Path, *, timeout: float = 60) -> None:
    await asyncio.to_thread(
        parquet.write_table, sample_orders(), directory / "orders.parquet"
    )
    plan = cf.Runtime().compile_stream_project(
        build_project(directory).canonical_json()
    )
    job = None
    try:
        async with asyncio.timeout(timeout):
            job = await cf.StreamingRunner(plan).start_async()
            outcome = await job.wait_async()
        if outcome.state != "completed":
            raise RuntimeError(f"unexpected postgresql job outcome: {outcome}")
        status = job.status()
        rows = sum(int(sink["delivered_rows"]) for sink in status["sinks"].values())
        if rows != 2:
            raise RuntimeError(f"unexpected postgresql delivered rows: {rows}")
        if status["delivery"]["output"]["effective"] != "at_least_once":
            raise RuntimeError("unexpected postgresql delivery guarantee")
        print(
            "Parquet -> postgresql: wrote 2 rows; "
            "expected (id, total) = (1, 20.0), (2, 60.0)"
        )
        print("delivery:", status["delivery"]["output"])
    finally:
        if job is not None:
            await job.cancel_async()


def main() -> None:
    available = {item.name for item in cf.Runtime().capabilities().connectors}
    if not {"file", "postgresql"} <= available:
        raise SystemExit(
            "Build connector-file and connector-postgresql; see examples/README.md."
        )
    if not os.environ.get("CALC_FLOW_PG_URL"):
        raise SystemExit("Set CALC_FLOW_PG_URL; see docs/connectors/postgresql.md.")
    with TemporaryDirectory(prefix="calc-flow-postgresql-sink-") as directory:
        asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    main()
