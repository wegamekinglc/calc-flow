"""Consume a pipe-delimited Kafka topic through a Python-registered decoder.

Prepare the broker and the calc-flow-example-orders-pipe topic described in
examples/README.md and produce the two sample records with
kafka-console-producer (text payloads), then run the consumer.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as parquet

import calc_flow as cf
from calc_flow import ProjectDocument, Runtime, StreamingRunner

TOPIC = "calc-flow-example-orders-pipe"
DECODER = {"name": "pipe-orders", "version": "1"}


def order_totals(orders: cf.TableExpr) -> cf.TableExpr:
    totals = orders.with_columns(
        total=cf.row.cast(orders["quantity"], "float64") * orders["price"]
    )
    return totals.filter(totals["quantity"] > 0).select("id", "total")


def decode_pipe_orders(payload: bytes) -> pa.RecordBatch:
    """Decode one ``id|quantity|price`` record into a single-row batch."""
    id_, quantity, price = payload.decode("utf-8").split("|")
    return pa.record_batch(
        [[int(id_)], [int(quantity)], [float(price)]],
        names=["id", "quantity", "price"],
    )


def build_project(directory: Path) -> ProjectDocument:
    schema = [
        cf.Field("id", "int64"),
        cf.Field("quantity", "int64"),
        cf.Field("price", "float64"),
    ]
    orders = cf.table_input("orders", schema=schema)
    graph = cf.Program(
        "kafka-custom-decoder", outputs={"totals": orders.pipe(order_totals)}
    )
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
                        "name": "kafka",
                        "version": "2.0.0",
                    },
                    "format": {"name": "custom", "version": "1"},
                    "options": {
                        "schema": [asdict(field) for field in schema],
                        "bootstrap_servers": os.environ.get(
                            "CALC_FLOW_KAFKA_BOOTSTRAP", "127.0.0.1:9092"
                        ),
                        "topic": TOPIC,
                        "partitions": [0],
                        "auto_offset_reset": "earliest",
                        "format": "custom",
                        "decoder": DECODER,
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
    runtime = Runtime()
    runtime.register_kafka_decoder(
        name=DECODER["name"], version=DECODER["version"], function=decode_pipe_orders
    )
    plan = runtime.compile_stream_project(build_project(directory).canonical_json())
    job = None
    try:

        async def wait_for_completion():
            nonlocal job
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
            return outcome

        outcome = await asyncio.wait_for(wait_for_completion(), timeout)
        if outcome.state != "completed":
            raise RuntimeError(f"unexpected kafka job outcome: {outcome}")
        totals = await asyncio.to_thread(read_totals, directory)
        if totals != [20.0, 60.0]:
            raise RuntimeError(f"unexpected kafka totals: {totals}")
        print("kafka custom decoder -> Parquet:", totals)
        print("delivery:", job.status()["delivery"]["output"])
    finally:
        if job is not None:
            await job.cancel_async()


def main() -> None:
    available = {connector.name for connector in Runtime().capabilities().connectors}
    if not {"file", "kafka"} <= available:
        raise SystemExit(
            "Build a wheel with connector-kafka and connector-file; "
            "see examples/README.md."
        )
    with TemporaryDirectory(prefix="calc-flow-kafka-custom-") as directory:
        asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    main()
