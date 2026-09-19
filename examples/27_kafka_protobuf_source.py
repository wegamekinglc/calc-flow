"""Consume a protobuf-encoded Kafka topic with the native source connector.

Prepare the broker and the calc-flow-example-orders-proto topic described in
examples/README.md, produce the two sample orders with
`python examples/27_kafka_protobuf_source.py --produce` (requires the
confluent-kafka package), then run the consumer without arguments.
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as parquet

import calc_flow as cf
from calc_flow import ProjectDocument, Runtime, StreamingRunner

DATA = Path(__file__).resolve().parent / "data"
TOPIC = "calc-flow-example-orders-proto"
MESSAGE = "calcflow.examples.Order"
SAMPLE_ORDERS = ((1, 2, 10.0), (2, 3, 20.0))


def order_totals(orders: cf.TableExpr) -> cf.TableExpr:
    totals = orders.with_columns(
        total=cf.row.cast(orders["quantity"], "float64") * orders["price"]
    )
    return totals.filter(totals["quantity"] > 0).select("id", "total")


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("the demo encoder supports non-negative fields only")
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            encoded.append(byte | 0x80)
        else:
            encoded.append(byte)
            return bytes(encoded)


def encode_order(order: tuple[int, int, float]) -> bytes:
    """Encode one sample Order with the protobuf wire format (demo only).

    Production producers should use generated protobuf classes; this keeps
    the example dependency-free for the three scalar fields of
    examples/data/orders.proto.
    """
    id_, quantity, price = order
    return (
        b"\x08"
        + _varint(id_)
        + b"\x10"
        + _varint(quantity)
        + b"\x19"
        + struct.pack("<d", price)
    )


def produce_orders() -> None:
    try:
        from confluent_kafka import Producer
    except ImportError:
        raise SystemExit(
            "Producing the sample orders requires the confluent-kafka package: "
            "uv run --with confluent-kafka python "
            "examples/27_kafka_protobuf_source.py --produce"
        ) from None
    producer = Producer(
        {
            "bootstrap.servers": os.environ.get(
                "CALC_FLOW_KAFKA_BOOTSTRAP", "127.0.0.1:9092"
            )
        }
    )
    for order in SAMPLE_ORDERS:
        producer.produce(TOPIC, encode_order(order))
    producer.flush()
    print(f"produced {len(SAMPLE_ORDERS)} protobuf orders to {TOPIC}")


def build_project(directory: Path) -> ProjectDocument:
    schema = [
        cf.Field("id", "int64"),
        cf.Field("quantity", "int64"),
        cf.Field("price", "float64"),
    ]
    orders = cf.table_input("orders", schema=schema)
    graph = cf.Program(
        "kafka-protobuf-source", outputs={"totals": orders.pipe(order_totals)}
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
                    "format": {"name": "protobuf", "version": "1"},
                    "options": {
                        "schema": [asdict(field) for field in schema],
                        "bootstrap_servers": os.environ.get(
                            "CALC_FLOW_KAFKA_BOOTSTRAP", "127.0.0.1:9092"
                        ),
                        "topic": TOPIC,
                        "partitions": [0],
                        "auto_offset_reset": "earliest",
                        "format": "protobuf",
                        "descriptor_set": str(DATA / "orders.pb"),
                        "message": MESSAGE,
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
            raise RuntimeError(f"unexpected kafka job outcome: {outcome}")
        totals = await asyncio.to_thread(read_totals, directory)
        if totals != [20.0, 60.0]:
            raise RuntimeError(f"unexpected kafka totals: {totals}")
        print("kafka protobuf -> Parquet:", totals)
        print("delivery:", job.status()["delivery"]["output"])
    finally:
        if job is not None:
            await job.cancel_async()


def main(argv: list[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args == ["--produce"]:
        produce_orders()
        return
    if args:
        raise SystemExit(f"unexpected arguments: {args}")
    available = {connector.name for connector in Runtime().capabilities().connectors}
    if not {"file", "kafka"} <= available:
        raise SystemExit(
            "Build a wheel with connector-kafka and connector-file; "
            "see examples/README.md."
        )
    with TemporaryDirectory(prefix="calc-flow-kafka-protobuf-") as directory:
        asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    main()
