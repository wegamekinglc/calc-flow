"""Compose two bounded native joins through explicit symbolic ordering."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    Cursor,
    Data,
    JoinStateLimits,
    JoinTimeBounds,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    SourceProvidedWatermarks,
    StreamingRunner,
    Watermark,
)
from calc_flow.symbolic import FeatureSet, Field, Program, table, table_input


def _ordered_input(name: str, event_time: str, value_name: str):
    return table_input(
        name,
        schema=[
            Field("account_id", "int64", nullable=False),
            Field(event_time, "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field(value_name, "float64", nullable=False),
        ],
        entity_by=["account_id"],
        event_time=event_time,
        sequence_by=["sequence"],
    )


def symbolic_program() -> Program:
    authorizations = _ordered_input(
        "authorizations", "authorized_at", "authorized_amount"
    )
    payments = _ordered_input("payments", "paid_at", "paid_amount")
    settlements = _ordered_input("settlements", "settled_at", "fee")
    bounds = JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2))
    limits = JoinStateLimits(10_000, 16 * 1024 * 1024, 100_000)

    matched = table.stream_join(
        authorizations,
        payments,
        left_keys=["account_id"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=bounds,
        limits=limits,
        left_prefix="authorization",
        right_prefix="payment",
        output_entity_by=["authorization__account_id"],
        output_event_time="authorization__authorized_at",
        output_sequence_by=[
            "authorization__sequence",
            "payment__sequence",
        ],
    )
    settled = table.stream_join(
        matched,
        settlements,
        left_keys=["authorization__account_id"],
        right_keys=["account_id"],
        left_event_time="authorization__authorized_at",
        right_event_time="settled_at",
        bounds=bounds,
        limits=limits,
        left_prefix="matched",
        right_prefix="settlement",
    )
    output = settled.with_columns(
        FeatureSet(
            [
                (
                    "net_amount",
                    settled["matched__authorization__authorized_amount"]
                    - settled["settlement__fee"],
                )
            ]
        )
    )
    return Program(
        "symbolic-relational-dag",
        inputs=[authorizations, payments, settlements],
        outputs=[("settled_matches", output)],
    )


class OneBatchSource:
    def __init__(self, value: pa.Table) -> None:
        self._value = value
        self._sent = False
        self._watermark_sent = False

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._sent = False
        self._watermark_sent = False

    async def next(self) -> Data | Watermark | None:
        if not self._sent:
            self._sent = True
            return Data(Batch.from_pyarrow(self._value), Cursor(b"1", {"offset": 1}))
        if not self._watermark_sent:
            self._watermark_sent = True
            return Watermark(datetime(2030, 1, 1, tzinfo=UTC))
        return None

    async def close(self) -> None:
        return None


class CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        value = batch.to_pyarrow()
        if value.num_rows:
            self.tables.append(value)

    async def close(self) -> None:
        return None


def _input_table(
    event_time: str, value_name: str, second: int, value: float
) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("account_id", pa.int64(), nullable=False),
            pa.field(event_time, pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field(value_name, pa.float64(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([7], type=pa.int64()),
            pa.array(
                [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([1], type=pa.uint64()),
            pa.array([value], type=pa.float64()),
        ],
        schema=schema,
    )


async def run() -> None:
    program = symbolic_program()
    runtime = Runtime()
    analysis = program.analyze(runtime, mode="stream")
    if analysis.issues:
        raise RuntimeError(analysis.issues)
    plan = program.compile_stream(runtime)
    sink = CollectSink()
    sources = {
        "authorizations.input": SourceBinding(
            OneBatchSource(
                _input_table("authorized_at", "authorized_amount", 10, 100.0)
            ),
            watermark_policy=SourceProvidedWatermarks(),
        ),
        "payments.input": SourceBinding(
            OneBatchSource(_input_table("paid_at", "paid_amount", 11, 100.0)),
            watermark_policy=SourceProvidedWatermarks(),
        ),
        "settlements.input": SourceBinding(
            OneBatchSource(_input_table("settled_at", "fee", 12, 2.5)),
            watermark_policy=SourceProvidedWatermarks(),
        ),
    }
    with TemporaryDirectory(prefix="calc-flow-relational-dag-") as directory:
        job = await StreamingRunner(
            plan,
            sources,
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(directory),
        ).start_async()
        outcome = await job.wait_async()
        if outcome.state != "completed":
            raise RuntimeError(outcome)

    output = pa.concat_tables(sink.tables)
    assert output["net_amount"].to_pylist() == [97.5]
    print(output.select(["matched__authorization__account_id", "net_amount"]))


if __name__ == "__main__":
    asyncio.run(run())
