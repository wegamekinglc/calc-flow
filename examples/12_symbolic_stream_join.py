"""Join two ordered symbolic streams through the native bounded join."""

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


def symbolic_program() -> Program:
    authorizations = table_input(
        "authorizations",
        schema=[
            Field("account_id", "int64", nullable=False),
            Field("authorized_at", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("amount", "float64", nullable=False),
        ],
        entity_by=["account_id"],
        event_time="authorized_at",
        sequence_by=["sequence"],
    )
    payments = table_input(
        "payments",
        schema=[
            Field("account_id", "int64", nullable=False),
            Field("paid_at", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("paid_amount", "float64", nullable=False),
        ],
        entity_by=["account_id"],
        event_time="paid_at",
        sequence_by=["sequence"],
    )
    joined = table.stream_join(
        authorizations,
        payments,
        left_keys=["account_id"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2)),
        limits=JoinStateLimits(10_000, 16 * 1024 * 1024, 100_000),
        left_prefix="authorization",
        right_prefix="payment",
    )
    matched = joined.with_columns(
        FeatureSet(
            [
                (
                    "amount_matches",
                    joined["authorization__amount"] == joined["payment__paid_amount"],
                )
            ]
        )
    )
    return Program(
        "symbolic-stream-join",
        inputs=[authorizations, payments],
        outputs=[("matches", matched)],
    )


class SegmentedSource:
    def __init__(self, value: pa.Table, segment: int) -> None:
        self._value = value
        self._segment = segment
        self._offset = 0
        self._watermark_emitted = False

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0
        self._watermark_emitted = False

    async def next(self) -> Data | Watermark | None:
        if self._offset < self._value.num_rows:
            end = min(self._offset + self._segment, self._value.num_rows)
            value = self._value.slice(self._offset, end - self._offset)
            self._offset = end
            return Data(
                Batch.from_pyarrow(value),
                Cursor(end.to_bytes(8, "big"), {"offset": end}),
            )
        if not self._watermark_emitted:
            self._watermark_emitted = True
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


def input_tables() -> tuple[pa.Table, pa.Table]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    authorization_schema = pa.schema(
        [
            pa.field("account_id", pa.int64(), nullable=False),
            pa.field("authorized_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("amount", pa.float64(), nullable=False),
        ]
    )
    authorizations = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(
                [
                    base + timedelta(seconds=10),
                    base + timedelta(seconds=20),
                    base + timedelta(seconds=30),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([1, 1, 1], type=pa.uint64()),
            pa.array([100.0, 50.0, 25.0], type=pa.float64()),
        ],
        schema=authorization_schema,
    )
    payment_schema = pa.schema(
        [
            pa.field("account_id", pa.int64(), nullable=False),
            pa.field("paid_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("paid_amount", pa.float64(), nullable=False),
        ]
    )
    payments = pa.Table.from_arrays(
        [
            pa.array([1, 2, 1], type=pa.int64()),
            pa.array(
                [
                    base + timedelta(seconds=12),
                    base + timedelta(seconds=16),
                    base + timedelta(seconds=20),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([1, 1, 2], type=pa.uint64()),
            pa.array([100.0, 45.0, 100.0], type=pa.float64()),
        ],
        schema=payment_schema,
    )
    return authorizations, payments


async def run() -> None:
    program = symbolic_program()
    runtime = Runtime()
    analysis = program.analyze(runtime, mode="stream")
    if analysis.issues:
        raise RuntimeError(analysis.issues)
    plan = program.compile_stream(runtime)
    authorizations, payments = input_tables()
    sink = CollectSink()
    with TemporaryDirectory(prefix="calc-flow-symbolic-join-") as directory:
        job = await StreamingRunner(
            plan,
            {
                "left": SourceBinding(
                    SegmentedSource(authorizations, 1),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "right": SourceBinding(
                    SegmentedSource(payments, 2),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(directory),
        ).start_async()
        outcome = await job.wait_async()
        if outcome.state != "completed":
            raise RuntimeError(outcome)

    output = pa.concat_tables(sink.tables).sort_by("authorization__account_id")
    print(output.select(["authorization__account_id", "amount_matches"]))


if __name__ == "__main__":
    asyncio.run(run())
