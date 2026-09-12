from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pyarrow as pa

import calc_flow
from calc_flow.runtime import JobStatus


def test_rolling_metrics_types_export_fixed_callback_units() -> None:
    for name in ("RollingMetrics", "RollingCallbackMetrics"):
        assert name in calc_flow.__all__
    callback = get_type_hints(calc_flow.RollingCallbackMetrics)
    assert callback["callback_duration_ns"] is int
    assert callback["send_wait_duration_ns"] is int
    assert callback["interrupted"] is int
    assert callback["output_rows_prepared"] is int
    assert "processing_duration_micros" not in callback
    metrics = get_type_hints(calc_flow.RollingMetrics)
    assert set(metrics) == {"data", "watermark", "end", "overflowed"}
    assert metrics["data"] is calc_flow.RollingCallbackMetrics
    assert metrics["overflowed"] is bool
    assert (
        get_type_hints(JobStatus)["rolling_metrics"]
        == dict[str, calc_flow.RollingMetrics]
    )


def test_managed_rolling_metrics_publish_eof_and_detached_status(
    tmp_path: Path,
) -> None:
    from calc_flow import (
        Batch,
        Cursor,
        Data,
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
    from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("price", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    result = quotes.with_columns(
        FeatureSet((("mean", ts.mean(quotes["price"], window=rows(2), min_periods=2)),))
    )
    plan = Program(
        "rolling-diagnostics", inputs=(quotes,), outputs=(("result", result),)
    ).compile_stream(Runtime())
    table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.timestamp("us", "UTC")),
            pa.array(["a", "a", "a"]),
            pa.array([1, 2, 3], type=pa.uint64()),
            pa.array([1.0, 2.0, 3.0]),
        ],
        schema=pa.schema(
            [
                pa.field("event_time", pa.timestamp("us", "UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("sequence", pa.uint64(), nullable=False),
                pa.field("price", pa.float64(), nullable=False),
            ]
        ),
    )
    events = (
        Data(Batch.from_pyarrow(table), Cursor(b"3", {"rows": 3})),
        Watermark(datetime(1970, 1, 1, microsecond=4, tzinfo=UTC)),
        None,
    )
    outputs: list[pa.Table] = []

    class Source:
        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=3,
                max_batch_bytes=4096,
                schema=table.schema,
                native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.events = iter(events)

        async def next(self) -> Data | Watermark | None:
            return next(self.events)

        async def close(self) -> None:
            return None

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            outputs.append(batch.to_pyarrow())

        async def close(self) -> None:
            return None

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                plan.source_binding_ids[0]: SourceBinding(
                    Source(), watermark_policy=SourceProvidedWatermarks()
                )
            },
            {plan.sink_binding_ids[0]: [SinkBinding.ordinary("collector", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        try:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=10)
            assert outcome.state == "completed"
            status = job.status()
            metrics = status["rolling_metrics"]
            assert len(metrics) == 1
            node_id, observation = next(iter(metrics.items()))
            assert not observation["overflowed"]
            assert observation["data"]["input_rows"] == 3
            assert observation["data"]["succeeded"] == 1
            assert observation["watermark"]["succeeded"] == 1
            assert observation["end"]["succeeded"] == 1
            for callback in ("data", "watermark", "end"):
                values = observation[callback]
                assert values["started"] == sum(
                    values[name]
                    for name in ("succeeded", "failed", "cancelled", "interrupted")
                )
                assert values["callback_duration_ns"] == sum(
                    value
                    for name, value in values.items()
                    if name.endswith("_duration_ns") and name != "callback_duration_ns"
                )
            observation["data"]["input_rows"] = 999
            assert job.status()["rolling_metrics"][node_id]["data"]["input_rows"] == 3
            assert pa.concat_tables(outputs)["mean"].to_pylist() == [None, 1.5, 2.5]
        finally:
            await job.cancel_async()

    asyncio.run(exercise())
