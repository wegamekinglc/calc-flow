from __future__ import annotations

import asyncio
from datetime import timedelta

import pyarrow as pa
import pytest

from benchmarks import engine_stream
from benchmarks.engine_comparison import EngineCase, expected_output, workload
from benchmarks.warm_stream import BASE
from calc_flow import Data, ReplayPositioning, SourceDeliveryCapability, Watermark
from scripts.benchmark_suite.catalog import engine_cases, stream_variant_cases


def test_interval_workload_has_64_keys_and_inclusive_five_second_oracle():
    data = workload(768, scenario="interval_join")
    assert data.entities == 64
    assert len(set(data.table["symbol"].to_pylist())) == 64
    result = expected_output(data, "interval_join")
    pairs = set(
        zip(result["sequence"].to_pylist(), result["right_sequence"].to_pylist())
    )
    assert (0, 5 * 64) in pairs
    assert (0, 6 * 64) not in pairs
    assert (5 * 64, 0) in pairs
    assert (6 * 64, 0) not in pairs
    assert result.num_rows == 64 * sum(
        min(11, 12 - max(0, tick - 5), tick + 6) for tick in range(12)
    )


@pytest.mark.parametrize(
    "backend", ("calc-flow-stream", "calc-flow-sql", "datafusion", "polars")
)
def test_interval_references_match_shared_oracle(tmp_path, backend):
    case = next(
        case
        for case in engine_cases(768)
        if case["backend"] == backend and case["scenario"] == "interval_join"
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


def test_interval_stream_accepts_on_time_out_of_order_rows(tmp_path):
    from dataclasses import replace

    from scripts.benchmark_suite.catalog import BATCH_ROWS

    async def exercise():
        data = workload(768, scenario="interval_join")
        order = [
            row
            for tick in (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
            for row in range(tick * 64, (tick + 1) * 64)
        ]
        table = data.table.take(pa.array(order))
        data = replace(data, table=table)
        events = engine_stream.stream_events(table, 64, batch_rows=BATCH_ROWS)
        assert events[1] == Watermark(BASE + timedelta(seconds=11, microseconds=1))
        result, _ = await engine_stream.run_stream(
            engine_stream.stream_plan("interval_join", table),
            {"left": events, "right": events},
            tmp_path,
            expected_output(data, "interval_join").num_rows,
        )
        pairs = [("sequence", "ascending"), ("right_sequence", "ascending")]
        actual = result.sort_by(pairs)
        expected = expected_output(data, "interval_join").sort_by(pairs)
        assert actual.column_names == expected.column_names
        assert all(
            actual[name].equals(expected[name]) for name in expected.column_names
        )

    asyncio.run(exercise())


def test_interval_evidence_waits_for_both_eviction_frontiers_outside_timing(
    monkeypatch, tmp_path
):
    class DelayedWatermarks(engine_stream._ReplaySource):
        async def next(self):
            event = await super().next()
            if isinstance(event, Watermark):
                await asyncio.sleep(0.5)
            return event

    monkeypatch.setattr(engine_stream, "_ReplaySource", DelayedWatermarks)
    case = next(
        case
        for case in engine_cases(768)
        if case["backend"] == "calc-flow-stream"
        and case["scenario"] == "interval_join"
        and not case.get("variant")
    )
    runner = EngineCase(case, tmp_path)
    try:
        sample = runner.sample()
        assert sample["correctness"]["passed"]
        statuses = tuple(sample["stream_evidence"]["interval_join"].values())
        assert len(statuses) == 1
        for side in ("left", "right"):
            assert statuses[0][side]["retained_rows"] > 0
            assert statuses[0][side]["evicted_rows"] > 0
    finally:
        runner.close()


def test_replay_source_seeks_exact_next_data_and_replays_equal_watermark():
    async def exercise():
        data = workload(20)
        events = engine_stream.stream_events(data.table, data.entities, batch_rows=10)
        source = engine_stream._ReplaySource(events, batch_rows=10)
        capabilities = source.capabilities()
        assert (
            capabilities.replay_positioning
            == ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK
        )
        assert capabilities.delivery == SourceDeliveryCapability.LOSSLESS
        await source.open(None)
        await source.push(events[0])
        first = await source.next()
        assert isinstance(first, Data)
        await source.push(events[1])
        watermark = await source.next()
        assert isinstance(watermark, Watermark)
        resumed = engine_stream._ReplaySource(events, batch_rows=10)
        await resumed.open(first.cursor)
        await resumed.push(events[1])
        assert await resumed.next() == watermark
        await resumed.push(events[2])
        second = await resumed.next()
        assert second.batch.to_pyarrow().equals(data.table.slice(10, 10))
        assert second.cursor.order > first.cursor.order
        assert resumed.position == 3

    asyncio.run(exercise())


def test_replay_source_rejects_unknown_or_tampered_positions():
    async def exercise():
        from calc_flow import Cursor

        events = engine_stream.stream_events(workload(20).table, 1, batch_rows=10)
        for cursor in (
            Cursor(b"unknown", {"event_index": 1}),
            Cursor((3).to_bytes(8, "big"), {"event_index": 1}),
        ):
            source = engine_stream._ReplaySource(events, batch_rows=10)
            with pytest.raises(ValueError, match="cursor"):
                await source.open(cursor)

    asyncio.run(exercise())


def test_replay_source_can_reopen_at_origin():
    async def exercise():
        events = engine_stream.stream_events(workload(20).table, 1, batch_rows=10)
        source = engine_stream._ReplaySource(events, batch_rows=10)
        await source.open(None)
        for event in events[:2]:
            await source.push(event)
            await source.next()
        await source.close()
        await source.open(None)
        assert source.position == 0
        await source.push(events[0])
        assert await source.next() == events[0]

    asyncio.run(exercise())


def test_checkpoint_cut_cancels_pending_work_on_runtime_failure():
    from types import SimpleNamespace

    async def exercise():
        cancelled = asyncio.Event()

        async def work():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def failed():
            return SimpleNamespace(
                state="failed", errors=("injected checkpoint failure",)
            )

        with pytest.raises(RuntimeError, match="injected checkpoint failure"):
            await asyncio.wait_for(engine_stream._with_running_job(work(), failed()), 1)
        assert cancelled.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "scenario", ("projection", "join", "interval_join", "asof_join")
)
@pytest.mark.parametrize("rows", (10, 4097))
def test_checkpoint_duration_variants_publish_nonterminal_epoch(
    tmp_path, scenario, rows
):
    case = next(
        case
        for case in stream_variant_cases(rows, checkpoint_duration=True)
        if case["scenario"] == scenario
        and case.get("checkpoint_interval_millis") == 100
        and case.get("batch_rows") == 1024
    )
    runner = EngineCase(case, tmp_path)
    try:
        sample = runner.sample()
        assert sample["correctness"]["passed"]
        dimensions = sample["stream_evidence"]
        assert dimensions["batch_rows"] == 1024
        assert dimensions["checkpoint_interval_millis"] == 100
        assert dimensions["replay_mode"] == "exact-cursor"
        assert dimensions["workload"] == "checkpoint-duration"
        assert dimensions["nonterminal_epochs"]
        assert dimensions["rows_before_checkpoint"] < case["rows"]
        assert sample["seconds"] >= 0.1
        if scenario == "join":
            assert dimensions["lookup_join"]["left"]["retained_rows"] == 0
            assert dimensions["lookup_join"]["left"]["evicted_rows"] == 0
    finally:
        runner.close()


def test_interval_stream_100k_rows_matches_oracle_and_bounded_retained_state(tmp_path):
    case = next(
        case
        for case in engine_cases(100_000)
        if case["backend"] == "calc-flow-stream"
        and case["scenario"] == "interval_join"
        and not case.get("variant")
    )
    runner = EngineCase(case, tmp_path)
    try:
        sample = runner.sample()
        assert sample["correctness"]["passed"]
        assert sample["correctness"]["rows"] > 1_000_000
        status = next(iter(sample["stream_evidence"]["interval_join"].values()))
        for side in ("left", "right"):
            assert 0 < status[side]["retained_rows"] <= 2 * case["batch_rows"] + 768
            assert status[side]["evicted_rows"] > 0
    finally:
        runner.close()


def test_checkpoint_ack_remains_valid_after_a_later_epoch_completes(
    monkeypatch, tmp_path
):
    from calc_flow import StreamingJob

    original = StreamingJob.status

    def later_epoch(job):
        status = original(job)
        checkpoint = status["checkpoint"]
        epoch = checkpoint["last_completed_epoch"]
        return {
            **status,
            "checkpoint": {
                **checkpoint,
                "last_completed_epoch": epoch + 1 if epoch is not None else None,
            },
        }

    monkeypatch.setattr(StreamingJob, "status", later_epoch)
    case = next(
        case
        for case in stream_variant_cases(10, checkpoint_duration=True)
        if case["scenario"] == "projection"
        and case.get("checkpoint_interval_millis") == 100
        and case["batch_rows"] == 1024
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()
