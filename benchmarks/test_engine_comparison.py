from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks import engine_comparison
from benchmarks.engine_comparison import (
    EngineCase,
    expected_output,
    sql_query,
    workload,
)
from scripts.benchmark_suite.catalog import STREAM_CASES, engine_cases
from scripts.benchmark_suite.process import child_environment


def test_sql_queries_reject_unknown_scenario_names():
    with pytest.raises(ValueError, match="unsupported SQL benchmark scenario"):
        sql_query("sma20; DROP TABLE input")


def test_polars_samples_collect_through_the_streaming_engine(monkeypatch):
    recorded: dict[str, object] = {}

    class RecordingResult:
        def to_arrow(self):
            return pa.table({"value": [1.0]})

    class RecordingPlan:
        def collect(self, **kwargs):
            recorded.update(kwargs)
            return RecordingResult()

    monkeypatch.setattr(
        engine_comparison, "_polars_plan", lambda data, scenario: RecordingPlan()
    )
    result = engine_comparison._polars(workload(101), "projection")()
    assert recorded == {"engine": "streaming"}
    assert result == pa.table({"value": [1.0]})


def test_polars_asof_case_matches_the_shared_oracle(tmp_path):
    case = next(
        case
        for case in engine_cases(101)
        if case["backend"] == "polars" and case["scenario"] == "asof_join"
    )
    runner = EngineCase(case, tmp_path)
    try:
        sample = runner.sample()
        assert sample["correctness"]["passed"]
        assert sample["correctness"]["rows"] == 101
    finally:
        runner.close()


def _polars_reference_samples(connection, root, rows, environment):
    os.environ.update(environment)
    import polars as pl

    samples = []
    for case in engine_cases(rows):
        if case["backend"] != "polars-1t":
            continue
        runner = EngineCase(case, root)
        try:
            samples.append(runner.sample()["correctness"]["passed"])
        finally:
            runner.close()
    connection.send(
        {"pid": os.getpid(), "threads": pl.thread_pool_size(), "samples": samples}
    )
    connection.close()


def _fresh_polars_samples(root, rows, site, source):
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_polars_reference_samples,
        args=(
            sending,
            root,
            rows,
            child_environment(site, source=source, polars_threads=1),
        ),
    )
    try:
        process.start()
        sending.close()
        process.join(timeout=30)
        if process.is_alive():
            raise TimeoutError("single-thread Polars reference worker timed out")
        if process.exitcode != 0 or not receiving.poll():
            raise RuntimeError(f"Polars reference worker exited {process.exitcode}")
        return receiving.recv()
    finally:
        sending.close()
        if process.is_alive():
            process.kill()
            process.join()
        receiving.close()
        process.close()


@pytest.mark.parametrize("rows", (10, 101))
def test_polars_single_thread_reference_matches_oracles_in_a_fresh_process(
    tmp_path, rows
):
    import calc_flow

    source = Path(__file__).resolve().parents[1]
    site = Path(calc_flow.__file__).resolve().parents[1]
    result = _fresh_polars_samples(tmp_path, rows, site, source)
    assert result["pid"] != os.getpid()
    assert result["threads"] == 1
    assert result["samples"] == [True] * 8


def test_single_thread_reference_rejects_a_process_with_a_larger_pool(
    monkeypatch, tmp_path
):
    import polars as pl

    monkeypatch.setattr(pl, "thread_pool_size", lambda: 32)
    case = next(case for case in engine_cases(101) if case["backend"] == "polars-1t")
    with pytest.raises(ValueError, match="single-thread.*one thread"):
        EngineCase(case, tmp_path)


@pytest.mark.parametrize("scenario", STREAM_CASES)
def test_ready_stream_repeated_samples_use_fresh_execution_plans(scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(10)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        for _ in range(3):
            assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


@pytest.mark.parametrize("scenario", ("join", "asof_join"))
def test_join_streams_complete_many_chunks_with_bounded_state(scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(320_000)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


def test_static_join_waits_for_delayed_dimension_progress(monkeypatch, tmp_path):
    from benchmarks import engine_stream
    from calc_flow import Data, Watermark

    class DelayedDimensionSource(engine_stream._ReadySource):
        static = False

        async def next(self):
            event = await super().next()
            if isinstance(event, Data):
                self.static = "factor" in event.batch.to_pyarrow().column_names
            elif self.static and isinstance(event, Watermark):
                await asyncio.sleep(2)
            return event

    monkeypatch.setattr(engine_stream, "_ReadySource", DelayedDimensionSource)
    case = next(
        case
        for case in engine_cases(320_000)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == "join"
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


def _static_join_status_measurement(left_counter, monkeypatch):
    from benchmarks import engine_stream
    from calc_flow import Batch, Watermark

    table = pa.table({"value": [1.0]})
    sink = engine_stream._CollectSink(1)
    sink.opened.set()
    clock_reads = []

    def clock():
        clock_reads.append(len(clock_reads))
        return clock_reads[-1] * 1_000_000

    monkeypatch.setattr(engine_stream.time, "perf_counter_ns", clock)

    class Source:
        def __init__(self, name):
            self.name = name
            self.ready = asyncio.Event()
            self.opened = asyncio.Event()
            self.ready.set()
            self.opened.set()

        async def push(self, _event):
            if self.name == "left":
                await sink.write(Batch.from_pyarrow(table))

    class Job:
        reads = 0

        def status(self):
            self.reads += 1
            quoted = self.reads >= 3
            if self.reads > 1:
                assert len(clock_reads) == 2, "status proof must follow timer stop"
            left = {"retained_rows": 0, "evicted_rows": 0}
            if quoted and left_counter:
                left[left_counter] = 1
            return {
                "stream_joins": {
                    "join": {
                        "left": left,
                        "right": {"watermark_micros": engine_stream.BASE_MICROS},
                        "emitted_match_rows": int(quoted),
                    }
                }
            }

    job = Job()
    sources = {name: Source(name) for name in ("left", "right")}
    streams = {
        "left": (object(),),
        "right": (object(), Watermark(engine_stream.BASE)),
    }
    result = asyncio.run(
        engine_stream._measure_ready(sources, sink, streams, job, static_join=True)
    )
    return result, job.reads


@pytest.mark.parametrize("left_counter", ("retained_rows", "evicted_rows"))
def test_static_join_rejects_post_quote_state_after_a_stale_snapshot(
    left_counter, monkeypatch
):
    with pytest.raises(RuntimeError, match="retained or evicted quote rows"):
        _static_join_status_measurement(left_counter, monkeypatch)


def test_static_join_accepts_the_causal_post_quote_status_outside_timing(monkeypatch):
    (table, seconds), reads = _static_join_status_measurement(None, monkeypatch)
    assert table == pa.table({"value": [1.0]})
    assert seconds == 0.001
    assert reads == 3


@pytest.mark.parametrize("binding", ("reference.input", "quotes.input"))
def test_asof_waits_for_delayed_chunk_watermarks(binding, monkeypatch, tmp_path):
    from benchmarks import engine_stream
    from calc_flow import Watermark

    names = iter(("reference.input", "quotes.input"))

    class DelayedWatermarkSource(engine_stream._ReadySource):
        def __init__(self):
            super().__init__()
            self.delay = next(names) == binding

        async def next(self):
            event = await super().next()
            if self.delay and isinstance(event, Watermark):
                self.delay = False
                await asyncio.sleep(2)
            return event

    monkeypatch.setattr(engine_stream, "_ReadySource", DelayedWatermarkSource)
    case = next(
        case
        for case in engine_cases(320_000)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == "asof_join"
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


def test_asof_small_batches_complete_without_reading_job_status(monkeypatch, tmp_path):
    from calc_flow import StreamingJob

    def status(_self):
        raise AssertionError("ASOF progress must await sink delivery")

    monkeypatch.setattr(StreamingJob, "status", status)
    case = next(
        case
        for case in engine_cases(4_097)
        if case["backend"] == "calc-flow-stream"
        and case["scenario"] == "asof_join"
        and case.get("batch_rows") == 1024
        and case.get("checkpoint_interval_millis") is None
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


@pytest.mark.parametrize("count", [64_001, 128_000])
@pytest.mark.parametrize(
    "scenario",
    tuple(
        scenario
        for scenario in STREAM_CASES
        if scenario not in ("window_sum", "asof_join")
    ),
)
def test_ready_stream_finalizes_every_chunk_before_eof(count, scenario, tmp_path):
    # Boundary sizes are correctness fixtures, not suite tiers: the catalog's
    # stream-join evidence cap must not drop the chunk-finalization coverage,
    # so the case is built here instead of looked up from engine_cases.
    case = {
        "id": f"engines/{count}/calc-flow-stream/{scenario}",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": scenario,
        "rows": count,
        "scope": "ready-enqueue-to-arrow",
    }
    runner = EngineCase(case, tmp_path)
    try:
        for _ in range(2):
            sample = runner.sample()
            assert sample["seconds"] > 0
            # The stream output cardinality matches the scenario oracle: one
            # row per input for row-local scenarios, one row per group for
            # group_by, and the filtered count for filter.
            assert sample["correctness"]["rows"] == runner.expected.num_rows
            assert sample["correctness"]["finite_rows"] > 0
    finally:
        runner.close()


def test_performance_prices_are_exact_eighths_with_bounded_magnitude():
    prices = workload(1_001).table["price"].to_numpy()
    assert np.all(prices * 8 == np.floor(prices * 8))
    assert np.all((prices >= 64) & (prices < 256))


@pytest.mark.parametrize("window", [64, 256])
def test_argmax_oracle_covers_full_periodic_windows(window):
    data = workload(20_000)
    actual = expected_output(data, f"argmax{window}")["value"].to_numpy()
    prices = data.table["price"].to_numpy()
    for entity in range(data.entities):
        series = prices[entity :: data.entities]
        for tick in (window - 1, window, 256, 257, 300):
            index = entity + tick * data.entities
            if index >= len(prices):
                continue
            trailing = series[max(0, tick - window + 1) : tick + 1]
            expected_age = len(trailing) - 1 - np.argmax(trailing)
            assert actual[index] == expected_age


@pytest.mark.parametrize(
    "scenario",
    (
        "average",
        "argmax64",
        "argmax256",
        "unique64",
        "cs_mean",
        "window_sum",
        "asof_join",
    ),
)
def test_new_stream_operators_match_independent_oracle(scenario, tmp_path):
    case = next(
        case
        for case in engine_cases(101)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    try:
        assert runner.sample()["correctness"]["passed"]
    finally:
        runner.close()


@pytest.mark.parametrize(
    "case",
    [
        case
        for rows in (10, 101)
        for case in engine_cases(rows)
        if case["backend"] != "polars-1t"
    ],
    ids=lambda case: case["id"],
)
def test_engine_outputs_match_independent_oracle(case: dict, tmp_path: Path):
    runner = EngineCase(case, tmp_path)
    try:
        result = runner.sample()
        assert result["seconds"] > 0
        assert result["correctness"]["passed"]
        if case["rows"] >= 20 or case["scenario"] not in ("sma20", "dual_sma"):
            assert result["correctness"]["finite_rows"] > 0
        else:
            assert result["correctness"]["finite_rows"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize("count", [10, 19, 20, 21, 41, 1_001])
def test_full_window_oracle_matches_direct_slices(count: int):
    data = workload(count)
    for scenario in ("sma20", "dual_sma"):
        expected = expected_output(data, scenario)
        prices = data.table["price"].to_numpy()
        direct = np.full(count, np.nan)
        for index in range(count):
            entity = index % data.entities
            history = prices[entity : index + 1 : data.entities]
            if len(history) >= 20:
                direct[index] = np.mean(history[-20:])
                if scenario == "dual_sma":
                    direct[index] = np.mean(history[-5:]) - direct[index]
        np.testing.assert_allclose(expected["value"].to_numpy(), direct, equal_nan=True)


def test_corrupted_output_is_rejected(tmp_path: Path):
    case = next(c for c in engine_cases(101) if c["backend"] == "ta-lib")
    runner = EngineCase(case, tmp_path)
    try:
        index = runner.expected.column_names.index("value")
        bad = runner.expected.set_column(
            index, "value", pa.array(np.zeros(runner.expected.num_rows))
        )
        with pytest.raises(AssertionError):
            runner.validate(bad)
    finally:
        runner.close()
