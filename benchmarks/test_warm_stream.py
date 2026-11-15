"""Correctness and accounting guards for warm StreamingRunner measurements."""

from __future__ import annotations

import asyncio
import collections
import json
import runpy
import sys
import threading
import time
import zoneinfo
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks import warm_stream as warm
from benchmarks.warm_stream import (
    BASE,
    SCHEMA,
    ScenarioConfig,
    WarmScenario,
    WarmSessionScenario,
    _expected,
    _input_table,
    _native_method_matcher,
    _NativeCallProbe,
    _prepared_events,
    _probe_overhead_seconds,
    _segment,
    _summary,
    _validate_output,
    _warm_program,
    ta_lib_reference,
)
from calc_flow.symbolic import Program


def test_warm_shared_helpers_do_not_require_optional_talib(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "talib", None)
    helpers = runpy.run_path(
        str(Path(__file__).with_name("rolling_indicator_comparison.py"))
    )
    values = helpers["expected_rolling_mean"](
        np.arange(64, dtype=np.float64), entities=64, window=20
    )
    np.testing.assert_array_equal(values, np.arange(64, dtype=np.float64))


def test_warm_oracle_uses_only_window_context() -> None:
    for indicator in ("rolling_mean", "dual_sma_spread"):
        values = _expected(
            ScenarioConfig(indicator=indicator),
            start=1_024_000,
            rows=64,
        )
        assert values.shape == (64,)
        assert np.isfinite(values).all()


def test_warm_summary_reports_tail_latency_and_original_samples() -> None:
    samples = [0.001, 0.002, 0.003, 0.004]
    summary = _summary(samples, 64)
    assert summary["median_seconds"] == 0.0025
    assert summary["p95_seconds"] == pytest.approx(0.00385)
    assert summary["samples_seconds"] == samples


def test_warm_validation_rejects_changed_identity_and_values() -> None:
    config = ScenarioConfig()
    expected = _expected(
        config,
        start=1_024,
        rows=64,
    )
    table = _segment(1_024, 64, 64).append_column("moving_average", pa.array(expected))
    kwargs = dict(
        config=config,
        start=1_024,
        rows=64,
    )
    assert _validate_output(table, **kwargs)["passed"]
    broken = table.set_column(1, "sequence", pa.array(np.zeros(64, dtype=np.uint64)))
    with pytest.raises(RuntimeError, match="sequence"):
        _validate_output(broken, **kwargs)
    broken = table.set_column(4, "moving_average", pa.array(expected + 1))
    with pytest.raises(RuntimeError, match="correctness"):
        _validate_output(broken, **kwargs)


@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
def test_warm_scenario_advances_one_runner_and_validates_every_append(
    tmp_path, indicator
) -> None:
    async def exercise() -> None:
        scenario = await WarmScenario.start(
            ScenarioConfig(history_rows=128, append_rows=64, indicator=indicator),
            tmp_path,
        )
        try:
            first = await scenario.sample(collect_gc=False)
            second = await scenario.sample(collect_gc=True)
            assert first["start_row"] == 128
            assert second["start_row"] == 192
            assert first["correctness"]["passed"]
            assert second["correctness"]["passed"]
            assert first["seconds"] > 0
            assert first["phases_seconds"]["to_pyarrow"] >= 0
        finally:
            assert (await scenario.finish())["state"] == "completed"

    asyncio.run(exercise())


def test_exact_projection_warm_region_preserves_stream_results(tmp_path) -> None:
    async def exercise() -> None:
        config = ScenarioConfig(
            history_rows=128,
            append_rows=64,
            indicator="dual_sma_spread",
            exact_projection=True,
        )
        document = _warm_program(config).to_project().root
        kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
        assert "expression" in kinds
        scenario = await WarmScenario.start(config, tmp_path)
        try:
            sample = await scenario.sample(collect_gc=False)
            assert sample["correctness"]["passed"]
            assert sample["phases_seconds"]["to_pyarrow"] >= 0
        finally:
            assert (await scenario.finish())["state"] == "completed"

    asyncio.run(exercise())


def test_warm_configuration_rejects_incomplete_ticks_and_invalid_sizes() -> None:
    for kwargs in ({"append_rows": 65}, {"history_rows": 0}, {"window": 0}):
        with pytest.raises(ValueError):
            ScenarioConfig(**kwargs)


@pytest.mark.parametrize("append_rows", (1, 4, 16, 64))
@pytest.mark.parametrize("append_entities", (1, 4, 16, 64))
@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
def test_sparse_appends_keep_all_history_entities_and_finalize_each_batch(
    tmp_path, append_rows, append_entities, indicator
) -> None:
    async def exercise() -> None:
        config = ScenarioConfig(
            history_rows=128,
            append_rows=append_rows,
            append_entities=append_entities,
            indicator=indicator,
        )
        scenario = await WarmScenario.start(config, tmp_path)
        try:
            for index in range(3):
                sample = await scenario.sample(collect_gc=False)
                assert sample["start_row"] == 128 + index * append_rows
                assert sample["correctness"]["passed"]
            status = scenario.job.status()
            assert all(node["late_rows"] == 0 for node in status["operators"].values())
        finally:
            assert (await scenario.finish())["state"] == "completed"

    asyncio.run(exercise())


def test_sparse_configuration_rejects_invalid_active_entity_counts() -> None:
    for count in (0, -1, 65, 1.5, True):
        with pytest.raises(ValueError, match="append_entities"):
            ScenarioConfig(append_rows=1, append_entities=count)


def test_prepared_watermarks_do_not_need_a_timezone_database(monkeypatch) -> None:
    def missing_timezone(_name):
        raise zoneinfo.ZoneInfoNotFoundError("no timezone database")

    monkeypatch.setattr(zoneinfo, "ZoneInfo", missing_timezone)
    for active, rows in ((None, 64), (1, 1), (64, 4)):
        config = ScenarioConfig(
            history_rows=128, append_entities=active, append_rows=rows
        )
        data, watermark = _prepared_events(config, 128, rows)
        assert data.batch.num_rows == rows
        seconds = (128 + rows - 1) // 64 if active is None else 128 + rows - 1
        assert watermark.at == BASE + timedelta(seconds=seconds)


@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
@pytest.mark.parametrize("active", (1, 4, 16, 64))
def test_sparse_oracle_matches_full_per_entity_history(indicator, active) -> None:
    config = ScenarioConfig(
        history_rows=128, append_rows=1, append_entities=active, indicator=indicator
    )
    total = config.history_rows + 32 * active + 7
    table = _input_table(config, 0, total).select(["symbol", "price"]).to_pydict()
    histories: dict[str, list[float]] = {}
    expected = []
    for symbol, price in zip(table["symbol"], table["price"], strict=True):
        values = histories.setdefault(symbol, [])
        values.append(price)
        slow = sum(values[-config.window :]) / len(values[-config.window :])
        fast = sum(values[-config.fast_window :]) / len(values[-config.fast_window :])
        expected.append(fast - slow if indicator == "dual_sma_spread" else slow)
    for start in (128, 129, 128 + 19 * active, 128 + 31 * active):
        rows = min(17, total - start)
        np.testing.assert_allclose(
            _expected(config, start=start, rows=rows),
            expected[start : start + rows],
            rtol=1e-10,
            atol=1e-10,
        )


def test_callback_diagnostics_are_opt_in_and_survive_terminal_cleanup(tmp_path) -> None:
    async def exercise() -> None:
        scenario = await WarmScenario.start(ScenarioConfig(history_rows=128), tmp_path)
        native = scenario.job._inner
        try:
            assert json.loads(native._take_callback_profile()) == {
                "records": [],
                "dropped": 0,
            }
            native._enable_callback_profiling()
            native._enable_callback_profiling()
            await scenario.sample(collect_gc=False)
        finally:
            assert (await scenario.finish())["state"] == "completed"
        profile = json.loads(native._take_callback_profile())
        assert profile["dropped"] == 0
        records = profile["records"]
        assert {record["callback"] for record in records} >= {
            "source.next",
            "_native_write",
        }
        for record in records:
            if record["outcome"] == "completed":
                assert 0 < record["attached_ns"] <= record["queued_ns"]
                assert record["queued_ns"] <= record["dispatched_ns"]
                assert (
                    record["dispatched_ns"]
                    <= record["completed_ns"]
                    <= record["elapsed_ns"]
                )
        assert json.loads(native._take_callback_profile())["records"] == []

    asyncio.run(exercise())


def _probed_push(native: collections.deque, value: int) -> dict[str, int]:
    prepared = list([value])
    native.append(sum(prepared))
    return {"rows": len(native)}


def _bound_push(native: collections.deque, value: int) -> int:
    append = native.append
    append(value)
    return len(native)


def _double_push(native: collections.deque, value: int) -> None:
    native.append(value)
    native.append(value)


def _probe_sample(probe: _NativeCallProbe, function, native) -> dict | str:
    probe.reset()
    started = time.perf_counter_ns()
    function(native, 1)
    return probe.phases(started, time.perf_counter_ns())


def test_native_call_probe_splits_samples_at_the_matching_native_call() -> None:
    codes = (_probed_push.__code__, _bound_push.__code__, _double_push.__code__)
    probe = _NativeCallProbe(codes, _native_method_matcher(collections.deque, "append"))
    assert probe.install() is None
    native: collections.deque = collections.deque()
    try:
        for function in (_probed_push, _bound_push):
            for _ in range(3):
                phases = _probe_sample(probe, function, native)
                assert set(phases) == {
                    "argument_handling",
                    "native_call",
                    "result_assembly",
                }
                assert all(value >= 0 for value in phases.values())
        assert _probe_sample(probe, _double_push, native) == (
            "observed 2 matching native calls"
        )
        assert _probe_sample(probe, lambda native, value: None, native) == (
            "observed 0 matching native calls"
        )
    finally:
        probe.close()
    probe.reset()
    _probed_push(native, 1)
    assert probe.calls == 0
    overhead = _probe_overhead_seconds(calls=200, rounds=2)
    assert overhead is not None and 0 <= overhead < 1e-3


def test_native_call_probe_reports_an_occupied_monitoring_tool() -> None:
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, "occupied")
    try:
        probe = _NativeCallProbe((), lambda callable_, arg0: False)
        assert "already in use" in probe.install()
    finally:
        sys.monitoring.free_tool_id(tool)


class _OracleSession:
    """Release rows at explicit watermarks with oracle values, like a session."""

    def __init__(self, config: ScenarioConfig, column: str) -> None:
        self._config = config
        self._column = column
        self._pending: list[pa.Table] = []
        self.emitted = 0
        self.closed = False
        self.calls: list[tuple[str, int, object]] = []

    def _record(self, operation: str, watermark: object = None) -> None:
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        self.calls.append((operation, threading.get_ident(), watermark))

    def _empty(self) -> pa.Table:
        schema = SCHEMA.append(pa.field(self._column, pa.float64()))
        return schema.empty_table()

    def push(self, inputs, *, watermark=None) -> dict[str, pa.Table]:
        self._record("push", watermark)
        (batch,) = inputs.values()
        assert list(inputs) == ["quotes"]
        self._pending.append(batch.to_pyarrow())
        if watermark is None:
            return {"indicators": self._empty()}
        released = pa.concat_tables(self._pending)
        self._pending = []
        values = _expected(self._config, start=self.emitted, rows=released.num_rows)
        self.emitted += released.num_rows
        return {"indicators": released.append_column(self._column, pa.array(values))}

    def finish(self) -> dict[str, pa.Table]:
        self._record("finish")
        return {"indicators": self._empty()}

    def close(self) -> None:
        self._record("close")
        self.closed = True


class _SessionProgram:
    def __init__(self, program: Program, session: _OracleSession) -> None:
        self._program = program
        self._session = session
        self.session_arguments: dict = {}

    def __getattr__(self, name: str):
        return getattr(self._program, name)

    def session(self, **arguments) -> _OracleSession:
        self.session_arguments = arguments
        return self._session


def test_session_driver_warms_samples_and_finishes_on_one_session_thread(
    monkeypatch,
) -> None:
    config = ScenarioConfig(
        history_rows=128,
        append_rows=64,
        history_segment_rows=32,
        indicator="dual_sma_spread",
        exact_projection=True,
    )
    session = _OracleSession(config, "dual_sma_spread")
    program = _SessionProgram(_warm_program(config), session)
    monkeypatch.setattr(warm, "_warm_program", lambda _config: program)

    async def exercise() -> None:
        scenario = await WarmSessionScenario.start(config)
        assert scenario.position == 128
        assert scenario.plan_identity["preflight"]["exact_projection"] is True
        assert scenario.phase_probe["installed"] is False
        first = await scenario.sample(collect_gc=True)
        second = await scenario.sample(collect_gc=False)
        assert [first["start_row"], second["start_row"]] == [128, 192]
        for sample in (first, second):
            assert sample["correctness"]["passed"]
            assert sample["seconds"] > 0
            assert sample["phases_seconds"] == {}
            assert sample["phase_attribution"].startswith("unavailable: ")
        terminal = await scenario.finish()
        assert terminal["state"] == "completed"
        assert terminal["finish_rows"] == {"indicators": 0}
        await scenario.cancel()

    asyncio.run(exercise())
    arguments = program.session_arguments
    assert set(arguments["watermarks"]) == {"quotes"}
    assert arguments["config"].checkpointing is False
    assert arguments["config"].edge_budget.max_rows == 64
    pushes = [watermark for operation, _, watermark in session.calls[:4]]
    # Sub-tick history segments carry a watermark only after a complete tick.
    assert [watermark is None for watermark in pushes] == [True, False] * 2
    assert [operation for operation, _, _ in session.calls[4:]] == [
        "push",
        "push",
        "finish",
        "close",
    ]
    assert len({thread for _, thread, _ in session.calls}) == 1
    assert session.calls[0][1] != threading.get_ident()
    assert session.closed


def test_session_driver_requires_the_checkpoint_off_exact_projection() -> None:
    with pytest.raises(ValueError, match="exact projection"):
        asyncio.run(WarmSessionScenario.start(ScenarioConfig(history_rows=128)))


@pytest.mark.parametrize(
    ("append_rows", "entities"), ((1, 1), (99, 1), (128, 64)), ids=str
)
def test_ta_lib_reference_times_each_paired_append_against_the_oracle(
    append_rows, entities
) -> None:
    pytest.importorskip("talib")
    config = ScenarioConfig(
        history_rows=4_096,
        append_rows=append_rows,
        entities=entities,
        indicator="dual_sma_spread",
        exact_projection=True,
    )
    reference = ta_lib_reference(config, samples=3, collect_gc=True)
    assert len(reference["samples_seconds"]) == 3
    assert all(value > 0 for value in reference["samples_seconds"])
    assert reference["history_rows_per_entity"] == 19
    assert reference["throughput_rows_per_second"] > 0


def test_ta_lib_reference_rejects_unsupported_layouts() -> None:
    for config in (
        ScenarioConfig(history_rows=128, indicator="rolling_decay"),
        ScenarioConfig(history_rows=128, append_rows=1, append_entities=1),
    ):
        with pytest.raises(ValueError, match="TA-Lib reference"):
            ta_lib_reference(config, samples=2, collect_gc=False)
    with pytest.raises(ValueError, match="two samples"):
        ta_lib_reference(ScenarioConfig(history_rows=128), samples=1, collect_gc=False)


@pytest.mark.skipif(
    not hasattr(Program, "session"), reason="requires the P1 Program.session API"
)
def test_session_driver_matches_the_runner_driver_on_a_real_session(
    tmp_path,
) -> None:
    config = ScenarioConfig(
        history_rows=128,
        append_rows=64,
        indicator="dual_sma_spread",
        exact_projection=True,
    )

    async def exercise() -> None:
        runner = await WarmScenario.start(config, tmp_path)
        session = await WarmSessionScenario.start(config)
        try:
            assert session.plan_identity == runner.plan_identity
            assert session.warm_output_digest == runner.warm_output_digest
            assert session.phase_probe["installed"]
            for _ in range(3):
                expected = await runner.sample(collect_gc=False)
                actual = await session.sample(collect_gc=True)
                assert actual["correctness"] == expected["correctness"]
            assert actual["phase_attribution"] == "native_call_probe"
            assert sum(actual["phases_seconds"].values()) == pytest.approx(
                actual["seconds"]
            )
        finally:
            assert (await runner.finish())["state"] == "completed"
            assert (await session.finish())["state"] == "completed"

    asyncio.run(exercise())
