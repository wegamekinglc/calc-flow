from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchmarks import engine_stream
from benchmarks.engine_comparison import EngineCase
from calc_flow import Data
from scripts.benchmark_suite.catalog import engine_cases


def _arrow_type(advance):
    class ArrowOutput:
        def __init__(self, table):
            self.table = table

        def to_pyarrow(self):
            advance("to-arrow", 3_000_000)
            return self.table

    return ArrowOutput


def _source_type(probe, advance, output):
    class Source(engine_stream._InteractiveSource):
        def __init__(self, *, max_batch_rows=64_000):
            super().__init__(max_batch_rows=max_batch_rows)
            self.ready = asyncio.Event()
            probe.source = self

        async def push(self, event):
            if probe.fail_push:
                raise RuntimeError("injected enqueue failure")
            if event is None:
                advance("eof", 3_000_000_000)
                if probe.extra_output:
                    await probe.sink.write(output(probe.expected.slice(0, 1)))
            elif isinstance(event, Data):
                assert self.ready.is_set(), "data arrived before source readiness"
                advance("enqueue", 1_000_000)
            else:
                advance("watermark", 2_000_000)
                await probe.sink.write(output(probe.expected))

    return Source


def _sink_type(probe):
    class Sink(engine_stream._CollectSink):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            probe.sink = self

    return Sink


def _runner_type(probe, advance):
    class Job:
        async def wait_async(self):
            advance("wait", 4_000_000_000)
            return SimpleNamespace(state=probe.outcome, errors=("injected",))

        async def cancel_async(self):
            advance("cancel", 5_000_000_000)

    def ready():
        advance("ready", 300_000_000)
        probe.source.ready.set()

    class Runner:
        def __init__(self, *_args, **_kwargs):
            advance("construct", 200_000_000)

        async def start_async(self):
            advance("start", 1_000_000_000)
            await probe.source.open(None)
            await probe.sink.open()
            if probe.deferred_ready:
                asyncio.get_running_loop().call_soon(ready)
            else:
                ready()
            return Job()

    return Runner


@pytest.fixture
def stream_probe(monkeypatch):
    probe = SimpleNamespace(
        clock=0,
        phases=[],
        source=None,
        sink=None,
        expected=None,
        outcome="completed",
        extra_output=False,
        fail_push=False,
        deferred_ready=False,
    )

    def advance(phase, elapsed):
        probe.phases.append(phase)
        probe.clock += elapsed

    concat = engine_stream.pa.concat_tables

    def materialize(tables):
        advance("concat", 4_000_000)
        return concat(tables)

    source = _source_type(probe, advance, _arrow_type(advance))
    monkeypatch.setattr(engine_stream, "_ReadySource", source)
    monkeypatch.setattr(engine_stream, "_CollectSink", _sink_type(probe))
    monkeypatch.setattr(engine_stream, "StreamingRunner", _runner_type(probe, advance))
    monkeypatch.setattr(engine_stream.pa, "concat_tables", materialize)
    monkeypatch.setattr("time.perf_counter_ns", lambda: probe.clock)
    return probe


def stream_case(tmp_path, probe):
    case = next(c for c in engine_cases(10) if c["backend"] == "calc-flow-stream")
    runner = EngineCase(case, tmp_path)
    probe.expected = runner.expected
    return runner


def test_stream_timer_excludes_startup_and_cleanup_but_includes_arrow(
    tmp_path, stream_probe
):
    runner = stream_case(tmp_path, stream_probe)
    try:
        sample = runner.sample()
        assert sample["correctness"]["passed"]
        assert sample["seconds"] == pytest.approx(0.010)
        assert stream_probe.phases == [
            "construct",
            "start",
            "ready",
            "enqueue",
            "watermark",
            "to-arrow",
            "concat",
            "eof",
            "wait",
            "cancel",
        ]
    finally:
        runner.close()


def test_stream_waits_for_source_readiness_before_timing(tmp_path, stream_probe):
    stream_probe.deferred_ready = True
    runner = stream_case(tmp_path, stream_probe)
    try:
        assert runner.sample()["seconds"] == pytest.approx(0.010)
    finally:
        runner.close()


def test_stream_rejects_output_arriving_after_the_timed_result(tmp_path, stream_probe):
    stream_probe.extra_output = True
    runner = stream_case(tmp_path, stream_probe)
    try:
        with pytest.raises(RuntimeError, match="row count"):
            runner.sample()
        assert stream_probe.phases[-1] == "cancel"
    finally:
        runner.close()


@pytest.mark.parametrize("failure", ["enqueue", "completion"])
def test_stream_cleans_up_failed_measurements(tmp_path, stream_probe, failure):
    stream_probe.fail_push = failure == "enqueue"
    stream_probe.outcome = "failed"
    runner = stream_case(tmp_path, stream_probe)
    try:
        with pytest.raises(RuntimeError, match="injected|stream failed"):
            runner.sample()
        assert stream_probe.phases[-1] == "cancel"
    finally:
        runner.close()
