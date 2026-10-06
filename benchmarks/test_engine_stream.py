from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchmarks import engine_stream
from benchmarks.engine_comparison import EngineCase
from benchmarks.warm_stream import BASE
from calc_flow import Batch, Cursor, Data, Watermark
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
            probe.sources.append(self)

        async def push(self, event):
            if probe.fail_push:
                raise RuntimeError("injected enqueue failure")
            if event is None:
                advance("eof", 3_000_000_000)
                if probe.extra_output:
                    await probe.sink.write(output(probe.expected.slice(0, 1)))
                probe.ended_sources += 1
                if probe.ended_sources == len(probe.sources):
                    probe.ended.set()
            elif isinstance(event, Data):
                self.static = "factor" in event.batch.to_pyarrow().column_names
                assert self.ready.is_set(), "data arrived before source readiness"
                advance("enqueue", 1_000_000)
            else:
                advance("watermark", 2_000_000)
                probe.watermark_micros = engine_stream.BASE_MICROS + (
                    event.at - BASE
                ) // engine_stream.timedelta(microseconds=1)
                # A multi-binding stream pushes one watermark per source; the
                # first delivers the workload, the rest find it complete.
                if not self.static and not probe.sink.rows:
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
        def status(self):
            advance("status", 99_000_000)
            return {
                "stream_joins": {
                    "join": {
                        "right": {"watermark_micros": probe.watermark_micros},
                        "left": {"retained_rows": 0, "evicted_rows": 0},
                    }
                }
            }

        async def wait_async(self):
            await probe.ended.wait()
            advance("wait", 4_000_000_000)
            return SimpleNamespace(state=probe.outcome, errors=("injected",))

        async def cancel_async(self):
            advance("cancel", 5_000_000_000)

    def ready():
        advance("ready", 300_000_000)
        for source in probe.sources:
            source.ready.set()

    class Runner:
        def __init__(self, *_args, **_kwargs):
            advance("construct", 200_000_000)

        async def start_async(self):
            advance("start", 1_000_000_000)
            for source in probe.sources:
                await source.open(None)
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
        watermark_micros=None,
        phases=[],
        sources=[],
        sink=None,
        expected=None,
        outcome="completed",
        extra_output=False,
        fail_push=False,
        deferred_ready=False,
        ended_sources=0,
        ended=asyncio.Event(),
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


def stream_case(tmp_path, probe, scenario="projection"):
    case = next(
        c
        for c in engine_cases(10)
        if c["backend"] == "calc-flow-stream" and c["scenario"] == scenario
    )
    runner = EngineCase(case, tmp_path)
    probe.expected = runner.expected
    return runner


def test_ready_source_applies_backpressure():
    async def exercise():
        source = engine_stream._ReadySource()
        await source.push("first")
        second = asyncio.create_task(source.push("second"))
        await asyncio.sleep(0)
        assert not second.done()
        assert await source.next() == "first"
        await asyncio.wait_for(second, 1)
        assert await source.next() == "second"

    asyncio.run(exercise())


def test_stream_timer_supports_the_two_source_join_binding(tmp_path, stream_probe):
    runner = stream_case(tmp_path, stream_probe, scenario="join")
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
            "status",
            "enqueue",
            "watermark",
            "to-arrow",
            "concat",
            "status",
            "eof",
            "eof",
            "wait",
            "cancel",
        ]
    finally:
        runner.close()


def test_static_join_waits_for_dimension_watermark_before_quotes(monkeypatch):
    states = iter((None, 99, 100))
    waits = []

    def status():
        return {
            "stream_joins": {
                "join": {
                    "right": {"watermark_micros": next(states)},
                    "left": {"retained_rows": 0, "evicted_rows": 0},
                }
            }
        }

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(engine_stream.asyncio, "sleep", sleep)
    asyncio.run(
        engine_stream._wait_static_dimension_progress(
            SimpleNamespace(status=status), 100
        )
    )
    assert waits == [0.001, 0.001]


def test_static_join_seals_dimension_before_feeding_any_quote():
    async def exercise():
        sink = engine_stream._CollectSink(2)
        await sink.open()
        batch = Batch.from_pyarrow(engine_stream.pa.table({"value": [1, 2]}))
        data = Data(batch, Cursor(b"rows", {}))
        watermark = Watermark(BASE)
        events = []
        acknowledged = False
        checks = 0

        class Source:
            opened = asyncio.Event()
            ready = asyncio.Event()

            def __init__(self, name):
                self.name = name
                self.opened.set()
                self.ready.set()

            async def push(self, event):
                events.append((self.name, event))
                if self.name == "left":
                    assert acknowledged, (
                        "quote arrived before dimension acknowledgement"
                    )
                    if isinstance(event, Data):
                        await sink.write(batch)

        def status():
            nonlocal acknowledged, checks
            checks += 1
            assert events[:2] == [("right", data), ("right", watermark)]
            acknowledged = checks >= 2
            return {
                "stream_joins": {
                    "join": {
                        "right": {
                            "watermark_micros": engine_stream.BASE_MICROS
                            if acknowledged
                            else None
                        },
                        "left": {"retained_rows": 0, "evicted_rows": 0},
                    }
                }
            }

        output, _seconds = await engine_stream._measure_ready(
            {name: Source(name) for name in ("left", "right")},
            sink,
            {name: (data, watermark) for name in ("left", "right")},
            SimpleNamespace(status=status),
        )
        assert output.equals(batch.to_pyarrow())
        assert [name for name, _event in events] == ["right", "right", "left", "left"]

    asyncio.run(exercise())


def test_static_join_limits_allow_only_dimension_and_one_batch():
    limits = engine_stream._join_limits(128)
    assert limits.max_state_rows_per_side == 128 + engine_stream.BATCH_ROWS
    assert limits.max_matches_per_input_batch == 128 + engine_stream.BATCH_ROWS


@pytest.mark.parametrize("metric", ("retained_rows", "evicted_rows"))
def test_static_dimension_gate_rejects_left_state(metric):
    def status():
        return {
            "stream_joins": {
                "join": {
                    "right": {"watermark_micros": 100},
                    "left": {"retained_rows": 0, "evicted_rows": 0, metric: 1},
                }
            }
        }

    with pytest.raises(
        RuntimeError, match="static Join retained or evicted quote rows"
    ):
        asyncio.run(
            engine_stream._wait_static_dimension_progress(
                SimpleNamespace(status=status), 100
            )
        )


def test_sink_waits_for_cumulative_delivery_across_output_chunks():
    async def exercise():
        sink = engine_stream._CollectSink(5)
        batch = Batch.from_pyarrow(engine_stream.pa.table({"value": range(5)}))
        waiting = asyncio.create_task(sink.wait_for_rows(3))
        try:
            await sink.write(Batch.from_pyarrow(batch.to_pyarrow().slice(0, 2)))
            await asyncio.sleep(0)
            assert not waiting.done()
            await sink.write(Batch.from_pyarrow(batch.to_pyarrow().slice(2, 1)))
            await asyncio.wait_for(waiting, 1)
            assert not sink.complete.is_set()
            waiting = asyncio.create_task(sink.wait_for_rows(5))
            await sink.write(Batch.from_pyarrow(batch.to_pyarrow().slice(0, 0)))
            await asyncio.sleep(0)
            assert not waiting.done(), (
                "an earlier delivery must not release a later wait"
            )
            await sink.write(Batch.from_pyarrow(batch.to_pyarrow().slice(3, 2)))
            await asyncio.wait_for(waiting, 1)
            await asyncio.wait_for(sink.wait_for_rows(5), 1)
            assert sink.complete.is_set()
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)

    asyncio.run(exercise())


def test_asof_lockstep_waits_for_sink_delivery_without_status_calls():
    async def exercise():
        expected = engine_stream.pa.table({"value": range(5)})
        sink = engine_stream._CollectSink(5)
        await sink.open()
        delivered = []
        tasks = []

        class Source:
            def __init__(self, name):
                self.name = name
                self.ready = asyncio.Event()
                self.opened = asyncio.Event()
                self.ready.set()
                self.opened.set()
                self.fed = 0
                self.pending = None

            async def push(self, event):
                if isinstance(event, Data):
                    assert sink.rows >= self.fed, (
                        "the next pair arrived before delivery"
                    )
                    self.pending = event.batch
                    self.fed += event.batch.num_rows
                elif self.name == "quotes.input":
                    tasks.append(asyncio.create_task(self.deliver()))

            async def deliver(self):
                await asyncio.sleep(0)
                await sink.write(self.pending)
                delivered.append(sink.rows)

        def status():
            raise AssertionError("status polling perturbs the timed operator")

        streams = {
            name: tuple(
                event
                for offset, count in ((0, 3), (3, 2))
                for event in (
                    Data(
                        Batch.from_pyarrow(expected.slice(offset, count)),
                        Cursor(offset.to_bytes(8, "big"), {"offset": offset}),
                    ),
                    Watermark(BASE),
                )
            )
            for name in ("reference.input", "quotes.input")
        }
        sources = {name: Source(name) for name in streams}
        try:
            table, _ = await asyncio.wait_for(
                engine_stream._measure_ready(
                    sources, sink, streams, SimpleNamespace(status=status)
                ),
                1,
            )
            assert table.equals(expected)
            assert delivered == [3, 5]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())


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
