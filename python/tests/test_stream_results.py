from __future__ import annotations

import asyncio
import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.stream as stream_module


class _Feed:
    def __init__(self, batches: list[pa.Table | cf.Batch | cf.Watermark]) -> None:
        self.batches = batches
        self.opened = 0
        self.closed = 0
        self.read = 0

    def __aiter__(self) -> _Feed:
        self.opened += 1
        return self

    async def __anext__(self) -> pa.Table | cf.Batch | cf.Watermark:
        if self.read == len(self.batches):
            raise StopAsyncIteration
        value = self.batches[self.read]
        self.read += 1
        return value

    async def aclose(self) -> None:
        self.closed += 1


def _source() -> cf.TableExpr:
    return cf.table_input("events", schema=pa.schema([("value", pa.int64())]))


def test_stream_enters_lazily_and_releases_source_at_eof() -> None:
    source = _source()
    feed = _Feed([pa.table({"value": [1, 2]}), pa.table({"value": [3]})])
    results = source.select(doubled=source["value"] * 2).stream(feed)
    assert feed.opened == 0

    async def run() -> None:
        with pytest.raises(RuntimeError, match="async with"):
            await anext(results)
        async with results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).to_pydict() == {"doubled": [2, 4, 6]}
        assert results.job.status()["state"] == "completed"
        assert results.job.status()["task_count"] == 0
        await results.aclose()
        with pytest.raises(RuntimeError, match="once"):
            await results.__aenter__()

    asyncio.run(run())
    assert (feed.opened, feed.closed, feed.read) == (1, 1, 2)


class _IdleFeed(_Feed):
    def __init__(self) -> None:
        super().__init__([])
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def __anext__(self) -> pa.Table:
        self.reading.set()
        await self.release.wait()
        raise StopAsyncIteration


class _FailingFeed(_Feed):
    async def __anext__(self) -> pa.Table:
        raise ValueError("private-row-payload-sentinel")


def test_stream_source_failure_wakes_consumer_with_safe_native_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    feed = _FailingFeed([])
    results = _source().stream(feed)

    async def run() -> None:
        before = asyncio.all_tasks()
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with results:
                await asyncio.wait_for(anext(results), 5)
        assert caught.value.category == "connector"
        assert caught.value.component_kind == "source"
        assert "private-row-payload-sentinel" not in str(caught.value)
        with pytest.raises(AttributeError):
            caught.value.category = "changed"
        assert results.job.status()["state"] == "recovery_required"
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert feed.closed == 1
    assert list(tmp_path.iterdir()) == []


def test_stream_cancels_idle_source_and_cleans_owned_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))

    async def run() -> None:
        feed = _IdleFeed()
        results = _source().stream(feed)
        before = asyncio.all_tasks()
        entered = asyncio.Event()

        async def consume() -> None:
            async with results:
                entered.set()
                await anext(results)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(feed.reading.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert feed.closed == 1
        assert results.job.status()["state"] == "cancelled"
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


def test_stream_rejects_concurrent_consumers_and_closes_on_early_exit() -> None:
    async def run() -> None:
        feed = _IdleFeed()
        results = _source().stream(feed)
        async with results:
            started = asyncio.Event()

            async def read() -> pa.Table:
                started.set()
                return await anext(results)

            pending = asyncio.create_task(read())
            await asyncio.wait_for(started.wait(), 5)
            with pytest.raises(RuntimeError, match="concurrent"):
                await anext(results)
            await results.aclose()
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(pending, 5)
        assert feed.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("table", "config", "path"),
    [
        (pa.table({"value": [1.0]}), None, r"stream.inputs.events.schema"),
        (
            pa.table({"value": [1, 2]}),
            cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(max_rows=1)),
            r"stream.inputs.events: batch exceeds edge_budget.max_rows",
        ),
        (
            pa.table({"value": [1, 2]}),
            cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(max_bytes=8)),
            r"stream.inputs.events: batch exceeds edge_budget.max_bytes",
        ),
    ],
)
def test_stream_rejects_invalid_batches_by_logical_source(
    table: pa.Table,
    config: cf.StreamRuntimeConfig | None,
    path: str,
) -> None:
    feed = _Feed([table])
    results = _source().stream(feed, config=config)

    async def run() -> None:
        with pytest.raises(cf.StreamingRuntimeError, match=path):
            async with results:
                await asyncio.wait_for(anext(results), 5)
        assert results.job.status()["state"] == "recovery_required"

    asyncio.run(run())
    assert feed.closed == 1


def test_stream_named_branches_consume_source_once_and_capture_mapping() -> None:
    source = _source()
    feed = _Feed([pa.table({"value": [1, 2]}), pa.table({"value": [3]})])
    program = cf.Program(
        "branches",
        outputs={
            "double": source.select(value2=source["value"] * 2),
            "large": source.filter(source["value"] >= 2).select("value"),
        },
    )
    inputs = {"events": feed}
    results = program.stream(inputs)
    inputs.clear()

    async def run() -> None:
        values = {"double": [], "large": []}
        async with results:
            async for output in results:
                assert isinstance(output, cf.StreamOutput)
                values[output.name].extend(output.table.column(0).to_pylist())
        assert values == {"double": [2, 4, 6], "large": [2, 3]}

    asyncio.run(run())
    assert (feed.opened, feed.closed, feed.read) == (1, 1, 2)


def _quotes() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {"ts": [1, 2, 3, 4], "symbol": ["a"] * 4, "price": [10.0, 12.0, 15.0, 14.0]},
        schema=schema,
    )


def test_stream_composite_rolling_continues_across_batches_with_fresh_runs() -> None:
    quotes = _quotes()
    source = cf.table_input(
        "quotes",
        schema=quotes.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    delta = cf.ts.delta(source["price"])
    output = source.select(delta=delta, mean_delta=cf.ts.mean(delta, window=cf.rows(2)))
    output = output.sql("SELECT delta, mean_delta FROM input")
    runtime = cf.Runtime()

    async def collect() -> dict[str, list[float | None]]:
        feed = _Feed([quotes.slice(0, 2), quotes.slice(2)])
        async with output.stream(feed, runtime=runtime) as results:
            tables = [table async for table in results]
        assert feed.closed == 1
        return pa.concat_tables(tables).to_pydict()

    expected = {
        "delta": [None, 2.0, 3.0, -1.0],
        "mean_delta": [None, 2.0, 2.5, 1.0],
    }
    for actual in (asyncio.run(collect()), asyncio.run(collect())):
        assert actual["delta"] == expected["delta"]
        assert actual["mean_delta"] == pytest.approx(expected["mean_delta"])


class _InfiniteFeed(_Feed):
    async def __anext__(self) -> pa.Table:
        self.read += 1
        return pa.table({"value": [self.read]})


def test_stream_full_output_queue_applies_backpressure_and_cancels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))

    async def run() -> None:
        blocked = asyncio.Event()
        original = stream_module._QueueSink.write

        async def write(sink, batch) -> None:
            if sink._queue.full():
                blocked.set()
            await original(sink, batch)

        monkeypatch.setattr(stream_module._QueueSink, "write", write)
        feed = _InfiniteFeed([])
        config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(1, 32))
        results = _source().stream(feed, config=config)
        before = asyncio.all_tasks()
        async with results:
            await asyncio.wait_for(blocked.wait(), 5)
            assert feed.read < 20
        assert results.job.status()["state"] == "cancelled"
        assert results.job.status()["task_count"] == 0
        assert feed.closed == 1
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


def test_stream_start_failure_closes_only_acquired_iterators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))

    class BrokenOpen(_Feed):
        def __aiter__(self) -> _Feed:
            self.opened += 1
            raise RuntimeError("private-open-payload-sentinel")

    feed = BrokenOpen([])
    results = _source().stream(feed)

    async def run() -> None:
        before = asyncio.all_tasks()
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            await results.__aenter__()
        assert "private-open-payload-sentinel" not in str(caught.value)
        assert asyncio.all_tasks() == before
        await results.aclose()

    asyncio.run(run())
    assert (feed.opened, feed.closed) == (1, 0)
    assert list(tmp_path.iterdir()) == []


def test_stream_context_exit_surfaces_failure_without_iteration() -> None:
    results = _source().stream(_FailingFeed([]))

    async def run() -> None:
        with pytest.raises(cf.StreamingRuntimeError, match="source"):
            async with results:
                outcome = await results.job.wait_async()
                assert outcome.state == "recovery_required"

    asyncio.run(run())


def test_stream_repeated_output_aliases_are_all_emitted() -> None:
    source = _source()
    shared = source.select(incremented=source["value"] + 1)
    program = cf.Program("shared", outputs={"first": shared, "second": shared})
    feed = _Feed([pa.table({"value": [1, 2]})])

    async def run() -> None:
        outputs = {}
        async with program.stream({"events": feed}) as results:
            async for output in results:
                outputs[output.name] = output.table.to_pydict()
        assert outputs == {
            "first": {"incremented": [2, 3]},
            "second": {"incremented": [2, 3]},
        }

    asyncio.run(run())
    assert feed.opened == feed.closed == 1


@pytest.mark.parametrize("same_root", [False, True])
def test_stream_rejects_multi_alias_sql_before_iterator_acquisition(
    same_root: bool,
) -> None:
    left = _source()
    right = (
        left
        if same_root
        else cf.table_input("other", schema=pa.schema([("value", pa.int64())]))
    )
    output = cf.sql(
        "SELECT l.value AS left_value FROM l JOIN r ON l.value = r.value",
        l=left,
        r=right,
    )
    feed = _Feed([pa.table({"value": [1]})])
    inputs = {"events": feed} if same_root else {"events": feed, "other": feed}

    async def run() -> None:
        with pytest.raises(cf.CompileError, match="multi.*SQL"):
            await output.stream(inputs).__aenter__()

    asyncio.run(run())
    assert feed.opened == feed.closed == 0


class _NativeSource:
    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.opened = 0
        self.closed = 0
        self.capability_calls = 0
        self.events = iter(
            [
                cf.Data(cf.Batch.from_pyarrow(table), cf.Cursor(b"1", {})),
                cf.Watermark(datetime(2026, 1, 1, tzinfo=UTC)),
            ]
        )

    def capabilities(self) -> cf.SourceCapabilities:
        self.capability_calls += 1
        return cf.SourceCapabilities(
            cf.ReplayPositioning.UNSUPPORTED,
            cf.SourceDeliveryCapability.LOSSY,
            max_batch_rows=self.table.num_rows,
            max_batch_bytes=self.table.nbytes,
            schema=self.table.schema,
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        self.opened += 1
        assert cursor is None

    async def next(self) -> cf.Data | cf.Watermark | None:
        return next(self.events, None)

    async def close(self) -> None:
        self.closed += 1


def test_stream_keeps_source_binding_capabilities_and_watermarks() -> None:
    native_source = _NativeSource(pa.table({"value": [4, 5]}))
    policy = cf.SourceProvidedWatermarks()
    binding = cf.SourceBinding(native_source, watermark_policy=policy)
    results = _source().stream(binding)

    async def run() -> None:
        async with results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).to_pydict() == {"value": [4, 5]}
        assert results.job.status()["state"] == "completed"

    asyncio.run(run())
    assert native_source.opened == native_source.closed == 1
    assert native_source.capability_calls == 1
    assert binding.source is native_source
    assert binding.watermark_policy is policy


def test_stream_metadata_normalization_preserves_caller_batch_and_buffers() -> None:
    schema = pa.schema(
        [pa.field("value", pa.int64(), metadata={b"unit": b"count"})],
        metadata={b"origin": b"test"},
    )
    table = pa.table({"value": [1, 2]}, schema=schema)
    batch = cf.Batch.from_pyarrow(table, metadata={"sequence": 7})
    feed = _Feed([batch])
    results = cf.table_input("events", schema=schema).stream(feed)

    async def run() -> None:
        async with results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).to_pydict() == {"value": [1, 2]}

    asyncio.run(run())
    assert table.schema.equals(schema, check_metadata=True)
    assert batch.to_pyarrow().schema.equals(schema, check_metadata=True)
    assert batch.metadata == {"sequence": 7}
    assert (
        batch.to_pyarrow().column(0).chunk(0).buffers()
        == table.column(0).chunk(0).buffers()
    )


def test_stream_cancellation_during_temp_creation_reclaims_created_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stream_module.tempfile.mkdtemp

    async def run() -> None:
        loop = asyncio.get_running_loop()
        created = asyncio.Event()
        release = threading.Event()

        def mkdtemp(*, prefix: str) -> str:
            root = original(prefix=prefix, dir=tmp_path)
            loop.call_soon_threadsafe(created.set)
            if not release.wait(5):
                raise RuntimeError("test did not release temp creation")
            return root

        monkeypatch.setattr(stream_module.tempfile, "mkdtemp", mkdtemp)
        feed = _Feed([])
        results = _source().stream(feed)
        before = asyncio.all_tasks()
        task = asyncio.create_task(results.__aenter__())
        await asyncio.wait_for(created.wait(), 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert feed.opened == feed.closed == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


def test_stream_sql_branches_share_rolling_input_and_consume_source_once() -> None:
    quotes = _quotes()
    source = cf.table_input(
        "quotes",
        schema=quotes.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    shared = source.with_columns(mean=cf.ts.mean(source["price"], window=cf.rows(2)))
    program = cf.Program(
        "sql-branches",
        outputs={
            "mean": shared.select("ts", "mean").sql("SELECT mean FROM input"),
            "double": shared.select("ts", adjusted=shared["mean"] * 2.0).sql(
                "SELECT adjusted FROM input"
            ),
        },
    )
    feed = _Feed([quotes.slice(0, 2), quotes.slice(2)])

    async def run() -> None:
        values = {"mean": [], "double": []}
        async with program.stream({"quotes": feed}) as results:
            async for output in results:
                values[output.name].extend(output.table.column(0).to_pylist())
        assert values["mean"] == pytest.approx([10.0, 11.0, 13.5, 14.5])
        assert values["double"] == pytest.approx([20.0, 22.0, 27.0, 29.0])

    asyncio.run(run())
    assert (feed.opened, feed.closed, feed.read) == (1, 1, 2)


def test_stream_preserves_cancel_when_source_close_also_fails() -> None:
    class BrokenClose(_IdleFeed):
        async def aclose(self) -> None:
            self.closed += 1
            raise ValueError("private-close-payload-sentinel")

    async def run() -> None:
        feed = BrokenClose()
        results = _source().stream(feed)
        entered = asyncio.Event()

        async def consume() -> None:
            async with results:
                entered.set()
                await anext(results)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(feed.reading.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


class _CancelStartResult:
    def __init__(self, awaitable, owner: asyncio.Task) -> None:
        self._awaitable = awaitable
        self._owner = owner

    def __await__(self):
        iterator = self._awaitable.__await__()
        observer = next(iterator)
        observer.add_done_callback(lambda _: self._owner.cancel())
        try:
            yield observer
        except asyncio.CancelledError as cancellation:
            observer = None
            return iterator.throw(cancellation)
        raise AssertionError("start-result observer was not cancelled")


class _StartProxy:
    def __init__(self, native, owner: asyncio.Task) -> None:
        self._native = native
        self._owner = owner

    def start_async(self):
        return _CancelStartResult(self._native.start_async(), self._owner)

    def __getattr__(self, name: str):
        return getattr(self._native, name)


def test_stream_cancellation_at_native_start_result_releases_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    original = cf.StreamingRunner.start_async

    async def start(runner):
        native = runner._inner
        runner._inner = _StartProxy(native, asyncio.current_task())
        try:
            return await original(runner)
        finally:
            runner._inner = native

    monkeypatch.setattr(cf.StreamingRunner, "start_async", start)

    async def run() -> None:
        before = asyncio.all_tasks()
        feed = _IdleFeed()
        results = _source().stream(feed)
        task = asyncio.create_task(results.__aenter__())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert feed.opened == feed.closed == 1
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


class _TempRootGate:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._create = stream_module.tempfile.mkdtemp
        self._loop = asyncio.get_running_loop()
        self.created = asyncio.Event()
        self.release = threading.Event()
        self.roots: list[str] = []

    def __call__(self, *, prefix: str) -> str:
        root = self._create(prefix=prefix, dir=self._directory)
        self.roots.append(root)
        self._loop.call_soon_threadsafe(self.created.set)
        if not self.release.wait(5):
            raise RuntimeError("test did not release temp creation")
        return root

    async def cleanup(self, results, tasks) -> None:
        self.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        if results._job is not None:
            await results.job.cancel_async()
            await results._waiter
        for root in self.roots:
            if Path(root).exists():
                await asyncio.to_thread(shutil.rmtree, root)


@pytest.mark.parametrize("close_count", [1, 2])
def test_stream_close_waits_for_pending_start_and_prevents_late_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_count: int,
) -> None:
    original_close = stream_module.StreamResults._close

    async def run() -> None:
        before = asyncio.all_tasks()
        gate = _TempRootGate(tmp_path)
        close_started, close_finished = asyncio.Event(), asyncio.Event()

        async def close(results) -> None:
            close_started.set()
            await original_close(results)
            close_finished.set()

        monkeypatch.setattr(stream_module.tempfile, "mkdtemp", gate)
        monkeypatch.setattr(stream_module.StreamResults, "_close", close)
        feed = _IdleFeed()
        results = _source().stream(feed)
        entry = asyncio.create_task(results.__aenter__())
        await asyncio.wait_for(gate.created.wait(), 5)
        closing = [asyncio.create_task(results.aclose()) for _ in range(close_count)]
        try:
            await asyncio.wait_for(close_started.wait(), 5)
            assert not close_finished.is_set()
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(entry, 5)
            await asyncio.wait_for(asyncio.gather(*closing), 5)
            assert list(tmp_path.iterdir()) == []
            assert feed.opened == feed.closed == 0
            assert asyncio.all_tasks() == before
        finally:
            await gate.cleanup(results, [entry, *closing])

    asyncio.run(run())


def test_stream_close_before_entry_never_acquires_sources() -> None:
    feed = _Feed([pa.table({"value": [1]})])
    results = _source().stream(feed)

    async def run() -> None:
        before = asyncio.all_tasks()
        await asyncio.gather(results.aclose(), results.aclose())
        with pytest.raises(RuntimeError, match="once"):
            await results.__aenter__()
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert feed.opened == feed.closed == 0


def test_stream_close_at_native_start_completion_cancels_before_entry_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    original_start = stream_module.StreamResults._start

    async def run() -> None:
        before = asyncio.all_tasks()
        closing = []

        async def start(results) -> None:
            await original_start(results)
            asyncio.get_running_loop().call_soon(
                lambda: closing.append(asyncio.create_task(results.aclose()))
            )

        monkeypatch.setattr(stream_module.StreamResults, "_start", start)
        feed = _IdleFeed()
        results = _source().stream(feed)
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(results.__aenter__(), 5)
            await asyncio.wait_for(asyncio.gather(*closing), 5)
            assert results.job.status()["state"] == "cancelled"
            assert results.job.status()["task_count"] == 0
            assert feed.opened == feed.closed == 1
            assert asyncio.all_tasks() == before
            assert list(tmp_path.iterdir()) == []
        finally:
            await results.aclose()
            await asyncio.gather(*closing, return_exceptions=True)

    asyncio.run(run())


class _OpenEndedFeed(_Feed):
    def __init__(self, batches: list[pa.Table | cf.Watermark]) -> None:
        super().__init__(batches)
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def __anext__(self) -> pa.Table | cf.Batch | cf.Watermark:
        if self.read == len(self.batches):
            self.waiting.set()
            await self.release.wait()
        return await super().__anext__()


async def _first_rows(results, count: int) -> pa.Table:
    tables: list[pa.Table] = []
    rows = 0
    async for table in results:
        tables.append(table)
        rows += table.num_rows
        if rows >= count:
            return pa.concat_tables(tables).slice(0, count)
    raise AssertionError("stream ended before the requested rows arrived")


def test_stream_composite_rolling_emits_while_source_remains_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    quotes = _quotes()
    source = cf.table_input(
        "quotes",
        schema=quotes.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    delta = cf.ts.delta(source["price"])
    output = source.select(delta=delta, mean_delta=cf.ts.mean(delta, window=cf.rows(2)))
    output = output.sql("SELECT delta, mean_delta FROM input")

    async def run() -> None:
        before = asyncio.all_tasks()
        feed = _OpenEndedFeed([quotes.slice(0, 2), quotes.slice(2)])
        results = output.stream(feed)
        try:
            async with results:
                await asyncio.wait_for(feed.waiting.wait(), 5)
                table = await asyncio.wait_for(_first_rows(results, 3), 5)
                assert table.to_pydict() == {
                    "delta": [None, 2.0, 3.0],
                    "mean_delta": [None, 2.0, 2.5],
                }
                assert not feed.release.is_set()
                assert feed.closed == 0
                assert results.job.status()["state"] == "running"
        finally:
            await results.aclose()
            assert (feed.opened, feed.closed, feed.read) == (1, 1, 2)
            assert not feed.release.is_set()
            assert results.job.status()["task_count"] == 0
            assert list(tmp_path.iterdir()) == []
            assert asyncio.all_tasks() == before

    asyncio.run(run())


def _temporal_source(name: str = "quotes") -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=_quotes().schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )


def _temporal_rows(times, symbols=None) -> pa.Table:
    return pa.table(
        {
            "ts": times,
            "symbol": ["a"] * len(times) if symbols is None else symbols,
            "price": [float(index) for index in range(len(times))],
        },
        schema=_quotes().schema,
    )


@pytest.mark.parametrize(
    "batches",
    [
        [_temporal_rows([2, 1])],
        [_temporal_rows([2]), _temporal_rows([1])],
        [_temporal_rows([2], ["a"]), _temporal_rows([1], ["b"])],
    ],
    ids=["within-batch", "across-batches", "across-entities"],
)
def test_stream_default_rejects_globally_descending_event_time(batches) -> None:
    source = _temporal_source()
    feed = _Feed(batches)
    results = source.select(delta=cf.ts.delta(source["price"])).stream(feed)

    async def run() -> None:
        before = asyncio.all_tasks()
        with pytest.raises(
            cf.StreamingRuntimeError,
            match=r"stream.inputs.quotes.event_time: expected nondecreasing.*explicit",
        ):
            async with asyncio.timeout(5), results:
                assert [table async for table in results] == []
        assert feed.closed == feed.opened == 1
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy, message",
    [
        ({"unknown": cf.DisabledWatermarks()}, "unknown"),
        ({"quotes": None}, "supported watermark policy"),
        ("invalid", "supported watermark policy"),
    ],
)
def test_stream_rejects_invalid_watermark_selection_before_source_open(
    policy, message
) -> None:
    feed = _Feed([_quotes()])
    results = _temporal_source().stream(feed, watermarks=policy)

    async def run() -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            await results.__aenter__()

    asyncio.run(run())
    assert feed.closed == feed.opened == 0


def test_stream_equal_times_and_empty_batches_preserve_open_frontier() -> None:
    source = _temporal_source()
    output = source.select("ts", "symbol", previous=cf.ts.lag(source["price"]))

    async def run() -> None:
        before = asyncio.all_tasks()
        feed = _OpenEndedFeed(
            [
                _temporal_rows([]),
                _temporal_rows([1], ["a"]),
                _temporal_rows([]),
                _temporal_rows([1], ["b"]),
                _temporal_rows([]),
                _temporal_rows([2], ["a"]),
            ]
        )
        results = output.stream(feed)
        async with asyncio.timeout(5), results:
            await feed.waiting.wait()
            table = await _first_rows(results, 2)
            assert table["ts"].cast(pa.int64()).to_pylist() == [1, 1]
            assert sorted(table["symbol"].to_pylist()) == ["a", "b"]
            assert table["previous"].to_pylist() == [None, None]
            assert feed.read == 6
            assert feed.closed == 0
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


class _PushFeed(_Feed):
    def __init__(self, batches) -> None:
        super().__init__(batches)
        self.waiting = asyncio.Event()
        self.pending: asyncio.Queue[pa.Table | cf.Watermark] = asyncio.Queue()

    async def __anext__(self) -> pa.Table | cf.Batch | cf.Watermark:
        if self.read < len(self.batches):
            return await super().__anext__()
        self.waiting.set()
        data = await self.pending.get()
        self.read += 1
        return data


def test_stream_explicit_disorder_emits_and_rejects_observed_inclusive_cutoff() -> None:
    source = _temporal_source()
    output = source.select("ts", previous=cf.ts.lag(source["price"]))
    policy = cf.BoundedOutOfOrderness(
        "ts", timedelta(microseconds=2), timedelta(milliseconds=100)
    )

    async def run() -> None:
        before = asyncio.all_tasks()
        feed = _PushFeed([_temporal_rows([2, 1, 4])])
        results = output.stream(feed, watermarks=policy)
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with asyncio.timeout(5), results:
                await feed.waiting.wait()
                table = await _first_rows(results, 2)
                assert table["ts"].cast(pa.int64()).to_pylist() == [1, 2]
                assert table["previous"].to_pylist() == [None, 1.0]
                # Observed output establishes that max(4) - delay(2) was published.
                feed.pending.put_nowait(_temporal_rows([2], ["b"]))
                await anext(results)
        assert caught.value.component_kind == "operator"
        assert caught.value.component_id.endswith("rolling")
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


def _watermark(microseconds: int) -> cf.Watermark:
    return cf.Watermark(
        datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)
    )


def test_stream_manual_watermarks_preserve_cursors_and_close_paused_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _temporal_source()
    output = source.select(previous=cf.ts.lag(source["price"]))
    cursors = []
    original = cf.SourceBinding._native_next

    async def next_event(binding):
        event = await original(binding)
        if event is not None and event[0] == "data":
            cursors.append(int.from_bytes(event[3], "big"))
        return event

    monkeypatch.setattr(cf.SourceBinding, "_native_next", next_event)

    async def run() -> None:
        feed = _OpenEndedFeed(
            [_temporal_rows([1]), _watermark(1), _temporal_rows([2]), _watermark(2)]
        )
        results = output.stream(feed, watermarks=cf.SourceProvidedWatermarks())
        async with asyncio.timeout(5), results:
            await feed.waiting.wait()
            table = await _first_rows(results, 2)
            assert table["previous"].to_pylist() == [None, 0.0]
            assert cursors == [1, 2]
            assert not feed.release.is_set()
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy",
    [
        None,
        cf.DisabledWatermarks(),
        cf.BoundedOutOfOrderness(
            "ts", timedelta(microseconds=1), timedelta(milliseconds=100)
        ),
    ],
    ids=["default", "disabled", "generated"],
)
def test_stream_rejects_manual_watermark_with_non_native_policy(policy) -> None:
    feed = _Feed([_watermark(2)])
    results = _temporal_source().stream(feed, watermarks=policy)

    async def run() -> None:
        with pytest.raises(
            cf.StreamingRuntimeError, match="Watermark requires SourceProvided"
        ):
            async with asyncio.timeout(5), results:
                await anext(results)
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_stream_default_rejects_null_event_time() -> None:
    feed = _Feed([_temporal_rows([None])])
    results = _temporal_source().stream(feed)

    async def run() -> None:
        with pytest.raises(
            cf.StreamingRuntimeError, match="expected non-null event time"
        ):
            async with asyncio.timeout(5), results:
                await anext(results)
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_stream_rejects_source_binding_policy_override_before_open() -> None:
    source = _NativeSource(_quotes())
    binding = cf.SourceBinding(source)
    results = _temporal_source().stream(binding, watermarks=cf.DisabledWatermarks())

    async def run() -> None:
        with pytest.raises(ValueError, match="cannot override SourceBinding"):
            await results.__aenter__()

    asyncio.run(run())
    assert source.opened == source.closed == source.capability_calls == 0


def test_stream_multiple_sources_require_logical_watermark_mapping() -> None:
    left, right = _temporal_source("left"), _temporal_source("right")
    program = cf.Program("two", outputs={"left": left, "right": right})
    feeds = {"left": _Feed([_quotes()]), "right": _Feed([_quotes()])}
    results = program.stream(feeds, watermarks=cf.DisabledWatermarks())

    async def run() -> None:
        with pytest.raises(ValueError, match="multiple inputs require a name mapping"):
            await results.__aenter__()

    asyncio.run(run())
    assert all(feed.opened == feed.closed == 0 for feed in feeds.values())


async def _named_rows(results, counts) -> dict[str, list[float | None]]:
    values = {name: [] for name in counts}
    async for output in results:
        values[output.name].extend(output.table["previous"].to_pylist())
        if all(len(values[name]) >= count for name, count in counts.items()):
            return values
    raise AssertionError("stream ended before the requested named rows arrived")


def test_stream_captures_policy_mapping_and_defaults_omitted_logical_source() -> None:
    left, right = _temporal_source("left_input"), _temporal_source("right_input")
    program = cf.Program(
        "two",
        outputs={
            "left": left.select(previous=cf.ts.lag(left["price"])),
            "right": right.select(previous=cf.ts.lag(right["price"])),
        },
    )

    async def run() -> None:
        feeds = {
            "left_input": _OpenEndedFeed([_quotes().slice(0, 2), _watermark(3)]),
            "right_input": _OpenEndedFeed([_quotes()]),
        }
        policies = {"left_input": cf.SourceProvidedWatermarks()}
        results = program.stream(feeds, watermarks=policies)
        policies["left_input"] = cf.DisabledWatermarks()
        async with asyncio.timeout(5), results:
            for feed in feeds.values():
                await feed.waiting.wait()
            assert await _named_rows(results, {"left": 2, "right": 3}) == {
                "left": [None, 10.0],
                "right": [None, 10.0, 12.0],
            }
        assert all(feed.opened == feed.closed == 1 for feed in feeds.values())
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


async def _observe_marker(results, timestamp: int) -> None:
    async for output in results:
        if output.name == "marker" and output.table["ts"].cast(
            pa.int64()
        ).to_pylist() == [timestamp]:
            return
    raise AssertionError("stream ended before source progress was observed")


async def _next_named_output(results, name: str) -> pa.Table:
    async for output in results:
        if output.name == name:
            return output.table
    raise AssertionError("stream ended before requested output arrived")


def test_stream_quiet_active_source_holds_shared_watermark_until_it_advances() -> None:
    left, right = _temporal_source("left_input"), _temporal_source("right_input")
    program = cf.Program(
        "two",
        outputs={
            "left": left.select(previous=cf.ts.lag(left["price"])),
            "right": right.select(previous=cf.ts.lag(right["price"])),
            "marker": right,
        },
    )

    async def run() -> None:
        before = asyncio.all_tasks()
        feeds = {
            "left_input": _PushFeed([_temporal_rows([1])]),
            "right_input": _PushFeed(
                [_temporal_rows([1]), _watermark(10), _temporal_rows([11])]
            ),
        }
        policies = {name: cf.SourceProvidedWatermarks() for name in feeds}
        results = program.stream(feeds, watermarks=policies)
        async with asyncio.timeout(5), results:
            for feed in feeds.values():
                await feed.waiting.wait()
            # The right source's post-watermark row proves its frontier was read.
            await _observe_marker(results, 11)
            assert results.job.status()["watermark_micros"] is None
            feeds["left_input"].pending.put_nowait(_watermark(10))
            table = await _next_named_output(results, "left")
            assert table["previous"].to_pylist() == [None]
            assert results.job.status()["watermark_micros"] == 10
        assert all(feed.opened == feed.closed == 1 for feed in feeds.values())
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


def test_stream_default_watermark_underflow_fails_without_forward_progress() -> None:
    source = _temporal_source()
    output = source.select(previous=cf.ts.lag(source["price"]))

    async def run() -> None:
        before = asyncio.all_tasks()
        feed = _OpenEndedFeed([_temporal_rows([-(1 << 63)])])
        results = output.stream(feed)
        with pytest.raises(cf.StreamingRuntimeError):
            async with asyncio.timeout(5), results:
                await feed.waiting.wait()
                assert [table async for table in results] == []
        assert results.job.status()["watermark_micros"] is None
        assert results.job.status()["task_count"] == 0
        assert feed.closed == 1
        assert asyncio.all_tasks() == before

    asyncio.run(run())


def test_stream_policies_apply_only_to_dynamic_inputs_with_static_parameters() -> None:
    import numpy as np

    source = _temporal_source()
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    values = cf.linalg.from_columns(source, columns=("price",), backend="numpy")
    output = cf.table.attach_columns(
        source, cf.linalg.matmul(values, weights), names=("score",)
    )
    runtime = cf.Runtime()
    cf.register_numpy(runtime)

    async def run() -> None:
        feed = _Feed([_quotes()])
        inputs = {
            "quotes": feed,
            "weights": cf.Batch.from_array(np.array([[2.0]]), backend="numpy"),
        }
        invalid = output.stream(
            inputs, runtime=runtime, watermarks={"weights": cf.DisabledWatermarks()}
        )
        with pytest.raises(ValueError, match="watermarks.weights.*dynamic input"):
            await invalid.__aenter__()
        assert feed.opened == 0
        results = output.stream(
            inputs, runtime=runtime, watermarks=cf.DisabledWatermarks()
        )
        async with asyncio.timeout(5), results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables)["score"].to_pylist() == [20.0, 24.0, 30.0, 28.0]
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_stream_generated_timer_with_full_output_queue_cleans_owned_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    source = _temporal_source()
    previous = source.select(previous=cf.ts.lag(source["price"]))
    program = cf.Program("fanout", outputs={"first": previous, "second": previous})

    async def run() -> None:
        before = asyncio.all_tasks()
        blocked = asyncio.Event()
        original = stream_module._QueueSink.write

        async def write(sink, batch) -> None:
            if sink._queue.full():
                blocked.set()
            await original(sink, batch)

        monkeypatch.setattr(stream_module._QueueSink, "write", write)
        feed = _OpenEndedFeed([_quotes()])
        config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(4, 256))
        results = program.stream({"quotes": feed}, config=config)
        async with asyncio.timeout(5), results:
            await feed.waiting.wait()
            await blocked.wait()
            assert feed.closed == 0
            assert not feed.release.is_set()
        assert results.job.status()["task_count"] == 0
        assert feed.closed == 1
        assert asyncio.all_tasks() == before
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())


def test_stream_rejects_invalid_iterable_value_by_logical_input() -> None:
    feed = _Feed([object()])
    results = _source().stream(feed)

    async def run() -> None:
        with pytest.raises(
            cf.StreamingRuntimeError, match="stream.inputs.events: expected Arrow"
        ):
            async with asyncio.timeout(5), results:
                await anext(results)
        assert feed.closed == 1
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())
