from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

import calc_flow as cf


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )


def _source(name: str) -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=_schema(),
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )


def _rows(
    times: list[int],
    *,
    sequences: list[int] | None = None,
    prices: list[float] | None = None,
) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "symbol": ["AAA"] * len(times),
            "time": times,
            "sequence": list(range(len(times))) if sequences is None else sequences,
            "price": [float(time) for time in times] if prices is None else prices,
        },
        schema=_schema(),
    )


def _joined(tolerance: int = 10, *, late_policy: str = "error") -> cf.TableExpr:
    return _source("trades").stream_asof_join(
        _source("quotes"),
        tolerance=timedelta(microseconds=tolerance),
        limits=cf.AsofStateLimits(1_000, 8_000_000),
        late_policy=late_policy,
        prefixes=("trade", "quote"),
    )


class _Feed:
    def __init__(self, values) -> None:
        self.values = iter(values)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.values)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed += 1


async def _collect(result, inputs, **options):
    stream = result.stream(inputs, **options)
    async with asyncio.timeout(10), stream:
        tables = [table async for table in stream]
    assert stream.job.status()["state"] == "completed"
    assert stream.job.status()["task_count"] == 0
    return pa.concat_tables(tables), stream.job.status()


@pytest.mark.parametrize(("tolerance", "expected"), [(10, 100.0), (4, None), (0, None)])
def test_asof_stream_selects_latest_history_and_preserves_unmatched_left(
    tolerance, expected
) -> None:
    left = _Feed([_rows([105])])
    right = _Feed([_rows([90, 100, 110])])
    output, status = asyncio.run(
        _collect(_joined(tolerance), {"trades": left, "quotes": right})
    )
    assert output["quote__price"].to_pylist() == [expected]
    assert output["trade__time"].cast(pa.int64()).to_pylist() == [105]
    assert all(
        field.nullable for field in output.schema if field.name.startswith("quote__")
    )
    assert left.closed == right.closed == 1
    asof = next(iter(status["stream_asof_joins"].values()))
    assert asof["emitted_left_rows"] == 1
    assert asof["pending_left_rows"] == 0


@pytest.mark.parametrize(
    "batches",
    [
        [_rows([100, 100], sequences=[9, 7], prices=[19.0, 17.0])],
        [
            _rows([100], sequences=[7], prices=[17.0]),
            _rows([100], sequences=[9], prices=[19.0]),
        ],
    ],
)
def test_asof_stream_tie_selection_does_not_depend_on_batch_or_arrival_order(
    batches,
) -> None:
    output, _ = asyncio.run(
        _collect(
            _joined(),
            {"trades": _Feed([_rows([105])]), "quotes": _Feed(batches)},
            watermarks={
                "trades": cf.SourceProvidedWatermarks(),
                "quotes": cf.SourceProvidedWatermarks(),
            },
        )
    )
    assert output["quote__sequence"].to_pylist() == [9]
    assert output["quote__price"].to_pylist() == [19.0]


def test_asof_collect_rejects_batch_even_with_finite_inputs() -> None:
    with pytest.raises(cf.CompileError, match="unsupported_mode"):
        _joined().collect({"trades": _rows([105]), "quotes": _rows([100])})


def test_asof_program_stream_preserves_logical_names_and_shares_state() -> None:
    joined = _joined()
    program = cf.Program(
        "fanout",
        outputs={
            "all": joined,
            "prices": joined.select("trade__price", "quote__price"),
        },
    )

    async def run() -> None:
        results = program.stream(
            {"trades": _Feed([_rows([105])]), "quotes": _Feed([_rows([100])])}
        )
        async with asyncio.timeout(10), results:
            outputs = [output async for output in results]
        assert {output.name for output in outputs} == {"all", "prices"}
        assert all(
            output.table["quote__price"].to_pylist() == [100.0] for output in outputs
        )
        assert len(results.job.status()["stream_asof_joins"]) == 1

    asyncio.run(run())


def _watermark(time: int) -> cf.Watermark:
    return cf.Watermark(datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=time))


class _PushFeed:
    def __init__(self, values) -> None:
        self.pending = asyncio.Queue()
        for value in values:
            self.pending.put_nowait(value)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.pending.get()
        if value is None:
            raise StopAsyncIteration
        return value

    async def aclose(self) -> None:
        self.closed += 1


async def _wait_for_watermarks(results, left: int, right: int) -> None:
    async with asyncio.timeout(5):
        while True:
            statuses = results.job.status()["stream_asof_joins"]
            if statuses:
                status = next(iter(statuses.values()))
                if (
                    status["left"]["watermark_micros"] == left
                    and status["right"]["watermark_micros"] == right
                ):
                    return
            await asyncio.sleep(0.001)


def test_asof_equal_watermarks_wait_and_same_time_candidate_can_still_win() -> None:
    async def run() -> None:
        before = asyncio.all_tasks()
        left = _PushFeed([_rows([105]), _watermark(105)])
        right = _PushFeed([_rows([100]), _watermark(105)])
        results = _joined().stream(
            {"trades": left, "quotes": right},
            watermarks={
                "trades": cf.SourceProvidedWatermarks(),
                "quotes": cf.SourceProvidedWatermarks(),
            },
        )
        async with asyncio.timeout(10), results:
            pending = asyncio.create_task(anext(results))
            await _wait_for_watermarks(results, 105, 105)
            assert not pending.done()
            right.pending.put_nowait(_rows([105], sequences=[9]))
            right.pending.put_nowait(_watermark(106))
            await _wait_for_watermarks(results, 105, 106)
            assert not pending.done()
            left.pending.put_nowait(_watermark(106))
            output = await pending
            assert output["quote__sequence"].to_pylist() == [9]
            assert output["quote__price"].to_pylist() == [105.0]
            left.pending.put_nowait(None)
            right.pending.put_nowait(None)
            assert [table async for table in results] == []
        assert left.closed == right.closed == 1
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


def test_asof_cancelling_wait_for_finality_closes_both_sources() -> None:
    async def run() -> None:
        before = asyncio.all_tasks()
        left = _PushFeed([_rows([105]), _watermark(105)])
        right = _PushFeed([_rows([100]), _watermark(105)])
        results = _joined().stream(
            {"trades": left, "quotes": right},
            watermarks={
                "trades": cf.SourceProvidedWatermarks(),
                "quotes": cf.SourceProvidedWatermarks(),
            },
        )

        entered = asyncio.Event()

        async def consume() -> None:
            async with results:
                entered.set()
                await anext(results)

        task = asyncio.create_task(consume())
        async with asyncio.timeout(10):
            await entered.wait()
            await _wait_for_watermarks(results, 105, 105)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert left.closed == right.closed == 1
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())


def test_asof_late_drop_is_counted_without_losing_accepted_left() -> None:
    feeds = {
        "trades": _Feed([_watermark(105), _rows([100, 105], sequences=[1, 2])]),
        "quotes": _Feed([_rows([100])]),
    }
    table, status = asyncio.run(
        _collect(
            _joined(late_policy="drop"),
            feeds,
            watermarks={
                "trades": cf.SourceProvidedWatermarks(),
                "quotes": cf.SourceProvidedWatermarks(),
            },
        )
    )
    assert table["trade__time"].cast(pa.int64()).to_pylist() == [105]
    asof = next(iter(status["stream_asof_joins"].values()))
    assert asof["left"]["late_rows"] == 1
    assert asof["left"]["accepted_rows"] == 1


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([_rows([100, 100], sequences=[1, 1])], "asof_duplicate_identity"),
        ([_watermark(105), _rows([100])], "asof_late_row"),
    ],
)
def test_asof_runtime_identity_and_late_errors_are_structured(values, reason) -> None:
    async def run() -> None:
        results = _joined().stream(
            {"trades": _Feed([_rows([105])]), "quotes": _Feed(values)},
            watermarks={
                "trades": cf.SourceProvidedWatermarks(),
                "quotes": cf.SourceProvidedWatermarks(),
            },
        )
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with asyncio.timeout(10), results:
                _ = [table async for table in results]
        assert caught.value.category == "operator"
        assert caught.value.reason_code == reason
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_asof_chain_and_sql_execute_natively_with_three_logical_inputs() -> None:
    chained = _joined().stream_asof_join(
        _source("reference"),
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(1_000, 8_000_000),
    )
    sql = chained.sql(
        "SELECT left__trade__time, left__quote__price, right__price FROM input"
    )
    output, status = asyncio.run(
        _collect(
            sql,
            {
                "trades": _Feed([_rows([105])]),
                "quotes": _Feed([_rows([100])]),
                "reference": _Feed([_rows([102])]),
            },
        )
    )
    assert output["left__quote__price"].to_pylist() == [100.0]
    assert output["right__price"].to_pylist() == [102.0]
    assert len(status["stream_asof_joins"]) == 2


def _inner(left, right, *, ordered: bool):
    return cf.table.stream_join(
        left,
        right,
        left_keys=["symbol"],
        right_keys=["symbol"],
        left_event_time="time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=10), timedelta()),
        limits=cf.JoinStateLimits(1_000, 8_000_000, 10_000),
        **(
            {
                "output_entity_by": ["left__symbol"],
                "output_event_time": "left__time",
                "output_sequence_by": ["left__sequence", "right__sequence"],
            }
            if ordered
            else {}
        ),
    )


def test_inner_to_asof_executes_complete_composite_identity_in_arrival_disorder() -> (
    None
):
    inner = _inner(_source("trades"), _source("quotes"), ordered=True)
    asof = inner.stream_asof_join(
        _source("reference"),
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(1_000, 8_000_000),
    )
    inputs = {
        "trades": _Feed([_rows([105])]),
        "quotes": _Feed([_rows([101, 100], sequences=[2, 1])]),
        "reference": _Feed([_rows([104, 102], sequences=[3, 2])]),
    }
    output, _ = asyncio.run(
        _collect(
            asof,
            inputs,
            watermarks={name: cf.SourceProvidedWatermarks() for name in inputs},
        )
    )
    assert output["right__price"].to_pylist() == [104.0, 104.0]
    assert output["left__right__sequence"].to_pylist() == [1, 2]


def test_asof_to_inner_preserves_all_inner_matches() -> None:
    inner = cf.table.stream_join(
        _joined(),
        _source("reference"),
        left_keys=["trade__symbol"],
        right_keys=["symbol"],
        left_event_time="trade__time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=10), timedelta()),
        limits=cf.JoinStateLimits(1_000, 8_000_000, 10_000),
    )
    inputs = {
        "trades": _Feed([_rows([105])]),
        "quotes": _Feed([_rows([100])]),
        "reference": _Feed([_rows([102, 104])]),
    }
    output, _ = asyncio.run(_collect(inner, inputs))
    assert sorted(output["right__price"].to_pylist()) == [102.0, 104.0]
    assert output["left__quote__price"].to_pylist() == [100.0, 100.0]


def test_post_asof_rolling_executes_once_for_each_final_left_row() -> None:
    joined = _joined()
    output = joined.with_columns(previous=cf.ts.lag(joined["trade__price"]))
    table, _ = asyncio.run(
        _collect(
            output,
            {"trades": _Feed([_rows([105, 106])]), "quotes": _Feed([_rows([100])])},
        )
    )
    assert table["previous"].to_pylist() == [None, 105.0]
    assert table["quote__price"].to_pylist() == [100.0, 100.0]


@pytest.mark.parametrize("right", [[_rows([105])], []])
def test_asof_zero_tolerance_and_empty_right_keep_exact_left_semantics(right) -> None:
    table, _ = asyncio.run(
        _collect(_joined(0), {"trades": _Feed([_rows([105])]), "quotes": _Feed(right)})
    )
    assert table["quote__price"].to_pylist() == ([105.0] if right else [None])


def test_asof_empty_left_produces_no_results() -> None:
    async def run() -> None:
        results = _joined().stream(
            {"trades": _Feed([]), "quotes": _Feed([_rows([100])])}
        )
        async with asyncio.timeout(10), results:
            assert [table async for table in results] == []
        status = next(iter(results.job.status()["stream_asof_joins"].values()))
        assert status["emitted_left_rows"] == 0
        assert status["state_rows"] == 0

    asyncio.run(run())


def test_asof_source_rejects_null_identity_before_operator_admission() -> None:
    invalid = pa.Table.from_pydict(
        {"symbol": ["AAA"], "time": [100], "sequence": [None], "price": [10.0]},
        schema=_schema(),
    )

    async def run() -> None:
        results = _joined().stream(
            {"trades": _Feed([_rows([105])]), "quotes": _Feed([invalid])}
        )
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with asyncio.timeout(10), results:
                await anext(results)
        assert caught.value.category == "connector"
        assert caught.value.component_kind == "source"
        assert caught.value.reason_code != "asof_invalid_input"
        assert results.job.status()["task_count"] == 0
        asof = next(iter(results.job.status()["stream_asof_joins"].values()))
        assert asof["right"]["accepted_rows"] == 0

    asyncio.run(run())


def test_asof_state_limit_error_is_explicit_and_emits_no_wrong_match() -> None:
    result = _source("trades").stream_asof_join(
        _source("quotes"),
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(1, 8_000_000),
    )

    async def run() -> None:
        results = result.stream(
            {"trades": _Feed([_rows([105, 106])]), "quotes": _Feed([_rows([100])])}
        )
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with asyncio.timeout(10), results:
                await anext(results)
        assert caught.value.reason_code == "asof_state_limit_exceeded"
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_post_asof_cross_section_executes_on_final_same_time_group() -> None:
    joined = _joined()
    group = cf.CrossSectionGroup(joined["trade__time"], None, ())
    output = joined.with_columns(
        centered=cf.cs.demean(joined["trade__price"], group=group)
    )
    table, _ = asyncio.run(
        _collect(
            output,
            {
                "trades": _Feed([_rows([105, 105], prices=[110.0, 130.0])]),
                "quotes": _Feed([_rows([100])]),
            },
        )
    )
    assert table["centered"].to_pylist() == [-10.0, 10.0]
    assert table["quote__price"].to_pylist() == [100.0, 100.0]


def test_asof_program_stream_keeps_independent_event_window_and_bypass_outputs() -> (
    None
):
    independent = _source("events")
    windowed = cf.window.tumbling(
        independent,
        event_time="time",
        size_micros=10,
        aggregates=[cf.window.count("price", output="count")],
    )
    program = cf.Program(
        "independent",
        outputs={"matched": _joined(), "windowed": windowed, "bypass": independent},
    )

    async def run() -> None:
        results = program.stream(
            {
                "trades": _Feed([_rows([105])]),
                "quotes": _Feed([_rows([100])]),
                "events": _Feed([_rows([1, 2])]),
            }
        )
        async with asyncio.timeout(10), results:
            outputs = [value async for value in results]
        grouped = {
            name: pa.concat_tables(
                [value.table for value in outputs if value.name == name]
            )
            for name in ("matched", "windowed", "bypass")
        }
        assert grouped["matched"]["quote__price"].to_pylist() == [100.0]
        assert grouped["windowed"]["count"].to_pylist() == [2]
        assert grouped["bypass"]["price"].to_pylist() == [1.0, 2.0]

    asyncio.run(run())


class _BoundSource:
    def __init__(self, time: int) -> None:
        self.opened = 0
        self.closed = 0
        self.events = iter(
            [
                cf.Data(cf.Batch.from_pyarrow(_rows([time])), cf.Cursor(b"1", {})),
                _watermark(106),
            ]
        )

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.UNSUPPORTED,
            cf.SourceDeliveryCapability.LOSSY,
            max_batch_rows=1,
            max_batch_bytes=1_000_000,
            schema=_schema(),
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        assert cursor is None
        self.opened += 1

    async def next(self):
        return next(self.events, None)

    async def close(self) -> None:
        self.closed += 1


def test_asof_stream_keeps_source_binding_policies_and_rejects_overrides() -> None:
    left, right = _BoundSource(105), _BoundSource(100)
    policy = cf.SourceProvidedWatermarks()
    bindings = {
        "trades": cf.SourceBinding(left, watermark_policy=policy),
        "quotes": cf.SourceBinding(right, watermark_policy=policy),
    }

    async def reject_override() -> None:
        results = _joined().stream(
            bindings, watermarks={"trades": cf.DisabledWatermarks()}
        )
        async with results:
            await anext(results)

    with pytest.raises(ValueError, match="cannot override SourceBinding"):
        asyncio.run(reject_override())
    assert left.opened == right.opened == 0
    table, _ = asyncio.run(_collect(_joined(), bindings))
    assert table["quote__price"].to_pylist() == [100.0]
    assert left.opened == left.closed == right.opened == right.closed == 1
    assert bindings["trades"].source is left
    assert bindings["trades"].watermark_policy is policy


def test_asof_accepts_row_local_inputs_and_explicit_derived_key_override() -> None:
    trades, quotes = _source("trades"), _source("quotes")
    left = trades.with_columns(alias=trades["symbol"], adjusted=trades["price"] * 2.0)
    right = quotes.filter(quotes["price"] > 95.0)
    result = left.stream_asof_join(
        right,
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(1_000, 8_000_000),
        keys=(["alias"], ["symbol"]),
    )
    table, _ = asyncio.run(
        _collect(
            result,
            {"trades": _Feed([_rows([105])]), "quotes": _Feed([_rows([90, 100])])},
        )
    )
    assert table["left__adjusted"].to_pylist() == [210.0]
    assert table["right__price"].to_pylist() == [100.0]
