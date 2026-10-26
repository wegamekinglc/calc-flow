from __future__ import annotations

import asyncio
import runpy
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import calc_flow as cf


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).parents[2] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "31_streaming_account_market_values.py"))


def test_account_market_value_feeds_use_requested_default_scale(example):
    assert example["demo_batches"].__kwdefaults__ == {
        "row_count": 10_000_000,
        "account_count": 100,
        "stock_count": 100,
        "batch_size": 1_000,
    }
    assert example["quote_batches"].__kwdefaults__ == {
        "row_count": 10_000,
        "stock_count": 100,
        "batch_size": 1_000,
    }
    trades = list(
        example["demo_batches"](
            row_count=80, account_count=2, stock_count=2, batch_size=4
        )
    )
    assert trades[0]["quantity"].to_pylist() == [0] * 4
    assert sum(batch.num_rows for batch in trades[1:]) == 80
    assert all(batch.num_rows == 4 for batch in trades)
    assert pc.count_distinct(pa.concat_tables(trades)["account"]).as_py() == 2
    assert pc.count_distinct(pa.concat_tables(trades)["stock"]).as_py() == 2
    quotes = list(example["quote_batches"](row_count=20, stock_count=2, batch_size=4))
    assert len(quotes) == 5
    assert all(batch.num_rows == 4 for batch in quotes)
    assert pc.count_distinct(pa.concat_tables(quotes)["stock"]).as_py() == 2


def legacy_oracle(example):
    # The old two-input batch path independently aggregates the full histories.
    return (
        cf.PipelineBuilder("legacy-account-values")
        .sql(
            "values",
            """
            WITH deltas AS (
                SELECT account, stock,
                       CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta
                FROM trades WHERE flow_type = 'trade'
            ), p AS (
                SELECT account, stock, SUM(delta) AS position
                FROM deltas GROUP BY account, stock
            ), q AS (
                SELECT stock,
                       LAST_VALUE(latest_price ORDER BY "timestamp") AS latest_price
                FROM quotes GROUP BY stock
            )
            """
            + example["VALUATION_SQL"],
            aliases=("trades", "quotes"),
        )
        .compile_batch()
    )


def test_account_values_reprice_live_positions_before_either_source_ends(example):
    async def run():
        oracle = legacy_oracle(example)
        trade_input = asyncio.Queue()
        quote_input = asyncio.Queue()
        closed = {"trades": asyncio.Event(), "quotes": asyncio.Event()}

        async def feed(queue, name):
            try:
                while True:
                    yield await queue.get()
            finally:
                closed[name].set()

        def trade(quantity, side="buy", flow_type="trade"):
            return pa.table(
                {
                    "account": ["alice"],
                    "stock": ["AAA"],
                    "quantity": [quantity],
                    "side": [side],
                    "price": [1.0],
                    "flow_type": [flow_type],
                },
                schema=example["TRADE_SCHEMA"],
            )

        def quote(price, timestamp):
            return pa.table(
                {"stock": ["AAA"], "latest_price": [price], "timestamp": [timestamp]},
                schema=example["QUOTE_SCHEMA"],
            )

        history = {"trades": [trade(10)], "quotes": [quote(2.0, 1_000_000)]}
        await trade_input.put(history["trades"][0])
        await quote_input.put(history["quotes"][0])
        async with example["market_value_stream"](
            feed(trade_input, "trades"), feed(quote_input, "quotes")
        ) as updates:

            async def next_value():
                table = await asyncio.wait_for(updates.__anext__(), 5)
                assert not closed["trades"].is_set()
                assert not closed["quotes"].is_set()
                expected = await oracle.execute_async(
                    {
                        name: cf.Batch.from_pyarrow(pa.concat_tables(tables))
                        for name, tables in history.items()
                    }
                )
                assert table.equals(expected.outputs["output"].to_pyarrow())
                return table["market_value"][0].as_py()

            assert await next_value() == 20.0

            async def update(name, table):
                history[name].append(table)
                await (trade_input if name == "trades" else quote_input).put(table)
                return await next_value()

            # Price changes alone must revalue existing positions while trades wait.
            assert await update("quotes", quote(3.0, 3_000_000)) == 30.0
            # An old quote cannot replace the latest price.
            assert await update("quotes", quote(99.0, 2_000_000)) == 30.0
            # Trades also revalue immediately at the latest known price.
            assert await update("trades", trade(4, side="sell")) == 18.0
            assert await update("trades", trade(999, flow_type="order")) == 18.0
            assert await update("trades", trade(6, side="sell")) == 0.0

        assert closed["trades"].is_set()
        assert closed["quotes"].is_set()

    asyncio.run(run())


def test_account_values_initialize_from_empty_aggregate_snapshots(example):
    async def run():
        trades = pa.Table.from_batches([], schema=example["TRADE_SCHEMA"])
        quotes = pa.Table.from_batches([], schema=example["QUOTE_SCHEMA"])
        results = example["make_program"]().stream(
            {"trades": trades, "quotes": quotes},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        )
        async with results:
            outputs = [output async for output in results]
        assert results.job.status()["task_count"] == 0
        assert [output.name for output in outputs] == ["market_values"]
        assert outputs[0].table.column_names == ["account", "market_value"]
        assert outputs[0].table.num_rows == 0

    asyncio.run(run())


def test_account_values_use_latest_timestamp_after_trade_stream_ends(example):
    trade_tables = tuple(
        example["demo_batches"](
            row_count=80, account_count=2, stock_count=2, batch_size=4
        )
    )
    quote_tables = tuple(
        example["quote_batches"](row_count=20, stock_count=2, batch_size=4)
    )

    async def run():
        trades_done = asyncio.Event()
        quotes_done = asyncio.Event()

        async def trades():
            try:
                for table in trade_tables:
                    yield table
            finally:
                trades_done.set()

        async def quotes():
            try:
                yield quote_tables[0]
                await trades_done.wait()
                for table in quote_tables[1:]:
                    yield table
            finally:
                quotes_done.set()

        result = await example["account_market_values"](trades(), quotes())
        assert trades_done.is_set()
        assert quotes_done.is_set()
        return result

    result = asyncio.run(asyncio.wait_for(run(), 5))
    assert result.column_names == ["account", "market_value"]
    assert result["account"].equals(pa.chunked_array([["account-000", "account-001"]]))
    assert result["market_value"].equals(pa.chunked_array([[23400.0, 23400.0]]))


def test_zero_positions_have_zero_market_value(example):
    async def run():
        return await example["account_market_values"](
            example["batches"](
                example["demo_batches"](
                    row_count=0, account_count=2, stock_count=2, batch_size=4
                )
            ),
            example["batches"](
                example["quote_batches"](row_count=20, stock_count=2, batch_size=4)
            ),
        )

    result = asyncio.run(asyncio.wait_for(run(), 5))
    assert result["account"].equals(pa.chunked_array([["account-000", "account-001"]]))
    assert result["market_value"].equals(pa.chunked_array([[0.0, 0.0]]))


def test_account_values_cli_prints_final_accounts_and_elapsed(
    example, monkeypatch, capsys
):
    clock = iter((10.0, 12.5))
    monkeypatch.setitem(
        example["main"].__globals__, "perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "31_streaming_account_market_values.py",
            "--rows",
            "80",
            "--accounts",
            "2",
            "--stocks",
            "2",
            "--batch-size",
            "4",
            "--quote-rows",
            "20",
        ],
    )
    example["main"]()
    output = capsys.readouterr().out
    assert output.count("pyarrow.Table") == 1
    assert "market_value: [[23400,23400]]" in output
    assert output.rstrip().endswith("total elapsed: 2.500 s")
