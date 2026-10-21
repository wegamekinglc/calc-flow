from __future__ import annotations

import asyncio
import itertools
import runpy
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).parents[2] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "31_streaming_account_market_values.py"))


def test_account_market_value_feeds_use_requested_default_scale(example):
    trades = example["demo_batches"]()
    initial = pa.concat_tables(list(itertools.islice(trades, 10)))
    assert initial.num_rows == 10_000
    assert pc.count_distinct(initial["account"]).as_py() == 100
    assert pc.count_distinct(initial["stock"]).as_py() == 100
    rows = 0
    for batch in trades:
        assert batch.num_rows == 1_000
        rows += batch.num_rows
    assert rows == 10_000_000
    quotes = list(example["quote_batches"]())
    assert len(quotes) == 10
    assert all(batch.num_rows == 1_000 for batch in quotes)
    assert pc.count_distinct(pa.concat_tables(quotes)["stock"]).as_py() == 100


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
