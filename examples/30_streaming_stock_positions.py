"""Calculate stock positions with native incremental streaming aggregation."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterator
from time import perf_counter

import pyarrow as pa
from stock_account_data import (
    ACCOUNT_COUNT,
    BATCH_SIZE,
    STOCK_COUNT,
    TRADE_COUNT,
    TRADE_SCHEMA,
    demo_batches,
)

import calc_flow as cf


def make_program() -> cf.Program:
    source = cf.table_input("trades", schema=TRADE_SCHEMA)
    trades = source.filter(source["flow_type"] == "trade")
    deltas = trades.select(
        "account",
        "stock",
        delta=cf.row.where(
            trades["side"] == "buy", trades["quantity"], -trades["quantity"]
        ),
    )
    positions = deltas.sql(
        "SELECT account, stock, SUM(delta) AS position "
        "FROM input GROUP BY account, stock"
    )
    return cf.Program("stock-positions", engine="streaming").output(
        "positions", positions
    )


async def trade_batches(batches: Iterator[pa.Table]) -> AsyncIterator[pa.Table]:
    """Replace this finite demo with your live async Arrow source."""
    for batch in batches:
        yield batch


async def run_stream(program: cf.Program, batches: Iterator[pa.Table]) -> pa.Table:
    last = None
    async with program.stream(
        {"trades": trade_batches(batches)},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        async for output in results:
            last = output.table
    if last is None:
        raise RuntimeError("stream produced no position snapshots")
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=TRADE_COUNT)
    parser.add_argument("--accounts", type=int, default=ACCOUNT_COUNT)
    parser.add_argument("--stocks", type=int, default=STOCK_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.rows < 0 or min(args.accounts, args.stocks, args.batch_size) <= 0:
        parser.error(
            "rows must be nonnegative; accounts, stocks and batch size positive"
        )
    started_at = perf_counter()
    batches = demo_batches(
        row_count=args.rows,
        account_count=args.accounts,
        stock_count=args.stocks,
        batch_size=args.batch_size,
    )
    result = asyncio.run(run_stream(make_program(), batches))
    elapsed = perf_counter() - started_at
    print(result)
    print(f"total elapsed: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
