"""Stream positions and quotes, then print each account's final market value."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from time import perf_counter

import pyarrow as pa
import pyarrow.compute as pc
from stock_account_data import (
    ACCOUNT_COUNT,
    BATCH_SIZE,
    STOCK_COUNT,
    TRADE_COUNT,
    TRADE_SCHEMA,
    demo_batches,
)

import calc_flow as cf

QUOTE_COUNT = 10_000
QUOTE_SCHEMA = pa.schema(
    [
        ("stock", pa.string()),
        ("latest_price", pa.float64()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
    ]
)
POSITION_SCHEMA = pa.schema(
    [("account", pa.string()), ("stock", pa.string()), ("position", pa.int64())]
)
LATEST_QUOTE_SCHEMA = pa.schema(
    [("stock", pa.string()), ("latest_price", pa.float64())]
)


def make_program() -> cf.Program:
    flows = cf.table_input("trades", schema=TRADE_SCHEMA)
    trades = flows.filter(flows["flow_type"] == "trade")
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
    quotes = cf.table_input("quotes", schema=QUOTE_SCHEMA)
    latest = quotes.sql(
        'SELECT stock, LAST_VALUE(latest_price ORDER BY "timestamp") AS latest_price '
        "FROM input GROUP BY stock"
    )
    return cf.Program(
        "account-market-values",
        engine="streaming",
        outputs={"positions": positions, "latest_quotes": latest},
    )


async def account_market_values(
    trades: AsyncIterator[pa.Table], quotes: AsyncIterator[pa.Table]
) -> pa.Table:
    snapshots = {
        "positions": pa.Table.from_batches([], schema=POSITION_SCHEMA),
        "latest_quotes": pa.Table.from_batches([], schema=LATEST_QUOTE_SCHEMA),
    }
    async with make_program().stream(
        {"trades": trades, "quotes": quotes},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        async for output in results:
            snapshots[output.name] = output.table

    # Streaming SQL accepts one alias. Join the two final Arrow snapshots at EOF.
    values = cf.sql(
        """
        SELECT p.account,
               SUM(p.position * q.latest_price) AS market_value
        FROM p LEFT JOIN q ON p.stock = q.stock
        GROUP BY p.account
        ORDER BY p.account
        """,
        p=cf.table_input("positions", schema=snapshots["positions"].schema),
        q=cf.table_input("latest_quotes", schema=snapshots["latest_quotes"].schema),
    )
    return await values.collect_async(snapshots)


def quote_batches(
    *,
    row_count: int = QUOTE_COUNT,
    stock_count: int = STOCK_COUNT,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pa.Table]:
    if row_count < 0 or min(stock_count, batch_size) <= 0:
        raise ValueError(
            "rows must be nonnegative; stocks and batch size must be positive"
        )

    stocks = pa.array([f"stock-{index:03d}" for index in range(stock_count)])
    rounds = row_count // stock_count
    for offset in range(0, row_count, batch_size):
        indices = pa.array(
            range(offset, min(offset + batch_size, row_count)), type=pa.int64()
        )
        cycle = pc.divide(indices, stock_count)
        stock_index = pc.subtract(indices, pc.multiply(cycle, stock_count))
        if rounds >= 2:
            # Deliver an older quote last to exercise timestamp ordering.
            cycle = pc.if_else(
                pc.equal(cycle, rounds - 2),
                rounds - 1,
                pc.if_else(pc.equal(cycle, rounds - 1), rounds - 2, cycle),
            )
        yield pa.table(
            {
                "stock": pc.take(stocks, stock_index),
                "latest_price": pc.add(
                    pc.add(10.0, pc.multiply(cycle, 0.5)),
                    pc.multiply(stock_index, 0.25),
                ),
                "timestamp": pc.multiply(pc.add(cycle, 1), 1_000_000),
            },
            schema=QUOTE_SCHEMA,
        )


async def batches(tables: Iterable[pa.Table]) -> AsyncIterator[pa.Table]:
    for table in tables:
        yield table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=TRADE_COUNT)
    parser.add_argument("--accounts", type=int, default=ACCOUNT_COUNT)
    parser.add_argument("--stocks", type=int, default=STOCK_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quote-rows", type=int, default=QUOTE_COUNT)
    args = parser.parse_args()
    if args.rows < 0 or min(args.accounts, args.stocks, args.batch_size) <= 0:
        parser.error(
            "rows must be nonnegative; accounts, stocks and batch size must be positive"
        )
    if args.quote_rows < args.stocks:
        parser.error("quote rows must cover every stock")

    started_at = perf_counter()
    trades = demo_batches(
        row_count=args.rows,
        account_count=args.accounts,
        stock_count=args.stocks,
        batch_size=args.batch_size,
    )
    quotes = quote_batches(
        row_count=args.quote_rows, stock_count=args.stocks, batch_size=args.batch_size
    )
    result = asyncio.run(account_market_values(batches(trades), batches(quotes)))
    elapsed = perf_counter() - started_at
    print(result)
    print(f"total elapsed: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
