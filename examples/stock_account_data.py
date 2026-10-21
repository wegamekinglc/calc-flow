"""Shared Arrow trade fixtures for the position and account-value examples."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.compute as pc

TRADE_COUNT = 10_000_000
ACCOUNT_COUNT = 100
STOCK_COUNT = 100
BATCH_SIZE = 1_000

TRADE_SCHEMA = pa.schema(
    [
        ("account", pa.string()),
        ("stock", pa.string()),
        ("quantity", pa.int64()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("flow_type", pa.string()),
    ]
)


def _mod(values: pa.Array, divisor: int) -> pa.Array:
    return pc.subtract(values, pc.multiply(pc.divide(values, divisor), divisor))


def _trade_template(account_count: int, stock_count: int) -> pa.Table:
    pair_count = account_count * stock_count
    indices = pa.array(range(pair_count * 10), type=pa.int64())
    pair = _mod(indices, pair_count)
    cycle = pc.divide(indices, pair_count)
    accounts = pa.array([f"account-{index:03d}" for index in range(account_count)])
    stocks = pa.array([f"stock-{index:03d}" for index in range(stock_count)])
    buy = pc.equal(_mod(cycle, 2), 0)
    flow = _mod(pc.add(pair, cycle), 10)
    return pa.table(
        {
            "account": pc.take(accounts, pc.divide(pair, stock_count)),
            "stock": pc.take(stocks, _mod(pair, stock_count)),
            "quantity": pc.if_else(buy, 200, 100),
            "side": pc.if_else(buy, "buy", "sell"),
            "price": pa.repeat(10.0, len(indices)),
            "flow_type": pc.if_else(
                pc.less(flow, 8),
                "trade",
                pc.if_else(pc.equal(flow, 8), "order", "cancel"),
            ),
        },
        schema=TRADE_SCHEMA,
    )


def demo_batches(
    *,
    row_count: int = TRADE_COUNT,
    account_count: int = ACCOUNT_COUNT,
    stock_count: int = STOCK_COUNT,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pa.Table]:
    """Yield zero seeds, then row_count business flows from an Arrow template."""
    if row_count < 0 or min(account_count, stock_count, batch_size) <= 0:
        raise ValueError(
            "rows must be nonnegative; accounts, stocks and batch size positive"
        )
    template = _trade_template(account_count, stock_count)
    pair_count = account_count * stock_count
    initial = template.slice(0, pair_count)
    initial = initial.set_column(
        2, TRADE_SCHEMA.field("quantity"), pa.repeat(0, pair_count)
    )
    initial = initial.set_column(
        5, TRADE_SCHEMA.field("flow_type"), pa.repeat("trade", pair_count)
    )
    for offset in range(0, pair_count, batch_size):
        yield initial.slice(offset, batch_size)
    for start in range(0, row_count, template.num_rows):
        count = min(template.num_rows, row_count - start)
        for offset in range(0, count, batch_size):
            yield template.slice(offset, min(batch_size, count - offset))
