"""Push ordered rows and recover a caller-driven stream session."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa

import calc_flow as cf

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("seq", pa.uint64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


def rows(start: int, prices: list[float]) -> pa.Table:
    return pa.table(
        {
            "ts": [
                BASE + timedelta(seconds=start + offset)
                for offset in range(len(prices))
            ],
            "symbol": ["A"] * len(prices),
            "seq": list(range(start, start + len(prices))),
            "price": prices,
        },
        schema=SCHEMA,
    )


def mean_values(table: pa.Table) -> list[float]:
    return table["mean_price"].to_pylist()


def push_quotes(session: cf.StreamSession[pa.Table]) -> tuple[pa.Table, pa.Table]:
    first = session.push(rows(0, [10.0, 12.0]), watermark=BASE + timedelta(seconds=1))
    return first, session.push(rows(2, [15.0]))


def recover_buffered_quote(output: cf.TableExpr) -> tuple[int, pa.Table]:
    config = cf.StreamRuntimeConfig(checkpointing=True)
    watermarks = cf.SourceProvidedWatermarks()
    with TemporaryDirectory(prefix="calc-flow-session-") as directory:
        checkpoint_root = Path(directory) / "managed"
        with output.session(
            config=config,
            watermarks=watermarks,
            checkpoints=cf.ManagedCheckpointRuntime(checkpoint_root),
        ) as session:
            push_quotes(session)
            epoch = session.checkpoint()

        # Reopening the same declaration and storage restores the buffered quote.
        with output.session(
            config=config,
            watermarks=watermarks,
            checkpoints=cf.ManagedCheckpointRuntime(checkpoint_root),
            restore=True,
        ) as restored:
            return epoch, restored.finish()


def main() -> None:
    quotes = cf.table_input(
        "quotes",
        schema=SCHEMA,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("seq",),
    )
    output = quotes.select(
        mean_price=cf.ts.mean(quotes["price"], window=cf.rows(2), min_periods=1)
    )
    with output.session(watermarks=cf.SourceProvidedWatermarks()) as session:
        first, second = push_quotes(session)
        final = session.finish()

    actual = (mean_values(first), mean_values(second), mean_values(final))
    expected = ([10.0, 11.0], [], [13.5])
    if actual != expected:
        raise RuntimeError(f"unexpected session means: {actual}")
    print(f"first push: {actual[0]}")
    print(f"second push: {actual[1]}")
    print(f"finish: {actual[2]}")

    epoch, restored_final = recover_buffered_quote(output)
    restored_values = mean_values(restored_final)
    if epoch != 1 or restored_values != expected[2]:
        raise RuntimeError(f"unexpected recovery at epoch {epoch}: {restored_values}")
    print(f"checkpoint epoch: {epoch}")
    print(f"restored finish: {restored_values}")


if __name__ == "__main__":
    main()
