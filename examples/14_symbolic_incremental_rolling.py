"""Build and run rolling indicators backed by the native incremental operator."""

from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, Runtime
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    row,
    rows,
    table_input,
    ts,
)

WINDOW_ROWS = 3


def rolling_indicators(quotes: TableExpr) -> FeatureSet:
    """Declare indicators without reading or mutating the input table."""
    window = rows(WINDOW_ROWS)
    price_mean = ts.mean(quotes["close"], window=window)
    price_stddev = ts.stddev(
        quotes["close"],
        window=window,
        min_periods=2,
        ddof=1,
    )
    return FeatureSet(
        (
            ("close_mean_3", price_mean),
            ("close_stddev_3", price_stddev),
            ("volume_sum_3", ts.sum(quotes["volume"], window=window)),
            (
                "distance_from_mean",
                row.coalesce(quotes["close"] / price_mean - 1.0, 0.0),
            ),
        )
    )


def indicator_program() -> Program:
    """Create one ordered per-symbol rolling-indicator program."""
    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    signals = quotes.with_columns(rolling_indicators(quotes))
    return Program(
        "incremental-rolling-indicators",
        inputs=(quotes,),
        outputs=(("signals", signals),),
    )


def quote_table() -> pa.Table:
    """Return interleaved quotes with one null sample for the AAA partition."""
    schema = pa.schema(
        (
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64(), nullable=False),
        )
    )
    return pa.table(
        {
            "event_time": [
                1_000_000,
                1_000_000,
                2_000_000,
                2_000_000,
                3_000_000,
                3_000_000,
            ],
            "sequence": [0, 0, 1, 1, 2, 2],
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 50.0, None, 51.0, 103.0, 49.0],
            "volume": [10.0, 20.0, 30.0, 25.0, 15.0, 35.0],
        },
        schema=schema,
    )


def require(condition: bool, message: str) -> None:
    """Keep example verification active under regular and optimized Python."""
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    runtime = Runtime()
    program = indicator_program()
    analysis = program.analyze(runtime, mode="batch")
    require(analysis.issues == (), f"unexpected analysis issues: {analysis.issues}")

    print(
        "rolling configuration: partition=symbol, order=(event_time, sequence), rows=3"
    )
    print(program.explain(runtime, mode="batch"))

    output = (
        program.compile_batch(runtime)
        .execute({"input": Batch.from_pyarrow(quote_table())})
        .outputs["output"]
        .to_pyarrow()
    )
    indicator_names = [
        "close_mean_3",
        "close_stddev_3",
        "volume_sum_3",
        "distance_from_mean",
    ]
    require(output.num_rows == 6, f"unexpected output rows: {output.num_rows}")
    require(
        output.column_names[-4:] == indicator_names,
        f"unexpected output columns: {output.column_names}",
    )
    require(
        output["close_mean_3"].to_pylist() == [100.0, 50.0, 100.0, 50.5, 101.5, 50.0],
        f"unexpected rolling means: {output['close_mean_3'].to_pylist()}",
    )
    print(
        output.select(["event_time", "symbol", "close", *indicator_names]).to_pydict()
    )


if __name__ == "__main__":
    main()
