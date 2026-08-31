"""Compose batch financial indicators with the symbolic declaration layer."""

from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, Runtime
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    cs,
    exact_time,
    row,
    rows,
    table_input,
    ts,
)


def financial_features(quotes: TableExpr) -> FeatureSet:
    """Build reusable declarations without reading or mutating any row data."""
    previous = ts.lag(quotes["price"])
    previous_log_price = ts.lag(row.log(quotes["price"]))
    momentum_1 = quotes["price"] / previous - 1.0
    log_return_1 = row.log(quotes["price"] / previous)
    price_mean_3 = ts.mean(quotes["price"], window=rows(3))
    price_stddev_3 = ts.stddev(
        quotes["price"],
        window=rows(3),
        min_periods=2,
        ddof=0,
    )
    volume_z = cs.zscore(
        quotes["volume"],
        group=exact_time(quotes["ts"]),
        min_samples=2,
    )
    return FeatureSet(
        (
            ("momentum_1", momentum_1),
            ("previous_log_price", previous_log_price),
            ("log_return_1", log_return_1),
            ("price_mean_3", price_mean_3),
            ("price_stddev_3", price_stddev_3),
            ("bollinger_upper_3", price_mean_3 + 2.0 * price_stddev_3),
            ("bollinger_lower_3", price_mean_3 - 2.0 * price_stddev_3),
            ("volume_z", volume_z),
            (
                "liquidity_adjusted_momentum",
                momentum_1 - volume_z * 0.01,
            ),
        )
    )


def require(condition: bool, message: str) -> None:
    """Keep example verification active under regular and optimized Python."""
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    quotes = table_input(
        "quotes",
        schema=(
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("price", "float64", nullable=False),
            Field("volume", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    program = Program(
        "symbolic-financial-features",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(financial_features(quotes))),),
    )
    runtime = Runtime()

    analysis = program.analyze(runtime, mode="batch")
    require(analysis.issues == (), f"unexpected analysis issues: {analysis.issues}")
    print(program.explain(runtime, mode="batch"))

    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
            pa.field("volume", pa.float64(), nullable=False),
        )
    )
    input_table = pa.table(
        {
            "ts": [1_000_000, 1_000_000, 2_000_000, 2_000_000],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "price": [100.0, 50.0, 102.0, 48.0],
            "volume": [10.0, 20.0, 30.0, 15.0],
        },
        schema=schema,
    )
    output = (
        program.compile_batch(runtime)
        .execute({"input": Batch.from_pyarrow(input_table)})
        .outputs["output"]
        .to_pyarrow()
    )

    require(output.num_rows == 4, f"unexpected output rows: {output.num_rows}")
    require(
        output.column_names[-9:]
        == [
            "momentum_1",
            "previous_log_price",
            "log_return_1",
            "price_mean_3",
            "price_stddev_3",
            "bollinger_upper_3",
            "bollinger_lower_3",
            "volume_z",
            "liquidity_adjusted_momentum",
        ],
        f"unexpected output columns: {output.column_names}",
    )
    print(output.select(output.column_names[-9:]).to_pydict())


if __name__ == "__main__":
    main()
