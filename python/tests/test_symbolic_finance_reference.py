"""Finance-inspired public symbolic acceptance vectors.

The fixture families are independently derived from the rolling and
cross-section coverage in ``alpha-miner/Finance-Python`` at commit
``3e33d3e70c3458b4c6dcf76b88df6148229b402c``. Calc Flow keeps its frozen
semantics: percentile is ``(rank - 1) / (n - 1)`` with singleton ``0.5``,
Arrow null is distinct from NaN, and declarations compile to native plans
rather than mutable ``push``/``value`` holders. CI never fetches or imports the
external project.
"""

from __future__ import annotations

import asyncio
import math

import pyarrow as pa
import pytest

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


def _ordered_quotes(*, with_sector: bool = False) -> TableExpr:
    fields = [
        Field("ts", "timestamp[us, UTC]", nullable=False),
        Field("symbol", "string", nullable=False),
        Field("seq", "uint64", nullable=False),
    ]
    if with_sector:
        fields.append(Field("sector", "string", nullable=False))
    fields.append(Field("price", "float64"))
    return table_input(
        "quotes",
        schema=tuple(fields),
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("seq",),
    )


def _execute(program: Program, table: pa.Table) -> pa.Table:
    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(table)}
    )
    return result.outputs["output"].to_pyarrow()


def _assert_optional_floats(
    actual: list[float | None], expected: list[float | None]
) -> None:
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert observed is None
        elif math.isnan(reference):
            assert observed is not None and math.isnan(observed)
        else:
            assert observed == pytest.approx(reference, rel=1e-10, abs=1e-12)


def _temporal_reference(
    rows_: list[tuple[int, str, int, float]],
) -> dict[str, list[float | None]]:
    history: dict[str, list[float]] = {}
    expected = {
        "previous_log_price": [],
        "simple_return": [],
        "log_return_2": [],
        "momentum_2": [],
        "mean_3": [],
        "stddev_3": [],
        "bollinger_upper": [],
        "bollinger_lower": [],
    }
    for _, symbol, _, price in rows_:
        prices = history.setdefault(symbol, [])
        values = _temporal_row(prices, price)
        for name, value in values.items():
            expected[name].append(value)
        prices.append(price)
    return expected


def _temporal_row(prices: list[float], price: float) -> dict[str, float | None]:
    previous = prices[-1] if prices else None
    previous_2 = prices[-2] if len(prices) >= 2 else None
    mean, stddev = _rolling_mean_std((*prices[-2:], price))
    upper, lower = _bollinger_bounds(mean, stddev)
    return {
        "previous_log_price": None if previous is None else math.log(previous),
        "simple_return": _optional_return(price, previous),
        "log_return_2": _optional_return(price, previous_2, logarithmic=True),
        "momentum_2": _optional_return(price, previous_2),
        "mean_3": mean,
        "stddev_3": stddev,
        "bollinger_upper": upper,
        "bollinger_lower": lower,
    }


def _optional_return(
    price: float,
    previous: float | None,
    *,
    logarithmic: bool = False,
) -> float | None:
    if previous is None:
        return None
    ratio = price / previous
    return math.log(ratio) if logarithmic else ratio - 1.0


def _rolling_mean_std(sample: tuple[float, ...]) -> tuple[float, float | None]:
    mean = math.fsum(sample) / len(sample)
    if len(sample) < 2:
        return mean, None
    variance = math.fsum((value - mean) ** 2 for value in sample) / len(sample)
    return mean, math.sqrt(variance)


def _bollinger_bounds(
    mean: float, stddev: float | None
) -> tuple[float | None, float | None]:
    if stddev is None:
        return None, None
    return mean + 2.0 * stddev, mean - 2.0 * stddev


def test_finance_style_returns_momentum_and_bollinger_match_reference() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    previous = ts.lag(price)
    previous_2 = ts.lag(price, periods=2)
    mean_3 = ts.mean(price, window=rows(3))
    stddev_3 = ts.stddev(price, window=rows(3), min_periods=2, ddof=0)
    features = FeatureSet(
        (
            ("simple_return", price / previous - 1.0),
            ("previous_log_price", ts.lag(row.log(price))),
            ("log_return_2", row.log(price / previous_2)),
            ("momentum_2", price / previous_2 - 1.0),
            ("mean_3", mean_3),
            ("stddev_3", stddev_3),
            ("bollinger_upper", mean_3 + 2.0 * stddev_3),
            ("bollinger_lower", mean_3 - 2.0 * stddev_3),
        )
    )
    program = Program(
        "finance-temporal-reference",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(features)),),
    )
    input_rows = [
        (1_000_000, "AAA", 1, 100.0),
        (1_000_000, "BBB", 1, 50.0),
        (2_000_000, "AAA", 2, 110.0),
        (2_000_000, "BBB", 2, 45.0),
        (3_000_000, "AAA", 3, 121.0),
        (3_000_000, "BBB", 3, 49.5),
        (4_000_000, "AAA", 4, 133.1),
        (4_000_000, "BBB", 4, 54.45),
    ]
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [item[0] for item in input_rows],
            "symbol": [item[1] for item in input_rows],
            "seq": [item[2] for item in input_rows],
            "price": [item[3] for item in input_rows],
        },
        schema=schema,
    )

    output = _execute(program, table)
    expected = _temporal_reference(input_rows)

    assert output["symbol"].to_pylist() == [item[1] for item in input_rows]
    for name, values in expected.items():
        _assert_optional_floats(output[name].to_pylist(), values)


def test_finance_style_ewma_and_macd_match_independent_reference() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    ema = ts.ema(price, span=3, min_periods=2)
    assert ema.digest == ts.ewma(price, span=3, min_periods=2).digest
    macd = ts.macd(price, fast_span=2, slow_span=4)
    features = FeatureSet(
        (
            ("ema_3", ema),
            ("macd_2_4", macd),
            ("ema_macd_3", ts.ema(macd, span=3)),
        )
    )
    program = Program(
        "finance-exponential-reference",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(features)),),
    )
    prices = [10.0, None, 14.0, math.nan, 18.0, 10.0]
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [index * 1_000_000 for index in range(1, len(prices) + 1)],
            "symbol": ["AAA"] * len(prices),
            "seq": list(range(1, len(prices) + 1)),
            "price": prices,
        },
        schema=schema,
    )

    output = _execute(program, table)

    _assert_optional_floats(
        output["ema_3"].to_pylist(),
        [None, None, 12.0, 12.0, 15.0, 12.5],
    )
    _assert_optional_floats(
        output["macd_2_4"].to_pylist(),
        [
            0.0,
            0.0,
            16.0 / 15.0,
            16.0 / 15.0,
            464.0 / 225.0,
            -1424.0 / 3375.0,
        ],
    )
    _assert_optional_floats(
        output["ema_macd_3"].to_pylist(),
        [0.0, 0.0, 8.0 / 15.0, 4.0 / 5.0, 322.0 / 225.0, 1703.0 / 3375.0],
    )


def _column_for_symbol(output: pa.Table, column: str, symbol: str) -> list[object]:
    return [
        output[column][index].as_py()
        for index, candidate in enumerate(output["symbol"].to_pylist())
        if candidate == symbol
    ]


def _ewma_reference(
    prices: list[float | None], span: int, min_periods: int
) -> list[float | None]:
    # Independent unadjusted recurrence: alpha = 2 / (span + 1), the first
    # valid sample seeds the accumulator, null/NaN inputs are ignored, and
    # outputs stay null until min_periods valid samples exist.
    alpha = 2.0 / (span + 1.0)
    valid_seen = 0
    accumulator: float | None = None
    expected: list[float | None] = []
    for price in prices:
        if price is not None and not math.isnan(price):
            valid_seen += 1
            accumulator = (
                price
                if accumulator is None
                else accumulator + alpha * (price - accumulator)
            )
        expected.append(accumulator if valid_seen >= min_periods else None)
    return expected


def test_finance_style_ewma_isolates_entities_under_interleaving() -> None:
    quotes = _ordered_quotes()
    features = FeatureSet((("ema_3", ts.ewma(quotes["price"], span=3, min_periods=2)),))
    program = Program(
        "finance-ewma-entity-isolation",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(features)),),
    )
    # Three entities with distinct price paths (including null and NaN gaps)
    # arrive round-robin so any shared-accumulator leak crosses entities.
    paths = {
        "AAA": [10.0, None, 14.0, math.nan, 18.0],
        "BBB": [100.0, 90.0, 95.0, 105.0, None],
        "CCC": [-5.0, 5.0, None, None, 7.0],
    }
    input_rows = []
    for index in range(5):
        for symbol, prices in paths.items():
            input_rows.append(
                ((index + 1) * 1_000_000, symbol, index + 1, prices[index])
            )
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [row[0] for row in input_rows],
            "symbol": [row[1] for row in input_rows],
            "seq": [row[2] for row in input_rows],
            "price": [row[3] for row in input_rows],
        },
        schema=schema,
    )

    output = _execute(program, table)

    for symbol, prices in paths.items():
        _assert_optional_floats(
            _column_for_symbol(output, "ema_3", symbol),  # type: ignore[arg-type]
            _ewma_reference(prices, 3, 2),
        )


def test_finance_style_rsi_composition_matches_independent_reference() -> None:
    quotes = _ordered_quotes()
    change = ts.delta(quotes["price"])
    gain = row.where(change > 0.0, change, 0.0)
    loss = row.where(change < 0.0, -change, 0.0)
    average_gain = ts.mean(gain, window=rows(3))
    average_loss = ts.mean(loss, window=rows(3))
    rsi = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    features = FeatureSet(
        (
            ("change", change),
            ("average_gain", average_gain),
            ("average_loss", average_loss),
            ("rsi_3", rsi),
        )
    )
    program = Program(
        "finance-rsi-reference",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(features)),),
    )
    prices = [100.0, 110.0, 105.0, 115.0, 100.0, 105.0]
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [index * 1_000_000 for index in range(1, len(prices) + 1)],
            "symbol": ["AAA"] * len(prices),
            "seq": list(range(1, len(prices) + 1)),
            "price": prices,
        },
        schema=schema,
    )

    output = _execute(program, table)

    assert output["change"].to_pylist() == [None, 10.0, -5.0, 10.0, -15.0, 5.0]
    _assert_optional_floats(
        output["average_gain"].to_pylist(),
        [0.0, 5.0, 10.0 / 3.0, 20.0 / 3.0, 10.0 / 3.0, 5.0],
    )
    _assert_optional_floats(
        output["average_loss"].to_pylist(),
        [0.0, 0.0, 5.0 / 3.0, 5.0 / 3.0, 20.0 / 3.0, 5.0],
    )
    _assert_optional_floats(
        output["rsi_3"].to_pylist(),
        [math.nan, 100.0, 200.0 / 3.0, 80.0, 100.0 / 3.0, 50.0],
    )


def test_finance_style_cross_section_uses_calc_flow_frozen_semantics() -> None:
    quotes = _ordered_quotes(with_sector=True)
    group = exact_time(quotes["ts"], partition_by=(quotes["sector"],))
    features = FeatureSet(
        (
            ("rank", cs.rank(quotes["price"], group=group)),
            ("percentile", cs.percentile(quotes["price"], group=group)),
            ("demean", cs.demean(quotes["price"], group=group)),
            ("zscore", cs.zscore(quotes["price"], group=group, ddof=0)),
            ("top", cs.top(quotes["price"], group=group, count=2)),
            (
                "bottom",
                cs.bottom(quotes["price"], group=group, count=1, include_ties=False),
            ),
            ("filled", cs.mean_fill(quotes["price"], group=group)),
        )
    )
    program = Program(
        "finance-cross-section-reference",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(features)),),
    )
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("sector", pa.string(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [1_000_000] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "seq": [1] * 6,
            "sector": ["multi", "multi", "multi", "multi", "single", "nan"],
            "price": [1.0, 2.0, 4.0, None, 9.0, math.nan],
        },
        schema=schema,
    )

    output = _execute(program, table)
    order = output["symbol"].to_pylist()
    by_symbol = {
        symbol: {
            name: output[name][index].as_py()
            for name in (
                "rank",
                "percentile",
                "demean",
                "zscore",
                "top",
                "bottom",
                "filled",
            )
        }
        for index, symbol in enumerate(order)
    }
    root_14 = math.sqrt(14.0)
    expected_multi = {
        "A": (1.0, 0.0, -4.0 / 3.0, -4.0 / root_14, False, True, 1.0),
        "B": (2.0, 0.5, -1.0 / 3.0, -1.0 / root_14, True, False, 2.0),
        "C": (3.0, 1.0, 5.0 / 3.0, 5.0 / root_14, True, False, 4.0),
        "D": (None, None, None, None, None, None, 7.0 / 3.0),
    }
    fields = ("rank", "percentile", "demean", "zscore", "top", "bottom", "filled")
    for symbol, expected in expected_multi.items():
        for field, reference in zip(fields, expected, strict=True):
            actual = by_symbol[symbol][field]
            if isinstance(reference, float):
                assert actual == pytest.approx(reference, rel=1e-10, abs=1e-12)
            else:
                assert actual is reference

    assert by_symbol["E"] == {
        "rank": 1.0,
        "percentile": 0.5,
        "demean": 0.0,
        "zscore": None,
        "top": True,
        "bottom": True,
        "filled": 9.0,
    }
    assert all(by_symbol["F"][name] is None for name in ("top", "bottom"))
    assert all(
        math.isnan(by_symbol["F"][name])
        for name in ("rank", "percentile", "demean", "zscore", "filled")
    )


def test_cross_section_mean_broadcasts_valid_group_sample() -> None:
    quotes = _ordered_quotes()
    group = exact_time(quotes["ts"])
    program = Program(
        "cross-section-mean",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    mean=cs.mean(quotes["price"], group=group, min_samples=3),
                    insufficient=cs.mean(quotes["price"], group=group, min_samples=4),
                ),
            ),
        ),
    )
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [1_000_000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "seq": [1] * 5,
            "price": [1.0, 2.0, 4.0, None, math.nan],
        },
        schema=schema,
    )

    output = _execute(program, table)
    assert output["mean"].to_pylist() == pytest.approx([7.0 / 3.0] * 5)
    assert output["insufficient"].to_pylist() == [None] * 5
    program.compile_stream(Runtime())


def test_cross_section_fraction_selection_uses_average_tie_rank() -> None:
    quotes = _ordered_quotes()
    group = exact_time(quotes["ts"])
    program = Program(
        "cross-section-fraction-selection",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    top_half=cs.top_quantile(
                        quotes["price"], group=group, fraction=0.5
                    ),
                    bottom_half=cs.bottom_quantile(
                        quotes["price"], group=group, fraction=0.5
                    ),
                ),
            ),
        ),
    )
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [1_000_000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "seq": [1] * 5,
            "price": [1.0, 2.0, 2.0, 4.0, None],
        },
        schema=schema,
    )

    output = _execute(program, table)
    assert output["top_half"].to_pylist() == [False, False, False, True, None]
    assert output["bottom_half"].to_pylist() == [True, False, False, False, None]
    program.compile_stream(Runtime())


def test_rolling_boolean_reductions_ignore_null_samples() -> None:
    quotes = _ordered_quotes()
    flag = quotes["price"] > 0.0
    program = Program(
        "rolling-boolean-reductions",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    all_true=ts.all_true(flag, window=rows(2)),
                    any_true=ts.any_true(flag, window=rows(2)),
                ),
            ),
        ),
    )
    schema = pa.schema(
        (
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.float64(), nullable=True),
        )
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            "symbol": ["A"] * 5,
            "seq": [1, 2, 3, 4, 5],
            "price": [0.0, None, None, 1.0, 2.0],
        },
        schema=schema,
    )

    output = _execute(program, table)
    assert output["all_true"].to_pylist() == [False, False, None, True, True]
    assert output["any_true"].to_pylist() == [False, False, None, True, True]
    program.compile_stream(Runtime())


def test_positive_window_reductions_ignore_nonpositive_and_missing_values() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    program = Program(
        "positive-window-reductions",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    positive_count=ts.count_positive(price, window=rows(3)),
                    positive_mean=ts.mean_positive(price, window=rows(3)),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000],
            "symbol": ["A"] * 6,
            "seq": [1, 2, 3, 4, 5, 6],
            "price": [-2.0, None, 2.0, math.nan, 4.0, 0.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    assert output["positive_count"].to_pylist() == [0, 0, 1, 1, 2, 1]
    _assert_optional_floats(
        output["positive_mean"].to_pylist(),
        [0.0, 0.0, 2.0, 2.0, 3.0, 4.0],
    )
    program.compile_stream(Runtime())


def test_cross_section_regression_residual_uses_pairwise_valid_rows() -> None:
    quotes = _ordered_quotes(with_sector=True)
    group = exact_time(quotes["ts"], partition_by=(quotes["sector"],))
    program = Program(
        "cross-section-regression-residual",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    residual=cs.residual(quotes["price"], quotes["seq"], group=group),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "seq": [1, 2, 3, 4, 5, 6],
            "sector": ["tech"] * 6,
            "price": [3.0, 5.0, 8.0, None, 11.0, math.nan],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("sector", pa.string(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    _assert_optional_floats(
        output["residual"].to_pylist(),
        [-7 / 35, -8 / 35, 26 / 35, None, -11 / 35, None],
    )
    program.compile_stream(Runtime())

    async def feed():
        yield table.slice(0, 2)
        yield table.slice(2)

    async def stream_residuals() -> list[float | None]:
        values: list[float | None] = []
        async with program.stream({"quotes": feed()}) as results:
            async for event in results:
                values.extend(event.table["residual"].to_pylist())
        return values

    _assert_optional_floats(
        asyncio.run(stream_residuals()), output["residual"].to_pylist()
    )


def test_cross_section_regression_residual_centers_large_predictors() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "cross-section-centered-regression",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    residual=cs.residual(
                        quotes["seq"],
                        quotes["price"],
                        group=exact_time(quotes["ts"]),
                    )
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "seq": [1, 2, 3, 4],
            "price": [1e12 + 1, 1e12 + 2, 1e12 + 3, 1e12 + 5],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)
    _assert_optional_floats(
        result["residual"].to_pylist(), [-7 / 35, 2 / 35, 11 / 35, -6 / 35]
    )


@pytest.mark.parametrize("magnitude", [1e308, 1e-200])
def test_cross_section_regression_residual_handles_opposite_extreme_values(
    magnitude: float,
) -> None:
    quotes = _ordered_quotes()
    program = Program(
        "cross-section-extreme-regression",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    residual=cs.residual(
                        quotes["price"], quotes["price"], group=exact_time(quotes["ts"])
                    )
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 1_000_000],
            "symbol": ["A", "B"],
            "seq": [1, 2],
            "price": [magnitude, -magnitude],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    assert _execute(program, table)["residual"].to_pylist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("x_scale", "dependent_value"), [(1e-100, 1e-250), (1e-150, 1e308)]
)
def test_cross_section_regression_residual_handles_extreme_scales(
    x_scale: float, dependent_value: float
) -> None:
    quotes = _ordered_quotes()
    program = Program(
        "cross-section-underflowed-covariance",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    residual=cs.residual(
                        quotes["price"],
                        row.cast(quotes["seq"], "float64") * x_scale,
                        group=exact_time(quotes["ts"]),
                    )
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 1_000_000],
            "symbol": ["A", "B"],
            "seq": [1, 2],
            "price": [0.0, dependent_value],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    assert _execute(program, table)["residual"].to_pylist() == [0.0, 0.0]


def test_rolling_order_unique_and_decay_match_reference() -> None:
    quotes = _ordered_quotes()
    value = quotes["price"]
    frame = rows(3)
    program = Program(
        "rolling-order-unique-decay",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    argmax=ts.argmax(value, window=frame),
                    argmin=ts.argmin(value, window=frame),
                    rank=ts.rank(value, window=frame),
                    quantile=ts.quantile(value, window=frame),
                    unique_count=ts.unique_count(value, window=frame),
                    decay=ts.decay(value, window=frame),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [index * 1_000_000 for index in range(1, 7)],
            "symbol": ["A"] * 6,
            "seq": list(range(1, 7)),
            "price": [2.0, 3.0, 3.0, 1.0, None, 4.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    assert output["argmax"].to_pylist() == [0, 0, 1, 2, 2, 0]
    assert output["argmin"].to_pylist() == [0, 1, 2, 0, 1, 2]
    assert output["rank"].to_pylist() == [0, 1, 1, 0, None, 1]
    _assert_optional_floats(
        output["quantile"].to_pylist(), [None, 1.0, 0.5, 0.0, None, 1.0]
    )
    assert output["unique_count"].to_pylist() == [1, 2, 2, 2, 2, 2]
    _assert_optional_floats(
        output["decay"].to_pylist(),
        [2.0, 13 / 5, 17 / 6, 2.0, 9 / 5, 14 / 5],
    )
    program.compile_stream(Runtime())

    async def feed():
        yield table.slice(0, 2)
        yield table.slice(2, 1)
        yield table.slice(3)

    async def stream_columns() -> dict[str, list[object]]:
        columns = {
            name: []
            for name in (
                "argmax",
                "argmin",
                "rank",
                "quantile",
                "unique_count",
                "decay",
            )
        }
        async with program.stream({"quotes": feed()}) as results:
            async for event in results:
                for name in columns:
                    columns[name].extend(event.table[name].to_pylist())
        return columns

    streamed = asyncio.run(stream_columns())
    for name in streamed:
        _assert_optional_floats(streamed[name], output[name].to_pylist())


def test_row_finance_math_functions_execute_in_batch_and_compile_stream() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    program = Program(
        "row-finance-math",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    sign=row.sign(price),
                    maximum=row.maximum(price, 0.0),
                    minimum=row.minimum(price, 0.0),
                    square=row.pow(price, 2.0),
                    acos=row.acos(price),
                    acosh=row.acosh(price),
                    asin=row.asin(price),
                    asinh=row.asinh(price),
                    ceil=row.ceil(price),
                    floor=row.floor(price),
                    rounded=row.round(price),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "symbol": ["A"] * 4,
            "seq": [1, 2, 3, 4],
            "price": [-2.0, 0.5, 2.0, None],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    assert output["sign"].to_pylist() == [-1.0, 1.0, 1.0, None]
    assert output["maximum"].to_pylist() == [0.0, 0.5, 2.0, 0.0]
    assert output["minimum"].to_pylist() == [-2.0, 0.0, 0.0, 0.0]
    assert output["square"].to_pylist() == [4.0, 0.25, 4.0, None]
    _assert_optional_floats(
        output["acos"].to_pylist(), [math.nan, math.acos(0.5), math.nan, None]
    )
    _assert_optional_floats(
        output["acosh"].to_pylist(), [math.nan, math.nan, math.acosh(2), None]
    )
    _assert_optional_floats(
        output["asin"].to_pylist(), [math.nan, math.asin(0.5), math.nan, None]
    )
    _assert_optional_floats(
        output["asinh"].to_pylist(),
        [math.asinh(-2), math.asinh(0.5), math.asinh(2), None],
    )
    assert output["ceil"].to_pylist() == [-2.0, 1.0, 2.0, None]
    assert output["floor"].to_pylist() == [-2.0, 0.0, 2.0, None]
    assert output["rounded"].to_pylist() == [-2.0, 1.0, 2.0, None]
    program.compile_stream(Runtime())


def test_row_finance_math_accepts_integer_columns() -> None:
    quotes = _ordered_quotes()
    sequence = quotes["seq"]
    program = Program(
        "row-integer-math",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    ceiling=row.ceil(sequence),
                    floor_value=row.floor(sequence),
                    rounded=row.round(sequence),
                    arccos=row.acos(sequence),
                    arcsin=row.asin(sequence),
                    arccosh=row.acosh(sequence),
                    arcsinh=row.asinh(sequence),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000],
            "symbol": ["A", "A"],
            "seq": [1, 2],
            "price": [1.0, 2.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    for name in ("ceiling", "floor_value", "rounded"):
        assert output[name].to_pylist() == [1.0, 2.0]
    _assert_optional_floats(output["arccos"].to_pylist(), [0.0, math.nan])
    _assert_optional_floats(output["arcsin"].to_pylist(), [math.pi / 2, math.nan])
    _assert_optional_floats(output["arccosh"].to_pylist(), [0.0, math.acosh(2)])
    _assert_optional_floats(
        output["arcsinh"].to_pylist(), [math.asinh(1), math.asinh(2)]
    )


def test_inverse_normal_cdf_matches_reference_probabilities() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "inverse-normal-cdf",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    normal_score=row.norminv(quotes["price"]),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "symbol": ["A"] * 4,
            "seq": [1, 2, 3, 4],
            "price": [0.025, 0.5, 0.975, None],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)["normal_score"].to_pylist()
    assert result[0] == pytest.approx(-1.959963984540054, abs=1e-8)
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[2] == pytest.approx(1.959963984540054, abs=1e-8)
    assert result[3] is None
    program.compile_stream(Runtime())


def test_inverse_normal_cdf_handles_domain_edges_and_tails() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "inverse-normal-domain",
        inputs=(quotes,),
        outputs=(("signals", quotes.with_columns(score=row.norminv(quotes["price"]))),),
    )
    probabilities = [-0.1, 0.0, 0.001, 0.999, 1.0, 1.1, math.nan, None]
    table = pa.table(
        {
            "ts": [index * 1_000_000 for index in range(1, 9)],
            "symbol": ["A"] * 8,
            "seq": list(range(1, 9)),
            "price": probabilities,
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    actual = _execute(program, table)["score"].to_pylist()
    assert actual[0] is None and actual[1] is None
    assert actual[2] == pytest.approx(-3.090232306167813, abs=1e-8)
    assert actual[3] == pytest.approx(3.090232306167813, abs=1e-8)
    assert actual[4] is None and actual[5] is None
    assert actual[6] is None and actual[7] is None


def test_pairwise_extrema_skip_nan_and_null_on_either_side() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    program = Program(
        "pairwise-extrema",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    left_max=row.maximum(price, 0.0),
                    right_max=row.maximum(0.0, price),
                    left_min=row.minimum(price, 0.0),
                    right_min=row.minimum(0.0, price),
                    both_max=row.maximum(price, price),
                    both_min=row.minimum(price, price),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "symbol": ["A"] * 4,
            "seq": [1, 2, 3, 4],
            "price": [math.nan, None, -2.0, 2.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)
    for name in ("left_max", "right_max"):
        assert result[name].to_pylist() == [0.0, 0.0, 0.0, 2.0]
    for name in ("left_min", "right_min"):
        assert result[name].to_pylist() == [0.0, 0.0, -2.0, 0.0]
    for name in ("both_max", "both_min"):
        assert result[name].to_pylist() == [None, None, -2.0, 2.0]


def test_isnan_and_sign_distinguish_nan_from_null() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "nan-sign",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    isnan=row.isnan(quotes["price"]),
                    sign=row.sign(quotes["price"]),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000],
            "symbol": ["A"] * 3,
            "seq": [1, 2, 3],
            "price": [math.nan, None, -1.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)
    assert result["isnan"].to_pylist() == [True, None, False]
    assert result["sign"].to_pylist() == [None, None, -1.0]


def test_boolean_windows_skip_nan_numeric_samples() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "boolean-numeric-nan",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    all_true=ts.all_true(quotes["price"], window=rows(2)),
                    any_true=ts.any_true(quotes["price"], window=rows(2)),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            "symbol": ["A"] * 5,
            "seq": [1, 2, 3, 4, 5],
            "price": [1.0, math.nan, 0.0, None, 2.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)
    assert result["all_true"].to_pylist() == [True, True, False, False, True]
    assert result["any_true"].to_pylist() == [True, True, False, False, True]


def test_cumulative_average_and_last_valid_survive_missing_values() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    program = Program(
        "cumulative-average-and-last",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    average=ts.average(price),
                    last_valid=ts.last(price),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            "symbol": ["A"] * 5,
            "seq": [1, 2, 3, 4, 5],
            "price": [2.0, None, 4.0, math.nan, 8.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)
    _assert_optional_floats(output["average"].to_pylist(), [2.0, 2.0, 3.0, 3.0, 14 / 3])
    assert output["last_valid"].to_pylist() == [2.0, 2.0, 4.0, 4.0, 8.0]
    program.compile_stream(Runtime())


def test_cumulative_average_handles_opposite_extreme_values() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "cumulative-extreme-average",
        inputs=(quotes,),
        outputs=(
            ("signals", quotes.with_columns(average=ts.average(quotes["price"]))),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000],
            "symbol": ["A", "A"],
            "seq": [1, 2],
            "price": [1e308, -1e308],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    assert _execute(program, table)["average"].to_pylist() == [1e308, 0.0]


def test_last_valid_replaces_infinity_with_new_finite_value() -> None:
    quotes = _ordered_quotes()
    program = Program(
        "last-after-infinity",
        inputs=(quotes,),
        outputs=(
            ("signals", quotes.with_columns(last_valid=ts.last(quotes["price"]))),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "symbol": ["A"] * 4,
            "seq": [1, 2, 3, 4],
            "price": [math.inf, 1.0, -math.inf, 2.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    output = _execute(program, table)["last_valid"].to_pylist()
    assert output == [math.inf, 1.0, -math.inf, 2.0]

    async def feed():
        for index in range(table.num_rows):
            yield table.slice(index, 1)

    async def stream_values() -> list[float | None]:
        values: list[float | None] = []
        async with program.stream({"quotes": feed()}) as results:
            async for event in results:
                values.extend(event.table["last_valid"].to_pylist())
        return values

    assert asyncio.run(stream_values()) == output


def test_rolling_order_treats_signed_zeros_as_equal() -> None:
    quotes = _ordered_quotes()
    price = quotes["price"]
    program = Program(
        "rolling-signed-zero",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                quotes.with_columns(
                    argmax=ts.argmax(price, window=rows(2)),
                    argmin=ts.argmin(price, window=rows(2)),
                    rank=ts.rank(price, window=rows(2)),
                    unique_count=ts.unique_count(price, window=rows(2)),
                ),
            ),
        ),
    )
    table = pa.table(
        {
            "ts": [1_000_000, 2_000_000],
            "symbol": ["A", "A"],
            "seq": [1, 2],
            "price": [-0.0, 0.0],
        },
        schema=pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64()),
            )
        ),
    )
    result = _execute(program, table)
    assert result["argmax"].to_pylist() == [0, 1]
    assert result["argmin"].to_pylist() == [0, 1]
    assert result["rank"].to_pylist() == [0, 0]
    assert result["unique_count"].to_pylist() == [1, 1]
