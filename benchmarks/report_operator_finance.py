"""Finance-Python transform adapter for the isolated legacy worker."""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from PyFin.api import Analysis as finance

from benchmarks.report_operator_inputs import columns


def frame_for(rows: int, entities: int) -> pd.DataFrame:
    data = columns(rows, entities)
    ticks = data.pop("tick")
    data["flag"] = data["flag"].astype(np.float64)
    return pd.DataFrame(data, index=ticks)


def _expression(name: str):
    x, y = "x", "y"
    return {
        "average": lambda: finance.AVG(x),
        "ma20": lambda: finance.MA(20, x),
        "ema20": lambda: finance.EMA(20, x),
        "macd_12_26": lambda: finance.MACD(12, 26, x),
        "correlation20": lambda: finance.MCORR(20, x, y),
        "decay20": lambda: finance.MADecay(20, x),
        "max20": lambda: finance.MMAX(20, x),
        "argmax20": lambda: finance.MARGMAX(20, x),
        "min20": lambda: finance.MMIN(20, x),
        "argmin20": lambda: finance.MARGMIN(20, x),
        "rank20": lambda: finance.MRANK(20, x),
        "quantile20": lambda: finance.MQUANTILE(20, x),
        "count20": lambda: finance.MCOUNT(20, x),
        "unique_count20": lambda: finance.MUCOUNT(20, x),
        "all_true20": lambda: finance.MALLTRUE(20, "flag"),
        "any_true20": lambda: finance.MANYTRUE(20, "flag"),
        "sum20": lambda: finance.MSUM(20, x),
        "variance20": lambda: finance.MVARIANCE(20, x),
        "stddev20": lambda: finance.MSTD(20, x),
        "count_positive20": lambda: finance.MNPOSITIVE(20, "signed"),
        "mean_positive20": lambda: finance.MAPOSITIVE(20, "signed"),
        "lag1": lambda: finance.SHIFT(x, 1),
        "delta1": lambda: finance.DELTA(x, 1),
        "return_simple1": lambda: finance.RETURNSimple(x),
        "return_log1": lambda: finance.RETURNLog(x),
        "cs_rank": lambda: finance.CSRank(x),
        "cs_top16": lambda: finance.CSTopN(x, 16),
        "cs_bottom16": lambda: finance.CSBottomN(x, 16),
        "cs_top_quantile25": lambda: finance.CSTopNQuantile(x, 0.25),
        "cs_bottom_quantile25": lambda: finance.CSBottomNQuantile(x, 0.25),
        "cs_mean": lambda: finance.CSMean(x),
        "cs_percentile": lambda: finance.CSQuantiles(x),
        "cs_zscore": lambda: finance.CSZScore(x),
        "cs_residual": lambda: finance.CSRes(x, y),
        "cs_mean_fill": lambda: finance.CSFillNA("fill"),
        "last": lambda: finance.LAST(x),
        "add": lambda: finance.CURRENT(x) + finance.CURRENT(y),
        "subtract": lambda: finance.CURRENT(x) - finance.CURRENT(y),
        "multiply": lambda: finance.CURRENT(x) * finance.CURRENT(y),
        "divide": lambda: finance.CURRENT(x) / finance.CURRENT(y),
        "power": lambda: finance.POW("positive", 2.5),
        "negate": lambda: -finance.CURRENT("signed"),
        "sign": lambda: finance.SIGN("signed"),
        "maximum": lambda: finance.MAXIMUM(x, y),
        "minimum": lambda: finance.MINIMUM(x, y),
        "sqrt": lambda: finance.SQRT("positive"),
        "exp": lambda: finance.EXP("signed"),
        "log": lambda: finance.LOG("positive"),
        "abs": lambda: finance.ABS("signed"),
        "acos": lambda: finance.ACOS("unit"),
        "acosh": lambda: finance.ACOSH("positive"),
        "asin": lambda: finance.ASIN("unit"),
        "asinh": lambda: finance.ASINH("signed"),
        "norminv": lambda: finance.NORMINV("unit"),
        "ceil": lambda: finance.CEIL("signed"),
        "floor": lambda: finance.FLOOR("signed"),
        "round": lambda: finance.ROUND("signed"),
        "where": lambda: finance.IIF("flag", x, y),
    }[name]()


def _aligned_values(
    frame: pd.DataFrame, output: pd.DataFrame, name: str, entities: int
) -> np.ndarray:
    if not output.index.equals(frame.index) or not output["symbol"].equals(
        frame["symbol"]
    ):
        raise ValueError("Finance-Python changed timestamp or entity output alignment")
    values = output["value"].to_numpy(dtype=np.float64, copy=False)
    if name == "cs_percentile":
        # Finance emits 1-based rank / N; Calc Flow maps the endpoints to 0 and 1.
        return (
            (values * entities - 1) / (entities - 1)
            if entities > 1
            else np.full_like(values, 0.5)
        )
    return values


def sample(
    frame: pd.DataFrame, name: str, entities: int, iterations: int
) -> tuple[np.ndarray, float]:
    """Time fresh public transforms; construction and validation are separate."""
    seconds = 0.0
    gc.collect()
    for _ in range(iterations):
        expression = _expression(name)
        started = time.perf_counter_ns()
        output = expression.transform(
            frame, name="value", category_field="symbol", dropna=False
        )
        seconds += (time.perf_counter_ns() - started) / 1e9
        values = _aligned_values(frame, output, name, entities)
    return values, seconds / iterations
