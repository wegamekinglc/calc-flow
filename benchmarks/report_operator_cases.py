"""Dependency-free inventory shared by the report and scheduled comparison."""

from __future__ import annotations

SHARD_FAMILY = "report-operators"
WORKLOAD_CONTRACT = "report-operators-complete-ticks-v1"
SCHEDULED_ROWS = 64_000
SCHEDULED_ENTITIES = 64
SCHEDULED_CHUNK_TICKS = 100
BATCH_BACKEND = "calc-flow-batch"
STREAM_BACKEND = "calc-flow-stream"
FINANCE_BACKEND = "finance-python"
BATCH_SCOPE = "native-expression-execute-to-arrow/report-operator-v1"
STREAM_SCOPE = "ready-enqueue-to-arrow/report-operator-v1"
FINANCE_SCOPE = "pandas-transform/report-operator-v1"
BACKEND_SCOPES = {
    BATCH_BACKEND: BATCH_SCOPE,
    STREAM_BACKEND: STREAM_SCOPE,
    FINANCE_BACKEND: FINANCE_SCOPE,
}

TIME_SERIES = (
    "average",
    "ma20",
    "ema20",
    "macd_12_26",
    "rsi20",
    "correlation20",
    "decay20",
    "max20",
    "argmax20",
    "min20",
    "argmin20",
    "rank20",
    "quantile20",
    "count20",
    "unique_count20",
    "all_true20",
    "any_true20",
    "sum20",
    "variance20",
    "stddev20",
    "count_positive20",
    "mean_positive20",
    "lag1",
    "delta1",
    "return_simple1",
    "return_log1",
)
CROSS_SECTION = (
    "cs_rank",
    "cs_top16",
    "cs_bottom16",
    "cs_top_quantile25",
    "cs_bottom_quantile25",
    "cs_mean",
    "cs_demean",
    "cs_percentile",
    "cs_zscore",
    "cs_residual",
    "cs_mean_fill",
)
ROW = (
    "last",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "negate",
    "sign",
    "maximum",
    "minimum",
    "sqrt",
    "exp",
    "log",
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "norminv",
    "ceil",
    "floor",
    "round",
    "where",
)
OPERATORS = TIME_SERIES + CROSS_SECTION + ROW
REPORT_INDICATOR_PREFIX = "report-operator/"
NO_FINANCE_EQUIVALENT = {
    "rsi20": "Wilder smoothing and warm-up differ from Finance-Python RSI",
    "cs_demean": "no Finance-Python equivalent for the report’s demean contract",
}
