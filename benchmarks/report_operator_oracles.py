"""Independent NumPy references for the report's complete-tick fixtures."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from benchmarks.report_operator_cases import CROSS_SECTION, ROW


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (1 + span)
    result = np.empty_like(values)
    result[0] = values[0]
    for tick in range(1, len(values)):
        result[tick] = alpha * values[tick] + (1 - alpha) * result[tick - 1]
    return result


def _rsi(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(values, np.nan)
    if len(values) <= period:
        return result
    delta = np.diff(values, axis=0)
    gains, losses = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    gain = gains[:period].mean(axis=0)
    loss = losses[:period].mean(axis=0)
    for tick in range(period, len(values)):
        if tick > period:
            gain = ((period - 1) * gain + gains[tick - 1]) / period
            loss = ((period - 1) * loss + losses[tick - 1]) / period
        total = gain + loss
        result[tick] = np.divide(
            100 * gain, total, out=np.full_like(total, 50.0), where=total != 0.0
        )
    return result


def _correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    denominator = np.sqrt((x_centered**2).sum(axis=0) * (y_centered**2).sum(axis=0))
    return np.divide(
        (x_centered * y_centered).sum(axis=0),
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator != 0,
    )


def _positive_mean(values: np.ndarray) -> np.ndarray:
    count = (values > 0).sum(axis=0)
    return np.divide(
        np.maximum(values, 0).sum(axis=0),
        count,
        out=np.zeros(values.shape[1]),
        where=count != 0,
    )


def _rolling(name: str, data: dict[str, np.ndarray], entities: int) -> np.ndarray:
    x, y, signed, flag = (
        data[key].reshape(-1, entities) for key in ("x", "y", "signed", "flag")
    )
    operations = {
        "ma20": lambda w: w.mean(axis=0),
        "decay20": lambda w: np.average(w, axis=0, weights=np.arange(21 - len(w), 21)),
        "max20": lambda w: w.max(axis=0),
        "argmax20": lambda w: len(w) - 1 - w.argmax(axis=0),
        "min20": lambda w: w.min(axis=0),
        "argmin20": lambda w: len(w) - 1 - w.argmin(axis=0),
        "rank20": lambda w: (w < w[-1]).sum(axis=0),
        "quantile20": lambda w: (w < w[-1]).sum(axis=0) / (len(w) - 1),
        "count20": lambda w: len(w),
        "unique_count20": lambda w: (
            (np.diff(np.sort(w, axis=0), axis=0) != 0).sum(axis=0) + 1
        ),
        "all_true20": lambda w: w.all(axis=0),
        "any_true20": lambda w: w.any(axis=0),
        "sum20": lambda w: w.sum(axis=0),
        "variance20": lambda w: w.var(axis=0, ddof=1),
        "stddev20": lambda w: w.std(axis=0, ddof=1),
        "count_positive20": lambda w: (w > 0).sum(axis=0),
        "mean_positive20": _positive_mean,
    }
    result = np.full_like(x, np.nan)
    values = {
        "all_true20": flag,
        "any_true20": flag,
        "count_positive20": signed,
        "mean_positive20": signed,
    }.get(name, x)
    min_ticks = (
        2 if name in {"quantile20", "variance20", "stddev20", "correlation20"} else 1
    )
    for end in range(min_ticks, len(x) + 1):
        start = max(0, end - 20)
        window = values[start:end]
        result[end - 1] = (
            _correlation(window, y[start:end])
            if name == "correlation20"
            else operations[name](window)
        )
    return result


def _time_series(name: str, data: dict[str, np.ndarray], entities: int) -> np.ndarray:
    x = data["x"].reshape(-1, entities)
    lag = np.vstack((np.full(entities, np.nan), x[:-1]))
    immediate = {
        "average": lambda: x.cumsum(axis=0) / np.arange(1, len(x) + 1)[:, None],
        "ema20": lambda: _ema(x, 20),
        "macd_12_26": lambda: _ema(x, 12) - _ema(x, 26),
        "rsi20": lambda: _rsi(x, 20),
        "lag1": lambda: lag,
        "delta1": lambda: x - lag,
        "return_simple1": lambda: x / lag - 1,
        "return_log1": lambda: np.log(x / lag),
    }
    if name in immediate:
        return immediate[name]()
    return _rolling(name, data, entities)


def _cross_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, group, counts = np.unique(values, return_inverse=True, return_counts=True)
    high = np.cumsum(counts)[group]
    low = high - counts[group] + 1
    return low, high, (low + high) / 2


def _selection_reference(name: str, values: np.ndarray) -> np.ndarray:
    size = len(values)
    low, high, average = _cross_ranks(values)
    return {
        "cs_rank": lambda: average,
        "cs_percentile": lambda: (
            (average - 1) / (size - 1) if size > 1 else np.full(1, 0.5)
        ),
        "cs_top16": lambda: size + 1 - high <= 16,
        "cs_bottom16": lambda: low <= 16,
        "cs_top_quantile25": lambda: (size + 1 - average) / size <= 0.25,
        "cs_bottom_quantile25": lambda: average / size <= 0.25,
    }[name]()


def _cross_section(name: str, data: dict[str, np.ndarray], entities: int) -> np.ndarray:
    x, y, fill = (data[key].reshape(-1, entities) for key in ("x", "y", "fill"))
    mean = x.mean(axis=1, keepdims=True)
    center = x - mean
    if name == "cs_mean":
        return np.broadcast_to(mean, x.shape)
    if name == "cs_demean":
        return center
    if name == "cs_zscore":
        scale = x.std(axis=1, keepdims=True)
        return np.divide(center, scale, out=np.full_like(x, np.nan), where=scale != 0)
    if name == "cs_residual":
        predictor = y - y.mean(axis=1, keepdims=True)
        variance = (predictor**2).sum(axis=1, keepdims=True)
        slope = np.divide(
            (center * predictor).sum(axis=1, keepdims=True),
            variance,
            out=np.full_like(variance, np.nan),
            where=variance != 0,
        )
        return center - slope * predictor
    if name == "cs_mean_fill":
        count = (~np.isnan(fill)).sum(axis=1, keepdims=True)
        valid_mean = np.divide(
            np.nansum(fill, axis=1, keepdims=True),
            count,
            out=np.full_like(count, np.nan, dtype=np.float64),
            where=count != 0,
        )
        return np.where(np.isnan(fill), valid_mean, fill)
    return np.array([_selection_reference(name, group) for group in x])


def _normal_inverse(probabilities: np.ndarray) -> np.ndarray:
    domain, indices = np.unique(probabilities, return_inverse=True)
    return np.array([NormalDist().inv_cdf(float(p)) for p in domain])[indices]


def _row(name: str, data: dict[str, np.ndarray]) -> np.ndarray:
    x, y, unit, positive, signed = (
        data[key] for key in ("x", "y", "unit", "positive", "signed")
    )
    return {
        "last": lambda: x,
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y,
        "power": lambda: positive**2.5,
        "negate": lambda: -signed,
        "sign": lambda: np.sign(signed),
        "maximum": lambda: np.maximum(x, y),
        "minimum": lambda: np.minimum(x, y),
        "sqrt": lambda: np.sqrt(positive),
        "exp": lambda: np.exp(signed),
        "log": lambda: np.log(positive),
        "abs": lambda: np.abs(signed),
        "acos": lambda: np.arccos(unit),
        "acosh": lambda: np.arccosh(positive),
        "asin": lambda: np.arcsin(unit),
        "asinh": lambda: np.arcsinh(signed),
        "norminv": lambda: _normal_inverse(unit),
        "ceil": lambda: np.ceil(signed),
        "floor": lambda: np.floor(signed),
        "round": lambda: np.sign(signed) * np.floor(np.abs(signed) + 0.5),
        "where": lambda: np.where(data["flag"], x, y),
    }[name]()


def oracle(name: str, data: dict[str, np.ndarray], entities: int) -> np.ndarray:
    """Return expected values in original sequence order; NaN marks warm-up nulls."""
    if name in ROW:
        return _row(name, data)
    if name in CROSS_SECTION:
        return _cross_section(name, data, entities).ravel()
    return _time_series(name, data, entities).ravel()
