"""Independent finite-input references, evaluated outside benchmark timers."""

from __future__ import annotations

import numpy as np


def ewma(values: np.ndarray, span: int) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    result[0] = values[0]
    alpha = 2.0 / (span + 1)
    for index in range(1, len(values)):
        result[index] = (1 - alpha) * result[index - 1] + alpha * values[index]
    return result


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    totals = np.convolve(values, np.ones(window), mode="full")[: len(values)]
    return totals / np.minimum(np.arange(len(values)) + 1, window)


def _wilder(samples: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(samples), np.nan)
    result[period - 1] = samples[:period].mean()
    for index in range(period, len(samples)):
        result[index] = (result[index - 1] * (period - 1) + samples[index]) / period
    return result


def rsi(values: np.ndarray, period: int) -> np.ndarray:
    differences = np.diff(values)
    gains = _wilder(np.maximum(differences, 0), period)
    losses = _wilder(np.maximum(-differences, 0), period)
    total = gains + losses
    fraction = np.divide(gains, total, out=np.full(len(gains), 0.5), where=total != 0)
    return np.r_[np.nan, 100 * fraction]


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    ranges = np.maximum.reduce(
        (
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
    )
    return np.r_[np.nan, _wilder(ranges, period)]


def macd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ewma(values, 12) - ewma(values, 26)
    signal = ewma(line, 9)
    return line, signal, line - signal
