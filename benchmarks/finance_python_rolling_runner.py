"""Isolated Finance-Python worker for the rolling comparison benchmark.

Finance-Python 0.9.10 does not build in Calc Flow's Python 3.13 environment,
so the parent benchmark drives this worker through a small JSON-lines protocol.
Input construction and the initial warm-up happen before the worker reports
ready. Each ``run`` command measures a fresh public ``MA(...).transform(...)``
call or a composed fast/slow MA spread over the already-built immutable input
frame.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PyFin.api.Analysis import AVG, MA, MARGMAX, MUCOUNT, CSMean

_ROLLING_MEAN = "rolling_mean"
_DUAL_SMA_SPREAD = "dual_sma_spread"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument(
        "--indicator",
        choices=(_ROLLING_MEAN, _DUAL_SMA_SPREAD),
        default=_ROLLING_MEAN,
    )
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--warm-output", type=Path, required=True)
    parser.add_argument(
        "--suite-scenario",
        choices=(
            "sma20",
            "dual_sma",
            "average",
            "argmax64",
            "argmax256",
            "unique64",
            "cs_mean",
        ),
    )
    args = parser.parse_args()
    if args.fast_window <= 0:
        parser.error("fast-window must be positive")
    if args.indicator == _DUAL_SMA_SPREAD and args.fast_window >= args.window:
        parser.error("fast-window must be positive and smaller than window")
    return args


def _input_frame(rows: int, entities: int, *, suite: bool = False) -> pd.DataFrame:
    sequence = np.arange(rows, dtype=np.uint64)
    entity_index = sequence % entities
    positions = sequence // entities
    symbol_names = np.asarray([f"S{index:03d}" for index in range(entities)])
    prices = (
        100.0
        + (sequence % 257).astype(np.float64) / 8.0
        + entity_index.astype(np.float64) / 8.0
        if suite
        else 100.0
        + ((sequence * np.uint64(17)) % np.uint64(1_000)).astype(np.float64) / 100.0
        + entity_index.astype(np.float64) / 100.0
    )
    return pd.DataFrame(
        {"price": prices, "symbol": symbol_names[entity_index]},
        index=positions,
    )


def _execute(
    frame: pd.DataFrame,
    window: int,
    *,
    indicator: str,
    fast_window: int,
    suite_scenario: str | None = None,
) -> np.ndarray:
    expression = {
        "average": lambda: AVG("price"),
        "argmax64": lambda: MARGMAX(64, "price"),
        "argmax256": lambda: MARGMAX(256, "price"),
        "unique64": lambda: MUCOUNT(64, "price"),
        "cs_mean": lambda: CSMean("price"),
    }.get(suite_scenario, lambda: MA(window, "price"))()
    if indicator == _DUAL_SMA_SPREAD or suite_scenario == "dual_sma":
        expression = MA(fast_window, "price") - MA(window, "price")
    result = expression.transform(
        frame,
        name="moving_average",
        category_field="symbol",
        dropna=False,
    )
    values = result["moving_average"].to_numpy(copy=False)
    if suite_scenario in ("sma20", "dual_sma"):
        values = np.array(values, copy=True)
        positions = np.arange(len(values)) // len(frame["symbol"].unique())
        values[positions < window - 1] = np.nan
    return values


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def _reply(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _validated_iterations(command: dict[str, object]) -> int:
    """Return one valid run count or reject the worker command."""
    iterations = command.get("iterations")
    if command.get("command") != "run" or type(iterations) is not int:
        raise ValueError(f"unsupported command: {command!r}")
    if iterations <= 0:
        raise ValueError(f"unsupported command: {command!r}")
    return iterations


def _timed_execute(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Measure repeated transforms while keeping GC outside the boundary."""
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        for _iteration in range(iterations):
            output = _execute(
                frame,
                args.window,
                indicator=args.indicator,
                fast_window=args.fast_window,
                suite_scenario=args.suite_scenario,
            )
        seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    finally:
        if gc_was_enabled:
            gc.enable()
    return output, seconds / iterations


def _serve(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    """Serve benchmark commands until the parent asks the worker to stop."""
    for line in sys.stdin:
        command = json.loads(line)
        if command == {"command": "stop"}:
            return
        iterations = _validated_iterations(command)
        output, seconds = _timed_execute(frame, args, iterations)
        _reply(
            {
                "event": "sample",
                "iterations": iterations,
                "rows": len(output),
                "seconds": seconds,
                "sha256": _digest(output),
            }
        )


def _prepare(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write the untimed warm output and announce immutable worker identity."""
    warm_output = _execute(
        frame,
        args.window,
        indicator=args.indicator,
        fast_window=args.fast_window,
        suite_scenario=args.suite_scenario,
    )
    np.save(args.warm_output, warm_output, allow_pickle=False)
    _reply(
        {
            "event": "ready",
            "fast_window": args.fast_window,
            "finance_python_version": version("Finance-Python"),
            "indicator": args.indicator,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "rows": len(warm_output),
            "sha256": _digest(warm_output),
            "suite_scenario": args.suite_scenario,
        }
    )


def main() -> None:
    args = _parse_args()
    frame = _input_frame(
        args.rows, args.entities, suite=args.suite_scenario is not None
    )
    _prepare(frame, args)
    _serve(frame, args)


if __name__ == "__main__":
    main()
