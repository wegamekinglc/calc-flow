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
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PyFin.api.Analysis import MA

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
    args = parser.parse_args()
    if args.fast_window <= 0:
        parser.error("fast-window must be positive")
    if args.indicator == _DUAL_SMA_SPREAD and args.fast_window >= args.window:
        parser.error("fast-window must be positive and smaller than window")
    return args


def _input_frame(rows: int, entities: int) -> pd.DataFrame:
    sequence = np.arange(rows, dtype=np.uint64)
    entity_index = sequence % entities
    positions = sequence // entities
    symbol_names = np.asarray([f"S{index:03d}" for index in range(entities)])
    prices = (
        100.0
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
) -> np.ndarray:
    expression = MA(window, "price")
    if indicator == _DUAL_SMA_SPREAD:
        expression = MA(fast_window, "price") - expression
    result = expression.transform(
        frame,
        name="moving_average",
        category_field="symbol",
        dropna=False,
    )
    return result["moving_average"].to_numpy(copy=False)


def _reply(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    args = _parse_args()
    frame = _input_frame(args.rows, args.entities)
    warm_output = _execute(
        frame,
        args.window,
        indicator=args.indicator,
        fast_window=args.fast_window,
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
        }
    )

    for line in sys.stdin:
        command = json.loads(line)
        if command == {"command": "stop"}:
            return
        iterations = command.get("iterations")
        if (
            command.get("command") != "run"
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError(f"unsupported command: {command!r}")
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
                )
            seconds = (time.perf_counter_ns() - started) / 1_000_000_000
        finally:
            if gc_was_enabled:
                gc.enable()
        _reply(
            {
                "event": "sample",
                "iterations": iterations,
                "rows": len(output),
                "seconds": seconds / iterations,
            }
        )


if __name__ == "__main__":
    main()
