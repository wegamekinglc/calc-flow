"""Measure 64 public one-row pushes after the native dual-SMA study's history."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pyarrow as pa

from benchmarks import warm_stream as warm
from calc_flow import (
    Batch,
    SourceProvidedWatermarks,
    StreamRuntimeConfig,
    Watermark,
)
from scripts.toolkit import worker_environment, write_json

NATIVE_STUDY = Path(__file__).parent / "stream-session/20261123-native-push"
HISTORY = 4096
APPENDS = 64


def _input_file(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def verify_output(actual: pa.Table, expected: pa.Table) -> None:
    if expected.num_columns != 5 or not actual.schema.equals(
        expected.schema, check_metadata=True
    ):
        raise ValueError("incorrect output schema")
    if not actual.select(range(4)).equals(
        expected.select(range(4)), check_metadata=True
    ):
        raise ValueError("output row identity or original columns changed")
    spread, wanted = actual.column(4), expected.column(4)
    if not spread.is_null().equals(wanted.is_null()):
        raise ValueError("output null placement changed")
    if not np.allclose(
        spread.to_numpy(), wanted.to_numpy(), rtol=1e-10, atol=1e-10, equal_nan=True
    ):
        raise ValueError("dual-SMA values differ from the independent reference")


def _verify_result(result: dict[str, pa.Table], name: str, expected: pa.Table) -> None:
    if set(result) != {name}:
        raise ValueError("unexpected logical session outputs")
    verify_output(result[name], expected)


def _prepared(table: pa.Table):
    return {warm.SESSION_INPUT: Batch.from_pyarrow(table)}, Watermark(
        table["event_time"][-1].as_py()
    )


def _program():
    return warm._warm_program(
        warm.ScenarioConfig(
            history_rows=HISTORY,
            append_rows=1,
            entities=1,
            indicator="dual_sma_spread",
            fast_window=5,
            window=20,
            history_segment_rows=HISTORY,
            exact_projection=True,
        )
    )


def observe(directory: Path, *, clock=perf_counter_ns) -> dict:
    source = _input_file(directory / "inputs.arrow")
    expected = _input_file(directory / "expected.arrow")
    if source.num_rows != HISTORY + APPENDS or expected.num_rows != source.num_rows:
        raise ValueError("expected 4,096 history rows and 64 appends")
    program = _program()
    if program.to_project(mode="stream").root != json.loads(
        (directory / "project.json").read_text()
    ):
        raise ValueError("public Program differs from the native session fixture")
    name = program.outputs[0][0]
    runtime = StreamRuntimeConfig(checkpointing=False)
    clocks = []
    with program.session(
        watermarks={warm.SESSION_INPUT: SourceProvidedWatermarks()}, config=runtime
    ) as session:
        inputs, mark = _prepared(source.slice(0, HISTORY))
        warm_result = session.push(inputs, watermark=mark)
        _verify_result(warm_result, name, expected.slice(0, HISTORY))
        for index in range(HISTORY, HISTORY + APPENDS):
            inputs, mark = _prepared(source.slice(index, 1))
            start = clock()
            result = session.push(inputs, watermark=mark)
            end = clock()
            clocks.append(end - start)
            _verify_result(result, name, expected.slice(index, 1))
            del result
        finish = session.finish()
        _verify_result(finish, name, expected.slice(source.num_rows, 0))
    if not source.equals(expected.select(range(4)), check_metadata=True):
        raise ValueError("session modified the caller input")
    return {
        "append_ns": clocks,
        "validated_rows": source.num_rows,
        "finish_rows": finish[name].num_rows,
    }


def main() -> None:
    directory, output = map(Path, sys.argv[1:])
    if not gc.isenabled() or sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError("sampling requires ordinary GC and no tracing or profile hook")
    write_json(
        output,
        {**observe(directory), "gc_enabled": True, "environment": worker_environment()},
        exclusive=True,
    )


if __name__ == "__main__":
    main()
