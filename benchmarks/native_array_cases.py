"""Prepared public API workloads; fixture construction stays outside timers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial

import numpy as np
import pyarrow as pa

import calc_flow as cf
import calc_flow.array as cfa
from benchmarks import native_array_oracles as oracle
from benchmarks.native_array_support import NativeCase
from benchmarks.support import BenchmarkScale

EAGER_CASES = (
    "elementwise",
    "where",
    "sum",
    "axis_mean",
    "matmul",
    "transpose_reshape",
    "rolling_mean",
    "ewma",
    "rsi",
    "atr",
    "macd",
)


def vector(rows: int) -> np.ndarray:
    return (np.arange(rows, dtype=np.float64) % 97) / 8 + 16


def _matrix(scale: BenchmarkScale) -> np.ndarray:
    size = scale.matrix_dimension
    return ((np.arange(size * size, dtype=np.float64) % 17) / 8).reshape(size, size)


def _vector_case(
    scale: BenchmarkScale,
    name: str,
    operation: Callable,
    reference: Callable,
    parameters: tuple = (),
) -> NativeCase:
    values = vector(scale.array_elements)
    native = cfa.from_numpy(values)
    expected = reference(values)
    expected = (expected,) if isinstance(expected, np.ndarray) else expected
    return NativeCase(
        f"native_{name}",
        "native-eager-api-v1",
        lambda: operation(native),
        expected,
        (values,),
        parameters,
    )


def _elementwise(scale: BenchmarkScale) -> NativeCase:
    one = cfa.from_numpy(np.array(1.0))
    return _vector_case(
        scale,
        "elementwise",
        lambda value: cfa.sqrt(cfa.add(cfa.multiply(value, value), one)),
        lambda value: np.sqrt(value * value + 1),
        (("expression", "sqrt(x*x+1)"),),
    )


def _where(scale: BenchmarkScale) -> NativeCase:
    threshold = cfa.from_numpy(np.array(20.0))
    zero = cfa.from_numpy(np.array(0.0))
    return _vector_case(
        scale,
        "where",
        lambda value: cfa.where_(cfa.greater_equal(value, threshold), value, zero),
        lambda value: np.where(value >= 20, value, 0),
        (("threshold", 20.0),),
    )


def _sum(scale: BenchmarkScale) -> NativeCase:
    return _vector_case(scale, "sum", cfa.sum, lambda value: np.asarray(value.sum()))


def _axis_mean(scale: BenchmarkScale) -> NativeCase:
    values = _matrix(scale)
    native = cfa.from_numpy(values)
    return NativeCase(
        "native_axis_mean",
        "native-eager-api-v1",
        lambda: cfa.mean(native, axis=1),
        (values.mean(axis=1),),
        (values,),
        (("axis", 1),),
    )


def _matmul(scale: BenchmarkScale) -> NativeCase:
    values = _matrix(scale)
    native = cfa.from_numpy(values)
    return NativeCase(
        "native_matmul",
        "native-eager-api-v1",
        lambda: cfa.matmul(native, native),
        (values @ values,),
        (values,),
        (("expression", "x@x"),),
    )


def _transpose_reshape(scale: BenchmarkScale) -> NativeCase:
    values = _matrix(scale)
    native = cfa.from_numpy(values)
    return NativeCase(
        "native_transpose_reshape",
        "native-eager-api-v1",
        lambda: cfa.reshape(cfa.transpose(native), (values.size,)),
        (values.T.reshape(-1),),
        (values,),
        (("expression", "reshape(transpose(x))"),),
    )


def _rolling(scale: BenchmarkScale) -> NativeCase:
    return _vector_case(
        scale,
        "rolling_mean",
        lambda value: cfa.rolling_mean(value, window=20),
        lambda value: oracle.rolling_mean(value, 20),
        (("window", 20), ("min_periods", 1), ("numerical_profile", "stable_v1")),
    )


def _ewma(scale: BenchmarkScale) -> NativeCase:
    return _vector_case(
        scale,
        "ewma",
        lambda value: cfa.ewma(value, span=7),
        lambda value: oracle.ewma(value, 7),
        (("span", 7), ("min_periods", 1)),
    )


def _rsi(scale: BenchmarkScale) -> NativeCase:
    return _vector_case(
        scale,
        "rsi",
        lambda value: cfa.rsi(value, period=14),
        lambda value: oracle.rsi(value, 14),
        (("period", 14),),
    )


def _atr(scale: BenchmarkScale) -> NativeCase:
    close = vector(scale.array_elements)
    high, low = close + 1, close - 1
    inputs = tuple(cfa.from_numpy(value) for value in (high, low, close))
    return NativeCase(
        "native_atr",
        "native-eager-api-v1",
        lambda: cfa.atr(*inputs, period=14),
        (oracle.atr(high, low, close, 14),),
        (high, low, close),
        (("period", 14),),
    )


def _macd(scale: BenchmarkScale) -> NativeCase:
    return _vector_case(
        scale,
        "macd",
        cfa.macd_components,
        oracle.macd,
        (
            ("fast_span", 12),
            ("slow_span", 26),
            ("signal_span", 9),
            ("min_periods", 1),
            ("signal_min_periods", 1),
        ),
    )


_EAGER = dict(
    zip(
        EAGER_CASES,
        (
            _elementwise,
            _where,
            _sum,
            _axis_mean,
            _matmul,
            _transpose_reshape,
            _rolling,
            _ewma,
            _rsi,
            _atr,
            _macd,
        ),
    )
)


def eager_case(name: str, scale: BenchmarkScale) -> NativeCase:
    return _EAGER[name](scale)


def ownership_case(kind: str, scale: BenchmarkScale) -> NativeCase:
    values = vector(scale.array_elements)
    if kind == "numpy":
        calculate = partial(cfa.from_numpy, values)
    elif kind == "arrow":
        data = pa.Array.from_buffers(
            pa.float64(), len(values), [None, pa.py_buffer(values.tobytes())]
        )
        calculate = partial(cfa.from_arrow, data)
    else:
        raise ValueError(f"unsupported native input: {kind}")
    return NativeCase(
        f"native_from_{kind}",
        "native-array-ingestion-v1",
        calculate,
        (values.copy(),),
        (values,),
        (("source", kind),),
    )


def _runtime() -> cf.Runtime:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    return runtime


def program_case(scale: BenchmarkScale) -> NativeCase:
    values = vector(scale.array_elements)
    native = cfa.from_numpy(values)
    source = cf.array_input("values", dtype="float64", shape=(None,))
    expression = ((source * source) + 1).mean()
    runtime = _runtime()
    return NativeCase(
        "native_array_expr_collect",
        "native-array-collect-v1",
        lambda: expression.collect_array({"values": native}, runtime=runtime),
        (np.asarray((values * values + 1).mean()),),
        (values,),
        (("planning", "fresh-per-call"), ("expression", "mean(x*x+1)")),
    )


def bridge_case(scale: BenchmarkScale) -> NativeCase:
    values = (np.arange(scale.table_rows * 4, dtype=np.float64) % 17).reshape(-1, 4)
    weights = np.arange(8, dtype=np.float64).reshape(4, 2) / 8
    data = pa.table({f"f{index}": values[:, index] for index in range(4)})
    source = cf.table_input("features", schema=data.schema)
    matrix = cf.linalg.from_columns(
        source, columns=tuple(data.column_names), backend="native"
    )
    parameter = cf.parameter(
        "weights", kind="array", backend="native", dtype="float64", shape=(4, 2)
    )
    attached = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, parameter), names=("p0", "p1")
    ).select("p0", "p1")
    runtime, native = _runtime(), cfa.from_numpy(weights)
    product = values @ weights
    return NativeCase(
        "native_table_array_table",
        "native-table-bridge-collect-v1",
        lambda: attached.collect(
            {"features": data, "weights": native}, runtime=runtime
        ),
        pa.table({"p0": product[:, 0], "p1": product[:, 1]}),
        (values, weights),
        (("planning", "fresh-per-call"), ("input_width", 4), ("output_width", 2)),
    )


def _stream_reference(values: np.ndarray, sequences: np.ndarray) -> pa.Table:
    output = np.empty((len(values), 3))
    for entity in range(4):
        selected = sequences % 4 == entity
        output[selected] = np.column_stack(oracle.macd(values[selected]))
    return pa.table(
        {
            "seq": sequences,
            "line": output[:, 0],
            "signal": output[:, 1],
            "histogram": output[:, 2],
        }
    )


async def _collect_stream(program, runtime, chunks) -> pa.Table:
    result = program.stream({"quotes": chunks}, runtime=runtime)
    async with result:
        tables = [event.table async for event in result]
    if result.job.status()["task_count"] != 0:
        raise AssertionError("native stream benchmark left live job tasks")
    return pa.concat_tables(tables)


def stream_case(scale: BenchmarkScale) -> NativeCase:
    rows = min(scale.table_rows, 20_000)
    prices, sequences = vector(rows), np.arange(rows, dtype=np.uint64)
    schema = pa.schema(
        [
            pa.field("entity", pa.uint64(), nullable=False),
            pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("close", pa.float64()),
        ]
    )
    data = pa.table(
        {
            "entity": sequences % 4,
            "time": sequences.astype(np.int64),
            "seq": sequences,
            "close": prices,
        },
        schema=schema,
    )
    source = cf.table_input(
        "quotes",
        schema=schema,
        entity_by=("entity",),
        event_time="time",
        sequence_by=("seq",),
    )
    parts = cf.ts.macd_components(source["close"])
    output = source.select(
        "seq", line=parts.line, signal=parts.signal, histogram=parts.histogram
    )
    program = cf.Program(
        "native-macd-benchmark", engine="streaming", outputs={"out": output}
    )
    runtime = cf.Runtime()
    chunks = tuple(
        pa.Table.from_batches([batch]) for batch in data.to_batches(max_chunksize=640)
    )
    return NativeCase(
        "native_macd_owned_stream",
        "native-table-stream-full-lifecycle-v1",
        lambda: asyncio.run(_collect_stream(program, runtime, chunks)),
        _stream_reference(prices, sequences),
        (prices, sequences),
        (
            ("batch_rows", 640),
            ("entities", 4),
            ("row_cap", 20_000),
            ("fast_span", 12),
            ("slow_span", 26),
            ("signal_span", 9),
        ),
    )
