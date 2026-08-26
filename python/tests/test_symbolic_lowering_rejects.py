from __future__ import annotations

import pytest

from calc_flow import Runtime
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    cs,
    exact_time,
    linalg,
    parameter,
    row,
    rows,
    table_input,
    ts,
    window,
)


def _ordered() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=False),
            Field("y", "float64", nullable=False),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _plain() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("x", "float64", nullable=False),
            Field("y", "float64", nullable=False),
        ],
    )


def _reject_message(error: pytest.ExceptionInfo[CompileError]) -> str:
    return str(error.value)


def test_stream_rolling_aggregate_is_rejected_not_made_batch_local() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet([("momentum", ts.mean(quotes["x"], window=rows(20)))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_stream(Runtime())

    message = _reject_message(excinfo)
    assert "outputs.signals.momentum" in message
    assert "unknown_primitive_version" in message
    assert "'mean'" in message


def test_batch_rolling_aggregate_is_rejected() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet([("momentum", ts.mean(quotes["x"], window=rows(20)))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError, match="unknown_primitive_version"):
        program.compile_batch(Runtime())


def test_cross_section_is_rejected() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet([("alpha", cs.zscore(quotes["x"], group=exact_time(quotes["ts"])))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_stream(Runtime())

    message = _reject_message(excinfo)
    assert "outputs.signals.alpha" in message
    assert "unknown_primitive_version" in message
    assert "'zscore'" in message


def test_sql_window_is_rejected_in_stream_mode() -> None:
    quotes = _ordered()
    windowed = window.tumbling(quotes, event_time="ts", size_micros=60_000_000)
    program = Program("p", inputs=[quotes], outputs=[("signals", windowed)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_stream(Runtime())

    message = _reject_message(excinfo)
    assert "unknown_primitive_version" in message
    assert "'window_tumbling'" in message


def test_array_outputs_are_rejected() -> None:
    quotes = _ordered()
    matrix = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    program = Program("p", inputs=[quotes], outputs=[("matrix", matrix)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())

    message = _reject_message(excinfo)
    assert "outputs.matrix" in message
    assert "'from_columns'" in message


def test_static_table_parameter_is_rejected_even_when_unused() -> None:
    quotes = _plain()
    reference = parameter(
        "reference",
        kind="table",
        schema=[Field("x", "float64", nullable=False)],
    )
    program = Program(
        "p",
        inputs=[quotes, reference],
        outputs=[("signals", quotes)],
    )

    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())

    message = _reject_message(excinfo)
    assert "static_inputs.reference" in message
    assert "unknown_primitive_version" in message


def test_static_array_parameter_is_rejected() -> None:
    quotes = _plain()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    program = Program(
        "p",
        inputs=[quotes, weights],
        outputs=[("signals", quotes)],
    )

    with pytest.raises(CompileError) as excinfo:
        program.compile_stream(Runtime())

    assert "static_inputs.weights" in _reject_message(excinfo)


def test_matmul_is_rejected() -> None:
    quotes = _ordered()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    matrix = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    scores = linalg.matmul(matrix, weights)
    program = Program("p", inputs=[quotes, weights], outputs=[("scores", scores)])

    with pytest.raises(CompileError, match="unknown_primitive_version"):
        program.compile_batch(Runtime())


def test_cast_to_non_portable_target_is_rejected() -> None:
    quotes = _plain()
    signals = quotes.with_columns(
        FeatureSet([("text", row.cast(quotes["x"], "string"))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())

    message = _reject_message(excinfo)
    assert "outputs.signals.text.cast.data_type" in message
    assert "unsupported_type" in message


def test_attach_columns_is_rejected() -> None:
    quotes = _ordered()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    matrix = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    scores = linalg.matmul(matrix, weights)
    from calc_flow.symbolic import table

    attached = table.attach_columns(quotes, scores, names=["score"])
    program = Program("p", inputs=[quotes, weights], outputs=[("signals", attached)])

    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())

    assert "unknown_primitive_version" in _reject_message(excinfo)
