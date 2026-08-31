from __future__ import annotations

import pytest

from calc_flow import ConfigError, Runtime, register_jax, register_numpy
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


def test_stream_rolling_pair_row_local_argument_is_materialized() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet(
            [
                (
                    "corr",
                    ts.correlation(quotes["x"] + 1.0, quotes["y"], window=rows(20)),
                )
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    program.compile_stream(Runtime())


def test_batch_rolling_pair_row_local_argument_is_materialized() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet(
            [
                (
                    "cov",
                    ts.covariance(quotes["x"], quotes["y"] * 2.0, window=rows(20)),
                )
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    program.compile_batch(Runtime())


def test_cross_section_winsorize_compiles_in_both_modes() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet(
            [
                (
                    "clipped",
                    cs.winsorize(
                        quotes["x"],
                        group=exact_time(quotes["ts"]),
                        lower=0.1,
                        upper=0.9,
                    ),
                )
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    program.compile_batch(Runtime())
    program.compile_stream(Runtime())


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


def test_attach_columns_requires_registered_matrix_provider() -> None:
    quotes = _ordered()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    matrix = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    scores = linalg.matmul(matrix, weights)
    from calc_flow.symbolic import table

    attached = table.attach_columns(quotes, scores, names=["score"])
    program = Program("p", inputs=[quotes, weights], outputs=[("signals", attached)])

    with pytest.raises(ConfigError) as excinfo:
        program.compile_batch(Runtime())

    assert "provider numpy:symbolic_matrix@1 is unavailable" in _reject_message(excinfo)


def test_symbolic_matrix_rejects_mixed_provider_backends() -> None:
    quotes = _plain()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    numpy_matrix = linalg.from_columns(quotes, columns=("x", "y"), backend="numpy")
    jax_matrix = linalg.from_columns(quotes, columns=("x", "y"), backend="jax")
    from calc_flow.symbolic import table

    program = Program(
        "mixed-matrix-backends",
        inputs=(quotes, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    quotes,
                    linalg.matmul(numpy_matrix + jax_matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)
    register_jax(runtime)

    with pytest.raises(CompileError, match="implicit cross-backend conversion"):
        program.compile_batch(runtime)


def test_symbolic_matrix_rejects_mixed_column_selections() -> None:
    quotes = _plain()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    x_matrix = linalg.from_columns(quotes, columns=("x",), backend="numpy")
    y_matrix = linalg.from_columns(quotes, columns=("y",), backend="numpy")
    from calc_flow.symbolic import table

    program = Program(
        "mixed-matrix-columns",
        inputs=(quotes, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    quotes,
                    linalg.matmul(x_matrix + y_matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(CompileError, match="one ordered column selection"):
        program.compile_batch(runtime)


def test_symbolic_matrix_rejects_undeclared_parameter_with_same_name() -> None:
    quotes = _plain()
    first = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    second = parameter(
        "weights", kind="array", backend="numpy", dtype="float32", shape=(1, 1)
    )
    matrix = linalg.from_columns(quotes, columns=("x", "y"), backend="numpy")
    from calc_flow.symbolic import table

    program = Program(
        "distinct-matrix-parameters",
        inputs=(quotes, first),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    quotes,
                    linalg.matmul(matrix, first) + second,
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(
        CompileError, match="referenced by program outputs but not declared"
    ):
        program.compile_batch(runtime)


def test_symbolic_matrix_rejects_non_weights_static_parameter() -> None:
    quotes = _plain()
    coefficients = parameter(
        "coefficients",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    matrix = linalg.from_columns(quotes, columns=("x", "y"), backend="numpy")
    from calc_flow.symbolic import table

    program = Program(
        "named-matrix-parameter",
        inputs=(quotes, coefficients),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    quotes,
                    linalg.matmul(matrix, coefficients),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(
        CompileError,
        match="requires the static array parameter name 'weights'",
    ):
        program.compile_batch(runtime)


def test_symbolic_matrix_rejects_missing_static_parameter() -> None:
    quotes = table_input(
        "quotes",
        schema=[Field("x", "float64", nullable=False)],
    )
    matrix = linalg.from_columns(quotes, columns=("x",), backend="numpy")
    from calc_flow.symbolic import table

    program = Program(
        "missing-matrix-parameter",
        inputs=(quotes,),
        outputs=(
            (
                "signals",
                table.attach_columns(quotes, matrix + 1.0, names=("score",)),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(CompileError, match="requires one static array parameter"):
        program.compile_batch(runtime)
