"""Independent acceptance vectors for SCE-06 Python lowering.

Authored by cf-tester during the independent verification of PR #198 and
absorbed into the shipped suite: non-goal rejections, lowering shape, batch
execution values, and caller-input immutability with independently chosen
fixtures and an independently derived reference model.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    duration,
    exact_time,
    rows,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import lower_program_document


def _ordered() -> object:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
            Field("v", "int64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _program(features: list[tuple[str, object]]) -> Program:
    quotes = _ordered()
    signals = quotes.with_columns(FeatureSet(features))
    return Program("p", inputs=[quotes], outputs=[("signals", signals)])


def _rejected(features: list[tuple[str, object]], mode: str = "batch") -> str:
    program = _program(features)
    with pytest.raises(CompileError) as excinfo:
        if mode == "batch":
            program.compile_batch(Runtime())
        else:
            program.compile_stream(Runtime())
    return str(excinfo.value)


def test_aggregate_catalog_kinds_stay_rejected() -> None:
    quotes = _ordered()
    aggregates = [
        ts.count(quotes["x"], window=rows(5)),
        ts.sum(quotes["x"], window=rows(5)),
        ts.mean(quotes["x"], window=rows(5)),
        ts.variance(quotes["x"], window=rows(5)),
        ts.stddev(quotes["x"], window=rows(5)),
    ]
    for aggregate in aggregates:
        message = _rejected([("feature", aggregate)])
        assert "unknown_primitive_version" in message


def test_duration_frame_and_correlation_stay_rejected() -> None:
    quotes = _ordered()
    message = _rejected([("m", ts.mean(quotes["x"], window=duration(1_000)))])
    assert "unknown_primitive_version" in message
    message = _rejected(
        [("c", ts.correlation(quotes["x"], quotes["v"], window=rows(5)))],
        mode="stream",
    )
    assert "unknown_primitive_version" in message


def test_cross_section_stays_rejected() -> None:
    quotes = _ordered()
    message = _rejected(
        [("alpha", cs.zscore(quotes["x"], group=exact_time(quotes["ts"])))]
    )
    assert "unknown_primitive_version" in message
    message = _rejected(
        [("r", cs.rank(quotes["x"], group=exact_time(quotes["ts"])))],
        mode="stream",
    )
    assert "unknown_primitive_version" in message


def test_composite_lag_delta_arguments_stay_rejected() -> None:
    quotes = _ordered()
    message = _rejected([("prev", ts.lag(quotes["x"] + 1.0))])
    assert "must be an input column" in message
    message = _rejected([("change", ts.delta(quotes["x"] * 2.0))])
    assert "must be an input column" in message

    derived = quotes.with_columns(FeatureSet([("y", quotes["x"] + 1.0)]))
    signals = derived.with_columns(FeatureSet([("prev", ts.lag(derived["y"]))]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])
    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())
    assert "must be an input column" in str(excinfo.value)


def test_probe_lowering_shape_with_periods_and_lateness() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("prev3", ts.lag(quotes["x"], periods=3)),
            ("dvol2", ts.delta(quotes["v"], periods=2)),
        ]
    )

    document = lower_program_document(
        program,
        Runtime(),
        "stream",
        allowed_lateness_micros=9,
        late_policy="drop",
    )

    rolling = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    assert len(rolling) == 1
    assert rolling[0]["operator"]["spec"] == {
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["seq"],
        "outputs": [
            {
                "kind": "lag",
                "primitive_version": 1,
                "input": "x",
                "output": "prev3",
                "periods": 3,
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "v",
                "output": "dvol2",
                "periods": 2,
            },
        ],
        "allowed_lateness_micros": 9,
        "late_policy": {"kind": "drop", "metrics_version": 1},
        "value_policy": "stateful_numeric_v1",
    }


def test_lowering_does_not_mutate_the_program() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    first = lower_program_document(program, Runtime(), "batch")
    second = lower_program_document(program, Runtime(), "batch")

    assert first == second


_PROBE_ROWS = [
    (12, "a", 3, 3.0, 120),
    (10, "a", 1, 1.0, 100),
    (15, "b", 4, 14.0, 800),
    (11, "a", 2, None, 110),
    (10, "b", 1, 10.0, 1000),
    (14, "a", 5, float("nan"), 140),
    (12, "a", 4, 3.5, 130),
    (13, "b", 3, 13.0, None),
    (11, "b", 2, 11.0, 900),
    (16, "a", 6, 6.0, 150),
]


# The independent reference oracle keeps per-entity lag/delta derivation in
# one flat loop for line-by-line auditability against the engine.
def _reference(rows_data, lag_periods, delta_periods):
    # #lizard forgives
    ordered = sorted(rows_data, key=lambda row: (row[0], row[1], row[2]))
    by_entity: dict[str, list] = {}
    for row in ordered:
        by_entity.setdefault(row[1], []).append(row)
    lag_of: dict[int, object] = {}
    delta_of: dict[int, object] = {}
    for entity_rows in by_entity.values():
        for index, row in enumerate(entity_rows):
            lag_of[id(row)] = (
                entity_rows[index - lag_periods][3] if index >= lag_periods else None
            )
            if index < delta_periods:
                delta_of[id(row)] = None
                continue
            reference = entity_rows[index - delta_periods][4]
            current = row[4]
            delta_of[id(row)] = (
                None if current is None or reference is None else current - reference
            )
    return [
        (row[0], row[1], row[2], lag_of[id(row)], delta_of[id(row)]) for row in ordered
    ]


def _probe_table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64()),
            pa.field("v", pa.int64()),
        ]
    )
    return pa.table(
        {
            "ts": pa.array(
                [row[0] for row in _PROBE_ROWS], type=pa.timestamp("us", tz="UTC")
            ),
            "symbol": [row[1] for row in _PROBE_ROWS],
            "seq": pa.array([row[2] for row in _PROBE_ROWS], type=pa.uint64()),
            "x": pa.array([row[3] for row in _PROBE_ROWS], type=pa.float64()),
            "v": pa.array([row[4] for row in _PROBE_ROWS], type=pa.int64()),
        },
        schema=schema,
    )


def _table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _assert_values_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got[:3] == want[:3]
        for column in (3, 4):
            if want[column] is None:
                assert got[column] is None
            elif isinstance(want[column], float) and math.isnan(want[column]):
                assert isinstance(got[column], float) and math.isnan(got[column])
            else:
                assert got[column] == want[column]


def test_batch_execution_matches_independent_reference_and_preserves_input() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("prev3", ts.lag(quotes["x"], periods=3)),
            ("dvol2", ts.delta(quotes["v"], periods=2)),
        ]
    )
    table = _probe_table()
    before = _table_bytes(table)

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(table)})

    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == ["ts", "symbol", "seq", "x", "v", "prev3", "dvol2"]
    ts_micros = output.column("ts").cast(pa.int64()).to_pylist()
    columns = output.drop_columns(["ts"]).to_pydict()
    actual = list(
        zip(
            ts_micros,
            columns["symbol"],
            columns["seq"],
            columns["prev3"],
            columns["dvol2"],
            strict=True,
        )
    )
    _assert_values_match(actual, _reference(_PROBE_ROWS, 3, 2))
    assert _table_bytes(table) == before, "caller-owned input table was mutated"
