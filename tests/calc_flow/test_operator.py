from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchKind
from calc_flow.context import RunContext
from calc_flow.operator import (
    ArrayExpressionOperator,
    ExpressionOperator,
    Port,
    SqlOperator,
    StatefulOperator,
    StatelessOperator,
)
from calc_flow.pipeline import Pipeline


def _identity(inputs: Mapping[str, Batch], context: RunContext) -> Mapping[str, Batch]:
    context.check_cancelled()
    return {"output": inputs["input"]}


def test_stateless_operator_runs_through_plan() -> None:
    batch = Batch.table(pa.table({"x": [1, 2]}))
    operator = StatelessOperator("identity", _identity)

    result = Pipeline("test").then(operator).compile().execute({"input": batch})

    assert result.output is batch


def test_stateful_operator_snapshot_restore_and_reset() -> None:
    class RunningSum(StatefulOperator):
        def process(self, inputs, context):
            total = self._state.get("sum", 0)
            total += sum(inputs["input"].table_payload["val"].to_pylist())
            self._state["sum"] = total
            return {"output": inputs["input"]}

    operator = RunningSum("accum")
    plan = Pipeline("test").then(operator).compile()
    plan.execute({"input": Batch.table(pa.table({"val": [10, 20]}))})

    assert operator.snapshot() == {"sum": 30}
    operator.restore({"sum": 5})
    plan.execute({"input": Batch.table(pa.table({"val": [2]}))})
    assert operator.snapshot() == {"sum": 7}
    operator.reset()
    assert operator.snapshot() == {}


def test_expression_operator_uses_datafusion() -> None:
    plan = Pipeline("test").then(ExpressionOperator("sum", "c = a + b")).compile()

    result = plan.execute({"input": Batch.table(pa.table({"a": [1, 2], "b": [3, 4]}))})

    assert result.output.table_payload.to_pylist() == [
        {"a": 1, "b": 3, "c": 4},
        {"a": 2, "b": 4, "c": 6},
    ]
    assert result.datafusion_metrics[0].node_id == "sum"


def test_expression_operator_projects_and_filters() -> None:
    operator = ExpressionOperator(
        "filter",
        select=("a", "b * 2 AS doubled"),
        filter_expression="a >= 2",
    )
    plan = Pipeline("test").then(operator).compile()

    result = plan.execute(
        {"input": Batch.table(pa.table({"a": [1, 2], "b": [10, 20]}))}
    )

    assert result.output.table_payload.to_pylist() == [{"a": 2, "doubled": 40}]


def test_expression_operator_requires_one_calculation_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ExpressionOperator("bad")
    with pytest.raises(ValueError, match="exactly one"):
        ExpressionOperator("bad", "a + 1", select=("a",))


def test_sql_operator_declares_named_table_inputs() -> None:
    operator = SqlOperator(
        "join",
        "select l.id, l.value + r.value as total "
        "from left_table l join right_table r on l.id = r.id",
        inputs=("left_table", "right_table"),
    )
    plan = Pipeline("test").add_node("join", operator).compile()

    result = plan.execute(
        {
            "left_table": Batch.table(pa.table({"id": [1], "value": [10]})),
            "right_table": Batch.table(pa.table({"id": [1], "value": [5]})),
        }
    )

    assert result.output.table_payload.to_pylist() == [{"id": 1, "total": 15}]


def test_operator_rejects_duplicate_ports() -> None:
    with pytest.raises(ValueError, match="duplicate input"):
        StatelessOperator(
            "bad",
            _identity,
            input_ports=(
                Port("value", BatchKind.TABLE),
                Port("value", BatchKind.TABLE),
            ),
        )


def test_port_rejects_schema_for_array() -> None:
    with pytest.raises(ValueError, match="only table"):
        Port("array", BatchKind.ARRAY, schema=pa.schema([("x", pa.int64())]))


def test_sql_operator_rejects_schema_for_unknown_alias() -> None:
    with pytest.raises(ValueError, match="unknown SQL inputs"):
        SqlOperator(
            "bad",
            "select * from input",
            inputs=("input",),
            input_schemas={"other": pa.schema([("x", pa.int64())])},
        )


def test_port_rejects_invalid_name_raw_values_and_wrong_kind() -> None:
    with pytest.raises(ValueError, match="invalid port name"):
        Port("not-valid", BatchKind.TABLE)

    port = Port("input", BatchKind.TABLE)
    with pytest.raises(TypeError, match="requires a Batch"):
        port.validate(object(), endpoint="test")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requires a table batch"):
        import numpy as np

        port.validate(Batch.array(np.asarray([1])), endpoint="test")


def test_operator_rejects_empty_name_and_invalid_udf_reference() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        StatelessOperator("", _identity)
    with pytest.raises(TypeError, match="UdfReference"):
        ExpressionOperator("bad", "value + 1", udfs=("bad",))  # type: ignore[arg-type]


def test_stateless_operator_without_function_is_descriptive() -> None:
    operator = StatelessOperator("unimplemented", fingerprint_config={"mode": "test"})
    plan = Pipeline("test").then(operator).compile()

    assert repr(operator) == "StatelessOperator(name='unimplemented')"
    assert operator.configuration() == {"mode": "test"}
    with pytest.raises(NotImplementedError, match="override process"):
        plan.execute({"input": Batch.table(pa.table({"value": [1]}))})


def test_specialized_operators_reject_invalid_construction() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        SqlOperator("sql", "select 1", inputs=())
    with pytest.raises(ValueError, match="numpy.*jax"):
        ArrayExpressionOperator("array", "x + 1", backend="other")
