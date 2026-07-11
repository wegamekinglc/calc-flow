from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchKind
from calc_flow.context import CancellationToken, RunCancelledError, RunContext
from calc_flow.operator import (
    ExpressionOperator,
    Port,
    SqlOperator,
    StatefulOperator,
    StatelessOperator,
)
from calc_flow.pipeline import Pipeline


def _identity(
    name: str,
    *,
    kind: BatchKind = BatchKind.TABLE,
    input_schema: pa.Schema | None = None,
    output_schema: pa.Schema | None = None,
) -> StatelessOperator:
    return StatelessOperator(
        name,
        lambda inputs, context: {"output": inputs["input"]},
        input_ports=(Port("input", kind, schema=input_schema),),
        output_ports=(Port("output", kind, schema=output_schema),),
    )


def test_linear_sugar_compiles_and_executes_in_order() -> None:
    pipeline = (
        Pipeline("test")
        .then(ExpressionOperator("add_b", "b = a + 1"))
        .add(ExpressionOperator("add_c", "c = b * 2"))
    )

    plan = pipeline.compile()
    result = plan.execute({"input": Batch.table(pa.table({"a": [1, 2]}))})

    assert result.output.table_payload.to_pylist() == [
        {"a": 1, "b": 2, "c": 4},
        {"a": 2, "b": 3, "c": 6},
    ]
    assert [node.node_id for node in plan.nodes] == ["add_b", "add_c"]


def test_graph_fan_out_returns_named_terminal_outputs() -> None:
    pipeline = (
        Pipeline("branch")
        .add_node("root", _identity("root"))
        .add_node("left", ExpressionOperator("left", "value = a + 1"))
        .add_node("right", ExpressionOperator("right", "value = a * 10"))
        .connect("root", "left")
        .connect("root", "right")
    )

    result = pipeline.compile().execute({"input": Batch.table(pa.table({"a": [2]}))})

    assert set(result.outputs) == {"left.output", "right.output"}
    assert result.outputs["left.output"].table_payload["value"].to_pylist() == [3]
    assert result.outputs["right.output"].table_payload["value"].to_pylist() == [20]


def test_graph_fan_in_executes_multi_table_sql() -> None:
    join = SqlOperator(
        "join",
        "select l.id, l.amount + r.amount as total "
        "from left_table l join right_table r on l.id = r.id",
        inputs=("left_table", "right_table"),
    )
    pipeline = (
        Pipeline("join")
        .add_node("left_source", _identity("left_source"))
        .add_node("right_source", _identity("right_source"))
        .add_node("join", join)
        .connect("left_source", "join", target_port="left_table")
        .connect("right_source", "join", target_port="right_table")
    )
    plan = pipeline.compile()

    result = plan.execute(
        {
            "left_source.input": Batch.table(
                pa.table({"id": [1, 2], "amount": [10, 20]})
            ),
            "right_source.input": Batch.table(
                pa.table({"id": [1, 2], "amount": [3, 4]})
            ),
        }
    )

    assert result.output.table_payload.to_pylist() == [
        {"id": 1, "total": 13},
        {"id": 2, "total": 24},
    ]
    assert result.datafusion_metrics[0].node_id == "join"


def test_compile_rejects_cycle() -> None:
    pipeline = (
        Pipeline("cycle")
        .add_node("a", _identity("a"))
        .add_node("b", _identity("b"))
        .connect("a", "b")
        .connect("b", "a")
    )

    with pytest.raises(ValueError, match="cycle"):
        pipeline.compile()


def test_compile_rejects_unknown_nodes_and_ports() -> None:
    with pytest.raises(ValueError, match="unknown source node"):
        Pipeline("bad").add_node("a", _identity("a")).connect("missing", "a").compile()

    with pytest.raises(ValueError, match="unknown output port"):
        (
            Pipeline("bad")
            .add_node("a", _identity("a"))
            .add_node("b", _identity("b"))
            .connect("a", "b", source_port="missing")
            .compile()
        )


def test_compile_rejects_multiple_connections_to_input() -> None:
    pipeline = (
        Pipeline("bad")
        .add_node("a", _identity("a"))
        .add_node("b", _identity("b"))
        .add_node("target", _identity("target"))
        .connect("a", "target")
        .connect("b", "target")
    )

    with pytest.raises(ValueError, match="more than one"):
        pipeline.compile()


def test_compile_rejects_table_to_array_edge() -> None:
    pipeline = (
        Pipeline("bad")
        .add_node("table", _identity("table"))
        .add_node("array", _identity("array", kind=BatchKind.ARRAY))
        .connect("table", "array")
    )

    with pytest.raises(TypeError, match="incompatible edge"):
        pipeline.compile()


def test_execute_validates_input_kind_and_schema() -> None:
    schema = pa.schema([pa.field("x", pa.int64())])
    plan = (
        Pipeline("typed")
        .then(_identity("typed", input_schema=schema, output_schema=schema))
        .compile()
    )

    with pytest.raises(TypeError, match="unexpected Arrow schema"):
        plan.execute({"input": Batch.table(pa.table({"x": ["wrong"]}))})


def test_execute_rejects_missing_and_unknown_inputs() -> None:
    plan = Pipeline("test").then(_identity("identity")).compile()

    with pytest.raises(ValueError, match="missing required"):
        plan.execute({})
    with pytest.raises(ValueError, match="unknown graph inputs"):
        plan.execute(
            {
                "input": Batch.table(pa.table({"x": [1]})),
                "extra": Batch.table(pa.table({"x": [1]})),
            }
        )


def test_execute_rejects_unknown_operator_output() -> None:
    operator = StatelessOperator(
        "bad", lambda inputs, context: {"other": inputs["input"]}
    )
    plan = Pipeline("test").then(operator).compile()

    with pytest.raises(ValueError, match="unknown outputs"):
        plan.execute({"input": Batch.table(pa.table({"x": [1]}))})


def test_execute_cancellation_prevents_node_execution() -> None:
    calls: list[bool] = []

    def process(
        inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]:
        calls.append(True)
        return {"output": inputs["input"]}

    token = CancellationToken()
    token.cancel()
    plan = Pipeline("test").then(StatelessOperator("node", process)).compile()

    with pytest.raises(RunCancelledError):
        plan.execute({"input": Batch.table(pa.table({"x": [1]}))}, cancellation=token)
    assert calls == []


def test_failed_run_rolls_back_stateful_nodes() -> None:
    class Counter(StatefulOperator):
        def process(self, inputs, context):
            self._state["calls"] = self._state.get("calls", 0) + 1
            return {"output": inputs["input"]}

    counter = Counter("counter")

    def fail(inputs, context):
        raise RuntimeError("boom")

    plan = (
        Pipeline("rollback")
        .then(counter)
        .then(StatelessOperator("fail", fail))
        .compile()
    )

    with pytest.raises(RuntimeError, match="boom"):
        plan.execute({"input": Batch.table(pa.table({"x": [1]}))})
    assert counter.snapshot() == {}


def test_plan_is_structurally_immutable_and_fingerprinted() -> None:
    first = Pipeline("test").then(_identity("identity")).compile()
    second = Pipeline("test").then(_identity("identity")).compile()

    assert first.fingerprint == second.fingerprint
    with pytest.raises(AttributeError, match="immutable"):
        first.name = "changed"
    with pytest.raises(TypeError):
        first.graph_inputs["other"] = first.graph_inputs["input"]  # type: ignore[index]


def test_pipeline_rejects_duplicate_node_ids_and_empty_graph() -> None:
    with pytest.raises(ValueError, match="empty pipeline"):
        Pipeline("empty").compile()

    pipeline = Pipeline("test").add_node("node", _identity("first"))
    with pytest.raises(ValueError, match="already"):
        pipeline.add_node("node", _identity("second"))


def test_run_result_contains_timings_and_metadata() -> None:
    plan = Pipeline("metrics").then(_identity("identity")).compile()
    result = plan.execute({"input": Batch.table(pa.table({"x": [1, 2]}))})

    timing = result.node_timings["identity"]
    assert timing.duration_ns > 0
    assert timing.input_rows == {"input": 2}
    assert timing.output_rows == {"output": 2}
    assert result.metadata.pipeline_fingerprint == plan.fingerprint
    assert result.metadata.finished_at >= result.metadata.started_at


def test_run_result_output_rejects_ambiguous_graph() -> None:
    result = (
        Pipeline("ambiguous")
        .add_node("root", _identity("root"))
        .add_node("left", _identity("left"))
        .add_node("right", _identity("right"))
        .connect("root", "left")
        .connect("root", "right")
        .execute({"input": Batch.table(pa.table({"value": [1]}))})
    )

    with pytest.raises(ValueError, match="exactly one"):
        _ = result.output


def test_pipeline_rejects_empty_names_and_non_json_fingerprint_config() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        Pipeline("")
    with pytest.raises(ValueError, match="node ID must not be empty"):
        Pipeline("test").add_node("", _identity("identity"))

    pipeline = Pipeline("bad-fingerprint").then(
        StatelessOperator(
            "node",
            fingerprint_config={"invalid": object()},
        )
    )
    with pytest.raises(TypeError, match="JSON-compatible"):
        pipeline.compile()


def test_compile_rejects_unknown_target_port_and_incompatible_schemas() -> None:
    with pytest.raises(ValueError, match="unknown target node"):
        Pipeline("bad").add_node("a", _identity("a")).connect("a", "missing").compile()

    with pytest.raises(ValueError, match="unknown input port"):
        (
            Pipeline("bad")
            .add_node("a", _identity("a"))
            .add_node("b", _identity("b"))
            .connect("a", "b", target_port="missing")
            .compile()
        )

    with pytest.raises(TypeError, match="incompatible Arrow schemas"):
        (
            Pipeline("bad-schema")
            .add_node(
                "a",
                _identity("a", output_schema=pa.schema([("x", pa.int64())])),
            )
            .add_node(
                "b",
                _identity("b", input_schema=pa.schema([("x", pa.string())])),
            )
            .connect("a", "b")
            .compile()
        )


def test_compile_rejects_graph_without_output_and_invalid_linear_sugar() -> None:
    with pytest.raises(ValueError, match="no reachable outputs"):
        Pipeline("no-output").then(
            StatelessOperator("sink", _identity, output_ports=())
        ).compile()

    pipeline = Pipeline("linear").then(_identity("source"))
    with pytest.raises(ValueError, match="linear sugar"):
        pipeline.then(
            SqlOperator(
                "join",
                "select * from left_table",
                inputs=("left_table", "right_table"),
            )
        )


def test_execute_rejects_naive_deadline_and_invalid_operator_results() -> None:
    plan = Pipeline("deadline").then(_identity("node")).compile()
    batch = Batch.table(pa.table({"value": [1]}))
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone"):
        plan.execute({"input": batch}, deadline=datetime.now())

    not_mapping = Pipeline("not-mapping").then(
        StatelessOperator("node", lambda inputs, context: [])  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="must return a mapping"):
        not_mapping.execute({"input": batch})

    missing = Pipeline("missing-output").then(
        StatelessOperator("node", lambda inputs, context: {})
    )
    with pytest.raises(ValueError, match="omitted required outputs"):
        missing.execute({"input": batch})


def test_pipeline_convenience_methods_delegate_to_compiled_plan() -> None:
    class Counter(StatefulOperator):
        def process(self, inputs, context):
            self._state["calls"] = self._state.get("calls", 0) + 1
            return {"output": inputs["input"]}

    counter = Counter("counter")
    pipeline = Pipeline("convenience").then(counter)
    batch = Batch.table(pa.table({"value": [1]}))

    pipeline.execute({"input": batch})
    assert pipeline.snapshot() == {"counter": {"calls": 1}}
    pipeline.restore({"counter": {"calls": 5}})
    assert counter.snapshot() == {"calls": 5}
    assert list(pipeline) == [counter]
    assert repr(pipeline) == "Pipeline(name='convenience', nodes=[counter])"

    pipeline.reset()
    assert counter.snapshot() == {}


def test_execution_plan_restore_rejects_unknown_nodes() -> None:
    plan = Pipeline("restore").then(_identity("node")).compile()

    with pytest.raises(ValueError, match="unknown nodes"):
        plan.restore({"missing": {"value": 1}})
