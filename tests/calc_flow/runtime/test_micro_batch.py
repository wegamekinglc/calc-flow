from __future__ import annotations

import pytest

from calc_flow.batch import Batch
from calc_flow.checkpoint import (
    CheckpointMismatchError,
    FileCheckpointStore,
)
from calc_flow.io import BatchingSource
from calc_flow.operator import ExpressionOperator, StatefulOperator, StatelessOperator
from calc_flow.pipeline import Pipeline
from calc_flow.runtime.micro_batch import MicroBatchRunner


class _RowCounter(StatefulOperator):
    def process(self, inputs, context):
        batch = inputs["input"]
        self._state["total_rows"] = self._state.get("total_rows", 0) + len(batch)
        return {"output": batch}


class _CollectSink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []

    def write(self, batch: Batch) -> None:
        self.batches.append(batch)


class _FailingSink:
    def write(self, batch: Batch) -> None:
        raise RuntimeError("sink failed")


def _source() -> BatchingSource:
    return BatchingSource(
        [{"value": index} for index in range(5)],
        source_id="records",
        max_rows=2,
    )


def test_micro_batch_runner_yields_results_and_delivers_sink(tmp_path) -> None:
    counter = _RowCounter("counter")
    plan = Pipeline("test").then(counter).compile()
    store = FileCheckpointStore(tmp_path)
    sink = _CollectSink()
    runner = MicroBatchRunner(plan, checkpoint_every=2, checkpoint_store=store)

    results = list(runner.run(_source(), sink))

    assert [result.output.num_rows for result in results] == [2, 2, 1]
    assert len(sink.batches) == 3
    assert counter.snapshot() == {"total_rows": 5}
    checkpoint = store.load("test")
    assert checkpoint is not None
    assert checkpoint.source_cursor == 5
    assert checkpoint.sequence == 2
    assert checkpoint.state == {"counter": {"total_rows": 5}}


def test_micro_batch_runner_recovers_state_and_source_cursor(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    first_counter = _RowCounter("counter")
    first = MicroBatchRunner(
        Pipeline("test").then(first_counter),
        checkpoint_every=1,
        checkpoint_store=store,
    )
    first_run = first.run(_source())

    first_result = next(first_run)
    first_run.close()
    assert first_result.output.table_payload["value"].to_pylist() == [0, 1]

    recovered_counter = _RowCounter("counter")
    recovered = MicroBatchRunner(
        Pipeline("test").then(recovered_counter),
        checkpoint_every=1,
        checkpoint_store=store,
    )
    remaining = list(recovered.run(_source()))

    assert [
        result.output.table_payload["value"].to_pylist() for result in remaining
    ] == [[2, 3], [4]]
    assert recovered_counter.snapshot() == {"total_rows": 5}


def test_micro_batch_runner_does_not_checkpoint_failed_sink(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    counter = _RowCounter("counter")
    runner = MicroBatchRunner(
        Pipeline("test").then(counter),
        checkpoint_every=1,
        checkpoint_store=store,
    )

    with pytest.raises(RuntimeError, match="sink failed"):
        list(runner.run(_source(), _FailingSink()))

    assert store.load("test") is None
    assert counter.snapshot() == {}


def test_micro_batch_runner_has_at_least_once_multi_sink_behavior(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    first_sink = _CollectSink()
    pipeline = (
        Pipeline("branch")
        .add_node(
            "root",
            StatelessOperator(
                "root", lambda inputs, context: {"output": inputs["input"]}
            ),
        )
        .add_node("left", ExpressionOperator("left", "left_value = value + 1"))
        .add_node("right", ExpressionOperator("right", "right_value = value * 2"))
        .connect("root", "left")
        .connect("root", "right")
    )
    runner = MicroBatchRunner(pipeline, checkpoint_every=1, checkpoint_store=store)

    with pytest.raises(RuntimeError, match="sink failed"):
        list(
            runner.run(
                BatchingSource([{"value": 1}], source_id="records", max_rows=1),
                {
                    "left.output": first_sink,
                    "right.output": _FailingSink(),
                },
            )
        )

    assert len(first_sink.batches) == 1
    assert store.load("branch") is None


def test_micro_batch_runner_rejects_stale_checkpoint(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    first = MicroBatchRunner(
        Pipeline("test").then(ExpressionOperator("calculate", "x = value + 1")),
        checkpoint_every=1,
        checkpoint_store=store,
    )
    list(first.run(_source()))

    changed = MicroBatchRunner(
        Pipeline("test").then(ExpressionOperator("calculate", "x = value + 2")),
        checkpoint_every=1,
        checkpoint_store=store,
    )

    with pytest.raises(CheckpointMismatchError):
        list(changed.run(_source()))


def test_micro_batch_runner_reset_clears_state_and_checkpoint(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    counter = _RowCounter("counter")
    runner = MicroBatchRunner(
        Pipeline("test").then(counter),
        checkpoint_every=1,
        checkpoint_store=store,
    )
    list(runner.run(_source()))

    runner.reset()

    assert counter.snapshot() == {}
    assert store.load("test") is None


def test_micro_batch_runner_rejects_invalid_settings_and_multi_input_plan(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="checkpoint_every"):
        MicroBatchRunner(
            Pipeline("test").then(_RowCounter("counter")),
            checkpoint_every=0,
        )

    multi_input = Pipeline("join").add_node(
        "join",
        StatelessOperator(
            "join",
            lambda inputs, context: {"output": next(iter(inputs.values()))},
            input_ports=(),
        ),
    )
    # A zero-input plan is also unsuitable for a source-driven runner.
    with pytest.raises(ValueError, match="exactly one"):
        MicroBatchRunner(multi_input, checkpoint_store=FileCheckpointStore(tmp_path))
