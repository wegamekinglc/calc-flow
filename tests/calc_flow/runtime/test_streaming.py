from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchMetadata
from calc_flow.checkpoint import FileCheckpointStore
from calc_flow.operator import StatefulOperator
from calc_flow.pipeline import Pipeline
from calc_flow.runtime.streaming import StreamingRunner


class _RunningMax(StatefulOperator):
    def process(self, inputs, context):
        batch = inputs["input"]
        for value in batch.table_payload["value"].to_pylist():
            current = self._state.get("max", float("-inf"))
            self._state["max"] = max(current, value)
        return {"output": batch}


class _CollectSink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []

    def write(self, batch: Batch) -> None:
        self.batches.append(batch)


class _FailingSink:
    def write(self, batch: Batch) -> None:
        raise RuntimeError("sink failed")


def _batch(*values: int, cursor: int | None = None) -> Batch:
    return Batch.table(
        pa.table({"value": values}),
        metadata=BatchMetadata(cursor=cursor),
    )


def test_streaming_runner_returns_run_result_and_delivers_sink(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    operator = _RunningMax("max")
    sink = _CollectSink()
    runner = StreamingRunner(Pipeline("test").then(operator), checkpoint_store=store)

    result = runner.step(_batch(5, 3, cursor=1), sink)

    assert result.output.table_payload["value"].to_pylist() == [5, 3]
    assert operator.snapshot() == {"max": 5}
    assert sink.batches == [result.output]
    checkpoint = store.load("test")
    assert checkpoint is not None
    assert checkpoint.sequence == 0
    assert checkpoint.source_cursor == 1


def test_streaming_runner_recovers_state_once(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    first_operator = _RunningMax("max")
    first = StreamingRunner(
        Pipeline("test").then(first_operator), checkpoint_store=store
    )
    first.step(_batch(10, cursor=1))

    recovered_operator = _RunningMax("max")
    recovered = StreamingRunner(
        Pipeline("test").then(recovered_operator), checkpoint_store=store
    )
    recovered.step(_batch(5, cursor=2))
    recovered.step(_batch(7, cursor=3))

    assert recovered_operator.snapshot() == {"max": 10}
    checkpoint = store.load("test")
    assert checkpoint is not None
    assert checkpoint.sequence == 2


def test_streaming_runner_rolls_back_failed_sink(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    operator = _RunningMax("max")
    runner = StreamingRunner(Pipeline("test").then(operator), checkpoint_store=store)

    with pytest.raises(RuntimeError, match="sink failed"):
        runner.step(_batch(10, cursor=1), _FailingSink())

    assert operator.snapshot() == {}
    assert store.load("test") is None


def test_streaming_runner_reset_clears_state_and_checkpoint(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    operator = _RunningMax("max")
    runner = StreamingRunner(Pipeline("test").then(operator), checkpoint_store=store)
    runner.step(_batch(10, cursor=1))

    runner.reset()

    assert operator.snapshot() == {}
    assert store.load("test") is None


def test_streaming_runner_rejects_raw_batch(tmp_path) -> None:
    runner = StreamingRunner(
        Pipeline("test").then(_RunningMax("max")),
        checkpoint_store=FileCheckpointStore(tmp_path),
    )

    with pytest.raises(TypeError, match="requires a Batch"):
        runner.step(pa.table({"value": [1]}))  # type: ignore[arg-type]
