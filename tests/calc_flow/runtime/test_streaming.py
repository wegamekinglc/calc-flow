from __future__ import annotations

import tempfile

from calc_flow.batch import Batch
from calc_flow.operator import StatefulOperator
from calc_flow.pipeline import Pipeline
from calc_flow.runtime.streaming import StreamingRunner


class _RunningMax(StatefulOperator):
    def apply(self, batch: Batch) -> Batch:
        for row in batch.to_pylist():
            cur = self._state.get("max", float("-inf"))
            self._state["max"] = max(cur, row["val"])
        return batch


def test_streaming_runner_step() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RunningMax("max_tracker")
        p = Pipeline("test").add(op)
        runner = StreamingRunner(p, checkpoint_dir=tmpdir)

        runner.step(Batch.from_pylist([{"val": 5}, {"val": 3}]))
        assert op.snapshot()["max"] == 5

        runner.step(Batch.from_pylist([{"val": 10}, {"val": 8}]))
        assert op.snapshot()["max"] == 10


def test_streaming_runner_step_returns_result() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Pipeline("test")
        runner = StreamingRunner(p, checkpoint_dir=tmpdir)

        batch = Batch.from_pylist([{"a": 1}])
        result = runner.step(batch)
        assert result is batch


def test_streaming_runner_reset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RunningMax("max_tracker")
        p = Pipeline("test").add(op)
        runner = StreamingRunner(p, checkpoint_dir=tmpdir)

        runner.step(Batch.from_pylist([{"val": 10}]))
        runner.reset()
        assert op.snapshot() == {}


def test_streaming_runner_reset_clears_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RunningMax("max_tracker")
        p = Pipeline("test").add(op)
        runner = StreamingRunner(p, checkpoint_dir=tmpdir)

        runner.step(Batch.from_pylist([{"val": 10}]))
        runner.reset()
        runner.step(Batch.from_pylist([{"val": 2}]))

        assert op.snapshot()["max"] == 2


def test_streaming_runner_recovers_checkpoint_once() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first_op = _RunningMax("max_tracker")
        first_pipeline = Pipeline("test").add(first_op)
        first_runner = StreamingRunner(first_pipeline, checkpoint_dir=tmpdir)
        first_runner.step(Batch.from_pylist([{"val": 10}]))

        recovered_op = _RunningMax("max_tracker")
        recovered_pipeline = Pipeline("test").add(recovered_op)
        recovered_runner = StreamingRunner(recovered_pipeline, checkpoint_dir=tmpdir)
        recovered_runner.step(Batch.from_pylist([{"val": 5}]))
        recovered_runner.step(Batch.from_pylist([{"val": 7}]))

        assert recovered_op.snapshot()["max"] == 10
