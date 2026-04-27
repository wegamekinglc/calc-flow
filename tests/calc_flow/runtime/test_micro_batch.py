from __future__ import annotations

import tempfile

from calc_flow.batch import Batch
from calc_flow.operator import StatefulOperator
from calc_flow.pipeline import Pipeline
from calc_flow.runtime.micro_batch import MicroBatchRunner


class _RowCounter(StatefulOperator):
    def apply(self, batch: Batch) -> Batch:
        self._state["total_rows"] = self._state.get("total_rows", 0) + batch.num_rows
        return batch


def test_micro_batch_runner() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RowCounter("counter")
        p = Pipeline("test").add(op)

        runner = MicroBatchRunner(
            p,
            batch_size=3,
            checkpoint_every=5,
            checkpoint_dir=tmpdir,
        )

        batches = [Batch.from_pylist([{"x": i}] * 3) for i in range(10)]
        runner.run(iter(batches))

        assert op.snapshot()["total_rows"] == 30


def test_micro_batch_runner_checkpoint_saved() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RowCounter("counter")
        p = Pipeline("test").add(op)

        runner = MicroBatchRunner(
            p,
            batch_size=10,
            checkpoint_every=2,
            checkpoint_dir=tmpdir,
        )

        batches = [Batch.from_pylist([{"x": i}] * 10) for i in range(5)]
        runner.run(iter(batches))

        from calc_flow.checkpoint import CheckpointManager

        mgr = CheckpointManager(tmpdir)
        cp = mgr.load("test")
        assert cp is not None
        assert cp.offset == 5


def test_micro_batch_runner_reset_clears_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        op = _RowCounter("counter")
        p = Pipeline("test").add(op)
        runner = MicroBatchRunner(p, checkpoint_every=1, checkpoint_dir=tmpdir)

        runner.run(iter([Batch.from_pylist([{"x": 1}])]))
        runner.reset()
        runner.run(iter([Batch.from_pylist([{"x": 2}])]))

        assert op.snapshot()["total_rows"] == 1


def test_micro_batch_runner_rejects_invalid_settings() -> None:
    p = Pipeline("test")

    import pytest

    with pytest.raises(ValueError, match="batch_size"):
        MicroBatchRunner(p, batch_size=0)

    with pytest.raises(ValueError, match="checkpoint_every"):
        MicroBatchRunner(p, checkpoint_every=0)
