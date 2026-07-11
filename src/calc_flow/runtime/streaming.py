from __future__ import annotations

from calc_flow.batch import Batch
from calc_flow.checkpoint import (
    CheckpointStore,
    FileCheckpointStore,
    validate_checkpoint,
)
from calc_flow.pipeline import ExecutionPlan, Pipeline, RunResult
from calc_flow.runtime.micro_batch import (
    SinkTarget,
    _coerce_plan,
    _make_checkpoint,
    _single_input_name,
    _write_outputs,
)


class StreamingRunner:
    """Execute one formed batch at a time with at-least-once checkpoints."""

    def __init__(
        self,
        pipeline: Pipeline | ExecutionPlan,
        *,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.plan = _coerce_plan(pipeline)
        self.checkpoint_store = (
            checkpoint_store if checkpoint_store is not None else FileCheckpointStore()
        )
        self._input_name = _single_input_name(self.plan)
        self._sequence = 0
        self._recovered = False

    def step(self, batch: Batch, sink: SinkTarget | None = None) -> RunResult:
        """Process, deliver, and checkpoint one batch before returning its result."""
        if not isinstance(batch, Batch):
            msg = "StreamingRunner.step() requires a Batch"
            raise TypeError(msg)
        self._recover_once()
        state_before = self.plan.snapshot()
        result = self.plan.execute({self._input_name: batch})
        try:
            _write_outputs(result, sink)
            checkpoint = _make_checkpoint(
                self.plan,
                cursor=batch.metadata.cursor,
                sequence=self._sequence,
            )
            self.checkpoint_store.save(checkpoint)
        except Exception:
            self.plan.restore(state_before)
            raise
        self._sequence += 1
        return result

    def reset(self) -> None:
        self._sequence = 0
        self.plan.reset()
        self.checkpoint_store.delete(self.plan.name)
        self._recovered = True

    def _recover_once(self) -> None:
        if self._recovered:
            return
        checkpoint = self.checkpoint_store.load(self.plan.name)
        if checkpoint is not None:
            validate_checkpoint(
                checkpoint,
                pipeline_name=self.plan.name,
                fingerprint=self.plan.fingerprint,
            )
            self.plan.restore(checkpoint.state)
            self._sequence = checkpoint.sequence + 1
        self._recovered = True
