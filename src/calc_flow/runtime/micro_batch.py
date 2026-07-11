from __future__ import annotations

from collections.abc import Iterator, Mapping

from calc_flow.batch import Batch, JSONValue
from calc_flow.checkpoint import (
    Checkpoint,
    CheckpointStore,
    FileCheckpointStore,
    validate_checkpoint,
)
from calc_flow.io import Sink, Source
from calc_flow.pipeline import ExecutionPlan, Pipeline, RunResult

type SinkTarget = Sink | Mapping[str, Sink]


def _coerce_plan(pipeline: Pipeline | ExecutionPlan) -> ExecutionPlan:
    return pipeline.compile() if isinstance(pipeline, Pipeline) else pipeline


def _single_input_name(plan: ExecutionPlan) -> str:
    if len(plan.graph_inputs) != 1:
        msg = "runners require an execution plan with exactly one graph input"
        raise ValueError(msg)
    return next(iter(plan.graph_inputs))


def _write_outputs(result: RunResult, sinks: SinkTarget | None) -> None:
    if sinks is None:
        return
    if isinstance(sinks, Mapping):
        unknown = set(sinks) - set(result.outputs)
        if unknown:
            msg = f"sinks configured for unknown graph outputs: {sorted(unknown)}"
            raise ValueError(msg)
        for name, sink in sinks.items():
            sink.write(result.outputs[name])
        return
    sinks.write(result.output)


def _make_checkpoint(
    plan: ExecutionPlan,
    *,
    cursor: JSONValue,
    sequence: int,
) -> Checkpoint:
    return Checkpoint(
        pipeline_name=plan.name,
        pipeline_fingerprint=plan.fingerprint,
        source_cursor=cursor,
        sequence=sequence,
        state=plan.snapshot(),
    )


class MicroBatchRunner:
    """Execute replayable source batches with at-least-once sink delivery."""

    def __init__(
        self,
        pipeline: Pipeline | ExecutionPlan,
        *,
        checkpoint_every: int = 100,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        if checkpoint_every <= 0:
            msg = "checkpoint_every must be greater than 0"
            raise ValueError(msg)
        self.plan = _coerce_plan(pipeline)
        self.checkpoint_every = checkpoint_every
        self.checkpoint_store = (
            checkpoint_store if checkpoint_store is not None else FileCheckpointStore()
        )
        self._input_name = _single_input_name(self.plan)

    def run(
        self,
        source: Source,
        sink: SinkTarget | None = None,
    ) -> Iterator[RunResult]:
        """Recover the source cursor, run batches, and yield every result."""
        checkpoint = self.checkpoint_store.load(self.plan.name)
        if checkpoint is None:
            cursor: JSONValue = None
            sequence = -1
        else:
            validate_checkpoint(
                checkpoint,
                pipeline_name=self.plan.name,
                fingerprint=self.plan.fingerprint,
            )
            self.plan.restore(checkpoint.state)
            cursor = checkpoint.source_cursor
            sequence = checkpoint.sequence

        since_checkpoint = 0
        pending: Checkpoint | None = None
        for batch in source.read(cursor):
            if not isinstance(batch, Batch):
                msg = "Source.read() must yield Batch objects"
                raise TypeError(msg)
            state_before = self.plan.snapshot()
            result = self.plan.execute({self._input_name: batch})
            next_sequence = sequence + 1
            try:
                _write_outputs(result, sink)
                pending = _make_checkpoint(
                    self.plan,
                    cursor=batch.metadata.cursor,
                    sequence=next_sequence,
                )
                since_checkpoint += 1
                if since_checkpoint == self.checkpoint_every:
                    self.checkpoint_store.save(pending)
                    since_checkpoint = 0
                    pending = None
            except Exception:
                self.plan.restore(state_before)
                raise

            sequence = next_sequence
            yield result

        if pending is not None:
            self.checkpoint_store.save(pending)

    def reset(self) -> None:
        self.plan.reset()
        self.checkpoint_store.delete(self.plan.name)
