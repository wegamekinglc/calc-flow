from __future__ import annotations

from collections.abc import Iterator

from calc_flow.batch import Batch
from calc_flow.checkpoint import CheckpointManager
from calc_flow.pipeline import Pipeline


class MicroBatchRunner:
    """Runs a pipeline in micro-batch mode.

    Processes data in batches ranging from dozens to tens of thousands
    of rows. Checkpoints are taken periodically for state recovery.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        batch_size: int = 1000,
        checkpoint_every: int = 100,
        checkpoint_dir: str = ".calc-flow-checkpoints",
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)
        if checkpoint_every <= 0:
            msg = "checkpoint_every must be greater than 0"
            raise ValueError(msg)
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self._checkpoints = CheckpointManager(checkpoint_dir)

    def run(self, source: Iterator[Batch]) -> None:
        """Process all batches from the source iterator."""
        offset = self._checkpoints.recover(self.pipeline)
        last_processed_offset = offset

        for i, batch in enumerate(source):
            if i < offset:
                continue

            self.pipeline.apply(batch)
            last_processed_offset = i + 1

            if last_processed_offset % self.checkpoint_every == 0:
                self._checkpoints.save(self.pipeline, last_processed_offset)

        if last_processed_offset > offset:
            self._checkpoints.save(self.pipeline, last_processed_offset)

    def reset(self) -> None:
        self.pipeline.reset()
        self._checkpoints.clear(self.pipeline.name)
