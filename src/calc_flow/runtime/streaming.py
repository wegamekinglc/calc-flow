from __future__ import annotations

from typing import Any

import pyarrow as pa

from calc_flow.checkpoint import CheckpointManager
from calc_flow.pipeline import Pipeline


class StreamingRunner:
    """Runs a pipeline in streaming mode.

    Processes one batch per round — each invocation of ``step()`` consumes
    a single item through the full pipeline. Suitable for real-time or
    near-real-time workloads.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        checkpoint_dir: str = ".calc-flow-checkpoints",
    ) -> None:
        self.pipeline = pipeline
        self._checkpoints = CheckpointManager(checkpoint_dir)
        self._round: int = 0
        self._recovered = False

    def step(self, data: pa.Table | Any) -> pa.Table | Any:
        """Process a single data item and return the result."""
        self._recover_once()
        result = self.pipeline.apply(data)
        self._round += 1
        self._checkpoints.save(self.pipeline, self._round)
        return result

    def reset(self) -> None:
        self._round = 0
        self.pipeline.reset()
        self._checkpoints.clear(self.pipeline.name)
        self._recovered = True

    def _recover_once(self) -> None:
        if self._recovered:
            return
        self._round = self._checkpoints.recover(self.pipeline)
        self._recovered = True
