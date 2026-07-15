"""Resume micro-batch processing from a committed source cursor."""

from __future__ import annotations

import gc
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    FileCheckpointStore,
    MicroBatchRunner,
    PipelineBuilder,
)


class ReplaySource:
    def __init__(self, values: list[int]) -> None:
        self._values = tuple(values)
        self._offset = 0

    def open(self, cursor: object) -> None:
        self._offset = 0 if cursor is None else int(cursor["offset"])

    def next(self) -> tuple[Batch, dict[str, int], int] | None:
        if self._offset == len(self._values):
            return None
        value = self._values[self._offset]
        self._offset += 1
        return (
            Batch.from_pyarrow(pa.table({"value": [value]})),
            {"offset": self._offset},
            self._offset,
        )


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-checkpoints-") as directory:
        store = FileCheckpointStore(directory)
        plan = (
            PipelineBuilder("recovery-example")
            .expression("calculate", "result = value + 1")
            .compile()
        )
        first_runner = MicroBatchRunner(
            plan,
            ReplaySource([1, 2, 3]),
            store,
            checkpoint_every=1,
        )
        first_result = first_runner.next()
        del first_runner
        gc.collect()

        recovered_runner = MicroBatchRunner(
            plan,
            ReplaySource([1, 2, 3]),
            store,
            checkpoint_every=1,
        )
        recovered_results = []
        while (result := recovered_runner.next()) is not None:
            recovered_results.append(result.outputs["output"].to_pyarrow().to_pylist())
        checkpoint = store.load_blocking("recovery-example")

        assert first_result is not None
        print("first committed batch:", first_result.outputs["output"].to_pyarrow())
        print("recovered batches:", recovered_results)
        print(
            "final source cursor:",
            checkpoint["source_cursor"] if checkpoint else None,
        )


if __name__ == "__main__":
    main()
