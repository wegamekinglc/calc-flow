"""Resume stateful micro-batch processing from a committed checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    BatchingSource,
    FileCheckpointStore,
    MicroBatchRunner,
    Pipeline,
    RunContext,
    StatefulOperator,
)


class RunningRowCount(StatefulOperator):
    def process(
        self, inputs: Mapping[str, Batch], _context: RunContext
    ) -> Mapping[str, Batch]:
        batch = inputs["input"]
        total_rows = self._state.get("total_rows", 0) + batch.num_rows
        self._state["total_rows"] = total_rows
        table = batch.table_payload.append_column(
            "running_rows",
            pa.array([total_rows] * batch.num_rows),
        )
        return {"output": batch.with_payload(table)}


def source() -> BatchingSource:
    return BatchingSource(
        [{"value": value} for value in range(1, 6)],
        source_id="numbers",
        max_rows=2,
    )


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-checkpoints-") as directory:
        store = FileCheckpointStore(directory)

        first_counter = RunningRowCount("count_rows")
        first_runner = MicroBatchRunner(
            Pipeline("recovery-example").then(first_counter),
            checkpoint_every=1,
            checkpoint_store=store,
        )
        first_runs = first_runner.run(source())
        first_result = next(first_runs)
        first_runs.close()

        recovered_counter = RunningRowCount("count_rows")
        recovered_runner = MicroBatchRunner(
            Pipeline("recovery-example").then(recovered_counter),
            checkpoint_every=1,
            checkpoint_store=store,
        )
        remaining_results = list(recovered_runner.run(source()))
        checkpoint = store.load("recovery-example")

        print("first committed batch:", first_result.output.table_payload.to_pylist())
        print(
            "recovered batches:",
            [result.output.table_payload.to_pylist() for result in remaining_results],
        )
        print("recovered state:", recovered_counter.snapshot())
        print("final source cursor:", checkpoint.source_cursor if checkpoint else None)


if __name__ == "__main__":
    main()
