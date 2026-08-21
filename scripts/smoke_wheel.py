from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

plan = (
    PipelineBuilder("wheel-smoke")
    .expression("calculate", "total = a + b")
    .compile_batch()
)
result = plan.execute({"input": Batch.from_pyarrow(pa.table({"a": [1], "b": [2]}))})
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3]
