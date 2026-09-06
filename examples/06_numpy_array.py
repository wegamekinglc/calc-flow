"""Run an optional NumPy Array API pipeline."""

from __future__ import annotations

import numpy as np

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy


def main() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    values = Batch.from_array(
        np.asarray([1.0, 2.0, 4.0, 6.0]),
        backend="numpy",
    )
    plan = (
        PipelineBuilder("numpy-array")
        .external(
            "center",
            "numpy",
            "expression",
            "1",
            {"expression": "x - mean(x)"},
        )
        .compile_batch(runtime)
    )

    run = plan.execute({"input": values})
    centered = run.outputs["output"].array.tolist()
    if centered != [-2.25, -1.25, 0.75, 2.75]:
        raise RuntimeError(f"unexpected centered values: {centered}")
    print(centered)


if __name__ == "__main__":
    main()
