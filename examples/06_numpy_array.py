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
        .compile(runtime)
    )

    run = plan.execute({"input": values})
    print(run.outputs["output"].array.tolist())


if __name__ == "__main__":
    main()
