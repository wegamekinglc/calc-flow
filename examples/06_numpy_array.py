"""Run an optional NumPy Array API pipeline."""

from __future__ import annotations

import numpy as np

from calc_flow import ArrayExpressionOperator, Batch, Pipeline


def main() -> None:
    values = Batch.array(np.asarray([1.0, 2.0, 4.0, 6.0]))
    plan = (
        Pipeline("numpy-array")
        .then(
            ArrayExpressionOperator(
                "center",
                "x - xp.mean(x)",
                backend="numpy",
            )
        )
        .then(
            ArrayExpressionOperator(
                "clip_lower_bound",
                "xp.maximum(x, -1.5)",
                backend="numpy",
            )
        )
        .compile()
    )

    run = plan.execute({"input": values})
    print(run.output.array_payload.tolist())


if __name__ == "__main__":
    main()
