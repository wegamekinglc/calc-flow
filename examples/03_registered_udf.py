"""Register and explicitly reference a trusted Python scalar UDF."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from calc_flow import Batch, PipelineBuilder, Runtime


def main() -> None:
    runtime = Runtime()

    def double_amount(amount: pa.Array) -> pa.Array:
        return pc.multiply(amount, 2)

    runtime.register_scalar_udf(
        provider="python",
        name="double_amount",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=double_amount,
    )

    plan = (
        PipelineBuilder("registered-udf")
        .expression(
            "calculate",
            "total = double_amount(amount)",
            udfs=(("python", "double_amount", "1"),),
        )
        .compile_batch(runtime)
    )
    run = plan.execute(
        {
            "input": Batch.from_pyarrow(
                pa.table({"amount": pa.array([100, 250, 400], type=pa.int64())})
            )
        }
    )

    print("registered catalog:", runtime.catalog())
    print("result:", run.outputs["output"].to_pyarrow().to_pylist())


if __name__ == "__main__":
    main()
