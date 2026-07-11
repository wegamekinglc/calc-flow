"""Register and explicitly reference a versioned DataFusion scalar UDF."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from calc_flow import (
    Batch,
    ExpressionOperator,
    Pipeline,
    UdfReference,
    UdfRegistry,
)


def main() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="apply_tax",
        version="1",
        input_fields=(pa.field("amount", pa.int64(), nullable=False),),
        return_field=pa.field("taxed_amount", pa.int64(), nullable=False),
        volatility="immutable",
        description="Add a fixed 20 percent tax to an integer amount.",
    )
    def apply_tax(amount: pa.Array) -> pa.Array:
        return pc.add(amount, pc.divide(pc.multiply(amount, 20), 100))

    plan = (
        Pipeline("registered-udf", udf_registry=registry)
        .then(
            ExpressionOperator(
                "calculate_tax",
                "total = apply_tax(amount)",
                udfs=(UdfReference("apply_tax", "1"),),
            )
        )
        .compile()
    )
    run = plan.execute(
        {"input": Batch.table(pa.table({"amount": pa.array([100, 250, 400])}))}
    )

    print("registered catalog:", registry.catalog())
    print("result:", run.output.table_payload.to_pylist())


if __name__ == "__main__":
    main()
