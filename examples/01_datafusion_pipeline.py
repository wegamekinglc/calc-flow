"""Build and execute a DataFusion expression pipeline with the v3 builder."""

from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, PipelineBuilder


def main() -> None:
    orders = Batch.from_pyarrow(
        pa.table(
            {
                "order_id": ["A-100", "A-101", "A-102"],
                "quantity": [3, 1, 4],
                "unit_price": [10, 12, 10],
            }
        )
    )

    plan = (
        PipelineBuilder("datafusion-quickstart")
        .expression("calculate_gross", "gross = quantity * unit_price")
        .expression(
            "large_orders",
            "",
            select=("order_id", "gross"),
            filter="gross >= 20",
        )
        .connect("calculate_gross", "large_orders")
        .compile_batch()
    )

    run = plan.execute({"input": orders})

    print(run.outputs["output"].to_pyarrow().to_pylist())
    print("node timings:", run.node_timings)


if __name__ == "__main__":
    main()
