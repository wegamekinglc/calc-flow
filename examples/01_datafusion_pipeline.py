"""Build and execute a linear DataFusion table pipeline."""

from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, ExpressionOperator, Pipeline


def main() -> None:
    orders = Batch.table(
        pa.table(
            {
                "order_id": ["A-100", "A-101", "A-102"],
                "quantity": [3, 1, 4],
                "unit_price": [10, 12, 10],
            }
        )
    )

    plan = (
        Pipeline("datafusion-quickstart")
        .then(ExpressionOperator("calculate_gross", "gross = quantity * unit_price"))
        .then(
            ExpressionOperator(
                "large_orders",
                select=("order_id", "gross"),
                filter_expression="gross >= 20",
            )
        )
        .compile()
    )

    run = plan.execute({"input": orders})

    print(run.output.table_payload.to_pylist())
    print(
        "node timings (ns):",
        {node_id: timing.duration_ns for node_id, timing in run.node_timings.items()},
    )


if __name__ == "__main__":
    main()
