"""Join two named Arrow inputs with one read-only DataFusion SQL query."""

from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, PipelineBuilder


def main() -> None:
    plan = (
        PipelineBuilder("orders-and-fees")
        .sql(
            "join",
            "SELECT orders.order_id, orders.amount - fees.fee AS net "
            "FROM orders JOIN fees ON orders.order_id = fees.order_id "
            "ORDER BY orders.order_id",
            aliases=("orders", "fees"),
        )
        .compile()
    )

    run = plan.execute(
        {
            "orders": Batch.from_pyarrow(
                pa.table({"order_id": [1, 2, 3], "amount": [75, 120, 40]})
            ),
            "fees": Batch.from_pyarrow(
                pa.table({"order_id": [1, 2, 3], "fee": [5, 12, 4]})
            ),
        }
    )

    print(run.outputs["output"].to_pyarrow().to_pylist())


if __name__ == "__main__":
    main()
