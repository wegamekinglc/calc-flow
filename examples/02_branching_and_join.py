"""Join two graph inputs and fan the result out to two calculations."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa

from calc_flow import (
    Batch,
    ExpressionOperator,
    Pipeline,
    RunContext,
    SqlOperator,
    StatelessOperator,
)


def identity(inputs: Mapping[str, Batch], _context: RunContext) -> Mapping[str, Batch]:
    return {"output": inputs["input"]}


def main() -> None:
    join = SqlOperator(
        "join_orders_and_fees",
        "SELECT o.order_id, o.amount, f.fee "
        "FROM orders_table o JOIN fees_table f ON o.order_id = f.order_id "
        "ORDER BY o.order_id",
        inputs=("orders_table", "fees_table"),
    )
    plan = (
        Pipeline("branching-join")
        .add_node("orders", StatelessOperator("orders", identity))
        .add_node("fees", StatelessOperator("fees", identity))
        .add_node("join", join)
        .add_node("report", ExpressionOperator("report", "net = amount - fee"))
        .add_node(
            "audit",
            ExpressionOperator("audit", "needs_review = amount >= 100"),
        )
        .connect("orders", "join", target_port="orders_table")
        .connect("fees", "join", target_port="fees_table")
        .connect("join", "report")
        .connect("join", "audit")
        .compile()
    )

    run = plan.execute(
        {
            "orders.input": Batch.table(
                pa.table({"order_id": [1, 2, 3], "amount": [75, 120, 40]})
            ),
            "fees.input": Batch.table(
                pa.table({"order_id": [1, 2, 3], "fee": [5, 12, 4]})
            ),
        }
    )

    for output_name, batch in run.outputs.items():
        print(output_name, batch.table_payload.to_pylist())


if __name__ == "__main__":
    main()
