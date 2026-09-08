"""Compose named SQL inputs and ordinary Python expression transforms."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def main() -> None:
    orders = pa.table({"order_id": [1, 2, 3], "amount": [75, 120, 40]})
    fees = pa.table({"order_id": [1, 2, 3], "fee": [5, 12, 4]})
    joined = cf.sql(
        "SELECT o.order_id, o.amount - f.fee AS net "
        "FROM o JOIN f ON o.order_id = f.order_id ORDER BY o.order_id",
        o=cf.table_input("orders", schema=orders.schema),
        f=cf.table_input("fees", schema=fees.schema),
    )
    output = joined.pipe(lambda t: t.select("order_id", doubled=t["net"] * 2))
    result = output.collect({"orders": orders, "fees": fees})
    expected = {"order_id": [1, 2, 3], "doubled": [140, 216, 72]}
    if result.to_pydict() != expected:
        raise RuntimeError(f"unexpected SQL join result: {result.to_pydict()}")
    print(result.to_pydict())


if __name__ == "__main__":
    main()
