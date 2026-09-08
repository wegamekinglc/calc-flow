"""Compose and compute order expressions through the Python API."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def large_orders(t: cf.TableExpr) -> cf.TableExpr:
    gross = t["quantity"] * t["unit_price"]
    enriched = t.with_columns(gross=gross, fee=cf.row.cast(gross, "float64") / 10.0)
    return enriched.filter(enriched["gross"] >= 20).select("order_id", "gross")


def main() -> None:
    orders = pa.table(
        {
            "order_id": ["A-100", "A-101", "A-102"],
            "quantity": [3, 1, 4],
            "unit_price": [10, 12, 10],
        }
    )
    rows = cf.compute(orders, large_orders).to_pylist()
    if rows != [
        {"order_id": "A-100", "gross": 30},
        {"order_id": "A-102", "gross": 40},
    ]:
        raise RuntimeError(f"unexpected filtered orders: {rows}")
    print(rows)

    t = cf.table_input("orders", schema=orders.schema)
    program = cf.Program(
        "order-outputs",
        outputs={
            "totals": t.select("order_id", gross=t["quantity"] * t["unit_price"]),
            "quantities": t.select("order_id", "quantity"),
        },
    )
    tables = program.collect({"orders": orders})
    if tables["totals"]["gross"].to_pylist() != [30, 12, 40]:
        raise RuntimeError("unexpected reusable program totals")
    print("named outputs:", list(tables))


if __name__ == "__main__":
    main()
