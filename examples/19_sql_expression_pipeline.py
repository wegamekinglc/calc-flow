"""Reuse table and column pipelines around a native SQL stage."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def add_gross(t: cf.TableExpr) -> cf.TableExpr:
    return t.with_columns(gross=cf.row.cast(t["quantity"], "float64") * t["price"])


def discounted(t: cf.TableExpr, rate: float) -> cf.TableExpr:
    return t.select("order_id", net=t["gross"] * (1.0 - rate))


def pipeline(t: cf.TableExpr) -> cf.TableExpr:
    return (
        t.pipe(add_gross)
        .sql("SELECT order_id, gross FROM input WHERE gross >= 20 ORDER BY order_id")
        .pipe(discounted, rate=0.1)
    )


def main() -> None:
    orders = pa.table(
        {
            "order_id": [1, 2, 3],
            "quantity": [2, 1, 3],
            "price": [10.0, 5.0, 10.0],
        }
    )
    result = cf.compute(orders, pipeline)
    expected = {"order_id": [1, 3], "net": [18.0, 27.0]}
    if result.to_pydict() != expected:
        raise RuntimeError(result.to_pydict())
    print(result.to_pydict())


if __name__ == "__main__":
    main()
