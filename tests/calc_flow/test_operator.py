from __future__ import annotations

import pyarrow as pa

from calc_flow.operator import StatefulOperator, StatelessOperator


def test_stateless_operator() -> None:
    table = pa.Table.from_pylist([{"x": 1}, {"x": 2}])
    op = StatelessOperator(
        "double",
        fn=lambda t: pa.Table.from_pylist([{"x": r["x"] * 2} for r in t.to_pylist()]),
    )
    result = op.apply(table)
    assert result.to_pylist() == [{"x": 2}, {"x": 4}]


def test_stateful_operator_snapshot_restore() -> None:
    class RunningSum(StatefulOperator):
        def apply(self, data: pa.Table) -> pa.Table:
            total = self._state.get("sum", 0)
            for row in data.to_pylist():
                total += row["val"]
            self._state["sum"] = total
            return data

    op = RunningSum("accum")
    op.apply(pa.Table.from_pylist([{"val": 10}, {"val": 20}]))  # sum = 30

    snap = op.snapshot()
    assert snap == {"sum": 30}

    op2 = RunningSum("accum")
    op2.restore(snap)
    op2.apply(pa.Table.from_pylist([{"val": 5}]))  # sum = 35
    assert op2.snapshot()["sum"] == 35


def test_stateful_operator_reset() -> None:
    class S(StatefulOperator):
        def apply(self, data: pa.Table) -> pa.Table:
            return data

    op = S("s")
    op._state = {"x": 42}
    op.reset()
    assert op.snapshot() == {}
