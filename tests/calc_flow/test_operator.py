from __future__ import annotations

from calc_flow.batch import Batch
from calc_flow.operator import StatefulOperator, StatelessOperator


def test_stateless_operator() -> None:
    batch = Batch.from_pylist([{"x": 1}, {"x": 2}])
    op = StatelessOperator(
        "double",
        fn=lambda b: Batch.from_pylist([{"x": r["x"] * 2} for r in b.to_pylist()]),
    )
    result = op.apply(batch)
    assert result.to_pylist() == [{"x": 2}, {"x": 4}]


def test_stateful_operator_snapshot_restore() -> None:
    class RunningSum(StatefulOperator):
        def apply(self, batch: Batch) -> Batch:
            total = self._state.get("sum", 0)
            for row in batch.to_pylist():
                total += row["val"]
            self._state["sum"] = total
            return batch

    op = RunningSum("accum")
    op.apply(Batch.from_pylist([{"val": 10}, {"val": 20}]))  # sum = 30

    snap = op.snapshot()
    assert snap == {"sum": 30}

    op2 = RunningSum("accum")
    op2.restore(snap)
    op2.apply(Batch.from_pylist([{"val": 5}]))  # sum = 35
    assert op2.snapshot()["sum"] == 35


def test_stateful_operator_reset() -> None:
    class S(StatefulOperator):
        def apply(self, batch: Batch) -> Batch:
            return batch

    op = S("s")
    op._state = {"x": 42}
    op.reset()
    assert op.snapshot() == {}
