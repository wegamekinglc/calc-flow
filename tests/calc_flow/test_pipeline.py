from __future__ import annotations

import pytest

from calc_flow.batch import Batch
from calc_flow.operator import StatefulOperator, StatelessOperator
from calc_flow.pipeline import Pipeline


class _AddCol(StatelessOperator):
    def __init__(self, name: str, col: str, val: int):
        super().__init__(name)
        self._col = col
        self._val = val

    def apply(self, batch: Batch) -> Batch:
        rows = batch.to_pylist()
        for r in rows:
            r[self._col] = self._val
        return Batch.from_pylist(rows)


def test_pipeline_apply() -> None:
    p = Pipeline("test")
    p.add(_AddCol("step1", "b", 10))
    p.add(_AddCol("step2", "c", 20))

    batch = Batch.from_pylist([{"a": 1}, {"a": 2}])
    result = p.apply(batch)

    rows = result.to_pylist()
    assert rows == [{"a": 1, "b": 10, "c": 20}, {"a": 2, "b": 10, "c": 20}]


def test_pipeline_snapshot_restore() -> None:
    class Accum(StatefulOperator):
        def apply(self, batch: Batch) -> Batch:
            self._state["n"] = self._state.get("n", 0) + batch.num_rows
            return batch

    p1 = Pipeline("test")
    acc = Accum("tracker")
    p1.add(acc)
    p1.apply(Batch.from_pylist([{"x": 1}] * 7))

    snap = p1.snapshot()
    assert snap == {"tracker": {"n": 7}}

    p2 = Pipeline("test")
    p2.add(Accum("tracker"))
    p2.restore(snap)
    p2.apply(Batch.from_pylist([{"x": 1}] * 3))
    assert p2.snapshot()["tracker"]["n"] == 10


def test_pipeline_restore_resets_missing_operator_state() -> None:
    class Accum(StatefulOperator):
        def apply(self, batch: Batch) -> Batch:
            self._state["n"] = self._state.get("n", 0) + batch.num_rows
            return batch

    op = Accum("tracker")
    p = Pipeline("test").add(op)
    p.apply(Batch.from_pylist([{"x": 1}] * 7))

    p.restore({})

    assert op.snapshot() == {}


def test_pipeline_rejects_duplicate_operator_names() -> None:
    p = Pipeline("test").add(_AddCol("step", "b", 10))

    with pytest.raises(ValueError, match="Operator name 'step'"):
        p.add(_AddCol("step", "c", 20))


def test_pipeline_iter() -> None:
    p = Pipeline("test")
    p.add(_AddCol("a", "x", 1))
    p.add(_AddCol("b", "y", 2))
    assert [op.name for op in p] == ["a", "b"]


def test_pipeline_reset() -> None:
    class S(StatefulOperator):
        def apply(self, batch: Batch) -> Batch:
            self._state["called"] = True
            return batch

    op = S("s")
    p = Pipeline("test").add(op)
    p.apply(Batch.from_pylist([{"x": 0}]))

    assert op.snapshot() == {"called": True}
    p.reset()
    assert op.snapshot() == {}
