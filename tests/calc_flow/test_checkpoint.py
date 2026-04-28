from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa

from calc_flow.checkpoint import Checkpoint, CheckpointManager
from calc_flow.operator import StatefulOperator
from calc_flow.pipeline import Pipeline


class _CountingOp(StatefulOperator):
    def apply(self, data: pa.Table) -> pa.Table:
        self._state["count"] = self._state.get("count", 0) + len(data)
        return data


def test_checkpoint_to_dict() -> None:
    cp = Checkpoint("p1", 42, {"op_a": {"x": 1}})
    d = cp.to_dict()
    assert d == {"pipeline": "p1", "offset": 42, "state": {"op_a": {"x": 1}}}


def test_checkpoint_manager_save_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        p = Pipeline("test")
        op = _CountingOp("cnt")
        p.add(op)
        p.apply(pa.Table.from_pylist([{"x": 1}] * 5))

        mgr.save(p, batch_offset=3)

        cp = mgr.load("test")
        assert cp is not None
        assert cp.offset == 3
        assert cp.state == {"cnt": {"count": 5}}

        # Verify file exists
        path = Path(tmpdir) / "test.json"
        assert path.exists()


def test_checkpoint_manager_load_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        assert mgr.load("nonexistent") is None


def test_checkpoint_manager_recover() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        p1 = Pipeline("test")
        p1.add(_CountingOp("cnt"))
        p1.apply(pa.Table.from_pylist([{"x": 1}] * 5))
        mgr.save(p1, batch_offset=5)

        p2 = Pipeline("test")
        op2 = _CountingOp("cnt")
        p2.add(op2)
        offset = mgr.recover(p2)

        assert offset == 5
        assert op2.snapshot()["count"] == 5


def test_checkpoint_manager_recover_no_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        p = Pipeline("test")
        offset = mgr.recover(p)
        assert offset == 0
