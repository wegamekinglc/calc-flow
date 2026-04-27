from __future__ import annotations

from collections.abc import Iterator

from calc_flow.batch import Batch
from calc_flow.context import Context
from calc_flow.operator import Operator


class Pipeline:
    """An ordered DAG of operators that process batches of data."""

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._operators: list[Operator] = []
        self._context = Context()

    def add(self, operator: Operator) -> Pipeline:
        if any(op.name == operator.name for op in self._operators):
            msg = (
                f"Operator name {operator.name!r} is already in pipeline {self.name!r}"
            )
            raise ValueError(msg)
        self._operators.append(operator)
        return self

    def apply(self, batch: Batch) -> Batch:
        """Apply all operators to a single batch in sequence."""
        result = batch
        for op in self._operators:
            result = op.apply(result)
        return result

    def snapshot(self) -> dict:
        """Return combined state of all stateful operators for checkpointing."""
        snapshot = {}
        for op in self._operators:
            state = op.snapshot()
            if state:
                snapshot[op.name] = state
        return snapshot

    def restore(self, checkpoint: dict) -> None:
        """Restore all operators from a checkpoint."""
        for op in self._operators:
            if op.name in checkpoint:
                op.restore(checkpoint[op.name])
            else:
                op.reset()

    def reset(self) -> None:
        for op in self._operators:
            op.reset()

    @property
    def context(self) -> Context:
        return self._context

    def __iter__(self) -> Iterator[Operator]:
        return iter(self._operators)

    def __repr__(self) -> str:
        ops = ", ".join(op.name for op in self._operators)
        return f"Pipeline(name={self.name!r}, operators=[{ops}])"
