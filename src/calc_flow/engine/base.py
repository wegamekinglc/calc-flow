from __future__ import annotations

from abc import ABC, abstractmethod

from calc_flow.batch import Batch, BatchKind


class Engine(ABC):
    """Abstract interface for a computation engine.

    An engine evaluates expressions against an immutable batch envelope.
    """

    input_kind: BatchKind

    @abstractmethod
    def evaluate(self, expression: str, data: Batch) -> Batch: ...

    def _require_kind(self, data: Batch) -> None:
        if not isinstance(data, Batch):
            msg = f"{type(self).__name__} requires a Batch input"
            raise TypeError(msg)
        if data.kind is not self.input_kind:
            msg = (
                f"{type(self).__name__} requires {self.input_kind.value} batches, "
                f"got {data.kind.value}"
            )
            raise TypeError(msg)
