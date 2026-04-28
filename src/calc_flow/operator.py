from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pyarrow as pa


class Operator(ABC):
    """Base class for a calculation node in the pipeline."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def apply(self, data: pa.Table | Any) -> pa.Table | Any: ...

    def reset(self) -> None:
        """Reset any internal state (called on recovery)."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return internal state for checkpointing."""
        return {}

    def restore(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class StatelessOperator(Operator):
    """An operator with no internal state — pure transform.

    May be constructed with a callable *fn*, or subclassed with a custom
    ``apply`` override.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[pa.Table | Any], pa.Table | Any] | None = None,
    ) -> None:
        super().__init__(name)
        self._fn = fn

    def apply(self, data: pa.Table | Any) -> pa.Table | Any:
        if self._fn is not None:
            return self._fn(data)
        msg = f"{type(self).__name__} must override apply or provide fn"
        raise NotImplementedError(msg)


class StatefulOperator(Operator):
    """An operator that maintains mutable state across batches (e.g. aggregations)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._state: dict[str, Any] = {}

    @abstractmethod
    def apply(self, data: pa.Table | Any) -> pa.Table | Any: ...

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    def restore(self, state: dict[str, Any]) -> None:
        self._state = dict(state)

    def reset(self) -> None:
        self._state.clear()
