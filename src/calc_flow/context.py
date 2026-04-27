from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


class Context(MutableMapping[str, Any]):
    """Execution context carrying configuration and accumulated metadata.

    Holds pipeline-level configuration, metrics, and shared state that
    spans across batches but is separate from operator-local state.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = dict(kwargs)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Context({list(self._data.keys())})"
