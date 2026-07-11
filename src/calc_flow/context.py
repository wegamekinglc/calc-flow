from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from calc_flow.engine.datafusion import DataFusionRuntime
from calc_flow.udf import UdfRegistrySnapshot


class RunCancelledError(RuntimeError):
    """Raised when execution is cancelled or exceeds its deadline."""


class CancellationToken:
    """Thread-safe cooperative cancellation signal shared by a run."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def _freeze_setting(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = "run setting keys must be strings"
                raise TypeError(msg)
            frozen[key] = _freeze_setting(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_setting(item) for item in value)
    return value


def _readonly_settings(settings: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _freeze_setting(settings or {})


@dataclass(frozen=True, slots=True)
class RunContext:
    """Execution-scoped services and controls passed to every operator."""

    datafusion: DataFusionRuntime
    udfs: UdfRegistrySnapshot
    run_id: str
    node_id: str | None
    cancellation: CancellationToken
    deadline: datetime | None
    settings: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        datafusion: DataFusionRuntime,
        *,
        udfs: UdfRegistrySnapshot | None = None,
        cancellation: CancellationToken | None = None,
        deadline: datetime | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> RunContext:
        if deadline is not None and deadline.tzinfo is None:
            msg = "deadline must include timezone information"
            raise ValueError(msg)
        return cls(
            datafusion=datafusion,
            udfs=udfs or UdfRegistrySnapshot(),
            run_id=uuid4().hex,
            node_id=None,
            cancellation=cancellation or CancellationToken(),
            deadline=deadline,
            settings=_readonly_settings(settings),
        )

    def for_node(self, node_id: str) -> RunContext:
        return RunContext(
            datafusion=self.datafusion,
            udfs=self.udfs,
            run_id=self.run_id,
            node_id=node_id,
            cancellation=self.cancellation,
            deadline=self.deadline,
            settings=self.settings,
        )

    def check_cancelled(self) -> None:
        if self.cancellation.cancelled:
            raise RunCancelledError(f"run {self.run_id} was cancelled")
        if self.deadline is not None and datetime.now(UTC) >= self.deadline:
            raise RunCancelledError(f"run {self.run_id} exceeded its deadline")
