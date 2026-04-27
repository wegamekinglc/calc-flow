from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from calc_flow.batch import Batch


class Engine(ABC):
    """Abstract interface for a computation engine.

    An engine evaluates expressions against data held in Calc Flow batches.
    Implementations may use dataframe libraries (pandas, polars, datafusion)
    or array libraries (numpy, jax).
    """

    @abstractmethod
    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch: ...
