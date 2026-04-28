from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa


class Engine(ABC):
    """Abstract interface for a computation engine.

    An engine evaluates expressions against data. Implementations may use
    dataframe libraries (pandas, polars, datafusion) which accept Arrow
    tables, or array libraries (numpy, jax) which accept Array API arrays.
    """

    @abstractmethod
    def evaluate(
        self, expression: str, data: pa.Table | Any, **kwargs: Any
    ) -> pa.Table | Any: ...
