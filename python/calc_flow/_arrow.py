"""Arrow protocol types and explicit optional adapter checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, Union

from calc_flow._compat import TypeAliasType

if TYPE_CHECKING:
    import pyarrow as pa

    from calc_flow._native import Batch


class ArrowCStream(Protocol):
    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


class ArrowRecordBatch(Protocol):
    def __arrow_c_array__(
        self, requested_schema: object = None
    ) -> tuple[object, object]: ...


class ArrowSchema(Protocol):
    def __arrow_c_schema__(self) -> object: ...


OutputFormat = Literal["pyarrow", "batch"]
TableData = TypeAliasType("TableData", Union[ArrowCStream, ArrowRecordBatch, "Batch"])
TableResult = TypeVar("TableResult", bound=Union["pa.Table", "Batch"])


def _require_pyarrow() -> object:
    try:
        import pyarrow
    except ImportError as error:
        raise ImportError(
            "PyArrow output and Python UDFs require calc-flow-python[pyarrow]"
        ) from error
    version = tuple(int(part) for part in pyarrow.__version__.split(".")[:2])
    import sys

    supported = (
        (21, 0) <= version < (22, 0)
        if sys.version_info < (3, 10)
        else (24, 0) <= version < (25, 0)
    )
    if not supported:
        raise ImportError(
            "unsupported PyArrow version; install calc-flow-python[pyarrow] "
            "(PyArrow 25 capsule lifetimes are not supported)"
        )
    return pyarrow


def _validate_output_format(output_format: OutputFormat) -> None:
    if output_format not in ("batch", "pyarrow"):
        raise ValueError("output_format: expected 'pyarrow' or 'batch'")
    if output_format == "pyarrow":
        _require_pyarrow()
