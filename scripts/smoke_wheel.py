"""Exercise a normally installed core wheel without optional array packages."""

from __future__ import annotations

import asyncio
import importlib.util
from importlib.metadata import metadata
from pathlib import Path

from smoke_wheel_recovery import check_recovery

import calc_flow as cf


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_check(
    all(
        "extra ==" in item
        for item in metadata("calc-flow-python").get_all("Requires-Dist", [])
    ),
    "core wheel has a required Python dependency",
)
for name in (
    "pyarrow",
    "numpy",
    "jax",
    "pydantic",
    "pydantic_core",
    "typing_extensions",
):
    _check(
        importlib.util.find_spec(name) is None,
        f"unexpected optional package: {name}",
    )

# IPC is read by bundled Rust Arrow, without importing a Python Arrow producer.
fixture = Path(__file__).with_name("fixtures") / "standalone.arrow"
batch = cf._native._batch_from_ipc(fixture.read_bytes())
_check(
    batch.num_rows == 2,
    "IPC fixture row count changed",
)


class Producer:
    def __arrow_c_stream__(self, requested_schema=None):
        return batch.__arrow_c_stream__(requested_schema)


source = cf.table_input(
    "input", schema=[cf.Field("a", "int64"), cf.Field("b", "int64")]
)
expression = source.select(total=source["a"] + source["b"])
output = expression.collect(Producer(), output_format="batch")
_check(
    isinstance(output, cf.Batch),
    "collect did not return Batch",
)
_check(
    output.num_rows == 2,
    "collect changed the row count",
)
_check(
    cf.compute(
        output, lambda t: t.filter(t["total"] != 3), output_format="batch"
    ).num_rows
    == 0,
    "compute returned incorrect totals",
)
_check(
    cf.Batch.from_arrow(output).num_rows == 2,
    "Arrow round trip changed the row count",
)
program = cf.Program("standalone", engine="sql", outputs={"sum": expression})
document = program.to_project()
_check(
    cf.ProjectDocument.model_validate_json(document.model_dump_json()).canonical_json()
    == document.canonical_json(),
    "project serialization changed the canonical document",
)
try:
    cf.ProjectDocument.model_validate({"format_version": 999})
except cf.ProjectValidationError as error:
    _check(bool(error.errors()), "invalid project omitted its issue records")
else:
    raise AssertionError("invalid project accepted")
_check(
    isinstance(
        program.execute({"input": batch}, output_format="batch")["sum"], cf.Batch
    ),
    "Program.execute did not return Batch",
)

for call in (lambda: output.to_pyarrow(), lambda: cf.compute(batch, lambda t: t)):
    try:
        call()
    except ImportError as error:
        _check(
            "calc-flow-python[pyarrow]" in str(error),
            "missing PyArrow diagnostic omitted the extra",
        )
    else:
        raise AssertionError("PyArrow adapter should report its missing extra")


async def check_async() -> None:
    _check(
        isinstance(
            await cf.compute_async(batch, lambda t: t, output_format="batch"), cf.Batch
        ),
        "compute_async did not return Batch",
    )
    _check(
        isinstance(
            await expression.collect_async(batch, output_format="batch"), cf.Batch
        ),
        "collect_async did not return Batch",
    )

    async def inputs():
        yield Producer()

    async with expression.stream(
        inputs(),
        output_format="batch",
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        observed = [part async for part in results]
    _check(
        sum(part.num_rows for part in observed) == 2,
        "stream changed the row count",
    )
    _check(
        all(isinstance(part, cf.Batch) for part in observed),
        "stream did not return Batch values",
    )


asyncio.run(check_async())
check_recovery(batch)

for register, extra in ((cf.register_numpy, "numpy"), (cf.register_jax, "jax")):
    try:
        register(cf.Runtime())
    except ImportError as error:
        _check(
            f"calc-flow-python[{extra}]" in str(error),
            "missing array backend diagnostic omitted the extra",
        )
    else:
        raise AssertionError("missing array backend accepted")
try:
    cf.Runtime().register_scalar_udf(
        provider="local",
        name="echo",
        version="1",
        input_types=["int64"],
        return_type="int64",
        volatility="immutable",
        function=lambda x: x,
    )
except ImportError as error:
    _check(
        "calc-flow-python[pyarrow]" in str(error),
        "missing PyArrow diagnostic omitted the extra",
    )
else:
    raise AssertionError("missing PyArrow UDF adapter accepted")
_check(
    not any(name in __import__("sys").modules for name in ("pyarrow", "numpy", "jax")),
    "optional array packages were imported",
)
print(
    "standalone wheel: import, declarations, compute, collect, Program, stream "
    "checkpoint recovery and missing-extra checks passed"
)
