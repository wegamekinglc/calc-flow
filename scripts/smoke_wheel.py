"""Exercise a normally installed core wheel without optional array packages."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import calc_flow as cf

for name in ("pyarrow", "numpy", "jax"):
    assert importlib.util.find_spec(name) is None, (
        f"unexpected optional package: {name}"
    )

# IPC is read by bundled Rust Arrow, without importing a Python Arrow producer.
fixture = Path(__file__).with_name("fixtures") / "standalone.arrow"
batch = cf._native._batch_from_ipc(fixture.read_bytes())
assert batch.num_rows == 2


class Producer:
    def __arrow_c_stream__(self, requested_schema=None):
        return batch.__arrow_c_stream__(requested_schema)


source = cf.table_input(
    "input", schema=[cf.Field("a", "int64"), cf.Field("b", "int64")]
)
expression = source.select(total=source["a"] + source["b"])
output = expression.collect(Producer(), output_format="batch")
assert isinstance(output, cf.Batch)
assert output.num_rows == 2
assert (
    cf.compute(
        output, lambda t: t.filter(t["total"] != 3), output_format="batch"
    ).num_rows
    == 0
)
assert cf.Batch.from_arrow(output).num_rows == 2
program = cf.Program("standalone", engine="sql", outputs={"sum": expression})
assert isinstance(
    program.execute({"input": batch}, output_format="batch")["sum"], cf.Batch
)

for call in (lambda: output.to_pyarrow(), lambda: cf.compute(batch, lambda t: t)):
    try:
        call()
    except ImportError as error:
        assert "calc-flow-python[pyarrow]" in str(error)
    else:
        raise AssertionError("PyArrow adapter should report its missing extra")


async def check_async() -> None:
    assert isinstance(
        await cf.compute_async(batch, lambda t: t, output_format="batch"), cf.Batch
    )
    assert isinstance(
        await expression.collect_async(batch, output_format="batch"), cf.Batch
    )

    async def inputs():
        yield Producer()

    async with expression.stream(
        inputs(),
        output_format="batch",
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        observed = [part async for part in results]
    assert sum(part.num_rows for part in observed) == 2
    assert all(isinstance(part, cf.Batch) for part in observed)


asyncio.run(check_async())

for register, extra in ((cf.register_numpy, "numpy"), (cf.register_jax, "jax")):
    try:
        register(cf.Runtime())
    except ImportError as error:
        assert f"calc-flow-python[{extra}]" in str(error)
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
    assert "calc-flow-python[pyarrow]" in str(error)
else:
    raise AssertionError("missing PyArrow UDF adapter accepted")
assert not any(
    name in __import__("sys").modules for name in ("pyarrow", "numpy", "jax")
)
print(
    "standalone wheel: import, declarations, compute, collect, Program, stream "
    "and missing-extra checks passed"
)
