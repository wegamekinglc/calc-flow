from __future__ import annotations

import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import uvicorn

from calc_flow.udf import UdfRegistry
from calc_flow.web.app import create_app

registry = UdfRegistry()


@registry.datafusion_scalar(
    name="double_value",
    version="1",
    input_fields=[pa.int64()],
    return_field=pa.int64(),
    description="Double a value",
)
def double_value(values):
    return pc.multiply(values, 2)


if __name__ == "__main__":
    projects = tempfile.mkdtemp(prefix="calc-flow-e2e-")
    uvicorn.run(
        create_app(project_directory=projects, udf_registry=registry),
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )
