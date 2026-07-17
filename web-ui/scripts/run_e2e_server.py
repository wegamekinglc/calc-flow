from __future__ import annotations

import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import uvicorn

from calc_flow import Runtime
from calc_flow_studio.app import create_app


def double_value(values: pa.Array) -> pa.Array:
    return pc.multiply(values, 2)


runtime = Runtime()
runtime.register_scalar_udf(
    provider="python",
    name="double_value",
    version="1",
    input_types=["int64"],
    return_type="int64",
    volatility="immutable",
    function=double_value,
)


if __name__ == "__main__":
    projects = tempfile.mkdtemp(prefix="calc-flow-e2e-")
    checkpoints = tempfile.mkdtemp(prefix="calc-flow-e2e-checkpoints-")
    uvicorn.run(
        create_app(
            project_directory=projects,
            checkpoint_directory=checkpoints,
            runtime=runtime,
        ),
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )
