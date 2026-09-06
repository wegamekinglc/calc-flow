from __future__ import annotations

import subprocess  # nosec B404 -- run the managed interpreter without a shell
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORRUPTED_INPUT = """
import runpy
import sys

import pyarrow as pa

column = sys.argv[2]
if column == "array":
    import numpy as np

    original_asarray = np.asarray

    def altered_array(values, *args, **kwargs):
        result = original_asarray(values, *args, **kwargs)
        if result.shape == (4,) and result.tolist() == [1.0, 2.0, 4.0, 6.0]:
            result = result.copy()
            result[0] = 100.0
        return result

    np.asarray = altered_array
else:
    original_table = pa.table

    def altered_table(data, *args, **kwargs):
        if isinstance(data, dict) and column in data:
            values = data[column]
            copied = (
                values.to_pylist() if hasattr(values, "to_pylist") else list(values)
            )
            data = {**data, column: [1000, *copied[1:]]}
        return original_table(data, *args, **kwargs)

    pa.table = altered_table

runpy.run_path(sys.argv[1], run_name="__main__")
"""


@pytest.mark.parametrize(
    ("example", "column"),
    (
        ("01_datafusion_pipeline.py", "quantity"),
        ("02_sql_join.py", "amount"),
        ("03_registered_udf.py", "amount"),
        ("04_continuous_runtime.py", "value"),
        ("05_async_execution.py", "a"),
        ("06_numpy_array.py", "array"),
        ("14_project_persistence.py", "a"),
    ),
)
def test_example_rejects_incorrect_results_with_python_optimization(
    example: str, column: str
) -> None:
    # The managed interpreter, script, and parameterized example/column pairs are
    # trusted test inputs. No external input or shell participates in this call.
    result = subprocess.run(  # noqa: E501  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        [sys.executable, "-O", "-c", CORRUPTED_INPUT, f"examples/{example}", column],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0, result.stdout
    assert "RuntimeError: unexpected" in result.stderr, result.stderr
