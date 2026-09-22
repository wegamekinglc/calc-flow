"""Load the current collection harness while retaining each checkout's tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_support():
    spec = importlib.util.spec_from_file_location(
        "benchmarks.support", ROOT / "benchmarks/support.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # Direct execution must not shadow stdlib statistics with our sibling module.
    directory = Path(__file__).resolve().parent
    sys.path[:] = [path for path in sys.path if Path(path).resolve() != directory]
    # Import the current plugin before pytest can discover the old scripts package.
    sys.path.insert(0, str(ROOT))
    from scripts.benchmark_suite import pytest_plugin

    sys.path.pop(0)
    load_support()
    import pytest

    return pytest.main(sys.argv[1:], plugins=[pytest_plugin])


if __name__ == "__main__":
    raise SystemExit(main())
