"""Verify the loaded native before release benchmark collection or execution."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.benchmark_suite.pytest_plugin import (
    pytest_collection_finish as pytest_collection_finish,
)
from scripts.toolkit import sha256_file, write_json


def pytest_sessionstart(session) -> None:
    from calc_flow import _native

    observed = {
        "native_sha256": sha256_file(Path(_native.__file__)),
        "native_path": _native.__file__,
    }
    write_json(Path(os.environ["CALC_FLOW_RELEASE_OBSERVED"]), observed)
    if observed["native_sha256"] != os.environ["CALC_FLOW_RELEASE_NATIVE"]:
        raise ValueError("benchmark loaded a different native than the sealed release")
