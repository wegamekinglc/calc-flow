from __future__ import annotations

import json
from pathlib import Path

import pyarrow.ipc as ipc

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1"


def test_v1_contract_manifest_references_readable_arrow_files() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert {case["name"] for case in manifest["cases"]} == {
        "expression_assignment",
        "sql_join",
        "empty_table",
        "metadata_round_trip",
        "state_rollback",
    }
    for relative_path in manifest["arrow_files"]:
        with ipc.open_file(FIXTURE_DIR / relative_path) as reader:
            assert reader.schema is not None
