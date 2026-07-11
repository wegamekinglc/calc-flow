from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLE_SCRIPTS = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.stem)
def test_example_script_runs(script: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(script, run_name="__main__")

    assert capsys.readouterr().out.strip()


def test_quickstart_notebook_is_clean_and_valid() -> None:
    notebook_path = ROOT / "examples" / "notebooks" / "datafusion_quickstart.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
