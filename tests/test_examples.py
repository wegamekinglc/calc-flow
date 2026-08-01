from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLE_SCRIPTS = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.stem)
def test_example_script_runs(script: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(script, run_name="__main__")

    assert capsys.readouterr().out.strip()
