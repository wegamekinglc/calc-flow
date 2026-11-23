"""Write the maintained dual-SMA graph, Arrow input, and independent NumPy oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks import warm_stream as warm


def generate(directory: Path) -> None:
    config = warm.ScenarioConfig(
        history_rows=4096,
        history_segment_rows=4096,
        append_rows=1,
        entities=1,
        indicator="dual_sma_spread",
        fast_window=5,
        window=20,
        exact_projection=True,
    )
    project = warm._warm_program(config).to_project(mode="stream")
    (directory / "project.json").write_text(
        json.dumps(project.root, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    rows = config.history_rows + 64
    inputs = warm._input_table(config, 0, rows)
    spread = warm._expected(config, start=0, rows=rows)
    expected = inputs.append_column(
        "dual_sma_spread", pa.array(spread, mask=np.isnan(spread))
    )
    for name, table in (("inputs", inputs), ("expected", expected)):
        with (
            pa.OSFile(str(directory / f"{name}.arrow"), "wb") as output,
            pa.ipc.new_file(output, table.schema) as writer,
        ):
            writer.write_table(table.combine_chunks())
