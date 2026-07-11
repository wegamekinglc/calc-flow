# Calc Flow examples

These examples are small, executable introductions to Calc Flow's public APIs.
All tabular calculations use Apache DataFusion; no pandas or Polars dataframe
backend is required.

Run the core examples from the repository root:

```bash
uv run python examples/01_datafusion_pipeline.py
uv run python examples/02_branching_and_join.py
uv run python examples/03_registered_udf.py
uv run python examples/04_micro_batch_recovery.py
uv run python examples/05_project_configuration.py
```

The optional array example needs NumPy:

```bash
uv run --extra numpy python examples/06_numpy_array.py
```

`notebooks/datafusion_quickstart.ipynb` contains the shortest table-pipeline
walkthrough for an interactive notebook. Select the repository's Python
environment as its kernel before running the cells.

## What each example demonstrates

- `01_datafusion_pipeline.py` — a linear DataFusion calculation, projection,
  filter, results, and node timings.
- `02_branching_and_join.py` — named graph inputs, a multi-table SQL join, and
  fan-out to two terminal outputs.
- `03_registered_udf.py` — a trusted, versioned, vectorized DataFusion scalar
  UDF referenced explicitly by a pipeline node.
- `04_micro_batch_recovery.py` — bounded record batches, stateful processing,
  checkpoint commit, and recovery from the saved source cursor.
- `05_project_configuration.py` — strict data-only project configuration,
  deterministic local persistence, validation, compilation, and execution.
- `06_numpy_array.py` — an optional NumPy Array API pipeline, kept separate
from DataFusion-backed table processing.
