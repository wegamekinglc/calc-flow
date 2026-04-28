# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Calc Flow is a micro-batch / streaming stateful calculation engine. Data flows
through pipelines as raw Apache Arrow tables or Array API arrays; computation is
delegated to pluggable dataframe or array engines. See `docs/introduction.md` for
requirements and data flow. This project lives under `workspace/calc-flow/` in
the parent repo — git conventions (branch naming, commit messages, PR titles)
are inherited from the parent repo's `CLAUDE.md`.

## Commands

```bash
uv sync                                  # install dependencies
uv run pytest                            # run all tests
uv run pytest -k checkpoint
uv run ruff check .
uv run ruff format --check .
uv run ruff format .                     # apply formatting
```

## Coding style

Target Python 3.13 or newer. Use four-space indentation, double quotes, and
modern type syntax (`list[str]`, `dict[str, Any]`, `A | B`). Keep `from
__future__ import annotations` in Python modules unless the project removes it
consistently. Prefer small, explicit modules over placeholder abstractions.
Keep dataframe behavior Arrow-backed and array behavior Array API-backed. Do not
add incomplete stubs, unused fixtures, unused CLIs, or placeholder modules just
to reserve future structure. See `.claude/rules/code-style.md`.

## Architecture

Data flows through the system as raw `pa.Table` (for dataframe operations) or
Array API arrays (for array computation). Construct Arrow tables with
`pa.Table.from_pylist`, `pa.table`, or `pa.Table.from_pandas`; pass arrays
directly to array engines.

### Operator, Pipeline, Checkpoint cycle

- **`Operator`** (ABC) — `apply(data) -> pa.Table | Any` is the sole abstract
  method. `snapshot() -> dict`, `restore(dict)`, `reset()` form the checkpoint
  lifecycle.
- **`StatelessOperator`** — pure transform. Construct with a `fn` callable, or
  subclass and override `apply`.
- **`StatefulOperator`** — maintains `self._state: dict` across items.
  `snapshot/restore/reset` operate on this dict. Subclass must implement `apply`.
- **`Pipeline`** — ordered operator sequence. `add()` enforces unique operator
  names (checkpoints key by name). `apply(data)` chains operators sequentially.
  `restore(checkpoint)` calls each operator's `restore` if present in the
  checkpoint, otherwise `reset()`.
- **`Checkpoint`** — value object: `(pipeline_name, offset, state)`. Has
  `to_dict/from_dict`.
- **`CheckpointManager`** — persists JSON to `{dir}/{pipeline_name}.json`.
  Atomic writes (write `.tmp`, rename). `recover()` restores state and returns
  offset, or 0 if no checkpoint exists. `clear()` deletes the file.

### Engines (`engine/`)

- **`Engine`** (ABC) — `evaluate(expression, data, **kwargs) -> pa.Table | Any`.
  Dataframe engines accept and return `pa.Table`; array engines accept and return
  Array API arrays.
- **`DataFrameEngine`** adds `sql(query, tables: dict[str, pa.Table]) -> pa.Table`
  (default: `NotImplementedError`).

Expression handling is centralized in `expression.py`:

- `split_assignment("c = a + b")` → `("c", "a + b")`. Returns `None` for
  non-assignment expressions. The regex guards against comparison operators
  (`==`, `!=`, `<=`, `>=`).
- `sql_projection("c = a + b", "input")` →
  `"SELECT *, (a + b) AS c FROM input"`. For non-assignment:
  `"SELECT (expr) AS result FROM input"`.

Engine implementations:

| Engine             | `evaluate`                                              | `sql`                                                 |
|--------------------|---------------------------------------------------------|-------------------------------------------------------|
| `PandasEngine`     | `df.eval()` with result-type handling                   | —                                                     |
| `PolarsEngine`     | Via `pl.SQLContext` + `sql_projection`                  | `pl.SQLContext` with named tables                     |
| `DataFusionEngine` | Delegates to `self.sql()`                               | `datafusion.SessionContext`, registers record batches |
| `NumpyEngine`      | `eval()` in scope with `{"x": arr, "xp": namespace}`    | —                                                     |
| `JaxEngine`        | Same as NumpyEngine but with `jax.numpy`                | —                                                     |

Array engines also expose a set of programmatic operation methods: `add`,
`subtract`, `multiply`, `divide`, `matmul`, `sum`, `mean`, `max`, `min`,
`transpose`, `reshape`. All accept and return raw Array API arrays.

### Runtime modes (`runtime/`)

- **`MicroBatchRunner`** — iterates a `source: Iterator[pa.Table | Any]`, applies
  pipeline to each, checkpoints at `checkpoint_every` intervals. On `run()`,
  recovers from last checkpoint offset first. `reset()` clears pipeline state and
  deletes the checkpoint file.
- **`StreamingRunner`** — one item per `step()` call. Recovers once on first
  `step()` (via `_recover_once`). Saves checkpoint after every step. `reset()`
  clears state and checkpoint, allows a fresh recovery on next `step()`.

### Test layout

Tests mirror the source tree under `tests/calc_flow/`. Every non-`__init__.py`
source module in `src/calc_flow/` must have a corresponding `test_<module>.py`
file. Name test functions `test_<behavior>()`. Add focused tests for public
behavior, regressions, and state recovery paths. Avoid tests that only preserve
unused scaffolding. Fixtures are defined locally in test files (no shared
`conftest.py`).
