# Repository Guidance

Calc Flow is a micro-batch and streaming stateful calculation engine. Data
flows through pipelines as raw Apache Arrow tables or Array API arrays, and
computation is delegated to pluggable dataframe or array engines. See
`docs/introduction.md` for requirements and data flow.

## Commands

```bash
uv sync --extra dev                      # install runtime and development dependencies
uv run pytest                            # run all tests
uv run pytest -k checkpoint              # run checkpoint-focused tests
uv run ruff check .                      # lint
uv run ruff format --check .             # check formatting
uv run ruff format .                     # apply formatting
```

## Git conventions

- Name feature branches `feature/<description>` and bug-fix branches
  `fix/<description>`. Use `main` as the base branch.
- Write commit messages with an imperative summary under 72 characters, then a
  blank line and a body explaining why when a body is useful. Do not add
  tool-specific attribution unless the user requests it.
- Prefix PR titles with one of `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
  `test:`, `style:`, `perf:`, or `ci:`. Keep them under 70 characters and use
  imperative mood.
- Use `gh pr create` or `gh pr edit`. PR bodies contain `## Summary` and
  `## Test plan` sections.

## Coding style

- Target Python 3.13 or newer.
- Use four-space indentation and double quotes.
- Keep `from __future__ import annotations` in Python modules unless the project
  removes it consistently.
- Use modern built-in type syntax such as `list[str]`, `dict[str, Any]`, and
  `A | B`.
- Prefer small, explicit modules over compatibility shims, duplicate abstraction
  layers, or placeholder abstractions.
- Keep dataframe behavior Arrow-backed and array behavior Array API-backed,
  matching `docs/introduction.md`.
- Do not add incomplete stubs, unused fixtures, unused CLIs, or placeholder
  modules merely to reserve future structure.

For Markdown tables, align columns with pipes and pad each separator row so its
dashes span the full column width, including the spaces around cell content.

## Architecture

Data flows through the system as raw `pa.Table` values for dataframe operations
or Array API arrays for array computation. Construct Arrow tables with
`pa.Table.from_pylist`, `pa.table`, or `pa.Table.from_pandas`; pass arrays
directly to array engines.

### Operator, Pipeline, Checkpoint cycle

- **`Operator`** (ABC) — `apply(data) -> pa.Table | Any` is the sole abstract
  method. `snapshot() -> dict`, `restore(dict)`, and `reset()` form the
  checkpoint lifecycle.
- **`StatelessOperator`** — pure transform. Construct with a `fn` callable, or
  subclass and override `apply`.
- **`StatefulOperator`** — maintains `self._state: dict` across items.
  `snapshot`, `restore`, and `reset` operate on this dict. Subclasses must
  implement `apply`.
- **`Pipeline`** — ordered operator sequence. `add()` enforces unique operator
  names because checkpoints key state by name. `apply(data)` chains operators
  sequentially. `restore(checkpoint)` calls each operator's `restore` when the
  checkpoint contains its name and otherwise calls `reset()`.
- **`Checkpoint`** — value object containing `pipeline_name`, `offset`, and
  `state`, with `to_dict` and `from_dict` methods.
- **`CheckpointManager`** — persists JSON to `{dir}/{pipeline_name}.json`.
  Writes atomically through a `.tmp` file and rename. `recover()` restores state
  and returns the offset, or `0` when no checkpoint exists. `clear()` deletes
  the file.

### Engines (`engine/`)

- **`Engine`** (ABC) — `evaluate(expression, data, **kwargs) -> pa.Table | Any`.
  Dataframe engines accept and return `pa.Table`; array engines accept and
  return Array API arrays.
- **`DataFrameEngine`** adds
  `sql(query, tables: dict[str, pa.Table]) -> pa.Table`, whose default raises
  `NotImplementedError`.

Expression handling is centralized in `expression.py`:

- `split_assignment("c = a + b")` returns `("c", "a + b")`. It returns `None`
  for non-assignment expressions, and its regex guards against comparison
  operators (`==`, `!=`, `<=`, `>=`).
- `sql_projection("c = a + b", "input")` returns
  `"SELECT *, (a + b) AS c FROM input"`. For a non-assignment expression it
  returns `"SELECT (expr) AS result FROM input"`.

Engine implementations:

| Engine             | `evaluate`                                              | `sql`                                                 |
| ------------------ | ------------------------------------------------------- | ----------------------------------------------------- |
| `PandasEngine`     | `df.eval()` with result-type handling                   | —                                                     |
| `PolarsEngine`     | Via `pl.SQLContext` and `sql_projection`                | `pl.SQLContext` with named tables                     |
| `DataFusionEngine` | Delegates to `self.sql()`                               | `datafusion.SessionContext`, registers record batches |
| `NumpyEngine`      | `eval()` in scope with `{"x": arr, "xp": namespace}`    | —                                                     |
| `JaxEngine`        | Same as NumpyEngine but with `jax.numpy`                | —                                                     |

Array engines also expose programmatic operations: `add`, `subtract`,
`multiply`, `divide`, `matmul`, `sum`, `mean`, `max`, `min`, `transpose`,
`reshape`. All accept and return raw Array API arrays.

### Runtime modes (`runtime/`)

- **`MicroBatchRunner`** — iterates a `source: Iterator[pa.Table | Any]`, applies
  the pipeline to each item, and checkpoints at `checkpoint_every` intervals.
  `run()` first recovers from the last checkpoint offset. `reset()` clears
  pipeline state and deletes the checkpoint file.
- **`StreamingRunner`** — processes one item per `step()` call. It recovers once
  on the first `step()` through `_recover_once` and saves a checkpoint after
  every step. `reset()` clears state and the checkpoint and permits fresh
  recovery on the next `step()`.

## Tests

- Mirror source files under `tests/calc_flow/`.
- Every source file in `src/calc_flow/`, except `__init__.py` and `__main__.py`,
  should have a corresponding `test_<module>.py` in the mirrored test path.
- Name test functions `test_<behavior>()`.
- Add focused tests for public behavior, regressions, and state-recovery paths.
  Avoid tests that only preserve unused scaffolding.
- Define fixtures locally in test files; do not add a shared `conftest.py`.

## Verification

Before considering a change complete, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
