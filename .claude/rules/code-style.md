# Code Style

Apply these rules to all changes in this repository.

## Python

* Target Python 3.13 or newer.
* Keep `from __future__ import annotations` in Python modules unless the project
  explicitly removes it everywhere.
* Use modern built-in type syntax: `list[str]`, `dict[str, Any]`, `A | B`.
* Prefer small, explicit modules over compatibility shims or duplicate
  abstraction layers.
* Keep dataframe behavior Arrow-backed and array behavior Array API-backed,
  matching `docs/introduction.md`.
* Do not add incomplete stubs, unused fixtures, unused CLIs, or placeholder
  modules just to reserve future structure.

## Tests

* Mirror source files under `tests/calc_flow/`.
* Every non-`__init__.py`, non-`__main__.py` source file in `src/calc_flow/`
  should have a corresponding `test_<module>.py` file in the mirrored test path.
* Keep tests focused on public behavior and regressions; avoid tests that only
  assert unused scaffolding exists.

## Markdown

* Align table columns with pipes and pad each column's separator row so dashes
  span the full column width (including the spaces on either side of the cell
  content). This keeps tables readable in plain text and consistent with the
  format used in `CLAUDE.md`.

## Verification

Run these before considering a change complete:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
