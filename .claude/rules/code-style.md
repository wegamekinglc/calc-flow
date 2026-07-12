# Code Style

Apply these rules to all changes in this repository.

## Python

* Target Python 3.13 or newer.
* Keep `from __future__ import annotations` in Python modules unless the project
  explicitly removes it everywhere.
* Use modern built-in type syntax: `list[str]`, `dict[str, Any]`, `A | B`.
* Prefer a functional style for transformations, validation, parsing, and
  calculations. Keep functions pure unless behavior requires owned state,
  identity, or lifecycle management.
* Never mutate a function's input parameters or caller-owned objects. Treat
  inputs as read-only and return new values; make a defensive copy when an
  underlying library requires mutation.
* Prefer functions over classes. Use a class only when identity, lifecycle,
  polymorphism, protocol implementation, resource ownership, or explicitly
  stateful behavior makes it the clearer boundary.
* Confine necessary mutation to clearly owned stateful boundaries such as
  `StatefulOperator`, runners, stores, and lifecycle managers. Never use that
  exception to mutate caller-owned input values.
* Prefer small, explicit modules over compatibility shims or duplicate
  abstraction layers.
* Keep dataframe behavior Arrow-backed and array behavior Array API-backed,
  matching `docs/introduction.md`.
* Do not add incomplete stubs, unused fixtures, unused CLIs, or placeholder
  modules just to reserve future structure.

## Web UI and backend

* Apply the functional-first, input-immutability, and function-over-class rules
  above to both TypeScript frontend code and Python backend code under
  `web-ui/`.
* Keep web I/O asynchronous whenever the framework or library supports it.
  Await network, file, stream, and process operations rather than blocking
  FastAPI's event loop or the browser main thread.
* Keep pure, CPU-local transformations synchronous. Do not mark a function
  `async` when it performs no asynchronous work.
* Update React state immutably. Use functional state updates when deriving a new
  value from previous state.
* Clean up streams, timers, requests, workers, and other asynchronous resources
  when their owning component, request, or run ends. Avoid floating promises and
  make cancellation and failure paths explicit.

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
