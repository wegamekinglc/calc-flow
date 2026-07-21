# Code Style

Apply these rules to all changes in this repository. Rust core style (Rust
2024, `unsafe_code = "forbid"`, `calc_flow::Result<T}`, async ownership,
rustdoc) is governed by the `Coding style` section of
[AGENTS.md](../../AGENTS.md); this file covers the Python adapters, the web
surfaces, Markdown, tests, and verification.

## Python

* Target Python 3.13 or newer. Do not add compatibility shims for the removed
  pure-Python v1 implementation (`src/calc_flow/` is gone).
* Keep `from __future__ import annotations` in Python modules unless the project
  explicitly removes it everywhere.
* Use modern built-in type syntax: `list[str]`, `dict[str, object]`, `A | B`.
* Prefer a functional style for transformations, validation, parsing, and
  calculations. Keep functions pure unless behavior requires owned state,
  identity, or lifecycle management.
* Never mutate a function's input parameters or caller-owned objects. Treat
  inputs as read-only and return new values; make a defensive copy when an
  underlying library requires mutation.
* Prefer functions over classes. Use a class only when identity, lifecycle,
  polymorphism, protocol implementation, resource ownership, or explicitly
  stateful behavior makes it the clearer boundary.
* Confine necessary mutation to clearly owned stateful boundaries such as the
  Python `Runtime`, runner/store adapters, and lifecycle managers. Never use
  that exception to mutate caller-owned input values.
* Keep pure, CPU-local transforms synchronous. Blocking convenience methods
  must reject a running event loop; await file, stream, process, and network
  I/O in servers and asyncio applications.
* Prefer small, explicit modules over compatibility shims or duplicate
  abstraction layers.
* Keep table behavior Arrow-backed and array behavior Array API-backed,
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

* Rust unit tests live beside source under `crates/calc-flow/src/`; integration
  tests live under `crates/calc-flow/tests/` and `crates/calc-flow-python/tests/`.
* Python binding and adapter tests live under `python/tests/`. Studio backend
  tests live under `web-ui/backend/tests/` (independent 85% coverage floor).
  Frontend unit tests use Vitest; browser workflows use Playwright.
* Define fixtures in focused test modules. Do not create a shared `conftest.py`.
* Name Python test functions `test_<behavior>()`. Keep tests focused on public
  behavior, regressions, and state recovery paths; avoid tests that only assert
  unused scaffolding exists.
* `tests/fixtures/v1/` is immutable historical v1 parity evidence, not a v2
  runtime or package path. Leave it unchanged.
* Every behavior change starts with a focused failing test that records the
  expected failure before implementation.

## Markdown

* Align table columns with pipes and pad each column's separator row so dashes
  span the full column width (including the spaces on either side of the cell
  content). This keeps tables readable in plain text and consistent with the
  format used in `CLAUDE.md`.

## Verification

Each surface has its own runner; there is no single project-wide test command.
Run the full command groups from the `Commands` section of
[AGENTS.md](../../AGENTS.md). The style-facing gates are:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

uv run ruff check .
uv run ruff format --check .
JAX_PLATFORMS=cpu uv run pytest python/tests -q

cd web-ui && npm run build && npm test
```
