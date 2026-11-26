---
name: code-style
description: Apply Calc Flow's code, test, example, documentation, and verification conventions when planning, implementing, reviewing, benchmarking, or documenting a repository change.
---

# Code Style

Apply these rules to every change and artifact in this repository. Read the
`Coding style` section of [AGENTS.md](../../../AGENTS.md) for the Rust core rules
(Rust 2024, `unsafe_code = "forbid"`, `calc_flow::Result<T>`, async ownership,
and rustdoc). Use the rules below for the Python adapters,
Markdown, tests, and verification.

## Python

- Target CPython 3.9 or newer for the core Python package. Do not add
  compatibility shims for the removed
  pure-Python v1 implementation (`src/calc_flow/` is gone).
- Keep `from __future__ import annotations` in Python modules unless the project
  explicitly removes it everywhere.
- Use modern built-in type syntax: `list[str]`, `dict[str, object]`, `A | B`.
- Prefer a functional style for transformations, validation, parsing, and
  calculations. Keep functions pure unless behavior requires owned state,
  identity, or lifecycle management.
- Never mutate a function's input parameters or caller-owned objects. Treat
  inputs as read-only and return new values; make a defensive copy when an
  underlying library requires mutation.
- Prefer functions over classes. Use a class only when identity, lifecycle,
  polymorphism, protocol implementation, resource ownership, or explicitly
  stateful behavior makes it the clearer boundary.
- Confine necessary mutation to clearly owned stateful boundaries such as the
  Python `Runtime`, runner/store adapters, and lifecycle managers. Never use
  that exception to mutate caller-owned input values.
- Keep CPU-local transforms synchronous and await file, stream, process, and
  network I/O in asyncio applications. Blocking convenience methods reject a
  running event loop. An existing `StreamSession` runs `push`, `advance`,
  `finish`, and `close` synchronously on the caller's thread, including inside
  a running loop. Construct or restore the session and call `checkpoint()`
  outside the loop.
- Prefer small, explicit modules over compatibility shims or duplicate
  abstraction layers.
- Use Arrow-backed tables. DataFusion executes SQL and general expressions,
  and supported operations use native Arrow or native array kernels. Array
  providers are optional and explicitly registered. See `docs/introduction.md`.
- Do not add incomplete stubs, unused fixtures, unused CLIs, or placeholder
  modules just to reserve future structure.

## Tests

- Keep Rust unit tests beside source under `crates/calc-flow/src/`; keep
  integration tests under `crates/calc-flow/tests/`. The `calc-flow-python`
  PyO3 binding has no separate Rust integration-test directory. Cover its
  behavior with inline unit tests and the Python suite under `python/tests/`.
- Keep Python binding and adapter tests under `python/tests/`.
- Define fixtures in focused test modules. Do not create a shared
  `conftest.py`.
- Name Python test functions `test_<behavior>()`. Keep tests focused on public
  behavior, regressions, and state recovery paths; avoid tests that only assert
  unused scaffolding exists.
- Preserve `tests/fixtures/v1/` unchanged. Treat it as immutable historical v1
  parity evidence, not a v2 runtime or package path.
- Start every behavior change with a focused failing test that records the
  expected failure before implementation.

## Examples

- Reconcile current guides and examples with the public entry points, options,
  and behavior in source. Keep the executable inventory in
  [examples/README.md](../../../examples/README.md) and the learning paths in
  [docs/examples.md](../../../docs/examples.md). Use explicit checks for
  executable Python examples so they still verify results under `python -O`.

## Markdown

- Align Markdown table columns and separator rows to the full column width,
  including cell padding.
- Remove trailing whitespace and end every Markdown file with a newline.

## Verification

Follow the local scope, CI responsibilities, and three full-test exceptions in
[AGENTS.md](../../../AGENTS.md#verification). Use the smallest checks for the
change and necessary formatter/linter/type checks for directly affected modules.
Full regression gates belong to regular CI; routine benchmarks run twice daily
on their own schedule. The Python package release verifies both distributions,
runs installed-wheel unit tests for portable wheels, and smoke-tests a clean
x86-64-v3 installation. The complete command groups in AGENTS.md remain CI or
explicitly scoped full-verification references.
Preserve the Rust 90% line coverage floor. Do not repeat
unchanged passing checks or rebuild native code only to review documentation.
For Markdown and agent guidance, check changed structure, synchronization,
links/anchors, whitespace, final newlines, and necessary existing examples.
After an authorized push, take at most one non-blocking CI snapshot; do not wait
or poll unless final results were explicitly requested. Pending permits handoff,
not merge; required failures remain blocking.
