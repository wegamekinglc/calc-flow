# Changelog

This file records fundamental changes to Calc Flow's public surfaces and
engine or Studio capabilities.

## 2026-08

- Add a crate-private ordered runtime envelope for existing `Batch` data and
  opaque watermark/epoch markers. Internal control dispatch preserves serial
  per-edge FIFO on supported single-ingress routes, keeps the runtime as the
  sole forwarding owner, fails closed before side effects at reachable
  multi-input nodes, and reuses plan snapshot and in-flight rollback
  machinery. This is internal groundwork only: public Rust, Python, Studio,
  runner, project, checkpoint, and graph-fingerprint contracts are unchanged.

## 2026-07

- Add a large, responsive Studio Data Source editor dialog with temporary
  per-source drafts. Inline JSON is validated and applied only on **Confirm**;
  Cancel, close, Escape, and backdrop dismissal discard edits. Keyboard focus
  starts in the editor, remains contained, and returns to the opener. The top
  action toolbar now uses consistently sized controls and intentional
  narrow-screen wrapping. Project, REST, OpenAPI, and engine contracts remain
  unchanged.
- Add frozen native Python `ExecutionOptions(settings, deadline)` for blocking
  and async plan execution. Its constructor remains positional-or-keyword,
  while plans receive it through the keyword-only `options=` parameter.
  Strict settings accept `settings=None` as empty and copy nested mappings in
  one pass with closed, redacted JSON validation. Any valid timezone-aware
  deadline is normalized to UTC with microseconds preserved. An absolute
  deadline continues while a same-plan run waits; queued cancellation remains
  isolated from the active run, and an observed deadline or accepted
  cancellation wins over a later provider error. Deadline expiry raises
  `calc_flow.CancelledError`; asyncio task cancellation remains
  `asyncio.CancelledError`, waits for native cleanup, and linearizes against a
  native result at handler entry. Existing two-argument providers remain
  compatible, while `accepts_context=True` opts into a frozen native
  `ProviderContext`; the public Python API does not expose the native
  cancellation token. These execution-options changes leave project and
  checkpoint formats unchanged.
- Add versioned Studio runtime and preview-worker discovery at
  `GET /api/v2/capabilities`, and model validation, run-state, and table/array
  result responses as closed unions. This is an intentional generated
  TypeScript client source break that requires consumers to regenerate and
  migrate; `/api/v2/catalog` remains a UDF-only compatibility route.
