# Changelog

This file records fundamental changes to Calc Flow's public surfaces and
engine or Studio capabilities.

## 2026-07

- Add frozen Python `ExecutionOptions(settings, deadline)` as a keyword-only
  option for blocking and async plan execution, with strict deep-copied JSON
  settings and cooperative, aware zero-offset UTC deadlines. Deadline expiry
  raises `calc_flow.CancelledError`; asyncio task cancellation remains
  `asyncio.CancelledError` and waits for native cleanup. Existing two-argument
  providers remain compatible, while `accepts_context=True` opts into a frozen
  `ProviderContext`; the public Python API does not expose the native
  cancellation token.
- Add versioned Studio runtime and preview-worker discovery at
  `GET /api/v2/capabilities`, and model validation, run-state, and table/array
  result responses as closed unions. This is an intentional generated
  TypeScript client source break that requires consumers to regenerate and
  migrate; `/api/v2/catalog` remains a UDF-only compatibility route.
