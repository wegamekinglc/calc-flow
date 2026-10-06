# Stream Join ingress progress — issue 363 Phase 0.2

## Design and scope

Expose the existing runtime-owned ingress watermark, idle and end state in
`StreamJoinSideStatus`, the Python native mapping and TypedDict, and Studio
progress events. Rust/Python use exact native integers; the new Studio
watermark uses the existing signed-decimal string contract. Existing Join
counters remain numeric. Join state layout 1 and checkpoint metadata do not
change. Standalone operators retain only transient observed ingress progress;
fresh or non-terminal managed startup projects runtime-owned progress before
acknowledgement. Terminal-manifest recovery completes before operator entry and
does not restore per-Join observations; this change adds no terminal restoration.

The static Join benchmark seals and acknowledges the dimension before quotes.
This readiness setup precedes timing. Timing begins with the first quote
enqueue and ends after Arrow concatenation; the zero retained/evicted quote
assertion follows timing. State limits permit the dimension plus one batch.
The scope becomes `ready-enqueue-to-arrow/bounded-feeds-v6`, and older native
stream scopes become new coverage. This boundary change is not a kernel
performance claim. SQL and warm comparison gates retain their existing scopes.

## TDD evidence

- Rust status and managed-entry tests failed with missing
  `watermark_micros`, `idle` and `ended` fields (E0609, exit 101).
- Static benchmark tests failed when a quote arrived before dimension
  acknowledgement, when the limit still permitted two batches, and when the
  new progress waiter was absent.
- Timer instrumentation failed at 0.211 seconds against the expected 0.010:
  dimension setup and two deliberately expensive status reads were included.
- Native Studio watermark normalization accepted invalid integer inputs;
  valid progress was rejected as unknown fields. OpenAPI lacked those fields.
- SSE omitted a nullable Join watermark; the test failed with KeyError.
- Python status type declarations lacked the new fields. TypeScript rejected
  the corresponding watermark property (TS2353).

## Verification

The focused checks passed:

- Two Rust core status tests, one PyO3 exact-value projection test, and two
  real native 320k-row static Join tests, including delayed dimension watermarks.
- 77 Python/benchmark/catalog/Studio tests and 49 subtests; seven frontend
  job-event tests and TypeScript compilation.
- Scoped Rust/PyO3 library Clippy with warnings denied, Ruff and formatting,
  expected OpenAPI/TypeScript regeneration, no project-schema drift, and whitespace.

The binding checks used a freshly rebuilt debug native module under `target/`,
not a sealed performance wheel. Specialist review found the terminal-manifest
documentation overclaim; the text now explicitly limits restored progress to
non-terminal running startup.

Full Rust/Python/Studio regression,
workspace coverage floors and cross-platform checks remain CI gates. No local
performance claim or remote mutation is part of this implementation artifact.
