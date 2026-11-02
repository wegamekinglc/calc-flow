# Working-tree review: configurable stream budgets

Branch: `fix/configurable-stream-budgets`; base: `7464235c` on `main`.
Scope: the tracked runtime, schema, example, documentation, and focused test
changes. No remote mutation or CI query was performed by the reviewer.

## Summary

Explicit stream budgets and source declarations can exceed one million rows.
The default remains one million rows; historical omitted project-v3 options
retain their existing values and fingerprints. Whole-message admission,
checked queue accounting, backpressure, and cancellation remain enforced.

The stock examples derive compatible budgets from configured batch and complete
snapshot sizes, construct Arrow columns directly, and yield during large batch
generation. Structured budget failures expose numeric costs through public
stream errors while withholding raw edge identifiers.

## Findings

No correctness, compatibility, security, or style blockers were identified.
The schema change removes only the row maximum; it does not alter defaults.
The remaining finite-run fixture limits are separate from stream admission.
Cancellation discards partial builders without advancing the accepted offset.
Valuation's SQL state budgets remain independent of its edge budgets.

Tests cover explicit large declarations, configured row and byte rejection,
whole-message backpressure, cancellation and reservation release, historical
project defaults, safe public diagnostics, and source-generation cancellation.
The Python regression uses a real five-million-row Arrow batch with small
buffers and checks whole-table delivery and terminal task cleanup.

Scoped complexity inspection found `project_runtime_failure` at CCN 7,
`preflight_failure_fields` at CCN 10, `edge_budget_detail` at CCN 1, and the
example's `runtime_config` at CCN 3. No new complexity suppression was added.

## Verification

The author observed the targeted large-budget, diagnostics, cancellation, and
Python binding regressions fail before their corresponding changes. The
focused Rust budget, source-preflight, redaction, queue, example, historical
project-option, and generated-schema checks now pass. The generated schema
diff was independently checked. Cargo formatting, changed-file Ruff, and the
maintained complexity ratchet pass. Scoped Clippy for the core library,
channel integration test, and both stock examples passes with warnings denied.

The isolated candidate CPython 3.13 debug wheel passed the new materialized
five-million-row Arrow case with an 8 MiB edge budget: one equal, intact output,
completed state, and zero remaining tasks. Both Rust stock examples completed
five million business rows in five-million-row batches with two accounts and
three stocks; valuation used nine quotes. Their independent built-in result
oracles passed. These are correctness checks, not performance comparisons.

Passing checks were not repeated. Full regression, coverage, cross-platform
checks, and CI approval remain separate from this local review. Required CI and
Codacy checks must pass before merging.

## Verdict

**Approve.** No blocking findings remain in the reviewed implementation. The
focused execution evidence covers the changed behavior and its directly
affected contracts; this approval does not assert that pending CI is green.
