# Working-tree review: numbered Rust examples

Base: `main` at `a8c8ab26`; reviewed the uncommitted working tree, including
the newly numbered files. There is no PR or CI result for this working tree.

## Summary

All 36 runnable Python examples have Rust counterparts with matching, unique
01–36 numbers and filename stems. Cargo feature gates, the example runner,
tests, and documentation follow the new names. The Rust examples preserve
result information and restore omitted diagnostics, including three-decimal
elapsed time for examples 30 and 31.

## Build and test results

- Core example compilation and Clippy passed; the final example-01 output
  adjustment also passed focused Clippy.
- All 23 numbered core Rust examples ran successfully with their assertions.
  The implementation's smoke checks also verified timing-line formatting,
  branch dictionaries, recovery cursors and epoch, and the event-window
  summary. Example 36's complete output matches its Python counterpart.
- Runner tests: 7 passed. Python example-validation tests: 10 passed.
  Python financial-example tests: 10 passed. Focused Ruff checks passed.
- Reviewer verification: Cargo metadata confirms exactly 36 matching
  Python/Rust example stems and preserved numbered connector feature gates.
  No stale renamed filenames or broken relative Markdown links were found.
  Whitespace and project-schema drift checks passed.
- Connector examples passed all-feature Clippy after supplying the existing
  curl headers through `C_INCLUDE_PATH`. The initial check failed in
  librdkafka's native C build before reaching example code; the successful
  retry changed no repository source or dependencies.
- File example 15 ran successfully for CSV, JSON Lines, and Parquet. Its
  original three totals/delivery lines were checked, bringing executed
  numbered Rust examples to 24. The other 12 examples require external
  services; they were compiled and linted but were not executed.
- Full workspace suites, coverage, and external-service example execution
  were not run for this review. No performance claim is established.

## Blocking issues

No remaining source findings. Two accidental documentation substitutions of
the `stream_asof_join` operator kind were corrected and rechecked. Stale
event-window discovery prose was also corrected.

## Style and coverage

No new style violations identified. Numbering, inventory completeness,
service exclusion, argument parity, and exit-code propagation have focused
runner coverage. Existing example assertions plus the recorded smoke checks
cover the changed printed output without changing runtime semantics.

## Documentation consistency

The inventory tables align, links resolve, numbered commands agree with
Cargo targets, and the five Python file renames retain identical contents.
Native Arrow table layout, physical runtime bindings, Rust provider identity,
and the NumPy/JAX arithmetic equivalents are explicitly documented.

## Verdict

**Approve.** No remaining blocking findings. Scoped local verification passed;
no claim is made that required CI gates passed or that this tree is merge-ready.
