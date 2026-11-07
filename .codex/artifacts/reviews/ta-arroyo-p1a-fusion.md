# Branch Review: P1a row-local runtime fusion

**Branch:** `feature/ta-arroyo-p1a-fusion` → `main`
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Status:** frozen product code and shared-harness delta reviewed; performance
and CI merge gates pending.

## Summary

The change expands existing runtime fusion to deterministic row-local arithmetic
and filters, retaining the existing task driver and collector. It adds a real
Python rolling/arithmetic/filter workload and a bounded selector to the maintained
warm-stream measurement tool. This report reviews the author-owned isolated
worktree at product commit `271a7a23` and the subsequent sustained-control
harness delta. No additional review worktree was created. Remote CI/Codacy
results are not inspected or claimed green by this reviewer handoff.

## Build and test results

- Rust: **Passed** for the selected fusion suite: 36 passed, zero failed and
  one existing ignored release-allocation diagnostic. All ten new P1a tests
  passed. The reviewer inspected `target/ta-arroyo-p1a-focused-green.log` in
  the root worktree. The first RED observed two admission failures; first
  GREEN passed three tests. The second RED observed seven passes and one
  failure at the real unknown-output-schema shape. Final crate lib/test Clippy
  passed with `--features benchmark-internals --lib --tests -- -D warnings`;
  the reviewer inspected its successful 3m32s log. Author evidence also records
  `cargo fmt --all --check` and no generated schema drift.
- Python author evidence: the initial affected warm-stream module passed 52
  tests and the profile script subsequently passed 27 tests. The subsequent
  sparse-shape guard independently failed with `DID NOT RAISE ValueError`, then
  passed its one focused test after the guard was implemented. The sustained
  scenario failed as unsupported, then passed one selected standard-scale test
  with nine deselected using `--benchmark-disable`; Ruff passed its source.
  Previously passing tests were not repeated.
- Reviewer verification: the baseline product source diff from `7a2463a8` to
  `8d8be7d9` is empty across all maintained `SOURCE_PATHS` entries. The reviewed
  tracked diff passed `git diff --check`.
- Product builds, benchmarks and full regression suites were not run by the
  reviewer. Already-passing author checks were not repeated.
- CI/Codacy: not inspected at this delta handoff; merge gates remain pending.

## Blocking issues

No correctness defect remains in the frozen product diff. The new production
logic consists of the internal expression eligibility method and its two fusion
admission call sites; the driver, emission collector, persistence format and
public graph identity are unchanged. Merge approval remains pending performance
evidence and applicable CI/Codacy checks. The two warm latency cases do not measure sustained pipeline
throughput. The new third preregistered control now addresses this direct risk
of moving expensive expressions into the same driver as rolling work. It must
be acquired before performance acceptance, within the same three-case budget.

## Reviewed invariants

- `prepare_operator_task_group` still binds the original per-node futures.
  Fusion selects bounded local internal edges, preserving logical task identity.
- `ChannelStreamCollector::emit` still validates declared output ports,
  computes envelope cost, validates senders and applies edge admission. Its
  successful acceptance remains the upstream emission/commit boundary.
- Allowing a computed tail to omit a declared output schema is compatible with
  this execution model: the known internal input schema and row-local/UDF proof
  remain required, and the original runtime output validation remains in place.
  Continuing through a computed node as a chain head still needs a known output
  schema. No startup schema planning is necessary for this adjustment.
- The row-local dependency proof rejects nested queries, aggregates, window
  functions, selected UDFs and non-immutable functions. The original exact-column
  projection proof remains the first short-circuit path.
- The warm oracle preserves retained row identity and independently computes
  rolling values before applying the new arithmetic and filter. Validation is
  outside individual latency samples, within the total acquisition budget.
- The reused baseline retains its actual commit/native identity; product-source
  equivalence to the logical main baseline is explicitly documented.

## Findings addressed during preparation

- The original default history/append sweep would have produced four cases
  instead of the two preregistered cases. The opt-in `--exact-matrix` selector
  now preserves the requested case set without changing the existing default
  sweep. Its focused RED/GREEN and compatibility tests were supplied by the
  author.
- Sparse appends could produce a completely filtered batch while the benchmark
  sink waits for nonempty output. The new indicator now rejects that unsupported
  shape before starting workers; the fixed dense-tick cases remain valid.
- The measurement protocol table was aligned to the repository Markdown rule.
- Candidate builds should use a frozen detached source worktree so later report
  commits cannot invalidate the build manifest's source-commit check.

## Test coverage and documentation

The real Python-exported unknown-output-schema shape selects one fused
physical driver and produces the same values, schema, metadata and controls as
the unfused graph. Tests cover volatile/stable functions, aggregates, windows,
subqueries, UDF/schema/topology exclusions, calculated expression tails, and an
unknown-output node refusing admission as the next chain head. Computed CAST
failures and sink budget refusal retain downstream task identity and upstream
accepted-emission counts. Full-edge cancellation releases credits. The ASOF
calculation test preserves both ingresses and results. Three-node recovery loads
an unfused checkpoint into fused execution and verifies pending values and
terminal checkpoints. Existing projection/ASOF tests passed in the selected
suite. The development plan records the justified unknown-tail design adjustment
and the warm-stream statistical contract separately from the release-pair gate.

## Verdict

**Comment Only — product correctness review has no remaining blocker;
performance and CI verification are pending.** This is not merge approval.

The sustained control now lives in the maintained
`benchmarks/test_stream_operators.py` as `rolling_arithmetic_filter`. It retains
the native Rolling → Expression boundary and feeds continuously through
`_feed`/`_run`. At `standard` scale the unchanged 20,000-row cap and 640-row
batch size provide 32 batches over 16 entities. `symbol < 'S008'` retains 10,000
rows. The oracle checks every retained sequence and transformed mean, before
and after the existing lifecycle timing boundary.

The reviewed task-local `target/ta-arroyo-p1a-performance/collect-p1a.py` selects
exactly that case and `standard` scale, retains the actual sealed build SHAs,
and validates native hashes and machine/dependency/workload identity. It invokes
40 real fresh pytest processes through the maintained command/environment
builders, enforces the selected-case inventory and saved raw samples, and uses
`release_pairs.collect_case/evaluate_case` without changing their two-round,
ten-pair order or verdict. Its result is separate from the warm bootstrap ratios.
The full Git tree for candidate build `7835a557` equals product commit `271a7a23`.

Two preparation advisories were sent before acquisition: hash all imported
statistics/process helpers rather than a partial harness list, and reserve enough
cleanup time for two serial 30-second warm worker closures (530-second acquisition
limit plus a 70-second reserve within 600 seconds). The root assigned these small
adapter/protocol updates to the performance author. Neither changes workloads,
sample counts or statistical rules. Acquisition has not started at this handoff.
