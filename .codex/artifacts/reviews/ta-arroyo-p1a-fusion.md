# Branch Review: P1a row-local runtime fusion

**Branch:** `feature/ta-arroyo-p1a-fusion` → `main`
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Status:** measured candidate `873e735a` reviewed; required performance
acceptance failed. Do not merge PR #563.

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
- The later Codacy test-helper extraction preserves all 13 rejection variants
  and assertions; its diff changes only `cfg(test)` code. Author evidence records
  one focused pass, formatting and final lib/test Clippy passes. No threshold
  suppression or production change was used.
- Reviewer verification: the baseline product source diff from `7a2463a8` to
  `8d8be7d9` is empty across all maintained `SOURCE_PATHS` entries. The reviewed
  tracked diff passed `git diff --check`.
- Product builds, benchmarks and full regression suites were not run by the
  reviewer. Already-passing author checks were not repeated.
- CI/Codacy: not inspected at this delta handoff; merge gates remain pending.

## Blocking issues

No value/ordering/recovery defect was found in the frozen product diff. The
production change is the expression eligibility method and its two admission
call sites; the driver, collector, state format and public graph identity stay
unchanged. However, the required target performance acceptance failed, and the
sustained-input control produced invalid identity evidence. These block merge
regardless of CI/Codacy results. The current candidate should be withdrawn under
its preregistered stopping rule, rather than changing sizes or samples to seek
favorable timings.

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

**Request Changes — do not merge.** The candidate has no demonstrated
performance improvement and the required sustained-input control is invalid.
The recommended disposition is to stop/withdraw this candidate with its evidence.
This review does not close or otherwise mutate the PR.

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
At setup review the full Git tree for build `7835a557` equaled product commit
`271a7a23`; final acquisition instead used the later sealed `873e735a` build.

Two preparation advisories were sent before acquisition: hash all imported
statistics/process helpers rather than a partial harness list, and reserve enough
cleanup time for two serial 30-second warm worker closures (530-second acquisition
limit plus a 70-second reserve within 600 seconds). The final protocol records the complete harness hash set and a 530-second
acquisition timeout. These updates do not change workloads, sample counts or
statistical rules. Final acquisition is audited below.

## Final measurement and admission audit

The single acquisition at harness/candidate commit `873e735a` is terminal. Its
journal reports 9.895 seconds before identity validation rejected the first
sustained-input baseline observation. No retry occurred. The reviewer inspected
`target/ta-arroyo-p1a-performance/acquisition-final/{warm.json,acquisition.json}`
and `sustained/failure.json` in the root worktree.

| Case                           | Paired median speedup | Bootstrap 95% interval | Evidence                               |
|--------------------------------|-----------------------|------------------------|----------------------------------------|
| rolling_mean                   | 0.941725              | [0.890114, 0.993488]   | 30 pairs; no improvement demonstrated  |
| rolling_mean_arithmetic_filter | 0.966894              | [0.896492, 1.080038]   | 30 pairs; target acceptance not met    |
| sustained arithmetic/filter    | unavailable           | unavailable            | First observation rejected by identity |

All warm sample oracles passed and both jobs completed for both cases. Neither
warm case meets the preregistered target requirement of median speedup >1.05
and lower confidence bound >1.0. The original mean control's interval does not
establish a confirmed slowdown greater than 5% under the registered control
rule, so it must not be labeled that kind of regression.

The sustained baseline has `extra_info.input_rows=20000` and `output_rows=10000`,
but its `workload_identity` contains only `rows_per_input` for row count. The
maintained validator correctly rejects the inconsistency. This is invalid
performance evidence, not a measured throughput regression. The preceding
harness review missed that descriptor mismatch; the failed check remains intact,
and no unvalidated timing or replacement run is used as acceptance evidence.

A read-only metadata diagnostic using the measured candidate installation and
common harness exported the actual warm arithmetic/filter Program. It contains
Rolling output fields `event_time`, `sequence`, `symbol`, `price` and nullable
Float64 `indicators__cf_roll_0`; the Expression input schema is identical, its
output schema is omitted, and its projection/filter use only those columns,
literals and row-local binary operations. There are no UDF references.

Every Rust admission condition matches that graph:

- Rolling has one ingress and one Error-policy output; the sole output edge is
  the internal edge to the Expression's `input` ingress. The tail has one input
  and one output.
- `config::rolling_node` validates configured outputs against the derived
  operator outputs. `build_runtime_nodes` copies ports from that same operator,
  and conversion moves the Rolling operator without changing its schema.
- Physical output projection pushdown only handles Join/ASOF, so it does not
  rewrite this Rolling schema. The known internal input schemas match exactly.
- Expression operator/runtime output declarations are both absent. The graph
  and node selected-UDF sets are empty, and `required_columns` accepts the
  concrete quoted identifiers, nested binary operations and literals.

The existing native fixture is not a byte-for-byte copy of this new harness
(the derived name and predicate differ), but it covers the same unknown-tail
admission shape. No missing real-application port or row-local proof was found.
The result therefore supports stopping this enabled candidate for lack of a
proven gain; it does not establish an unimplemented fusion path that could be
fixed and then treated as the same performance experiment.
