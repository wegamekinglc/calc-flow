# Continuous Streaming M3 Implementation Evidence

## Scope and authority

This artifact records implementation evidence for the M3 delta approved in
PR #85. The controlling order is:

1. `../specs/m3-delta-spec.md`;
2. `../api-notes/m3-delta-api-note.md`;
3. `../critiques/m3-delta-critique.md`;
4. compatible M2 behavior at `main@88243e9565795cd0bce01a155f75e735f4b47728`.

M3 adds crate-private progress preparation, one job-scoped progress driver,
generated/source-provided watermarks, idle/reactivation and multi-input
aggregation, complete transient execution tracing, receipt settlement,
status, and exact-coordinate in-memory snapshot/restore/replay. It does not
add a public Rust, Python, REST/OpenAPI, window, durable-checkpoint, or public
A6 surface.

## Delivery plan and implementation

- **M3.1 policy and preflight:** `progress/prepare.rs` and `job.rs` provide
  side-effect-free descriptors, schema/policy validation, the full capability
  matrix, immutable binding order, and prepared/config/fence fingerprints
  before connector open or task spawn. AC-01–AC-09 cover this step.
- **M3.2 driver and progress:**
  `progress/{driver,aggregate,generated,trace,types}.rs` provide one owner for
  admission, logical time, finite inbox fences, ready ordering, timers,
  aggregate progress, complete lifecycle trace, gate closure, and exactly-once
  receipt settlement. Source and operator tasks use the prepared driver and
  multi-ingress progress. AC-10–AC-33 and AC-49–AC-80 cover this step.
- **M3.3 snapshot and status:** `progress/{snapshot,status}.rs` capture exact
  prepared/config, upstream, trace, gate/fence, allocator, aggregate, and timer
  coordinates at a receipt-quiescent boundary. Restore validates every
  duplicate coordinate before governed effects. Status is observational and
  contains no late-row metric. AC-34–AC-44 and AC-81–AC-85 cover this step.
- **M3.4 integration:** `runner.rs` and `soak.rs` expose crate-private progress
  evidence to the standard two-source slow-sink soak. The main plan and
  `docs/runtime-envelope.md` use the PR #85 milestone partition.

## Acceptance mapping

The normative AC-01–AC-85 table and exact suggested names remain in
`../specs/m3-delta-spec.md` §14. Eighty behavior criteria are implemented by
tests with those exact names. The five repository/scope criteria use the
following explicit equivalent gates because their subject is the complete
diff or a complete regression suite rather than one runtime value:

- **AC-45 `m3_builds_without_window_operator`:** `rg` rejects
  `operator::window`, `window::`, `mod window`, and `LateDataPolicy`
  dependencies in all M3 runtime files.
- **AC-46 `m3_preserves_public_rust_api`:** the diff from `main` is empty for
  `lib.rs`, `error.rs`, `operator/stream.rs`, and `pipeline/stream.rs`; all new
  M3 values are `pub(crate)` at most.
- **AC-47 `m3_preserves_external_surface_baselines`:** the diff from `main` is
  empty for `python/`, `web-ui/openapi.json`, and
  `web-ui/src/api/schema.d.ts`; regenerated contracts are unchanged.
- **AC-48 `m2_runtime_regression_suite_remains_green`:** the complete Rust
  harness, including all core integration/bench targets and the isolated PyO3
  Rust target, passes.
- **AC-75 `m3_delta_prior_decisions_and_scope_firewall_remain_intact`:** the
  source/dependency diffs above, no-late behavior tests, exact transient
  snapshot tests, full-trace tests, and reconciled M3/M4/M5/post-M5
  documentation jointly enforce the firewall.

Additional milestone evidence:

- `progress_recording_stress_captures_one_hundred_seed_artifacts` records and
  independently reproduces 100 artifacts containing the seed, ordered raw
  attempts, logical-clock trace, prepared/config identity, full execution
  trace, and terminal result across natural-End, cancellation, selected-error,
  and pre-key driver-phase-failure paths.
- `progress_replay_is_deterministic_from_exact_coordinate` replays the same
  exact captured coordinate and complete trace 100 times.
- `seeded_paused_time_stress_runs_one_hundred_full_graph_schedules` preserves
  the M2 full-graph lifecycle/backpressure/cancellation matrix.

## Local verification

- **Rust formatting:** pass, `cargo fmt --all --check`.
- **Workspace Clippy:** pass for all targets/features with warnings denied.
- **Rust harness:** pass, 256 core unit tests (255 passed and the opt-in soak
  ignored), all
  integration/examples/bench targets, and 76 isolated PyO3 Rust tests.
- **Rust coverage:** pass, 92.14% lines against the 90% workspace floor; PyO3
  used the managed interpreter loader path.
- **Rustdoc:** pass with warnings denied.
- **Python:** pass, 436 tests plus Ruff check and format check.
- **Studio backend:** pass, 150 passed and 4 skipped with 93.88% coverage
  against the 85% floor.
- **Studio frontend:** pass, API sync/build, 182 Vitest tests, 4 Playwright
  tests, and production npm audit with zero vulnerabilities.
- **Supply chain:** pass, RustSec policy, cargo-deny
  advisories/bans/licenses/sources, and release-helper unit tests.
- **Generated contracts:** pass; project schema, OpenAPI, and generated
  TypeScript are unchanged.
- **Scope baselines:** pass; public Rust, Python, REST/OpenAPI, M4 window, M5
  durability, and public A6 diffs are empty.
- **Source hygiene:** pass, `git diff --check`, with no generated
  `_native*.so` in source.
- **Standard soak:** pass on exact runtime implementation commit
  `2e55d906ea1ee95c811636865ca802d1ba4d7704`, published as remote commit
  `574475b35ccab85f54e04455411dae795bac980a` with the identical tree
  `c7e024932e0bd31244075dfe4d053f5c9703a62c`; 1,200 measured seconds and 120
  samples, followed by bounded graceful drain and convergence.
- **GitHub review/Codacy:** PR #86 passed Codacy with zero issues on the prior
  remote head. Every completed Copilot review round reported success; every
  inline and suppressed actionable finding was fixed or, for the source-budget
  finding, audited against the existing whole-job preflight test and documented
  at the validated live-source boundary. The final suppressed findings now fail
  exhausted source frontiers before progress admission and require literal
  UTF-8 release-config reads. A final exact-head review is required before
  merge.

The local environment was already synchronized and `uv pip check` reported no
conflicts. The execution policy rejected a fresh `uv sync`; every locked `uv`
test/build command nevertheless completed in that managed Python 3.13.9
environment, and CI remains the authoritative clean-environment sync check.

## Standardized soak evidence

The only merge-gate soak is
`twenty_minute_two_source_slow_sink` with
`CALC_FLOW_STREAM_SOAK=1`. Its structured `calc-flow.m3-soak-log.v1` result
uses the standard 500 ms sink-write delay so bounded edges remain saturated
while the required lossless trace stays within the existing RSS-growth gate.
It records the exact commit, deterministic seed/config, observed duration and
RSS gate, admission/drain/fence/timer/trace counts, gate generations and close
cuts, settlement disposition counts and latency, peak unsettled receipts and
timer/trace sizes, bounded-edge/backpressure evidence, delivery conservation,
task/queue/resource convergence, and terminal outcome.

The exact runtime implementation-head result passed with `passed=true`, all
three bounded edges saturated and blocked, 1,330 accepted batches conserved at
both sinks with no missing or duplicate delivery, and 1,332 of 1,332 receipts
settled as commit-success. Terminal queue depth, charged rows/bytes, tasks,
unsettled receipts, and timer entries were all zero. The run completed through
`GracefulShutdown`; its first post-warmup and final five-minute median RSS
values were 38,142 KiB and 23,812 KiB, respectively, with a
`-72.60983507032795 MiB/hour` least-squares slope. The structured trace
contained 3,998 records and retained both terminal gate cuts. PR review, CI,
Codacy, and merge evidence is also recorded in the implementation PR.
