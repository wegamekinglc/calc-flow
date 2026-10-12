# Original-wheel 10M group_by diagnosis

The October 12 morning suite and one sealed-wheel recovery both find a 10–12%
regression between `89e51141` and `8ee5b71b`. Correctness passes. Unchanged window
source and a local WSL non-reproduction do not explain or dismiss the failure.

Before collecting more evidence, fix this narrow scope:

| Case   | Product inputs                         | Connection to the failure                   | Fixed observations               |
|--------|----------------------------------------|---------------------------------------------|----------------------------------|
| A/A    | Original baseline wheel twice          | Check process/host bias                     | 2 rounds × 10 pairs + warmups    |
| B/B    | Original candidate wheel twice         | Check process/host bias                     | 2 rounds × 10 pairs + warmups    |
| A/B    | Original baseline / candidate wheels   | Reproduce the failing binary comparison     | 2 rounds × 10 pairs + warmups    |
| A      | Original baseline native module        | Locate baseline instruction-address costs   | One 30s CPU sampling window      |
| B      | Original candidate native module       | Locate added instruction-address costs      | One 30s CPU sampling window      |

All five cases use only the historical 10M `calc-flow-stream/group_by` workload.
They reuse the common harness, inputs, oracles, bounded subprocess ownership,
AB/BA order and paired regression rule. Keep two rounds, ten pairs, one warmup
per side per round, and the existing +5% threshold; never filter or retry samples
to obtain a favorable verdict. A completed diagnostic can report a regression.

The collection budget is 600 seconds, including wheel hashing/installation,
fixture creation, preflight, warmup, oracle checks, startup and cleanup. Work is
cancelled at 550 seconds, reserving 50 seconds to settle owned tasks. Native build
estimate: zero; original artifacts are reused. Workflow bootstrap, dependency/tool
installation and artifact transfer are separate setup, outside measured workloads.
Retain partial evidence on timeout/failure. Dispatch once after focused validation
and specialist review; do not dispatch another full shard or complete suite.

Pin the original run `38178694560`, artifact IDs `11716974975` and `11716848795`,
full source/wheel/native seals, Python 3.13.16, unchanged dependency lock, and
common harness SHA256 `e30672098c018ef1a650492403bf5ae4813d0182e2b4921cb9233b4d26b8b2e6`.
Require actual GitHub-hosted Linux with four logical CPUs and full affinity.
Preserve the existing Tokio/Polars 32-thread settings and reject native-build
overlap. Record CPU model/cache details, environment, all IPC, input Arrow hashes,
process exits, worker completion and 250ms resource observations.

This workload enables checkpoints with a 24-hour interval. Its ready-to-output
timer excludes the terminal checkpoint, but successful final job completion is
still required. Label it LOW, never OFF. There is no new runtime memory pool cap.

Sample exact original stripped modules with Linux perf at 49Hz, `cpu-clock:u`,
strict frequency and DWARF stack unwinding with 16KiB stack dumps. Never rebuild
with debug/frame-pointer flags and call it the original binary. The isolated CI
runner temporarily sets `perf_event_paranoid=1`, records its previous value and
restores it on shell exit. No fallback frequency or profiler is selected after a
failure. Keep raw `perf.data`, raw/decoded events, native maps, tool/version and
command logs. Reject lost/throttled traces or missing native instruction addresses.

Original wheels have no static symbol table. These profiles locate addresses;
they cannot claim resolved Rust functions or causal attribution yet. Any later
symbol mapping must prove matching executable bytes/layout, or identify a new
build and reproduce its own comparison before attributing the original regression.
Profile workloads include work outside the paired timer; their timing samples
never enter A/A, B/B or A/B. Perf does not supply a direct sampling-lag metric.
Arrow IPC fixture files are local collection intermediates; artifact upload keeps
their hashes, deterministic generator provenance and IPC requests/responses.

Required validation: focused RED/GREEN tests for seals/scope, loss detection,
permission refusal, preflight failure and cleanup; existing diagnostic tests;
Ruff and workflow validation; final specialist review. Production code and the
benchmark gate remain unchanged until evidence supports a specific fix.

## Authorized repair follow-up, October 12

The user explicitly requested further repairs and a fully runnable benchmark.
The previous fixed original-wheel diagnostic is complete and is not retried.
Earlier no-full-suite/no-full-shard wording applies to that completed diagnostic
phase; this new user request authorizes the separate complete-suite repair check.
This follow-up tests a concrete production change and the complete maintained
workflow. It does not resume the paused eight-target optimization goal.

Production scope is the four temporary lookup maps in window aggregation.
Use the existing randomized AHash implementation instead of standard SipHash;
keep canonical encodings, exact equality, state accounting, row update order,
ordered installation, checkpoint serialization and public contracts unchanged.
The original-wheel profile identifies hash-table processing as a lead, not the
established cause of the historical regression. Only new paired measurements
can support a performance claim for this change.

PR #412 Windows CI has four timeout failures. Its sensitive-log test recursively
runs Cargo without the parent's feature set, recompiling while timed tests run.
Start the current test executable directly and require the child to retain that
exact executable. Keep redaction checks, deadlines and all test coverage. This
removes concurrent native compilation and feature mismatch from the test harness;
Windows CI still must confirm whether the four timeouts are resolved. The logs
do not prove that this caused all four failures.

### Verification and measurement contract

- Reuse the existing all-feature Rust cache. Estimate the focused unit rebuilds
  separately at 3–5 minutes each; do not run the whole workspace locally.
- Run existing window tests before and after the hash refactor; preserve their
  null/full-width/composite/string, collision, ordering, accounting and restored
  continuation coverage. Run the sensitive-log child regression separately.
- The final measurement is one complete `benchmark-suite.yml` dispatch, with
  explicit baseline main `e538099228b54882fe3b9fcc72d802e9998221ed` and the
  reviewed immutable repair commit as candidate. Build both ordinary release
  wheels with locked dependencies; no profile-build timing is substituted.
- Include every maintained shard: 22 total, with 409 engine cases and 26 warm
  cases plus discovered Python/Rust/Studio/frontend/lifecycle cases. The window
  cases exercise the changed scratch maps; other cases check workflow completeness
  and known regressions under the user's explicit broad verification request.
- Retain two rounds of ten AB/BA pairs for historical engine/warm cases, unchanged
  correctness oracles and the maintained +5% regression rule. Other families
  retain their existing sample counts and informational/gating semantics.
- Budget remote release builds separately at 60 minutes per arm in parallel.
  Measurement jobs retain their existing 180-minute limits, running on independent
  runners, plus prepare/summary. The overall dispatch budget is 260 elapsed minutes,
  including runner queues, setup, artifact transfer and cleanup; cancel the owned
  dispatch and retain evidence if this budget is exhausted. Job timeout values
  alone do not bound queue time or promise a 250-minute actual completion. This
  explicitly requested complete-suite verification is broader than the ordinary
  600-second feature-measurement budget.
- Do not remove slow cases, relax thresholds, filter samples or retry until green.
  Retain all failures, partial results, native seals, exact refs and CPU metadata.
  A correctness/infrastructure failure requires diagnosis and a concrete repair
  before any justified new run. An unresolved regression remains a failure.
- Report OFF, LOW and ON according to the actual case. Historical group_by remains
  LOW (24-hour interval, terminal capture outside the timer), not OFF. Existing
  OFF and paced checkpoint/recovery cases retain their distinct scopes. No default
  runtime memory-pool cap is introduced.
- Completion requires the entire suite to produce its consolidated report with
  no missing cases, correctness/infrastructure errors or confirmed regressions,
  and green required CI plus final specialist review before merge.
