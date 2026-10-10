# Issue 406: explicit checkpoint modes and complete lifecycle timing

## Scope

This is the next increment of PR #407, following `3588cbd6`. The earlier
[Join report](issue406-join-hotpaths.md) remains a historical measurement of
that increment. PR #405 is not a merge dependency.

Rust adds `StreamingRunner::without_checkpoints`; Python adds the strict
boolean `StreamRuntimeConfig(checkpointing=False)`. The default remains enabled.
An explicit runner must pair enabled mode with managed checkpoint storage and
disabled mode with no backend. Expression and Program streaming use the same
configuration and create no temporary checkpoint directory when disabled.
Python connector-backed project configuration remains project-owned.

Disabled mode is fixed before startup. It skips storage, restore, checkpoint
coordination, recovery journals, and periodic/manual/terminal snapshots.
Ordinary sinks retain awaited output and cleanup, with best-effort effective
delivery. Exactly-once requests, epoch-based sinks, and immutable source history
are rejected before connector lifecycle calls. Live state, edge and SQL budgets,
watermarks, eviction, and cancellation remain active.

Join omits pending checkpoint records; ASOF omits checkpoint credit and index
workspace; SQL omits dirty-group tracking; Window releases emitted accumulators
without retaining snapshot payloads. Rejected output does not commit state
changes. Enabled paths keep their existing accounting and recovery behavior.

Successful enabled EOF previously reused an internal manual-request termination
sentinel as public checkpoint failure status. The fix only records that failure
when the checkpoint task actually failed. Real errors and cancellation retain
their existing paths. The comparison baseline receives this same isolated fix,
so complete-lifecycle validation does not admit a known-invalid baseline.

## Measurement protocol

The predeclared specification is
[explicit checkpoint modes](../specs/issue406-checkpoint-modes.md). The five
cases use one interval Join workload, 200,000 rows per side, 8,192-row batches:
baseline/candidate at 24 hours, baseline/candidate at 100 milliseconds, and
candidate disabled. Each has two rounds of ten measured samples plus one warmup
per round. The total budget is 600 seconds including startup, fixtures, oracle,
manifest inspection, and process cleanup. Release builds are recorded separately.

The baseline is `3588cbd6` plus the isolated successful-EOF status fix. There is
no historical disabled baseline. Source and native hashes identify both builds.
Version comparisons use matched checkpoint modes and the maintained paired
statistics; comparisons between candidate modes only describe configuration cost.

The new `benchmarks/checkpoint_cycles.py` timer starts with ready sources before
first data and ends after output collection, EOF, natural completion, final Arrow
concatenation, and owned cleanup. Data-output and completion boundaries are also
recorded. Failed/cancelled/timeout samples retain evidence and cannot become
accepted samples. A parent deadline includes blocking fixture/oracle work and
settles only its owned worker process groups.

Retained manifest-v3 evidence distinguishes terminal from nonterminal epochs and
records nonempty Join state. Retention makes observed counts lower bounds.
Missing periodic coverage remains inconclusive. No sleeps, manual checkpoints,
or additional samples manufacture checkpoint coverage.

These finite-job diagnostics do not verify a 20-epoch steady-state target or a
Polars throughput target. Engineering goals remain separate: disabled P50 at
most 1.25x same-window Polars 1T; sustained enabled throughput at least 90% of
matching disabled throughput at 1 second and 75% at 100 milliseconds. None is
an acceptance claim for this increment.

## Regression evidence

Four operator regressions failed before implementation: Join pending records
were retained, ASOF checkpoint credit was allocated, Window prepared snapshots
remained, and SQL dirty-group records were allocated. All four passed after the
mode-aware changes. Core tests first exposed dropped SQL budget and sink timeout,
unrejected epoch-based sinks, incorrect effective delivery, and the successful
EOF failure status. The focused core tests and lifecycle controls now pass.

Python tests failed against the prior API and native module for the expected
missing mode and strict native/config-storage boundary. The completed candidate
wheel passed all 88 selected Python cases, including the 17 new checkpoint-mode
cases, stream results, native stubs, Program execution, and explicit runner
recovery/cancellation controls.

Evidence is retained under the original checkout's
`target/issue406-checkpoint-modes/`, including red/green logs, the isolated EOF
patch, fixed performance plan, build logs, wheel/native hashes, raw samples,
worker cleanup records, and statistical summaries.

## Focused validation

The current Rust binary passed 36 distinct focused tests: eight new core mode
cases, seven lifecycle/checkpoint controls, four operator bookkeeping cases,
and seventeen enabled-mode operator controls. The controls cover ASOF cancelled
prefix recovery and workspace/owner accounting, SQL delta restore and rejected
output, Join cancellation/restore and sparse tombstones, and Window snapshot
budget release and compensated aggregate restore. No full module or workspace
suite was run for this increment.

`cargo clippy --locked --offline -p calc-flow -p calc-flow-python --lib --
-D warnings` passed. Rust formatting, focused Ruff checks, generated-contract
drift, and whitespace checks passed. The new measurement helper passed fifteen
focused unit tests, including terminal failure, timeout/cancellation, retained
manifest coverage, and cleanup failure behavior.

The earlier test-target Clippy diagnostic for the compiler-generated libtest
array remains unresolved; production-library lint does not clear that gate.
The pinned toolchain and workspace lints are unchanged. Full regression,
coverage, and cross-platform results remain CI responsibilities, and the PR
must not be presented as merge-ready with unresolved required checks.

The binding test target initially exposed an old constructor test argument that
needed `Some` after the native backend became optional. After adapting that
fixture, `cargo check --locked --offline -p calc-flow-python --tests` passed.
This change is inside `cfg(test)` and does not affect the measured release code.

The first Python run was interrupted after hanging before runner construction.
An independent standard-library `asyncio.to_thread`/shield example reproduced
the execution sandbox's callback/exit failure; the same example exited normally
outside the sandbox. All 88 wheel-backed Python checks then passed outside it
in 1.87 seconds. The benchmark uses that validated execution environment.

## Build provenance

Baseline and candidate release builds took 301.3 and 338.0 seconds. One failed
1.9-second candidate attempt reused a stale baseline core rlib in the shared
Cargo target directory. Removing only the three local workspace crates' release
artifacts forced a correct candidate rebuild; both final wheels are isolated.
The baseline native SHA256 starts `a2e665e51b49`, and the candidate starts
`bcc6f1e1f17d`; full hashes and loaded-module verification are in provenance.

Cargo's cache cleanup reordered only the `flatbuffers`/`fs2` lockfile entries.
Parsed package identities, versions, checksums, and dependency edges were
verified identical, and the original order was restored. Provenance retains the
build-time lock hash and the normalized dependency equivalence. The only source
change after the candidate build is the binding's `cfg(test)` constructor
argument above; all release behavior and the benchmark helper remain frozen.

## Paired results

The single fixed run completed in **98.77 seconds of the 600-second budget**.
All 100 measured jobs and ten warmups passed the full 2,198,080-row Arrow oracle
with zero maximum absolute error, completed naturally without task or checkpoint
errors, and settled their owned resources. All ten fresh worker processes exited
successfully. No case, sample count, workload size, or failed observation was
replaced, retried, or expanded.

The machine was WSL2 on an Intel Core i9-13900HX, with 32 logical CPUs and 32
Tokio workers; OMP, OpenBLAS, and MKL were fixed at one thread. No owned build or
test overlapped measurement. Host scheduling remains uncontrolled on this shared
virtualized environment. Both wheels ran outside the diagnosed broken execution
sandbox. Matching mode comparisons have matching observed machine, dependency,
and workload identities and loaded native hashes verified against their seals.

Times below are milliseconds. Full-cycle P50/P95 cover ready-before-data through
natural completion, Arrow concatenation, and owned job cleanup. Startup,
fixture/planning, oracle, manifest inspection, and checkpoint-file removal are
outside each sample timer but inside the total budget. P95 values are descriptive.

| Version and mode    | Full P50 | Full P95 | CPU P50 |
|---------------------|----------|----------|---------|
| Baseline, 24h       | 403.17   | 451.70   | 398.76  |
| Candidate, 24h      | 404.13   | 447.53   | 394.33  |
| Baseline, 100ms     | 884.12   | 929.19   | 449.16  |
| Candidate, 100ms    | 880.41   | 920.25   | 453.48  |
| Candidate, disabled | 263.25   | 288.01   | 333.57  |

The maintained two-round paired-median rule gives the following percentage
changes for candidate versus baseline in the same mode. Brackets are each
round's exact 97.85% median confidence interval, not P95 latency bounds.

| Mode  | Round 1 change [interval] | Round 2 change [interval] | Verdict      |
|-------|---------------------------|---------------------------|--------------|
| 24h   | -1.76% [-5.26%, +5.42%]   | -2.21% [-4.73%, +2.23%]   | inconclusive |
| 100ms | +1.21% [-3.25%, +5.54%]   | -1.19% [-12.29%, +1.83%]  | inconclusive |

Neither comparison triggers the maintained regression gate, which requires both
rounds' lower bounds to exceed +5%. Both remain **inconclusive** because one
round's upper bound exceeds +5%; these observations establish neither an enabled
mode speedup nor equivalence. There is no disabled historical baseline.

Within the candidate, disabled P50 is 34.86% lower than 24h and 70.10% lower than
100ms. These are descriptive configuration differences. Equivalently, finite-job
throughput at 24h and 100ms is 65.14% and 29.90% of disabled throughput. Changing
the durability contract is not a code-version speedup. The 100ms number includes
this finite job's terminal checkpoint and does not establish the separate
steady-state throughput target.

Candidate phase P50 / P95 values, in milliseconds:

| Mode     | Data output     | EOF drain       | Arrow and owned cleanup |
|----------|-----------------|-----------------|-------------------------|
| 24h      | 309.89 / 328.16 | 82.56 / 133.54  | 0.74 / 0.91             |
| 100ms    | 722.12 / 814.43 | 156.13 / 164.94 | 0.75 / 1.07             |
| Disabled | 261.00 / 286.15 | 1.09 / 1.63     | 0.71 / 0.95             |

The 24h job still performs a terminal checkpoint: its EOF-drain P50 is 82.56ms,
compared with 1.09ms when disabled. Its data-output P50 also remains above disabled
(309.89ms versus 261.00ms). This is evidence that lowering the periodic frequency
does not remove checkpoint lifecycle and bookkeeping costs; it does not isolate
any one journal, barrier, or filesystem operation's contribution.

All 20 measured on-mode samples per version retained a committed nonterminal
manifest containing nonempty Join state, in addition to the terminal manifest.
The retained evidence exposes one nonterminal manifest per sample, with 640
retained Join rows in the baseline and 640 to 8,512 in the candidate. Terminal
epoch identifiers range from 7 to 8 in the baseline
and 6 to 7 in the candidate; identifiers and retained manifest counts are not a
total periodic-epoch count. Low-frequency samples retain only terminal epoch 1.
Disabled samples have no epoch and zero checkpoint files or bytes. All ten
warmups meet the same mode-specific validity requirements.

Retained checkpoint bytes have P50 3,418 at 24h, 115,493 at 100ms, and zero when
disabled; these are files left after settlement, not cumulative bytes written.
Worker-lifetime peak RSS ranges from 585.43 to 621.43 MiB across cases and includes
fixture creation, oracle validation, and warmup. It is not independent per-job
allocation or memory-savings evidence.

`summary.json` preserves all phase quantiles, paired intervals, terminal epochs,
and retained-state coverage. The ten case/round JSON files retain every warmup,
measured observation, status, manifest hash, correctness result, and loaded
identity. `execution.json` records fixed ordering, budget, and process settlement;
`evidence-sha256.json` seals these artifacts and the driver scripts under the
original checkout's `target/issue406-checkpoint-modes/`.

No sustained 20-epoch stream, recovery run, Polars comparison, one-second
checkpoint interval, or ASOF workload was measured. The disabled and enabled
engineering goals above remain unverified. The highest-priority additional
coverage is a maintained continuous checkpoint-cycle case with nonempty retained
state and full phase/restore evidence; disabled SQL/ASOF/Window timing coverage
can follow their existing focused correctness controls in a separately scoped
measurement task.

## Final specialist review

The read-only specialist review found no outstanding implementation, API,
documentation, or measurement-methodology defects. It independently audited all
110 jobs, native and evidence hashes, clean terminal states, retained checkpoint
coverage, and process settlement. Its verdict is comment-only: this increment
is suitable for commit and PR handoff, while required CI and the unresolved
test-target Clippy gate still prevent a merge-ready claim.
