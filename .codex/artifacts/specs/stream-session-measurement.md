# P1 StreamSession — measurement commands

Date: 2026-11-15. Status: pre-registered; **not run**. Protocol:
[stream-session.md §8](stream-session.md#8-measurement-protocol-pre-registered);
critique item 3 (B3) and S6 in
[the critique](../critiques/stream-session.md).
Harness: branch `feature/stream-session-harness` (commits `302b0085` port of
the #616 pieces, `2a7c7ae6` session driver and explicit cases).

The candidate wheel does not exist yet. It must be built from
`feature/stream-session` after the implementation lands and after the harness
commits are merged into that branch (step 2). Nothing below was executed except
the baseline manifest check in "Builds".

## Cases

Each row is one case; every case runs 30 alternating AB/BA pairs in a fresh
pair of persistent worker processes. Shared configuration: warm dual-SMA
spread, window 20, fast window 5, history 4,096 rows in one 4,096-row segment,
exact projection, checkpointing off, `SourceProvidedWatermarks`, per-append
watermark at the last appended timestamp, forced GC before every sample,
`TOKIO_WORKER_THREADS=4` in both workers.

| Case | Workload              | Baseline arm | Candidate arm | Reference | Pre-registered gate               | Connection to the change                         |
|------|-----------------------|--------------|---------------|-----------|-----------------------------------|--------------------------------------------------|
| 1    | 1 row × 1 entity      | runner       | session       | TA-Lib    | candidate p50 ≤ 20 µs (§8.4)      | per-push overhead the session removes            |
| 2    | 99 rows × 1 entity    | runner       | session       | —         | speedup CI vs 1.0                 | small-batch session path                         |
| 3    | 6,400 rows × 64 ents  | runner       | session       | TA-Lib    | speedup CI vs 1.0                 | large-batch control: operator work dominates     |
| 4    | 1 row × 1 entity      | runner       | runner        | —         | regression iff CI high < 0.95     | runner cost of the new session edge variants     |

Seeded case order (`--seed 20261115`): 3, 2, 1, 4. The bootstrap uses the
maintained harness's fixed seed 771031 with 20,000 resamples.

## Builds (reported separately from the budget)

| Build     | Source                                         | Status                                   | Estimate                               |
|-----------|------------------------------------------------|------------------------------------------|----------------------------------------|
| Baseline  | `b9a46e12` (product-identical to `39cab2fb`)   | reuse; manifest loads and verifies today | 0                                      |
| Candidate | `feature/stream-session` with harness merged   | build after implementation lands         | ≈10–15 min (#616's build: 10 min 12 s) |

Baseline manifest:
`target/compiled-region-performance/baseline-build/build.json` (wheel
`e7af99f8…`, native module `68be91fe…`, rustc 1.94.0, Python 3.13.9,
`Cargo.lock` `d2b0e0ea…`). `compare` refuses to run when the candidate's
`Cargo.lock`, rustc or Python differ; report that as **inconclusive** and stop
for a decision rather than rebuilding the baseline.

## Commands

Shell variables (absolute paths):

```bash
ROOT=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow
CANDIDATE=$ROOT/target/worktrees/stream-session
PY=$ROOT/.venv/bin/python
WORK=$ROOT/target/stream-session-p1
```

1. Precondition: `$CANDIDATE` is a clean checkout of `feature/stream-session`
   with the implementation committed and no other build, test or benchmark
   running on the machine.
2. Merge the harness into the candidate branch **before** building; the
   build manifest pins `git_sha`, so a later merge invalidates the wheel:

   ```bash
   git -C "$CANDIDATE" merge --no-ff feature/stream-session-harness
   ```

3. Build the sealed candidate release wheel (not part of the 600 s budget):

   ```bash
   cd "$CANDIDATE"
   "$PY" scripts/profile_warm_stream.py build \
     --source "$CANDIDATE" \
     --target "$CANDIDATE/target" \
     --output "$WORK/candidate-build"
   ```

4. Run the four cases inside one budget. Both workers import the harness from
   `$CANDIDATE`; the report records its SHA-256 per harness file:

   ```bash
   cd "$CANDIDATE"
   "$PY" scripts/profile_warm_stream.py compare \
     --baseline-build "$ROOT/target/compiled-region-performance/baseline-build/build.json" \
     --candidate-build "$WORK/candidate-build/build.json" \
     --output benchmarks/stream-session/p1-20261115.json \
     --work-dir "$WORK/work" \
     --samples 30 --seed 20261115 --budget-seconds 590 \
     --history-rows 4096 --history-segment-rows 4096 \
     --window 20 --fast-window 5 \
     --indicators dual_sma_spread --gc-modes forced --exact-projection \
     --case name=1,append_rows=1,entities=1,drivers=runner:session,reference=ta_lib,p50_limit_us=20 \
     --case name=2,append_rows=99,entities=1,drivers=runner:session \
     --case name=3,append_rows=6400,entities=64,drivers=runner:session,reference=ta_lib \
     --case name=4,append_rows=1,entities=1,drivers=runner:runner,regression_below=0.95
   ```

   This writes `benchmarks/stream-session/p1-20261115.json` (raw samples,
   phases, references, worker environments and exit receipts, builds, harness
   hashes, verdicts) and `benchmarks/stream-session/p1-20261115.md`. Worker
   temporaries and receipt files stay under `$WORK/work`.

## Budget

600 s wall-clock for all cases: `--budget-seconds 590` covers manifest checks,
worker installation and startup, warm-up, oracle and digest checks, samples,
references, finish and cleanup; 10 s is reserved for interpreter start and
report writing. When the budget expires the harness cancels the running case,
settles both workers, records the remaining cases as unmeasured and sets
`status: stopped`. Do not add cases or samples or rerun for a different result.

Estimate from the #616 artifacts (five `compare` invocations of two 20-pair
points each, fresh workers per point: 24.4 s in total):

| Case | Startup | Warm-up | 30 pairs + oracle | Reference | Cleanup | Estimate (upper) |
|------|---------|---------|-------------------|-----------|---------|------------------|
| 1    | ≈2 s    | < 0.1 s | ≈0.5 s            | < 0.1 s   | ≈0.5 s  | ≈3 s (≤ 15 s)    |
| 2    | ≈2 s    | < 0.1 s | ≈0.5 s            | —         | ≈0.5 s  | ≈3 s (≤ 15 s)    |
| 3    | ≈2 s    | < 0.1 s | ≈1.5 s            | ≈0.2 s    | ≈0.5 s  | ≈5 s (≤ 20 s)    |
| 4    | ≈2 s    | < 0.1 s | ≈0.5 s            | —         | ≈0.5 s  | ≈3 s (≤ 15 s)    |

Total ≈15 s, upper estimate 65 s, of the 590 s budget.

## Verdicts (computed into each case's `verdict`)

- Case 1 acceptance: `candidate_p50_within_limit` (p50 ≤ 20 µs).
- Cases 1–3: `paired_speedup` is `slower` when the speedup CI upper bound is
  below 1.0, `faster` when the lower bound exceeds 1.0, otherwise
  `inconclusive` (not a regression).
- Case 4: `confirmed_regression` when the CI upper bound is below 0.95.
- If case 1 misses 20 µs, the session phase table locates the cost; report it
  and stop for a decision (plan §7.2) instead of expanding the measurement.
- The machine is WSL2 (virtualized); state that in the report. A fingerprint,
  dependency or toolchain mismatch is inconclusive, not a result.

## Protocol decisions taken while preparing the harness

1. **History length.** The #616 headline results labelled "history 4,096"
   were measured at history 1,024,000: `matrix_points` crosses every append
   size with H=1,024,000, and `measure.py` read `cases[0]`, which was that
   point in all five reports (`case-*.json`). The ≈599 µs one-row figure is
   therefore an H=1,024,000 number. This protocol measures H=4,096 as §8.2
   states, through explicit `--case` selections that never cross matrix
   points. The rolling window is bounded, so the difference is expected to be
   small, but the #616 figures are context, not this protocol's baseline.
2. **Session thread.** Each session is built, warmed, sampled and finished on
   one dedicated worker thread with no running event loop, as a synchronous
   caller would use it. This avoids depending on the §6.3 running-loop
   exception and satisfies §6.1 (construction rejects a running loop).
3. **Same prepared input.** Both drivers time the table `Batch` and the
   `Watermark` object from `_prepared_events`; the session's
   `{"quotes": batch}` mapping is built before the timer starts.
4. **Phase timers without library instrumentation.** `sys.monitoring` CALL and
   C_RETURN events on the loaded `calc_flow` Python code timestamp the native
   `_StreamSession.push` call: `argument_handling` is Python work before it,
   `native_call` includes PyO3 argument conversion, the engine call and Arrow
   export, and `result_assembly` is Python work after it up to the returned
   `pa.Table`. The two callbacks are inside every candidate session sample
   (conservative for the 20 µs gate); each worker calibrates their cost
   (≈0.4 µs per sample on this machine) into
   `start[1].phase_probe.overhead_seconds_estimate`. A sample whose native call
   is not observed exactly once falls back to total-only timing with a recorded
   reason.
5. **TA-Lib reference.** Unpaired, in the idle baseline worker after the 30
   pairs: sample *i* recalculates `window - 1` rows per entity before the
   *i*-th paired append plus that append, returns only the appended rows and
   matches the warm oracle outside the timed call. GC is collected before and
   disabled inside each timed call (#616 ran it in the controller without a
   collection).
6. **Zero late rows.** P1 sessions expose no `status()`. The plan preflight
   pins `late_policy=error`, so any late row fails the session, and `finish`
   must flush zero rows.
7. **Plan identity.** The session arm reports the identity of the same program
   compiled by `_prepared_plan`; the session's internal plan fingerprint is not
   exposed, so identity proves the same program and compile, not the session's
   own plan instance.
8. **Strict parity.** Warm and per-sample Float64 digests, including the
   canonical output schema and metadata, must be equal across the runner and
   session arms. A difference aborts the case; treat it as a session parity
   defect to fix, not as a measurement result.
9. **Coupled checkpointing.** As in #616, `exact_projection` also turns runner
   checkpointing off; the session driver requires it. Both drivers use
   `EdgeBudget(max_rows=max(segment, append), max_bytes=64 MiB)`.

## Not measured

Memory/RSS comparison (recorded at finish only), normal-GC mode, `advance` and
`finish` latency, `Ordered` and `BoundedOutOfOrderness` watermark modes,
sparse multi-entity appends, Union/fan-out graphs and the NumPy path.
