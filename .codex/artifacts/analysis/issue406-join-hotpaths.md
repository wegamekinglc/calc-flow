# Issue 406: first Join hot-path implementation

## Scope

The standalone implementation includes the owned parallel-probe cost gate,
algebraic key-arena sizing, probe-key reuse after an owned probe declines,
and column-wise materialization across multiple shared payload chunks. It keeps
row ordering, V1 charges, memory ownership, cancellation, checkpoint formats,
and public APIs intact. No new checkpoint-disabled mode is introduced here.

The branch is based directly on main `588726c8`. The cost gate and arena-sizing
code originally developed in #405 are included in this PR, together with their
supporting tests; merging #405 is not required. Its unrelated lockfile ordering
and historical report are excluded, and incidental test weakening is reverted.
The unused `arena_measure_cells` counter and its vacuous zero-count test are
also excluded; existing framing, allocation, and charge tests remain.

The recorded performance comparison remains baseline
`ac74dc54ed5f2b326e7344b698f2f07502cefdb4` versus the measured implementation
at `ff7ddba6`. Both already include the cost gate and algebraic sizing, so the
reported deltas isolate key reuse and output gathering. They are not gains
relative to main. Production code is preserved when making the PR standalone.

The cost gate sums opposite-side run lengths for distinct probe keys. It is
a conservative proxy, not a complete output-work estimate: duplicate probe
keys can produce substantially more fill work than the estimate counts.
Repeated-key fanout needs a separate focused measurement before tuning this
heuristic. This packaging change does not alter its behavior or claim a gain
for that unmeasured shape.

## Acceptance

- The cost gate keeps slim probes serial; algebraic sizing preserves V1 charges.
- Cost-gated and worker-budget fallback reuse the funded native keys. One
  native key-arena frame is built per admitted row, with unchanged output and
  eventual reservation release.
- Single-source flat columns retain `take`; multiple shared sources use
  `interleave`, with no per-row column views. Cover two record batches in one
  event, multiple retained chunks, both incoming sides, nulls, reordered
  selections, and offsets above `u32`.
- Dictionary and nested payloads keep the existing compacting path, including
  removal of unreferenced dictionary values. Output buffers remain independent
  of large unselected input payloads.
- Focused Join tests and package lint pass; generated contracts have no drift.

## Performance protocol

The predeclared plan and raw evidence live in
`target/issue406-join-hotpaths-performance/` in the original repository root.
Four cases compare baseline and candidate: Join 1M with 64k batches and
interval Join 100k/side with 8k batches, each at 24h and 100ms checkpoint
intervals. Each case has two rounds of ten AB/BA pairs on release builds.
The total measurement budget is 600 seconds including warmup, correctness,
process startup, and cleanup. Builds are recorded separately.

The 24h mode is only a low-frequency baseline. Both modes retain journal
maintenance and terminal snapshots. Ready-output timing does not charge an
in-flight checkpoint's remaining work and cannot establish steady-state
checkpoint throughput. Wall time, process CPU, per-thread runtime, RSS,
end-adjacent checkpoint status, and output correctness are retained.

## Remaining stages

1. Define a true checkpoint-disabled lifecycle and full-cycle checkpoint-on
   acceptance before claiming the investigation's separate throughput targets.
2. Optimize shared batch admission and key representation with charge parity.
3. Reassess larger state, ASOF merge, and parallel execution changes using
   profiles of the resulting implementation.

The investigation's proposed target remains P50 at most 1.25 times same-window Polars 1T
when checkpoints are truly disabled. Proposed sustained checkpoint-on budgets
are at least 90% of matching disabled throughput at 1s and 75% at 100ms.
These remain engineering goals, not demonstrated results of this change.

## Regression evidence

Before implementation, `slim_probes_without_dispatch_stay_serial` failed with
16,384 arena frames instead of 8,192. The existing worker-admission refusal
fixture, extended with the same complexity assertion, independently failed
with 16,384 instead of 8,192. Logs are `red-keys.log` and `red-budget.log` in
the original repository's `target/issue406-join-hotpaths-build/`.

The implementation carries optional funded `NativeKeys` through both serial
fallback boundaries. Worker recovery returns the keys only after all owned
work settles; successful parallel output keeps the same ownership path.
Native-to-SQL fallback still drops scratch and refunds its unused index.

Five gather regressions then failed on the old production path. The mixed-parent
fixture observed 12 row-column views instead of zero; the multi-event/chunk
fixture observed 192 instead of zero. Three low-level cases observed zero
interleave calls instead of one. The nested dictionary safety cases passed.
After key reuse, both key complexity regressions passed in that same run.
These results are recorded in `red-gather.log`.

Multi-source materialization assigns source IDs by first appearance in the
matched chunk and uses a pointer-identity lookup only for deduplication.
Selectors keep the original pair order. The type allowlist is unchanged;
flat output remains independently allocated. Selectors and source lookup
are local to one bounded output chunk.

## Focused validation

On the original measured tree `ff7ddba6`,
`cargo test --locked -p calc-flow --lib operator::join:: -- --test-threads=2`
passed all 214 selected tests. This includes the new complexity and output
regressions, both serial fallback directions with complete credit release,
worker refusal/cancellation, match and memory budgets, and existing Join
checkpoint capture/restore tests. Full workspace and cross-platform gates
remain CI responsibilities.

The original `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
check failed: pinned Rust/Clippy 1.88 reported `large_stack_arrays` with no
source span for the generated libtest table. The unchanged `ac74dc54`
comparison baseline reproduced the same diagnostic, captured in
`clippy-baseline.jsonl`. That historical baseline has 2,050 tests and the
original measured candidate has 2,055; the generated reference array crosses the
16,384-byte threshold at 2,049 entries on this platform. This matches
[upstream Clippy issue 13774](https://github.com/rust-lang/rust-clippy/issues/13774).
No lints or pinned toolchain were weakened. The unresolved test-target lint
gate prevents a green/merge-ready claim. This diagnosis was reproduced on
`ac74dc54`, not main: importing the additional tests can cross the generated
harness threshold relative to main. The standalone branch still requires
its own CI result; removal of the vacuous test does not resolve that gate.

The separate production check
`cargo clippy --locked -p calc-flow --lib -- -D warnings` passed.
`cargo fmt --all --check`, generated-contract drift checks, and
`git diff --check` also passed.

Baseline and candidate release wheel builds took 407.4 and 419.5 seconds
respectively, separate from the measurement budget. Candidate Rust source
hashes were verified unchanged after build; wheel/native hashes, Cargo lock
hash, compiler version, and source patch hash are recorded in `provenance.json`.

## Paired performance results

The four fixed cases completed in 84.1 seconds, within the 600-second budget.
All 160 measured samples passed the Arrow oracle and ended as
`completed/natural_end` with zero task errors. All owned measurement workers
were settled. The whole-case builds above are excluded from this duration.

Each row below pools 20 samples for descriptive P50 values. Paired changes
and 97.85% median intervals remain separate by round; no pooled ratio is
used as an acceptance verdict. These are baseline `ac74dc54` versus the
frozen candidate, not the older source tree in the initial investigation.

| Case                                              | Baseline wall ms | Candidate wall ms | Baseline CPU ms | Candidate CPU ms | Round 1 paired change [interval] | Round 2 paired change [interval] |
|---------------------------------------------------|------------------|-------------------|-----------------|------------------|----------------------------------|----------------------------------|
| Join 1M, 24h                                      | 135.6            | 105.1             | 151.8           | 120.3            | -21.5% [-29.2%, -18.7%]          | -22.8% [-24.9%, -17.3%]          |
| Join 1M, 100ms (insufficient checkpoint coverage) | 215.6            | 104.9             | 160.5           | 122.0            | -49.9% [-51.9%, -44.9%]          | -51.1% [-53.9%, -49.0%]          |
| Interval 100k/side, 24h                           | 206.0            | 157.4             | 247.2           | 197.0            | -24.6% [-26.4%, -20.5%]          | -22.4% [-24.2%, -20.8%]          |
| Interval 100k/side, 100ms                         | 373.5            | 253.1             | 259.4           | 198.0            | -32.0% [-39.4%, -29.3%]          | -32.1% [-34.3%, -16.7%]          |

Both low-frequency cases show repeatable reductions in output-window wall
time and process CPU. All low-frequency samples had zero completed/in-flight
checkpoints and zero snapshot files at the end-adjacent observation. This
confirms gains in the data path with periodic snapshots absent; journal
maintenance remains enabled.

**The Join 100ms figure is not evidence of a 50% checkpoint-on throughput
improvement.** Every baseline sample had completed one epoch, whereas 18/20
candidate samples had completed none at the end-adjacent observation. The
optimized operation is near the interval boundary. Its finite-window time
changes partly because checkpoint work moves beyond that boundary. This case
fails the intended checkpoint-completion coverage and cannot accept the
checkpoint-on throughput goal, despite favorable timing intervals. No extra
rows, cases, or samples were added to seek a favorable acceptance result.

For interval Join at 100ms, both versions observed completed epochs: baseline
2–3 and candidate 1–2. However, 20/20 baseline and 19/20 candidate samples
still had an in-flight epoch. Their tail work is excluded. The roughly 32%
short-window reduction is useful diagnostic evidence, not a sustained
checkpoint-throughput verdict. All end status reads occur after the wall
timer and before EOF; they do not timestamp individual epoch completions.

Worker-lifetime peak RSS ranges across case/version maxima were 413–433 MiB.
These cumulative high-water marks include warmup and setup; they do not
establish per-sample allocation savings or memory equivalence. The existing
post-EOF checkpoint `failure_category=internal` observation remains present;
output correctness and job completion do not replace durable-recovery
acceptance. No ASOF, 10M, true checkpoint-disabled, or steady-state throughput
claim is made by this increment.

The next benchmark change should run across complete nonterminal checkpoint
cycles with nonempty state and timestamp epoch completion. Lower-frequency
and true-disabled results must remain separately labelled.

Evidence files under the original root's
`target/issue406-join-hotpaths-performance/`: `plan.json`, `candidate.patch`,
`provenance.json`, `summary.json`, `checkpoint-evidence.json`,
`execution.json`, `measurement.log`, and per-version/round sample JSON.
`measure.py`, `run.py`, and `summarize.py` retain the bounded reproduction
steps; `evidence-sha256.json` indexes the frozen evidence.

## Final specialist review

The read-only specialist review found no introduced correctness or resource
ownership blockers. It checked the implementation, 214 passing Join tests,
production-library Clippy, formatting, contract drift, paired sample results,
and source provenance. The checkpoint coverage limits above remain part of
the result, including the rejected Join 100ms throughput interpretation.

The initial verdict was comment-only and suitable for an implementation
handoff, not merge approval. Test-target Clippy remains unresolved after the
diagnostic reproduced on the historical comparison baseline. These findings
preceded the commit and PR handoff;
the standalone branch requires its own CI results. PR #405 is no longer a
merge dependency.

## Standalone verification

Compared with `ff7ddba6`, all production code is unchanged. The Rust diff is
limited to restoring three test fixtures/assertions and removing the unused
test-only counter and its redundant test. Restoring main's lockfile only
changes dependency ordering; parsed package versions, checksums, and dependency
edges are identical. The final diff against main excludes that lockfile and
the old probe-cost branch report.

The three restored tests passed individually with `cargo test --locked
--offline -p calc-flow --lib <name> -- --test-threads=1`:

- `status_tracks_ingress_watermark_idle_reactivation_and_end`
- `test_native_key_types_match_sql_order_in_both_directions`
- `test_native_probe_resolves_each_distinct_key_once_before_both_window_passes`

The initial rebuild took 2m30s. Each selected invocation ran exactly one test,
with 2,053 others filtered out. The standalone libtest target therefore has
2,054 tests; the earlier 214-test Join run and 2,055 total refer to the original
measured tree, not a new full run. Logs are retained in
`target/issue406-join-hotpaths-build/standalone-targeted.log` in the original
repository root. `cargo fmt --all --check` passed after these edits. The
previous performance measurements were not repeated for this packaging change.
