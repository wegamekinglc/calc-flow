# Issue 406: pinned main, checkpoint modes, and Polars 1T

The previous-source cache has **no confirmed regression** in the valid paired
off-mode comparison. Its confidence intervals do not establish a speedup or
equivalence. Exact-main enabled runs all retained the known terminal checkpoint
failure, so their timing comparison with current code is diagnostic only.

This report records the user-requested pinned-main/current table. Keep that table,
with source revisions, mode, timing boundary and actual validity, in every later
PR 407 performance update. An earlier feature increment is a separate control
and must never replace main in that table.

## Revisions and workload

- Pinned main: `588726c8bce0f1c57b90077064834c4a472ee5f2`.
- Current production: `6d079f82c16a091b91e2996ecda8fd8c7a9c38c0`.
- Incremental off control: `e3940c7df2c347d1ffd2aceb9b7d817dd5598ff1`.

Current production caches the last source ID while constructing one Join output
gather. Consecutive rows from that source reuse its resolved position; a source
transition performs the existing lookup. The test counter is compiled only for
tests. The code commit passed 26 focused tests, Rust formatting and all-feature
core lib/tests Clippy; the specialist code review reported no findings.

The maintained interval-join fixture contains 200,000 rows per side, 64 keys,
inclusive ±5-second bounds and 8,192-row native input batches. Every observation
checks all 2,198,080 output rows against the maintained Arrow oracle, including
both row identities and computed values. Maximum absolute error was zero.
The deterministic integer-time grid permits the maintained Polars reference's
eleven shifted equality joins. This establishes output parity for this fixture.

Native timing starts with empty, ready sources and sink. It includes all event
feeding and output, natural EOF, full Arrow concatenation and awaited owned
cleanup. Compilation/startup, manifest inspection and the full oracle are outside
that timer, but inside the total measurement budget. A narrow target-only observer
uses the maintained lifecycle helpers and performs strict terminal validation
after this common endpoint for every native arm. Main's error therefore cannot
silently shorten its timed work by skipping Arrow concatenation.

## Main versus current: finite native lifecycle

Times are milliseconds. Each cell summarizes 20 measured samples in two rounds;
each fresh worker also ran one excluded warmup.

| Checkpoint mode | Main p50 | Main p95  | Main lifecycle  | Current p50 | Current p95 | Current lifecycle |
|-----------------|----------|-----------|-----------------|-------------|-------------|-------------------|
| True off        | N/A      | N/A       | Unsupported API | 257.796     | 303.933     | 20/20 valid       |
| Enabled, 1 s    | 524.397  | 592.047   | 20/20 invalid   | 390.983     | 443.503     | 20/20 valid       |
| Enabled, 100 ms | 950.554  | 1,095.198 | 20/20 invalid   | 721.931     | 784.880     | 20/20 valid       |

Main does not expose true checkpoint-off configuration. No long checkpoint
interval is relabeled as off, and main was built without a source patch.

All 40 measured enabled-main observations, and their four warmups, finished
`completed/natural_end` with zero tasks/errors and a retained terminal manifest,
but reported `checkpoint.failure_category = internal`. They fail strict lifecycle
validation. Both EOF and post-cleanup statuses are retained verbatim. Diagnostic
continuation required this exact failure shape, no earlier checkpoint failure,
no active/unknown checkpoint fields, successful full oracle and completed
cleanup. Other operation, state, oracle, manifest or cleanup failures would have
stopped execution. No accepted main/current speedup or regression verdict applies.

Machine, dependency and workload fingerprints match within each same-mode pair.
For completeness, the diagnostic paired-median changes were −26.491% and
−24.528% for 1 s, and −28.200% and −26.772% for 100 ms. Those intervals and every
raw timing remain in `summary.json`; the failed baseline lifecycle prevents an
accepted performance claim regardless of the apparent timing difference.

## Valid incremental control

| True-off revision  | p50 ms  | p95 ms  | CPU p50 ms | Lifecycle   |
|--------------------|---------|---------|------------|-------------|
| `e3940c7d` control | 262.611 | 322.240 | 334.721    | 20/20 valid |
| `6d079f82` current | 257.796 | 303.933 | 328.841    | 20/20 valid |

The relative change in aggregate wall medians is −1.834%. The maintained verdict uses
aligned per-pair changes, rather than that aggregate ratio:

| Round | Paired median change | Confidence interval | Coverage    |
|-------|----------------------|---------------------|-------------|
| 1     | −3.119%              | [−4.992%, +0.199%]  | 97.8515625% |
| 2     | −2.347%              | [−7.661%, +1.153%]  | 97.8515625% |

The two-round maintained rule confirms regression only when both lower bounds
exceed +5%. Its result here is `no-confirmed-regression`. Both intervals include
zero; this is not established speedup or equivalence. Matching contract-v2
machine/dependency/workload fingerprints were required before applying the
paired-median rule. No minimum-ratio substitution or additional acceptance rule
was used.

## Polars reference and distinct goals

| Reference         | Timing scope                                               | p50 ms  | p95 ms  | CPU p50 ms |
|-------------------|------------------------------------------------------------|---------|---------|------------|
| Polars 1T, 1.44.2 | Prepared lazy plan: streaming collect and Arrow conversion | 233.027 | 275.392 | 231.731    |

Polars' actual thread pool was checked to contain one thread. Its lazy plan and
Arrow input conversion are prepared before timing; collection and output Arrow
conversion are timed. Calc Flow runs the complete finite stream lifecycle with
32 Tokio workers. The descriptive current-off/Polars p50 ratio is **1.106×**.
The execution, checkpoint and ownership scopes differ, so this ratio does not
establish the goal for a comparable streaming execution contract.

| Goal                                    | Observation from this run                                              | Status                        |
|-----------------------------------------|------------------------------------------------------------------------|-------------------------------|
| True off ≤1.25× comparable Polars 1T    | 1.106× for the distinct finite-native/bulk-Polars scopes               | Pending comparable scope      |
| Sustained 1 s throughput ≥90% of off    | Finite p50 throughput ratio 65.94%; no periodic checkpoints            | Pending sustained measurement |
| Sustained 100 ms throughput ≥75% of off | Finite p50 throughput ratio 35.71%; periodic evidence in 20/20 samples | Pending sustained measurement |

These seven finite workloads do not establish sustained throughput, recovery or
the complete scheduled lifecycle acceptance contract. No additional workloads
were run to fill those gaps.

## Checkpoint and resource evidence

Both 1 s arms had no nonterminal manifest and completed terminal epoch 1. They
exercise enabled checkpoint bookkeeping and terminal publication, not a periodic
1 s checkpoint workload. Both 100 ms arms retained a nonempty nonterminal
manifest in all 20 measured jobs, plus the terminal manifest. Main's terminal
epochs were 7–8; current's were 6–7. Default retention leaves one nonterminal
manifest per sample; it does not prove cumulative epoch counts or one continuous
run. Current off and the e394 control produced no state files or epochs.

| Native arm     | Retained checkpoint bytes, median | Retained bytes, maximum | Worker lifetime peak RSS, MiB |
|----------------|-----------------------------------|-------------------------|-------------------------------|
| Main 1 s       | 3,418                             | 3,418                   | 627.398                       |
| Current 1 s    | 3,418                             | 3,418                   | 602.633                       |
| Main 100 ms    | 115,493                           | 1,502,378               | 612.840                       |
| Current 100 ms | 317,612.5                         | 3,182,142               | 610.516                       |
| e394 off       | 0                                 | 0                       | 586.449                       |
| Current off    | 0                                 | 0                       | 586.184                       |

Checkpoint cuts differ as jobs progress. These physical retained-byte snapshots
are not cumulative writes, fsync counts or evidence of a disk regression. RSS
includes fixture construction and full Arrow oracles; Polars' corresponding
worker lifetime peak was 701.543 MiB. No allocation or steady-state memory claim
is made.

## Profile, execution and provenance

The preceding current-off profile used the exact reused e394 native, one warmup
and ten measured jobs at 1,000 Hz. All eleven full oracles, natural completions,
off-state checks and cleanup checks passed. Direct monotonic boundaries selected
2,952 timed CPU samples, with zero unresolved native leaf samples: 2,936 during
data processing, nine during EOF and seven during Arrow/cleanup.

`materialize_output_record` accounted for 18.63% inclusive samples;
`output_gather::column` for 12.33%; `SideGather::new` for 5.86% self/6.17%
inclusive. A fixed offline 10 ms boundary trim retained 18.92% materialization
and 5.99%/6.32% `SideGather` shares. Eviction remained 10.06% self/16.73%
inclusive, and Arrow byte gathering 7.11% self/9.25% inclusive. These sampled
stacks support investigating output gathering but do not isolate an individual
map-lookup instruction. Profile-perturbed p50 wall/CPU values, 262.184/339.677 ms,
were excluded from performance comparisons.

Seven workloads ran two fresh-worker rounds, ten measurements and one warmup per
worker: 140 measurements and 14 warmups. Six native cases produced 132 native
jobs; Polars produced 22 collections. All 154 observations passed their complete
Arrow oracles. The 44 main native observations remained lifecycle-invalid; the
other 88 native observations were valid. All fourteen workers and the summary
process exited successfully, every owned process settled, and no cleanup signal,
retry or extra sample was needed.

Measured execution alternated the entire forward/reverse order at each index:
main 1s, current 1s, main 100ms, current 100ms, e394 off, current off, Polars.
Each native pair remained adjacent. The machine was WSL2 on an i9-13900HX with
32 logical processors/32 Tokio workers and one OpenMP/BLAS/Polars thread. Owned
builds and workloads were idle before launch; Windows host load and power mode
were uncontrolled. Paired timing uncertainty therefore remains relevant.

The profile plus its external preflight consumed 7.936413 seconds. The comparison,
its charged external process check, fixtures, startup, warmups, oracles, manifest
inspection, file cleanup, analysis and final sealing consumed 125.862591 seconds.
The total was **133.799004 / 600 seconds**. Unused time was not spent on additional
measurements. Separate release build costs were 384.867593 seconds for exact main
and 344.836340 seconds for current; the e394 native was reused.

All releases use the same rustc 1.88.0 compiler, CPython 3.13 default ABI,
optimized release settings, locked dependency graph and unstripped symbols.
The old cached main ABI3 wheel was rejected because its build features differed.
Current source was exported with a binary diff and immutable archive; 777
Rust/Python/release paths matched code commit `6d079f82`, allowing only equivalent
Cargo.lock ordering. The control's production-source equivalence was audited
separately, including its test-only differences.

| Arm                  | Native SHA-256                                                     |
|----------------------|--------------------------------------------------------------------|
| Exact main           | `83c1984005258b702fabbb322d27bb9a0bf7842823d7ddabc4e9660b40997d5f` |
| Current              | `1a1c6420acaff0947de694683886da4cfabbd6d9755dbbe06fce86db4e02cd66` |
| e394 control/profile | `008208d4b2d10ccbd20d634e35fd8e1e4ec1bba3367cd9f289a9a699dcea273e` |

Raw evidence resides in `target/issue406-main-comparison/` in the original
repository checkout. `provenance.json` records source/wheel/native hashes,
compiler and matching build settings. `summary.json` contains exact per-case
statistics, actual lifecycle counts, paired intervals, phase/resource summaries
and goal limitations. Per-case JSON files retain every warmup/sample, full
identity, oracle result, status and manifest evidence. Profile captures, symbols
and clock windows are retained separately.

The comparison preparation seal is
`af617d87e938e81802f40a34b59a84c45fe708ceeeeccd576faa49171bad22c6`.
`comparison-evidence-sha256.json` seals the 19 result/control documents;
`comparison-budget-final.json` charges the full controller lifetime through
sealing. Five synthetic observer tests passed, covering equal invalid-main work,
unexpected failures, missing manifests, actual-valid-main preservation and
cancellation settlement. Three clock-domain tests and two launcher-signal tests
also passed; focused Python syntax and Ruff checks passed. No native pilot ran.

## Coverage advisory

The maintained interval-join fixture exercises the changed output-gather path;
its e394/current off pair supplies the direct performance control. The retained
1s/100ms cases cover the requested finite lifecycle modes. Existing correctness
tests cover source transitions and output parity.

For later targeted coverage, prioritize a maintained output-gather workload with
frequently alternating source owners, where the last-source cache has fewer hits.
That would characterize its fallback cost independently of this fixture's runs
of consecutive owners. The separate absolute targets need comparable execution
boundaries and sustained periodic checkpoint workloads. Those are follow-up
measurement scopes, not claims or acceptance results from this run.
