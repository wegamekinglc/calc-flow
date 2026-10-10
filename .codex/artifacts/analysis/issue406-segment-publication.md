# Issue 406: batched segment publication

The maintained paired rule classifies the 100 ms checkpoint-on finite lifecycle
as **improved** and checkpoint-off as **no confirmed regression**. A separate
continuous candidate run retained 58 consecutive nonempty periodic checkpoints
and passed a real durable restore plus the full Arrow output oracle. These are
focused results on the recorded WSL2 environment, not steady-state or portable
throughput guarantees.

## Implementation and validation

`StateLineageBackend::publish_segments` preflights conflicts and duplicates;
its provided default preserves existing committed-read and staged-publication
contracts. Local publication uses one owned worker and one sync per affected
committed directory, while retaining file, staging, creation and manifest
boundaries. Managed staging distinguishes durable carries, visible files needing
confirmation, and newly staged files. Session caches and pins advance only
after successful publication; cancellation settles the whole admitted batch.

Five baseline behavior failures were observed before implementation: repeated
directory syncs, a visible-file retry incorrectly succeeding after sync failure,
premature carried-cache advancement, omitted directory-creation sync retries,
and cancellation settling only the first segment. The final change passes
38 distinct focused Rust tests and its new API doctest, including strict legacy
backends, multi-segment operator ACK ordering, sink staging, cross-session retry,
ownership, restore, retention and checkpoint-off controls.

All 15 checkpoint-helper tests and their 71 existing assertions are preserved.
The helper refactor resolves the five reported complexity findings without
changing timing or cleanup behavior. Core library/test Clippy with all features,
warning-free rustdoc, targeted Ruff/complexity checks, formatting and generated
contract checks pass. Full coverage and cross-platform gates remain CI work;
these local results do not establish merge readiness.

## Workload, budget and provenance

The changed hot path groups local state-segment publication within one
snapshot/epoch. The preceding interval-Join investigation identified repeated
directory synchronization as a material checkpoint-on cost. This experiment
measures the resulting lifecycle; it does not count this candidate's syscalls.

The predeclared five cases use `interval_join`, 200,000 input rows per side and
8,192-row batches. Four cases compare baseline/candidate with checkpoint-off
(`checkpointing=False`, no managed backend) and checkpoint-on (100 ms, two
retained epochs). Each mode has two rounds of ten alternating baseline/candidate
pairs, with one warmup in each fresh case/round worker: 80 measured and eight
warmup finite jobs. The fifth case has one fixed paced prefix and one restored
suffix, with no warmup or timing comparison. No cases, samples or retries were
added.

All 90 native job lifecycles, fixture creation, preflight, startup, warmup,
oracles, restore, inspection, analysis, sealing and cleanup completed in
**78.72097 seconds**, within the 600-second budget. The paired phase took
71.51926 seconds, the sustained worker phase 7.11574 seconds and analysis
0.08410 seconds. `summary.json.execution_elapsed_seconds` describes only the
paired phase; `execution.json` and the controller's final record describe the
whole attempt. Final source/helper assertions ran inside the global timer.
Every worker exited successfully; no cleanup signals or unsettled workers
remained.

Baseline is `53f44fad3f6a323fcf32ab689592099846d99146`. Its previously sealed
candidate artifact was reused after proving that only the changelog and an
analysis document differed from this baseline; release sources were identical.
The candidate was exported with its untracked production file and binary patch.
Its release build took **363.87138 seconds**, plus 0.46576 seconds of static
provenance validation, separately from measurement. Both arms use the same
compiler, locked dependency graph, release settings, unstripped build mode and
42 byte-identical Python adapter files. Final production-file hashes matched
the built export. Both arms used the same frozen maintained fixture/oracle and
refactored checkpoint-cycle helper.

- Baseline native SHA-256:
  `86ba1e3b82a3e1e34985c8db66cec9820a9ce6fcb52531c21a856c28f4bce976`.
- Candidate native SHA-256:
  `008208d4b2d10ccbd20d634e35fd8e1e4ec1bba3367cd9f289a9a699dcea273e`.
- Candidate source archive SHA-256:
  `ae06d90cf79de35f52fd069022bdc07af83ce718dd41db69ccc68a8226b97ac4`.
- Common `checkpoint_cycles.py` SHA-256:
  `057d02aa03628cd1c2d3eb0fe1e488c517e1ae708c6cae4bd3bb23d8193976e8`.

Machine, dependency and workload fingerprints match within each compared mode.
The machine is WSL2 on an i9-13900HX with 32 logical CPUs, 32 Tokio workers and
single-thread BLAS settings. Owned builds/checks had settled before timing;
Windows host load and power behavior were not controlled. Alternating execution
reduces drift but does not establish independent pairs.

## Separate checkpoint-mode comparisons

Timers cover ready input through natural EOF, Arrow concatenation and owned
cleanup. Startup, fixture/oracle creation and manifest inspection count toward
the global budget, outside these sample timers. Values below are milliseconds;
paired changes and intervals are percentages, with negative values faster.

| Mode  | Baseline p50 | Candidate p50 | Round 1 median [interval] | Round 2 median [interval] | Maintained verdict      |
| ----- | ------------ | ------------- | ------------------------- | ------------------------- | ----------------------- |
| Off   | 239.43       | 237.21        | -1.03 [-2.86, +0.71]      | -0.51 [-2.19, +2.34]      | No confirmed regression |
| 100ms | 869.04       | 721.48        | -12.79 [-21.43, -9.99]    | -16.07 [-21.58, -10.93]   | Improved                |

The maintained `paired_round` order-statistic intervals have 97.85% coverage
for ten pairs. The regression gate requires both rounds' lower bounds above
+5%; the improvement rule requires both upper bounds below -5%. These are
intervals for median per-pair change, not ratios of aggregate medians. The
observed aggregate p50 changes are -0.93% off and -16.98% on; off does not
establish a speedup or equivalence.

All 88 finite jobs matched the 2,198,080-row Arrow oracle, ended naturally,
and had zero remaining tasks/task errors and successful owned cleanup. All
off runs wrote no checkpoint state or epochs. All 40 measured on runs retained
nonempty nonterminal state and a committed terminal manifest. Their terminal
epochs were six or seven; these short jobs do not supply the continuous
checkpoint acceptance below.

On-mode process-CPU p50 was 403.51 → 393.70 ms; wall p95 was
894.59 → 783.38 ms. Maximum worker-lifetime RSS was 616.79 → 647.83 MiB.
Retained managed-file median/max bytes were 115,493 / 2,575,938 baseline and
317,612.5 / 8,055,393 candidate. These descriptive resource values include
different checkpoint cut positions and fixture/warmup lifetimes; they are not
cumulative write counts or independent allocation measurements. Off wall p95
was 251.37 → 261.83 ms despite its slightly lower observed p50.

## Continuous checkpoint and restore evidence

The fifth case feeds the first 24 of 25 data batches per side at fixed absolute
250 ms intervals, independent of observed checkpoint counts. Its explicit
runtime uses 100 ms checkpoints and retains 128 epochs. The reused `EngineCase`
fixture metadata retains `checkpoint_interval_millis=null`; the actual runtime
override is recorded in the plan and `sustained.py::start_job`. Sealed raw
metadata has not been rewritten.

The first job produced actual durable, nonterminal, nonempty manifest documents
for **every epoch 1–58**. There were 25 observed advancing monotone cursor
tuples: 24 complete two-source batch frontiers plus one split-source
intermediate frontier at epochs 35/36. The count includes the first positive
frontier; it does not mean 25 complete paired batch admissions. Empty/repeated
cursor snapshots cannot substitute for source progress, and forced/restored
epochs do not contribute to the 58-epoch count.

After 196,608 rows per side and the exact prefix output/watermark frontier,
explicit cut **59** completed. The first job was cancelled and fully settled.
The latest actual manifest was epoch **59**, and the new native job independently
reported restored epoch **59**. Both opened cursors matched the selected
document: order `0000000000030000`, payload rows `196608`, event index `47`.
Replaying the remaining watermark and one data batch per side reached natural
EOF and terminal epoch **60**. Prefix plus restored suffix matched all
2,198,080 oracle rows with maximum absolute error zero.

The first job ended `cancelled/explicit_cancel`; the restored job ended
`completed/natural_end`. Both had zero remaining tasks/task errors, no metrics
overflow, no failed or indeterminate checkpoint publication, and completed
owned cleanup. All 60 retained manifest documents and 115 referenced segments
were preserved; segment paths, lengths and SHA-256 values were verified after
writers settled.

Observed startup/paced-prefix/explicit-cut/cancel/restore-to-ready/suffix-EOF
durations were **41.46 / 5,805.29 / 140.42 / 8.39 / 111.05 / 132.06 ms**.
The diagnostic through its full oracle took 6.54302 seconds and 1.74702 seconds
of process CPU. The whole worker took 6.75454 seconds. A single restore gives
one latency observation, not restore quantiles.

The 575 status observations covered 58 periodic checkpoint starts through
durable sink commit. Conservative sampled lower/upper-bound p50 values were
42.91 / 53.28 ms and p95 values 75.84 / 86.14 ms. Actual poll gaps ranged
10.11–49.81 ms; the largest status call took 0.153 ms. Bounds include these
intervals and microsecond truncation. `last_completed_epoch` marks durable
sink commit, not source-acknowledgement completion or exact native phase totals.

The settled managed root contained **4,715,416 bytes**: 4,218,392 unique
referenced segment bytes and 497,024 manifest bytes. These are retained
physical bytes, not total writes, retry/temp-file traffic or fsync counts.
RSS was 196.40 MiB before startup and 505.00 MiB at oracle completion, where
the recorded worker high-water mark was 579.98 MiB. The summary's `rss_after`
field is the earlier inspection boundary: 392.36 MiB, peak 393.19 MiB, before
the oracle. These values include fixtures and retained output; the final marks
also include oracle work. The paced diagnostic establishes its
correctness/lifecycle invariants; it does not establish throughput or compare
per-checkpoint performance.

## Coverage and limits

Maintained interval-Join checkpoint/recovery scenarios exercise the changed
operator snapshot/publication path; this focused continuous case adds direct
evidence across repeated publication and a real restore. A future maintained
continuous scenario belongs alongside `benchmarks/engine_stream.py` checkpoint
cases and the unified lifecycle contract. A lower-priority coverage gap is the
custom-backend fallback and multi-segment sink publication path, which these
local-backend measurements do not time.

Low-frequency/1-second checkpoints, Polars comparisons, absolute throughput
targets, ASOF, other row/batch sizes, cumulative write-volume/fsync reductions,
and the scheduled full lifecycle acceptance set remain **unmeasured**. No broad
matrix or scheduled-suite verdict is inferred from these five cases.

Raw evidence is under `target/issue406-segment-publication/`: the predeclared
plan, build/reuse provenance, eight paired worker reports, `sustained.json`,
retained state, phase/controller records, summary and hash seals. Independent
read-only review verified 206 evidence hashes and four final seals. Previous
`target/issue406-next/` artifacts remain unchanged.
