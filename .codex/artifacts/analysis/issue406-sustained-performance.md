# Issue 406: historical sustained measurements and staging diagnosis

This report preserves completed measurements from the earlier sustained-workload
task. The later seven-workload final protocol was **never executed** and is
superseded by the complete eight-row acceptance task described in
`../specs/issue406-performance-acceptance.md`. Historical measurements below
remain evidence for their recorded revisions and scopes, not fresh results for
the current candidate.

The initial valid runtime (`6d079f82`) has an observed prepared-input processing
ratio of **1.159× Polars 1T**, within the declared 1.25× target. Its single
saturated checkpoint observations reach **74.90% of off at 1 s** and **42.12% at
100 ms**, below the respective 90% and 75% targets. These are descriptive WSL2
observations; they do not establish replicated sustained performance verdicts.

A separate 100 ms diagnostic found four staging-owner directory fsync calls per
four-segment snapshot. The selected increment batches those confirmations at one
owner snapshot boundary while preserving file synchronization and the other
durability boundaries. This report contains no measurement of that increment.

## Revisions, scope and fixed inventory

- Exact main: `588726c8bce0f1c57b90077064834c4a472ee5f2`.
- Initial valid runtime:
  `6d079f82c16a091b91e2996ecda8fd8c7a9c38c0`.
- Initial checkout: `b9a12255`; production equivalence to `6d079f82` was audited.
- The superseded final protocol did not measure a candidate.

The maintained interval join has 64 keys, inclusive ±5-second bounds and batches
of 8,192 rows. A 200,000-row-per-source block produces exactly 2,198,080 rows.
Every observation validates both row identities and computed values against the
full deterministic oracle. Polars' eleven shifted equality joins provide exact
output parity on this integer-time grid; they do not provide the native stream
and checkpoint lifecycle.

The initial inventory was two finite processing arms, one long job for each of
off/1 s/100 ms, and the identical off job under a CPU profiler. The later
checkpoint diagnostic repeated exactly the 100 ms long job with CPU sampling and
fsync instrumentation. The profiling observations are excluded from throughput
comparisons. Neither stage extended its fixed inventory or retried a timing run.

## Comparable finite processing

Both finite timers start from prepared immutable input and stop after complete
Arrow output. Native planning and Polars lazy-plan construction are outside the
timer; full exact oracles are outside both timers. Native EOF/terminal and owned
cleanup have separate clocks and strict validation. Two rounds of ten alternating
pairs completed, plus one excluded warmup per arm and round.

| Arm                | Processing p50 | Processing p95 | Process CPU p50 |
|--------------------|----------------|----------------|-----------------|
| Initial native off | 245.20 ms      | 254.08 ms      | 314.22 ms       |
| Polars 1T          | 211.59 ms      | 225.77 ms      | 210.07 ms       |

The p50 ratio is 1.15886. Paired descriptive median deltas are +15.89% and
+18.27%, with confidence intervals [+9.92%, +29.62%] and [+9.74%, +27.49%]. The
observed ratio meets the target; these intervals do not establish a confidence
bound below 1.25×. Native terminal and cleanup p50 are 1.046 and 0.184 ms.
The superseded protocol proposed a full-cycle off control with a different
timing scope; it was never executed and cannot refresh this processing/Polars
ratio.

## Initial saturated jobs

Each continuously fed job contains 80 disjoint 200,000-row blocks per source,
16,000,000 rows per source and exactly 175,846,400 output rows. IDs and timestamps
advance between blocks; prices retain their local template. There are no feeder
sleeps or per-block output barriers. After the fixed four-block warmup prefix,
the actual steady output delta is 167,054,080 rows. The bounded bitmap oracle
checks every row without retaining all output tables. Retention is explicitly
**32 epochs**, compared with the runtime default of 2.

| Mode   | Steady wall | Output rows/s | Fraction of off | Validator wall | Validator thread CPU | Peak RSS |
|--------|-------------|---------------|-----------------|----------------|----------------------|----------|
| Off    | 22.603 s    | 7.391 million | 100.00%         | 9.357 s        | 9.254 s              | 680 MiB  |
| 1 s    | 30.177 s    | 5.536 million | 74.90%          | 9.556 s        | 9.441 s              | 700 MiB  |
| 100 ms | 53.660 s    | 3.113 million | 42.12%          | 9.774 s        | 9.674 s              | 740 MiB  |

Synchronous validation occupies 41.4%, 31.7% and 18.2% of the respective wall
intervals and remains included in throughput. It cannot be subtracted to infer
engine throughput. Source push waits mostly express bounded backpressure, not
an isolated source cost. RSS includes fixture creation and validation storage.

Both enabled jobs prove 30 consecutive advancing, nonempty checkpoints whose
conservatively observed start through completed retirement lies inside steady
time: epochs 2–31 for 1 s and 477–506 for 100 ms. Terminal and boundary-crossing
epochs are excluded. Retained manifests, referenced segment lengths/checksums and
source cursors passed verification. Off produced no epoch or state file.

| Mode   | Retirement p50 bounds | Retirement p95 bounds | Retained bytes | Terminal wall | Cleanup wall |
|--------|-----------------------|-----------------------|----------------|---------------|--------------|
| 1 s    | 163.03–174.11 ms      | 180.00–190.98 ms      | 3,398,495      | 79.15 ms      | 0.235 ms     |
| 100 ms | 105.74–116.64 ms      | 129.51–140.56 ms      | 35,358,820     | 152.17 ms     | 0.240 ms     |

Bounds come from 10 ms status observations; they overlap processing. One long
job per mode is not the maintained 20-round lifecycle/recovery acceptance set.
Recovery and replicated sustained performance acceptance remain unclaimed.

## CPU and fsync diagnosis

The separate off profile has 31,775 samples in the declared steady interval at
1,000 Hz, with zero unresolved native leaf samples. Tokio threads contribute
63.83% and the Python thread 36.17%. Eviction is 12.23% inclusive/6.85% self,
BTreeMap insertion 4.82% self, Arrow byte gathering 4.67% self, and output
materialization 11.18% inclusive. The fixed 10 ms trim leaves eviction at 12.23%.
NumPy remainder contributes 8.31% self. Inclusive shares overlap; the absence of
Python frames prevents attribution to individual Python callbacks.

The checkpoint diagnostic repeated the fixed 100 ms job. All 175,846,400 output
rows and owned lifecycle checks passed, with 30 qualifying epochs, 453–482.
Its instrumented steady interval is 56.698 s and is excluded from ratio tables.
CPU sampling contains 42,992 steady samples with zero unresolved native leaf
samples: manifest validation 7.12% inclusive, canonical manifest checksum 5.59%,
strict decode 5.38%, retention 5.22%, and fsync CPU 0.14%. The duplicated
post-decode validation pass is measurable at 3.71%; every such sample has
retention ancestry. It was not selected for this increment.

A narrowly scoped interposer records actual fsync/fdatasync return values,
monotonic wall intervals, thread CPU and path, with bounded loss detection and
preserved errno. Its compile/smoke controls verified successful and failed
forwarding, multithreaded records and fail-closed initialization. All 6,929 records
are complete; 6,506 lie wholly inside steady time, with no boundary crossing,
dropped or nested call. They are all successful fsync calls under the owned
state directory. Their intervals do not overlap: sum and union are both
32.344614 s; thread CPU totals 0.214271 s. Call wall p50/p95 are 4.807/6.927 ms.
These are observed syscall durations, not kernel I/O attribution, whole-job
blocked time, or a duration that can be subtracted from the job.

| Observed path class         | Calls | Summed wall | Distinct paths | Relevant shape                             |
|-----------------------------|-------|-------------|----------------|--------------------------------------------|
| Staged file                 | 1,808 | 8.860 s     | 1,808          | Four new snapshot files per epoch          |
| Staging owner directory     | 1,808 | 8.436 s     | 452            | Exactly four calls per directory           |
| Staging epoch directory     | 452   | 2.184 s     | 452            | One per epoch                              |
| Staging lineage directory   | 452   | 3.277 s     | 1              | Directory-creation boundary                |
| Committed owner directory   | 542   | 2.558 s     | 1              | Publication/retention path; phases overlap |
| Committed lineage directory | 90    | 0.384 s     | 1              | All observed phase labels ambiguous        |
| Manifest file               | 452   | 2.320 s     | 452            | One per epoch                              |
| Manifest directory          | 902   | 4.325 s     | 1              | Publication/retention path; phases overlap |

Phase labels use surrounding status brackets. A transition or disagreeing
bracket remains ambiguous. The trace establishes the repeated owner-directory
calls; source review and durability tests determine which boundary can safely
be batched. The selected change must retain every file sync and separate
ancestor, committed-directory, manifest and retention boundary. No removable
wall-time saving is inferred from the table.

## Required pinned-main comparison

This historical finite table uses a common ready-through-EOF, complete Arrow
concat and owned-cleanup timer. It is preserved from the prior evidence and
labels the measured current runtime as `6d079f82`. It is not the new finite
processing timer and does not yet describe the final candidate.

| Mode   | Main p50 / p95       | Historical 6d p50 / p95 | Main lifecycle                         | Historical 6d lifecycle |
|--------|----------------------|-------------------------|----------------------------------------|-------------------------|
| Off    | N/A                  | 257.80 / 303.93 ms      | True checkpoint-off API unsupported    | 20/20 valid             |
| 1 s    | 524.40 / 592.05 ms   | 390.98 / 443.50 ms      | 20/20 invalid: internal EOF checkpoint | 20/20 valid             |
| 100 ms | 950.55 / 1,095.20 ms | 721.93 / 784.88 ms      | 20/20 invalid: internal EOF checkpoint | 20/20 valid             |

Main remains the exact unpatched `588726c8`; its timing is diagnostic and cannot
supply a valid lifecycle performance gate. The superseded protocol proposed
fresh main/current 100 ms and current off arms, plus a valid 6d/current 100 ms
control; none was executed. Both enabled rows above remain historical.

## Superseded final protocol: never executed

The withdrawn protocol specified seven workloads: four finite arms and three
long jobs. Each finite arm would have had two rounds of ten samples plus one
warmup per round. The proposed forward order was
main 100 ms → candidate 100 ms → valid 6d 100 ms → candidate off, reversed at
alternate indices. Candidate 100 ms samples would have been shared between the
two adjacent comparisons by round/index. The compatible valid incremental pair
would have used the maintained paired-median confidence rule; no standalone minimum-ratio
substitution or extra all-intervals-below-threshold requirement applies.

The proposed long jobs were off, 1 s, and 100 ms with the fixed 80-block worker
and oracle. There are no final-stage runtime results, paired intervals, or
resource summaries. The complete eight-row acceptance task replaces this
inventory and budget; this report makes no promise to run the withdrawn stage.

## Environment, accounting and evidence

Measurements ran on WSL2, Intel i9-13900HX, 32 logical CPUs and Tokio workers,
with one OpenMP/BLAS/Polars thread. Known competing builds/workloads were absent
at process preflight; Windows host scheduling and power remain uncontrolled.
Fingerprint compatibility is required within each classified pair. Modes and
backends have different workload identities and supply descriptive ratios only.

The initial attempt consumed 165.374710 s. The diagnostic, including its analysis,
sealing and cleanup, consumed 66.767839 s. The bounded offline path classification
consumed 0.727044 s. Total charged so far: **232.869592 / 600 s**, with all owned
processes settled. The withdrawn final stage had a 240 s cap: workload deadline
220 s, analysis 230 s, and settlement/sealing 240 s. It consumed no measurement
time. The separate 29.272956 s settlement reserve also remained unused. Build
cost is separate. These historical allowances do not authorize measurements
under the replacement task.

Evidence directories in the original checkout:

- `target/issue406-sustained-performance/`: initial raw reports, exact phase
  clocks, profiler capture/symbols, strict oracle/manifest evidence and summary.
- `target/issue406-checkpoint-diagnostic/`: CPU profile, bounded raw syscall
  trace, path/phase classification and charged analysis budget.
- `target/issue406-final-comparison/`: superseded preparation only; no final
  measurements were executed.
- `target/issue406-main-comparison/`: separately sealed historical main table.

Initial preparation/evidence seals are
`1581824fe2b547962872029aeed59cf891ac51e2ef29d80cea360e5ba8ce0ef4` and
`9b11ab1ff71d570711c3849a0cfc7f1c293d8c6e6591709d6ac4a107fb0fffdf`.
Diagnostic preparation/evidence seals are
`274e761cd25ce4d3699a7d5991b569240e945c70e8b0380255f02c47f3e6c707` and
`d2f39158fc1baa46363712a890f0ca6dd66989ded7588885c4023a83bcbdae53`.
The path-analysis seal covers all six derived/input artifacts; independent
review recomputed every seal and confirmed the 452×4 owner-directory pattern.

## Coverage advisory

The maintained interval-join stream and lifecycle cases exercise checkpoint
snapshot staging. The new bounded sustained helper adds full mathematical
output validation, continuously advancing sources and completed-retirement
coverage. Its focused tests cover duplicates/missing output, input isolation,
clock bounds, missed epochs and retirement; driver controls cover unpaced feed,
collector failure and cancellation settlement. Focused lint and format checks
passed. The workflow includes the sustained helper and checkpoint-cycle tests.

For future prioritization, several owner-specific snapshots, or a backend batch
spanning multiple staging directories, would characterize amortization beyond
this fixture's four segments. Recovery
and cross-platform durability remain correctness/CI responsibilities; these
measurements do not substitute for either acceptance contract.
