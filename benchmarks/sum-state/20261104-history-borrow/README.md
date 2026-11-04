# History-borrow candidate: focused verification

Collection status: **completed**; 120 formal observations.
The artifact manifest records the first-phase 471.105224 seconds plus this
phase's collection, analysis, publication and cleanup within 600 seconds.

The final source changes integer SUM history seeding to borrow the prior
immutable groups by stable slot, avoiding the per-candidate history copy.
The Rust harness, fixtures, affinity, compact state budget and two rounds of
ten alternating pairs are unchanged. These three stream cases directly
exercise the changed state path. **Finite control was not measured on this
final source.** Its earlier inconclusive result does not establish final
finite equivalence or absence of regression.

## Timings and exact maintained verdicts

| Case               | Baseline P50 ms | Candidate P50 ms | P50 change | Verdict      |
|--------------------|-----------------|------------------|------------|--------------|
| stock-stream-1cpu  | 364.328         | 290.357          | -20.30%    | improved     |
| stock-stream-8cpu  | 378.398         | 293.215          | -22.51%    | improved     |
| sparse-stream-1cpu | 37.855          | 35.931           | -5.08%     | inconclusive |

| Case               | Round | Paired median | Median interval    | Coverage  |
|--------------------|-------|---------------|--------------------|-----------|
| stock-stream-1cpu  | 1     | -19.99%       | [-22.36%, -16.17%] | 97.85156% |
| stock-stream-1cpu  | 2     | -20.41%       | [-23.20%, -19.35%] | 97.85156% |
| stock-stream-8cpu  | 1     | -20.06%       | [-32.62%, -18.87%] | 97.85156% |
| stock-stream-8cpu  | 2     | -21.63%       | [-29.31%, -13.73%] | 97.85156% |
| sparse-stream-1cpu | 1     | -4.46%        | [-11.19%, -3.69%]  | 97.85156% |
| sparse-stream-1cpu | 2     | -5.46%        | [-12.31%, +5.07%]  | 97.85156% |

## CPU and whole-process memory

| Case               | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|--------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu  | baseline  | 400.481 | 362.087    | 1.000     | 231.45           |
| stock-stream-1cpu  | candidate | 322.692 | 290.211    | 1.000     | 232.96           |
| stock-stream-8cpu  | baseline  | 410.459 | 424.941    | 1.135     | 245.52           |
| stock-stream-8cpu  | candidate | 317.885 | 342.161    | 1.166     | 248.88           |
| sparse-stream-1cpu | baseline  | 42.316  | 37.853     | 1.000     | 70.40            |
| sparse-stream-1cpu | candidate | 38.280  | 35.909     | 1.000     | 71.86            |

Verdicts use the existing paired-median confidence rule and 5% threshold.
P50 ratios and paired medians are distinct statistics. Inconclusive timing
does not establish improvement, regression or equivalence. No cases, pairs
or retries were added to obtain a favorable verdict. Peak RSS includes
IPC preload, retained outputs and validation; it is not operator-only memory.

## Profiles and remaining work

Accepted current candidate profiles: 0.
Profile budget disposition: predeclared remaining-time admission refused; timings prioritized.
The predeclared candidate-only capture requires at least 25 seconds remaining
after formal timings; otherwise it is skipped to preserve complete timings
and publication. A capture has six jobs, excludes the first, and verifies
every per-thread timestamp before selecting five exact processing windows.

The baseline profile is retained from the **earlier window cohort**, with the
same baseline binary, common Rust harness, fixture and CPU resources. It is
qualitative context rather than a newly paired elapsed-speedup comparison.
All accepted stacks and exact self/inclusive target counts are preserved.
Inclusive paths overlap; an absent or inlined symbol does not mean zero cost.
Percentages describe user-mode on-CPU samples, not wall-time attribution.
Off-CPU waits, hardware counters and memory bandwidth were not measured.
Composite-key rewrites, group sharding and finite filter/projection changes
remain unimplemented candidates; no performance benefit for them is claimed.

## Scope, budget and complete evidence

Dense input: 2M business rows (1.6M trades), plus 10k zero seed trades.
Sparse input initializes 10k groups and sends 20 batches of 100 rows:
80 trades and 20 non-trades per batch, touching 80 trade groups each batch
and 100 distinct groups across the fixture. Each stream emits 21 complete
10k-group snapshots. All retained snapshots are validated after the primary
timer; absolute times differ from the older final-only sink diagnostic.
Native affinity, thread settings and clean lifecycle are checked each time.
Eight logical CPUs correspond to four reported core IDs on this WSL2 host;
host load and power state are unobservable. Builds are timed separately.

- baseline: `199d70b4156591153eb6b15cc2c8ffc238b3348f`; binary SHA256 `65c13e041bc26f49492fde6dbda8860ac2e6d07866c1ff3040e786895bf1efda`; Release build 692.276s.
- candidate: `84186554b078e25e71003a74fedc91c63b2b7f70`; binary SHA256 `17259e26f471722696cbc9c1af67ebc901d1d3c2c6f67b49f78f77c655137ab2`; Release build 592.259s.

[summary.json](summary.json) retains every case, resource statistic, build
identity, fixture/oracle digest and environment/tool version/hash. Read all
formal rows directly in [samples.csv](samples.csv).
[complete-evidence.json.gz](complete-evidence.json.gz) includes all raw
warmup/formal/profile rows, commands, resource receipts, lifecycle metrics,
source seals and intermediate snapshot digests.
[Profile summary](profile-summary.json) and [targets](profile-targets.json)
retain
accepted current-profile and historical-context attribution. Artifact hashes
and cumulative timing are in the [manifest](evidence-manifest.json).
The [first complete four-case collection](../20261104/README.md) remains
unchanged, including its non-confirmed streaming results and imprecise finite
control. See [reproduction](../README.md#history-borrow-followup).
