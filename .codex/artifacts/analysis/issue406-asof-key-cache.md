# Issue 406: reuse ASOF right-key admission dictionary IDs

PR #407 merged as `acd10e0e`. The follow-up workflow fix PR #408 merged
as `89e51141` after all 37 CI checks passed;
its current complete benchmark is run `38162461866`, against merged main
`acd10e0e`. The earlier run `38158920926` was cancelled after dependency-stage
hangs; its cancellation is not a passing result. Runtime performance targets
remain unmet; that workflow fix changes no runtime source.

A new diagnostic profile reused measured product `67df5dac`, native SHA-256
`573bd1e185edad150a967ae2c457f3549624f15b19c8cd79f19a9de643c8ce8c`.
Its runtime sources match merged main; the only crate diff is an integration
test. Binary and execution-controller identities remain distinct.
Four preregistered OFF cases each ran one warmup and five processing windows
at 4,000 Hz. Total acquisition, correctness, analysis and sealing took
26.477 / 180 seconds; build cost was zero. All owned processes settled
without forced kills. WSL2 host isolation remains unverified.

- `row-1/join_sealed/1000000/batch-64000/off/cpu-1`: 1904 samples; native stacks 1815; windows [384, 378, 412, 365, 365].
- `row-1/join_sealed/1000000/batch-64000/off/cpu-32`: 2001 samples; native stacks 1852; windows [387, 402, 406, 393, 413].
- `row-3/join_retained/4000000/batch-64000/off/cpu-1`: 16734 samples; native stacks 16011; windows [3500, 3498, 3325, 3203, 3208].
- `row-6/asof/1000000/batch-64000/off/cpu-1`: 5209 samples; native stacks 5000; windows [1013, 1036, 1050, 1067, 1043].

Join admission append has 14.65% / 12.04% inclusive samples at 1 / 32 CPUs,
and probe-key preparation 17.44% / 16.79%. ASOF admission preparation has
44.62%, identity construction 20.79%, and installation 21.96%.
These inclusive regions overlap and must not be added. The profile supplies
no speedup estimate or acceptance verdict. Evidence and frozen plan are under
`target/issue406-postmerge-profile/`; all raw profiles and symbol sidecars are retained.

## Selected increment

ASOF already builds distinct string encodings and per-row dictionary IDs.
Right admission still probes `InputKeys` for every accepted row and expands
hashes into a per-row vector. Cache the resolved admission-key index and hash
per distinct dictionary entry, scoped to one fully accepted record batch.
Count every admitted row in right capacities even when its key lookup is reused.

Keep the existing path for mixed-late, non-string, high-cardinality input and
optional cache admission refusal. Reserve cache storage before allocation,
never retain a cache across independently encoded record batches, and preserve
identity/duplicate validation, row order, resident-key ownership and cancellation.
Default memory remains unbounded; explicit caller budgets remain authoritative.

The focused RED must observe 4,096 interner probes for 64 distinct keys where
the optimization requires only 64. Add controls for independently ordered batch
dictionaries, mixed late rows, explicit budget refusal and wide key ownership.
Use the existing identity/lifecycle regressions; do not relax validation to
reach a timing goal. Measure only affected ASOF paths and necessary controls
with fixed pairs after implementation. Full eight-row and OFF/LOW/ON acceptance
continues separately; no target is marked achieved by this source increment.

## Focused validation

The initial test failed with exactly 4,096 interner calls against the required
64. The implementation passes all 63 admission tests when run serially, plus
the canonical string dictionary hashing/sliced-row test. The new narrow and
wide-key checks each observe 64 calls with unchanged canonical identities and
right capacities; all tracked pool reservations retire after use.

The first local 32-thread admission run exhausted the existing bounded process
infrastructure and aborted. This was diagnosed before the serial rerun; no
resource bound or test was weakened. Final specialist source review approves
the implementation. Library Clippy, scoped formatting and generated-contract
checks pass. Windows and Linux CI then identified a potentially wrapping
`usize` to `i64` cast in the new dictionary-order test. The fixture now uses
checked conversion. Linux also reproduced the pinned-toolchain generated
libtest `large_stack_arrays` diagnostic: adding five tests increases the
2,048-entry reference array from 16,384 to 16,424 bytes. A minimal standalone
Clippy reproduction passes with 2,048 tests and fails with 2,053.

Move five existing checkpoint-OFF lifecycle tests into the existing
`checkpointing_mode_contract` integration target, keeping all test bodies and
21 assertions. Its fixture now constructs the same plan, sources and sinks
through the public runner constructor. Two private context/hash controls stay
unit tests. The all-feature unit harness returns to 2,048 entries; no lint or
toolchain requirement is relaxed. Test-inclusive core Clippy passes in
124 seconds. All six integration tests pass after a 176-second build
(0.02-second test execution). Independent migration review confirms unchanged
fixtures and equivalent public construction. Final CI remains outstanding.

## Focused paired measurement

The ordinary candidate release is product `59518ad6`, native SHA-256
`ed65a2ee1cf5fafeb9affe767481ba7f67e4b25561e61b15e1e2e3701edab43c`.
One audited release build took 364.689 seconds, separately from measurement.
Baseline reuses the original `67df5dac` product and native identity above;
its runtime and Python sources equal merged main `acd10e0e`. The preserved
source diff contains only a Rust integration-test correction. Both production
wheels retain matching compiler, Python, lock fingerprint and build environment.

Before acquisition, five cases, two rounds of ten adjacent AB/BA pairs,
one warmup per arm per round and a 600-second all-inclusive budget were frozen.
Each observation retains the maintained full-output oracle, checkpoint status,
cleanup, product and affinity validation. Runtime memory remains unbounded.
The processing clock excludes fixture construction, startup, terminal checkpoint,
correctness and cleanup; those costs all count toward the collection budget.

Four complete cases have 20 observations per arm. Acquisition stopped on the
fifth case's baseline warmup; no samples were retried or added. Total wall time,
including sealing and owned cleanup, was 100.775 seconds. No forced kill or
surviving owned process was recorded. The complete-set verdict is unavailable.

The small-batch control cannot run on the selected ordinary releases: the
maintained 1,024-row ASOF harness enables dispatch trace, and its native method
is compiled only with `benchmark-internals`. The ordinary baseline's method is
absent. This is a measurement selection error, not evidence of a runtime
regression. The existing instrumented row-7 report remains historical evidence;
there is no new small-batch or harness-overhead comparison in this increment.

| ASOF 1M per input, 64k batches | Main p50   | Candidate p50 | Pooled change | Round 1 paired median, CI95 | Round 2 paired median, CI95 | Maintained verdict      |
|--------------------------------|------------|---------------|---------------|-----------------------------|-----------------------------|-------------------------|
| OFF, 1 CPU                     | 250.711 ms | 243.454 ms    | -2.89%        | -6.48%, [-7.32%, -0.05%]    | -3.29%, [-6.74%, -1.85%]    | no-confirmed-regression |
| OFF, 32 CPU                    | 282.280 ms | 277.229 ms    | -1.79%        | -1.06%, [-4.17%, +2.15%]    | -0.88%, [-5.77%, +3.04%]    | no-confirmed-regression |
| LOW, 1 CPU                     | 254.065 ms | 244.516 ms    | -3.76%        | -2.58%, [-7.85%, +2.02%]    | -2.18%, [-5.61%, +4.33%]    | no-confirmed-regression |
| ON, 1 CPU                      | 261.510 ms | 256.953 ms    | -1.74%        | -2.19%, [-5.66%, +2.54%]    | -2.87%, [-7.43%, +26.15%]   | inconclusive            |

These timings establish no >5% improvement or confirmed >5% regression under
the maintained paired rule. OFF and LOW have no processing checkpoint activity.
ON uses a 100 ms interval and has processing checkpoint activity in every sample;
advancing committed state is observed in 20/20 main and 17/20 candidate samples.
The remaining candidate samples do not establish advancing periodic state cost.
Host isolation on WSL2 remains unverified. The cache eliminates repeated interner
probes, but this limited timing benefit does not explain the remaining ASOF gap.

The ASOF OFF targets remain 100 / 30 ms at 1 / 32 CPUs. Keeping the existing
LOW throughput target of at least 90% of OFF implies 111.11 / 33.33 ms at those
absolute goals; ON at least 75% implies 133.33 / 40 ms. Current single-thread
LOW and ON are above these absolute targets despite remaining close to current
OFF. No target is marked achieved.

Evidence, frozen plan, source-equivalence diff, raw observations and the
unchanged collection seal are under `target/issue406-asof-key-cache-performance/`.
The local report is a focused subset, not a replacement for the eight-row matrix.

## Full target inventory

All eight rows stay visible. Only row 6 has a new paired main/candidate comparison.
Other main values below are the earlier measured product `67df5dac`, whose runtime
matches merged main. They are not fresh measurements of this candidate. Pre-#407
main `588726c8` failed lifecycle or fixture validation on rows 1-7; its valid
enabled small-batch comparison is retained in PR #407.

| Workload                        | Merged-main runtime, prior run | This candidate                            | OFF / component target     |
|---------------------------------|--------------------------------|-------------------------------------------|----------------------------|
| Sealed Join 1M, 1 / 32 CPU      | 103.33 / 93.10 ms              | Not remeasured; Join code unchanged       | ≤60 / ≤15 ms               |
| Sealed Join 10M, 1 / 32 CPU     | 1094.44 / 962.35 ms            | Not remeasured; Join code unchanged       | ≤600 / ≤150 ms             |
| Retained Join 4M / 1M           | 4.467×                         | Not remeasured; Join code unchanged       | ≤4.4×                      |
| Capture / restore per row       | 0.186 / 0.437 µs               | Not remeasured; checkpoint code unchanged | ≤0.1 / ≤0.2 µs             |
| Handler during compaction       | 1.049× steady                  | Not remeasured; Join code unchanged       | ≤2× steady                 |
| ASOF 1M, 1 / 32 CPU             | New paired 250.71 / 282.28 ms  | New paired 243.45 / 277.23 ms             | ≤100 / ≤30 ms              |
| ASOF harness, 1,024-row batches | Prior OFF residual 488.66 ms   | Unmeasured; trace release required        | ≤2× floor, prior 424.39 ms |
| Join full 1,024-row callback    | OFF 91.89; enabled 89.83 µs    | Not remeasured; Join code unchanged       | ≤50 µs                     |

Earlier retained-state LOW/ON throughput misses its 90%/75% OFF goals. That
finding remains open. No historical value above is silently carried forward as
a new candidate measurement or a complete acceptance pass.
