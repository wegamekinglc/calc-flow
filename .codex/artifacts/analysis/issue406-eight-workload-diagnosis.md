# Issue 406: complete eight-workload diagnosis

All eight workloads were attempted against pinned main `588726c8` and candidate
runtime `e8a28d99` (execution harness `427426b2`, native harness build `a2f31da0`).
Of 64 supported metric arms, 52 produced both rounds of ten samples and two
warmups; 12 metrics were blocked by eight physical workload failures. The 60
physical arms account for 1,040 measured calls, 104 successful warmups, and eight
failed warmups. No failed or completed attempt was repeated. This is coverage of
the complete inventory, not a passing acceptance result.

Values are p50 across 20 runs unless stated otherwise. Row 8 takes the median
of the 976 full 1,024-row callbacks within each run, then the median across
runs; the final 576-row callback is excluded. Main has no true OFF mode. Its LOW and ON
streaming observations marked † are processing diagnostics with failed EOF
checkpoint completion; they are not valid complete-lifecycle comparators. The
first table labels different policies explicitly; the second compares the same
policies. Historical issue figures are not reused as fresh measurements.


Row 7 is an instrumented trace/replay residual, not an ordinary release-wheel
measurement. ASOF end-to-end and projection floor use the same instrumented
feature wheel; matched native replay follows actual dispatch/state geometry.
H = median(ASOF settled end-to-end) − median(matched native replay), with no
paired residual confidence interval. The projection floor retains its existing, shorter sink timing
boundary. OFF observes H 363.28 ms against 2×floor 344.91 ms; candidate LOW
observes H 360.75 ms against 368.55 ms. ON end-to-end includes checkpoint work,
so its native residual is not an applicable harness-only metric.

| Workload                                 | Fresh main 588726c8                 | Candidate runtime e8a28d99        | Target / observation                    |
|------------------------------------------|-------------------------------------|-----------------------------------|-----------------------------------------|
| Join 1M, 1 / 32 CPU                      | LOW: 125.6 / 305.8 ms †             | OFF: 87.1 / 89.5 ms               | ≤60 / ≤15 ms; above                     |
| Join 10M, 1 / 32 CPU                     | LOW: 3355.7 / 5132.5 ms †           | OFF: 933.8 / 916.6 ms             | ≤600 / ≤150 ms; above                   |
| Retained Join 4M ÷ 1M                    | LOW: 4.711× †; ON failed            | OFF: 4.697×; ON failed            | ≤4.4×; above                            |
| Join capture / restore, per row          | Checkpoint preparation failed       | Checkpoint preparation failed     | ≤0.1 / ≤0.2 µs; capture/restore blocked |
| Join handler during compaction           | Fixture checkpoint failed           | Fixture checkpoint failed         | ≤2× steady; handler comparison blocked  |
| ASOF 1M per input, 1 / 32 CPU            | LOW: 292.7 / 319.2 ms †             | OFF: 300.1 / 320.3 ms             | ≤100 / ≤30 ms; above                    |
| ASOF harness overhead, 1,024-row batches | LOW: H 361.1 ms; 2×floor 386.0 ms † | OFF: H 363.3 ms; 2×floor 344.9 ms | H ≤2×projection floor; above            |
| Join full 1,024-row callback             | Enabled: 92.52 µs                   | OFF: 90.93 µs; enabled: 89.55 µs  | ≤50 µs; above                           |

| Case                           | Main LOW †, ms | Candidate LOW, ms | Main ON †, ms   | Candidate ON, ms | Candidate OFF, ms |
|--------------------------------|----------------|-------------------|-----------------|------------------|-------------------|
| row 1, 1,000,000, 1 CPU        | 125.58         | 89.10             | 210.13          | 94.53            | 87.07             |
| row 1, 1,000,000, 32 CPU       | 305.77         | 89.83             | 383.75          | 91.15            | 89.45             |
| row 2, 10,000,000, 1 CPU       | 3355.70        | 934.77            | 3611.02         | 1195.38          | 933.81            |
| row 2, 10,000,000, 32 CPU      | 5132.48        | 938.86            | 5425.91         | 1150.25          | 916.56            |
| row 3, 1,000,000, 1 CPU        | 416.91         | 369.40            | FAILED (warmup) | FAILED (warmup)  | 255.32            |
| row 3, 4,000,000, 1 CPU        | 1963.98        | 1790.99           | FAILED (warmup) | FAILED (warmup)  | 1199.20           |
| row 6, 1,000,000, 1 CPU        | 292.69         | 292.13            | 293.32          | 292.07           | 300.12            |
| row 6, 1,000,000, 32 CPU       | 319.16         | 317.45            | 319.41          | 321.14           | 320.30            |
| row 7, ASOF settled E2E, 1 CPU | 704.30         | 702.18            | 900.61          | 824.86           | 702.67            |

OFF disables checkpoints completely. LOW uses a 24-hour interval and still
performs terminal checkpoint completion. ON uses 100 ms. Relative throughput
goals are LOW ≥90% of OFF and ON ≥75% of OFF. Retained Join LOW observes 69.1%
at 1M and 67.0% at 4M, below its goal. Candidate Join 1M ON has no advancing processing
checkpoint in any accepted observation; its periodic impact is unmeasured.
Candidate Join 10M ON has mixed coverage (11/20 at 1 CPU, 12/20 at 32 CPU). Candidate ASOF ON has
20/20 at 1 CPU, 2/20 at 32 CPU; candidate ASOF with 1,024-row batches has 6/20
(main has 16/20, with the lifecycle limitation noted above). Do not
interpret configuration-only or mixed-coverage results as uniform periodic cost.
Native component “enabled” is not a periodic-frequency arm.

The capture/restore and compaction fixtures fail before their timed operation,
with `Internal { message: "V2 IPC constructor credit admission failed" }`.
Restore cannot be measured without a captured snapshot. This shared limitation
must not be reported as an equivalent or noise-level result. Source inspection
proves per-row constructor reservations exceed the fixed 1 GiB pool at 448,000
retained rows for this four-column fixture; this is not a universal row limit.
The measured runtime also has a separate wire-buffer growth problem; its source
correction below has not yet established 4M capacity.

The fixed 4,500-second measurement budget was satisfied at 3,161.762 seconds,
including earlier failed preparation, both continuation corrections, reviews,
startup, correctness work, cleanup, and evidence sealing. All owned processes
settled without forced kills. Five native/instrumented-wheel builds separately
cost 1,345.667 seconds; existing ordinary wheels were reused. WSL2 host isolation
is unverified. The frozen report conservatively marks the overall assessment
incomplete; resource guards did not identify an unsuitable host.

Evidence roots are `measurement/`, `measurement-remaining/`, and
`measurement-final/` under `target/issue406-complete-acceptance/` in the parent checkout. Seeded observation IDs remain relative
to their original root. Final evidence seal:
`cbaf8b5e6c512b3a52c8d03a6d132fb3396b9628aea08b811d2b3c4c71df9a86`.
The detailed report retains p95, median intervals, diagnostic paired intervals,
phase clocks, checkpoint bytes, actual dispatch replay links, and all failed
attempts. Its generic “unmeasured” labels for failed components should be read
with the explicit failure states above.

## CPU profile follow-up

Four diagnostic profiles reused the measured ordinary candidate wheel: sealed
Join 1M with 1 and 32 CPUs, retained Join 4M with 1 CPU, and ASOF 1M per input
with 1 CPU. Every case used one warmup and five measured processing windows at
4,000 Hz. The fixed budget was 180 seconds, including preparation, correctness
checks, symbol analysis and cleanup; execution and sealing took 27.216 seconds.
No new build was needed. All owned process groups settled without forced kills.

Profile clocks were aligned against the raw observations' monotonic processing
start/end times, excluding warmup, setup, correctness and terminal checkpoint
work. These diagnostic observations do not replace the acceptance timings.
Profile evidence lives under `target/issue406-acceptance-profile/measurement/`;
the frozen plan SHA-256 is
`e5e230eb65877aeaacb027af1e43069ce9ae5a044073e5f78b5d60a03cc264ce`.


The processing windows contained 1,841 / 1,938 / 23,509 / 5,900 CPU samples
respectively; every window has native-stack coverage. The largest measured
regions are:

| Case                 | Inclusive sampled CPU cost                                    |
|----------------------|---------------------------------------------------------------|
| Sealed Join, 1 CPU   | Key interning 25.15%; output materialization 20.91%           |
| Sealed Join, 32 CPUs | Key interning 23.94%; output materialization 19.66%           |
| Retained Join 4M     | BTreeMap insertion 22.93%; probe-key table assembly 17.18%    |
| ASOF, 1 CPU          | Input admission 69.19%; identities 30.07%; encode_rows 17.03% |

Inclusive regions overlap and cannot be summed. For sealed Join with 32 CPUs,
one Tokio worker accounts for 91.69–94.38% of Tokio samples within each window;
the worker identity changes across runs. Neither sealed Join profile has gather
worker samples. ASOF's one-CPU profile assigns 7.56% to gather workers. Unknown
self-frame names account for 10.21–16.32%, mainly libc; only 0 / 2 / 4 / 2 native
samples lack names. This supports prioritizing serial admission and key work,
not an off-CPU or memory-bandwidth diagnosis.

The retained Join probe assembly includes per-row RecordBatch slicing and
RowView destruction. Source inspection also found that probe keys retain their
whole batch's conservative construction reservation after transient vectors and
hash tables have been freed. Separating scratch and resident-key lifetimes is a
candidate improvement; a focused allocation/lifetime regression and a new
measurement must establish its benefit. CPU sampling alone does not prove why
the 1M-to-4M ratio exceeds the target.

The detailed profile report is
`target/issue406-acceptance-profile/profile-evidence.md` (SHA-256
`0ba14fa27fe9dd648f01bbdfe1bb85b1e05d96e2ff93dec53eb829c8e1aee576`).
Its raw-derived caller/TID evidence SHA-256 is
`5ed58960f9ccf80f9505b8eada0c56447abd42331a431a43db5bf09387b7bed6`.
All 12 raw profile, symbol sidecar and summary hashes match the collector seal.

## First corrective increment

Focused tests reproduced checkpoint admission failures before the production
change: dense contiguous rows exhausted generic constructor credit; sparse
mixed arrays exceeded retained credit; cancellation did not retire bounded
scratch owners; and the asynchronous prepare/capture path reproduced the
constructor error. The reviewed fix in `a9fe388c` coalesces contiguous selections
only when they share the exact array owner and consecutive offsets, and bounds
fallback concatenation by run count. It must retain cancellation, detached
output buffers, mixed-array ownership, deterministic equivalent recovery and prepaid
allocation. The separate wire-buffer growth correction is described below. This diagnosis
does not certify a later runtime revision or complete the development plan.

Constructor validation now passes all four new regressions, the existing
compaction guard, deterministic prepared-capture checks and 16 checkpoint
compaction tests. The mixed-array fixture needed 128 MiB: the first candidate
run demonstrated an unchanged generic admission requirement of 94,065,860
bytes, above its initially estimated 64 MiB. The dense and asynchronous
constructor regressions still use 16 MiB; production budgets are unchanged.
Library Clippy, scoped formatting and whitespace checks pass. Test-target Clippy remains failed
on the previously recorded Clippy 1.88 generated-libtest `large_stack_arrays`
diagnostic; it is not a passing check. The retained historical baseline log
contains the same dummy-span diagnostic. This does not establish CI success.

## Retained probe-key credit correction

Commit `9be7bd3f` preserves the full original construction admission threshold,
then separates transient batch descriptors from retained key ownership. It drops
the interner hash/table scratch before refunding surplus credit and retains
checked prepaid capacity for unique key buffers, Arc controls, batch IDs, key
vectors and the later serial probe-window slots. A key alias can outlive the
batch without pinning its conservative per-row construction reservation.

The new RED regression left enough headroom for two full construction charges
minus one byte. With one key alias retained after the first batch descriptors
dropped, the old code refused the second native probe. The fix passes, still
refuses initial admission at one byte below the original threshold, and retains
at most 4 KiB for the repeated-key owner in the 16,384-row fixture. Existing
allocation tracking checks that actual live bytes remain funded at every tested
ownership cut. Ten native lookup tests and five dictionary tests pass; library
Clippy, scoped format, whitespace and generated-contract checks pass. Test-target
Clippy retains only the known generated-libtest failure described above.

The source and focused evidence were independently reviewed. Validation and raw
RED/GREEN logs are under `target/issue406-native-key-credit/`. Performance after
this correction has not been measured; the tables above still describe runtime
`e8a28d99`. Avoiding the expensive fallback seen in the profile remains a
measurement hypothesis, not an asserted speedup.

## Exact checkpoint wire-buffer capacity

Commit `988a5972` computes checked sizes before allocating base and delta indices
and payload frames. Base indices use `32 + sum(72 + key_length)` bytes; delta
indices use `40 + sum_upserts(72 + key_length) + sum_tombstones(24 + key_length)`;
payload frames use `32 + 8 * row_count + IPC_length`. Each allocation is fallible
and prepaid. Underfill and overfill fail closed; cancellation remains observable.
Arrow IPC construction retains its existing geometric buffer policy.

The RED tests reproduced an 816-byte frame growing to 1,024 bytes and a 297-byte
delta growing to 512 bytes, plus tight-budget base growth and cancellation
admission failures. Nineteen focused tests now pass, including comparisons with
the previous geometric builder using identical prepared rows and IPC bytes.
Library Clippy, scoped formatting, contract and whitespace checks pass.
Test-target Clippy still has the recorded generated-libtest failure.

This correction does not refund encoder funding highwater or change global pool
limits. Existing state, held history and replacement snapshots can coexist, so
exact output sizes alone do not prove capacity at 4M retained rows. The planned
fixed diagnostic must observe simultaneous reservations through fresh/repeated
preparation, four dirty epochs, base replacement, serial full-state restores and
final alias release. Its source and restore pools must remain distinct, and
reservation peaks must not be presented as heap or RSS measurements. Validation
logs are under the feature worktree's
`target/issue406-complete-acceptance/wire-validation.json`. The eight-row timing
tables still describe `e8a28d99`; no speedup is attributed to this unmeasured fix.
