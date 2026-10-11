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

| Workload                                   | Fresh main 588726c8                   | Candidate runtime e8a28d99          | Target / observation                      |
|--------------------------------------------|---------------------------------------|-------------------------------------|-------------------------------------------|
| Join 1M, 1 / 32 CPU                        | LOW: 125.6 / 305.8 ms †               | OFF: 87.1 / 89.5 ms                 | ≤60 / ≤15 ms; above                       |
| Join 10M, 1 / 32 CPU                       | LOW: 3355.7 / 5132.5 ms †             | OFF: 933.8 / 916.6 ms               | ≤600 / ≤150 ms; above                     |
| Retained Join 4M ÷ 1M                      | LOW: 4.711× †; ON failed              | OFF: 4.697×; ON failed              | ≤4.4×; above                              |
| Join capture / restore, per row            | Checkpoint preparation failed         | Checkpoint preparation failed       | ≤0.1 / ≤0.2 µs; capture/restore blocked   |
| Join handler during compaction             | Fixture checkpoint failed             | Fixture checkpoint failed           | ≤2× steady; handler comparison blocked    |
| ASOF 1M per input, 1 / 32 CPU              | LOW: 292.7 / 319.2 ms †               | OFF: 300.1 / 320.3 ms               | ≤100 / ≤30 ms; above                      |
| ASOF harness overhead, 1,024-row batches   | LOW: H 361.1 ms; 2×floor 386.0 ms †   | OFF: H 363.3 ms; 2×floor 344.9 ms   | H ≤2×projection floor; above              |
| Join full 1,024-row callback               | Enabled: 92.52 µs                     | OFF: 90.93 µs; enabled: 89.55 µs    | ≤50 µs; above                             |

| Case                             | Main LOW †, ms   | Candidate LOW, ms   | Main ON †, ms     | Candidate ON, ms   | Candidate OFF, ms   |
|----------------------------------|------------------|---------------------|-------------------|--------------------|---------------------|
| row 1, 1,000,000, 1 CPU          | 125.58           | 89.10               | 210.13            | 94.53              | 87.07               |
| row 1, 1,000,000, 32 CPU         | 305.77           | 89.83               | 383.75            | 91.15              | 89.45               |
| row 2, 10,000,000, 1 CPU         | 3355.70          | 934.77              | 3611.02           | 1195.38            | 933.81              |
| row 2, 10,000,000, 32 CPU        | 5132.48          | 938.86              | 5425.91           | 1150.25            | 916.56              |
| row 3, 1,000,000, 1 CPU          | 416.91           | 369.40              | FAILED (warmup)   | FAILED (warmup)    | 255.32              |
| row 3, 4,000,000, 1 CPU          | 1963.98          | 1790.99             | FAILED (warmup)   | FAILED (warmup)    | 1199.20             |
| row 6, 1,000,000, 1 CPU          | 292.69           | 292.13              | 293.32            | 292.07             | 300.12              |
| row 6, 1,000,000, 32 CPU         | 319.16           | 317.45              | 319.41            | 321.14             | 320.30              |
| row 7, ASOF settled E2E, 1 CPU   | 704.30           | 702.18              | 900.61            | 824.86             | 702.67              |

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

| Case                   | Inclusive sampled CPU cost                                      |
|------------------------|-----------------------------------------------------------------|
| Sealed Join, 1 CPU     | Key interning 25.15%; output materialization 20.91%             |
| Sealed Join, 32 CPUs   | Key interning 23.94%; output materialization 19.66%             |
| Retained Join 4M       | BTreeMap insertion 22.93%; probe-key table assembly 17.18%      |
| ASOF, 1 CPU            | Input admission 69.19%; identities 30.07%; encode_rows 17.03%   |

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


## Actual 4M checkpoint follow-up: stopped before timing

The next frozen candidate is `9063f2fe8f870edb509ef38be2b9734a1392947c`,
including all three corrections and the reviewed fixed-history diagnostic.
Its ordinary wheel and feature-only native binary passed source, dependency,
compiler/profile, symbol and package-content audits. Build/audit time was
614.567 seconds, separate from measurement. A source-inventory checker initially
misclassified the unchanged `fuzz/Cargo.lock`; the failed audit, original checker
and cache replay were preserved. Correcting only the root-lockfile exclusion
allowed audit recovery without rebuilding the wheel or repeating Cargo replay.

The fixed 900-second follow-up attempted its first correctness invocation once.
It stopped at 8.065 seconds, including startup allowance, checks, settlement and
sealing. All 4,000,000 left rows and 64 dimension rows were admitted. The actual
bounded reservation observer reported these shared-pool totals:

| Observation                                  | Bytes           |
|----------------------------------------------|-----------------|
| Unchanged DataFusion pool limit              | 1,073,741,824   |
| Live reservation after complete admission    | 100,990,496     |
| Peak reservation during admission            | 327,112,480     |
| Live / peak at failed fresh preparation      | 997,664,473     |
| Rejected additional write reservation        | 134,217,216     |
| Required total if that request were granted  | 1,131,881,689   |
| Final reservation after owned cleanup        | 0               |

The failure is `V2 checkpoint IPC payload encoding failed`, in `fresh_prepare`;
the rejected consumer is `sql-incremental:stream-join-v2-write`. This is an
observed reservation-capacity failure, not an RSS measurement or proof of a
memory leak. The raw source must determine which simultaneously live buffers
and retained credit account for that request before another correction.

No retained-Join timing, capture/restore sample or compaction-handler sample
ran. Repeated preparation, dirty history, replacement snapshots and restores
remain unvalidated at 4M. Every owned process settled without forced kills, and
the observed pool returned to zero. Do not rerun this unchanged failure or infer
a speedup from the smaller correctness tests. All eight timing rows above still
refer to `e8a28d99`; current-source acceptance remains incomplete.

Evidence is under the parent checkout's
`target/issue406-checkpoint-key-followup/measurement/`. The execution-plan SHA-256
is `9b6938635595557ed559ee54f4a71ce8db9e59b445c296feec83be588e9ffcd4`;
the raw evidence seal SHA-256 is
`76b537af8823df01bf9a3dc25a87a9cd2b89531143b445d6769a8cb7b56545bd`.
`completion.json` binds that seal, the stopped result and its budget record.

## IPC owner lifetime and remaining capacity requirements

A focused 8,192-row, four-column regression reproduced the same IPC encoding
failure with an unchanged 2,400,000-byte pool. Commit `8a740b1e` separates Arrow
message generation from output copying: schema, dictionary and record messages
own their encoded byte vectors; their generator tracker and compression context
are destroyed, followed by the compacted Arrow batch, before output-buffer growth.
The batch's independent reservation owner can therefore retire at its actual
last use. PaidBuffer prepayment, encoder scratch/high-water, memory limits and
v2 index/payload formats remain unchanged.

The tight-pool test now succeeds. Complete bytes match the previous StreamWriter
for flat, dictionary, nested and null data. Phase-boundary cancellation, alias
lifetime, allocation accounting and prepared checkpoint controls are included in
22 passing focused tests. Library Clippy and scoped formatting pass; test-target
Clippy retains only the documented generated-libtest failure. This is a
correctness/resource result, not a new
performance sample or a completed 4M checkpoint.

Source arithmetic establishes why another whole-lifecycle run cannot pass on
this correction alone. For the fixed 4M left schema, each independent serialized
base requires at least 340,000,032 index bytes plus 160,000,036 payload bytes.
Keeping two copies plus the observed 100,990,496-byte admitted-state reservation
requires at least 1,100,990,632 bytes, exceeding 1 GiB by 27,248,808 bytes before
right-side data, controls or scratch. The current writer keeps an earlier
prepared base until replacement completes, and a held E4 snapshot must remain
valid throughout rebase. This lower bound is source-derived; the actual sealed
9063 run stopped earlier, at fresh preparation.

Do not spend another release build on the expectation that the phase split or
constructor chunking alone removes this lower bound. Next designs must reuse an
unchanged prepared cut and share immutable checkpoint payloads safely during
eligible history rebase, with reservation ownership matching the bytes retained.
The restore fold needs a separate bounded-memory design before claiming the
complete 4M lifecycle. Its existing reservation for two locator trees and one
payload-row set alone exceeds 6,576,106,860 bytes at 4,000,064 upserts under the
pinned whole-node-per-entry budget. This is another source-derived admission
lower bound, not an observed restore failure in the stopped 9063 run. Expiration
container budgeting must also be included in the complete restore bound. Every unsupported history shape must preserve existing
correctness through the fallback; wire-format or validation changes require
separate review. Final acceptance still includes all eight workload rows and
checkpoint modes.

Validation logs and hashes are recorded in the feature worktree
`target/issue406-complete-acceptance/ipc-owner-validation.json`; the exact
source accounting worksheet is `ipc-owner-source-diagnosis.json` beside it.

## Reuse an unchanged fresh prepared cut

Commit `f9adf9e2` avoids submitting another checkpoint writer when the complete
fresh preparation Cut still matches: generation, captured epoch, revision and
dirty revision. The existing cancellation, retirement and advancing-epoch
checks remain before reuse. This applies to fresh prepared bases; it does not
extend to prepared deltas or replace dirty history.

The recorded RED prepared a real state, reserved the remaining 1 GiB pool and
repeated the unchanged preparation. The old writer failed constructor admission;
the new path succeeds without allocating a second base. Three new tests check
identical owners and checkpoint bytes, complete restored state, each stale Cut
field independently, and cancellation while waiting for retirement. Seven
existing writer regressions also pass. Library Clippy, scoped formatting and
contract checks pass; test-target Clippy retains the known generated-libtest
diagnostic. Evidence is in the feature worktree's
`target/issue406-complete-acceptance/prepared-reuse-validation.json`.

These ten tests establish reuse semantics and resource ownership, not a new
performance result. The eight-row comparison still measures `e8a28d99`. Held E4
history during a replacement base and bounded restore remain separate capacity
requirements; the next full release build and 4M diagnostic remain held until
their simultaneous allocation bounds are resolved.
