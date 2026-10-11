# Issue 406: complete eight-workload diagnosis

Latest runtime evidence: the [fresh a681e7da attempt](issue406-fresh-acceptance.md)
stopped on the candidate's 4M retained-ON warmup at epoch 11. It contains all
eight target rows, fresh pinned-main/current mode comparisons, and complete
cff/current retained-OFF controls. Rows 4–8 remain unmeasured in that attempt;
earlier measurements below keep their original revisions. OFF scaling observes
4.011×, while sealed Join 1M/10M still miss their absolute targets. The candidate
failure and all final acceptance verdicts remain unresolved. No retry or fill-in
was performed, and all owned processes settled.

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

| Workload                                     | Fresh main 588726c8                     | Candidate runtime e8a28d99            | Target / observation                        |
|----------------------------------------------|-----------------------------------------|---------------------------------------|---------------------------------------------|
| Join 1M, 1 / 32 CPU                          | LOW: 125.6 / 305.8 ms †                 | OFF: 87.1 / 89.5 ms                   | ≤60 / ≤15 ms; above                         |
| Join 10M, 1 / 32 CPU                         | LOW: 3355.7 / 5132.5 ms †               | OFF: 933.8 / 916.6 ms                 | ≤600 / ≤150 ms; above                       |
| Retained Join 4M ÷ 1M                        | LOW: 4.711× †; ON failed                | OFF: 4.697×; ON failed                | ≤4.4×; above                                |
| Join capture / restore, per row              | Checkpoint preparation failed           | Checkpoint preparation failed         | ≤0.1 / ≤0.2 µs; capture/restore blocked     |
| Join handler during compaction               | Fixture checkpoint failed               | Fixture checkpoint failed             | ≤2× steady; handler comparison blocked      |
| ASOF 1M per input, 1 / 32 CPU                | LOW: 292.7 / 319.2 ms †                 | OFF: 300.1 / 320.3 ms                 | ≤100 / ≤30 ms; above                        |
| ASOF harness overhead, 1,024-row batches     | LOW: H 361.1 ms; 2×floor 386.0 ms †     | OFF: H 363.3 ms; 2×floor 344.9 ms     | H ≤2×projection floor; above                |
| Join full 1,024-row callback                 | Enabled: 92.52 µs                       | OFF: 90.93 µs; enabled: 89.55 µs      | ≤50 µs; above                               |

| Case                               | Main LOW †, ms     | Candidate LOW, ms     | Main ON †, ms       | Candidate ON, ms     | Candidate OFF, ms     |
|------------------------------------|--------------------|-----------------------|---------------------|----------------------|-----------------------|
| row 1, 1,000,000, 1 CPU            | 125.58             | 89.10                 | 210.13              | 94.53                | 87.07                 |
| row 1, 1,000,000, 32 CPU           | 305.77             | 89.83                 | 383.75              | 91.15                | 89.45                 |
| row 2, 10,000,000, 1 CPU           | 3355.70            | 934.77                | 3611.02             | 1195.38              | 933.81                |
| row 2, 10,000,000, 32 CPU          | 5132.48            | 938.86                | 5425.91             | 1150.25              | 916.56                |
| row 3, 1,000,000, 1 CPU            | 416.91             | 369.40                | FAILED (warmup)     | FAILED (warmup)      | 255.32                |
| row 3, 4,000,000, 1 CPU            | 1963.98            | 1790.99               | FAILED (warmup)     | FAILED (warmup)      | 1199.20               |
| row 6, 1,000,000, 1 CPU            | 292.69             | 292.13                | 293.32              | 292.07               | 300.12                |
| row 6, 1,000,000, 32 CPU           | 319.16             | 317.45                | 319.41              | 321.14               | 320.30                |
| row 7, ASOF settled E2E, 1 CPU     | 704.30             | 702.18                | 900.61              | 824.86               | 702.67                |

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

| Case                     | Inclusive sampled CPU cost                                        |
|--------------------------|-------------------------------------------------------------------|
| Sealed Join, 1 CPU       | Key interning 25.15%; output materialization 20.91%               |
| Sealed Join, 32 CPUs     | Key interning 23.94%; output materialization 19.66%               |
| Retained Join 4M         | BTreeMap insertion 22.93%; probe-key table assembly 17.18%        |
| ASOF, 1 CPU              | Input admission 69.19%; identities 30.07%; encode_rows 17.03%     |

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

| Observation                                    | Bytes             |
|------------------------------------------------|-------------------|
| Unchanged DataFusion pool limit                | 1,073,741,824     |
| Live reservation after complete admission      | 100,990,496       |
| Peak reservation during admission              | 327,112,480       |
| Live / peak at failed fresh preparation        | 997,664,473       |
| Rejected additional write reservation          | 134,217,216       |
| Required total if that request were granted    | 1,131,881,689     |
| Final reservation after owned cleanup          | 0                 |

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

## Separate segment backing from encoder workspace

Commit `640d8c84` attaches prepaid capacity and ownership controls to each encoded
segment. Base inventory controls remain separate, and encoder completion refunds
only temporary funding after its buffers retire. Retargeted indexes receive new
funding while old aliases stay valid. Restored owned segments preserve opaque
source owners and fund additional wrapper controls; unowned segments receive
independent backing funding. Capture wrappers also retain their own controls, so
a held snapshot segment need not represent only its byte length.

Four focused RED cases exposed sibling backing remaining pinned, row-dependent
scratch surviving encoder completion, retarget retaining unrelated Base controls,
and unowned restored aliases failing independent retirement. They now pass;
34 unique related tests and core library Clippy pass. Source/evidence hashes are
in the feature worktree's
`target/issue406-complete-acceptance/segment-credit-validation.json`.

At the worksheet revision, source analysis identified a separate E1 blocker:
the synchronous pending path reserved 960,014,640 bytes for copied operations and 863,999,352
bytes for materialized left rows before payload encoding. Their 1,824,013,992-byte
sum already exceeded 1 GiB. The later borrowed-pending increment below resolves
this copy term; shared-payload rebase and bounded restore remain required. The proposed rebase and restore leading
terms (941,016,759 and 928,234,338 bytes) exclude unresolved owners and overlap;
these are conditional calculations, not measured admission or sufficient bounds.
Funding a new append-grown expiration representation in ordinary source state
would also exceed fresh/rebase capacity and requires an explicit combined remedy.
The full 4M release/measurement hold remains.

The read-only worksheet is in the parent checkout at
`target/issue406-ipc-lifetime-followup/capacity-upper-bounds.md`, SHA-256
`66ef5fb963aad21d9c9ae4da1f66a44cb8810057a6f4bd06c2e2f1855df3c0eb`.
No new performance claim follows from these source bounds or focused tests.

## Focused CI corrections

Commit `25b6eb78` decomposes the acceptance harness after its observed complexity
ratchet failure, preserving validator order, progressive failure evidence and
native-overlay recipe boundaries. The recipe identity changes with the source;
old sealed measurements and attestations remain immutable. All 102 unique
focused tests pass, as do Ruff and the unchanged complexity ratchet.

Commit `fe40a96c` corrects the five observed Rust CI fixture failures: ordinary
sinks for the disabled-checkpoint benchmark, checkpoint-aware inner state for a
manual-snapshot soak, readiness-based graceful shutdown, and three canonical
publication-path hooks. The two soak failures were reproduced locally before the
fix, and all five cases pass afterward. Windows is not yet validated. Scoped test
Clippy still fails solely on the known generated-libtest `large_stack_arrays`
diagnostic; this unresolved check blocks merge. No production persistence logic,
coverage threshold or lint requirement is weakened.

## Intern repeated string keys before canonical framing

Commit `0d2b23c7` specializes the local batch dictionary for one non-null Utf8 or
LargeUtf8 key. It compares borrowed values with their type tag, including forced
hash collisions, and constructs canonical key bytes only on a local miss. Global
dictionary hashing, canonical bytes, exact admission arithmetic and resident
ownership remain unchanged; composite/null/non-fitting inputs retain their
existing path. No Arrow input owner escapes through a borrowed lookup key.

Both runtime work-count REDs now pass: 14 rows with six distinct values construct
six frames, and the mixed-tag collision fixture constructs 12 for 28 rows. The
owned-key encoding count was already D; the removed work is N transient frames
and D copies, replaced by D direct constructions. All nine focused tests and
library Clippy pass. Evidence is in the feature worktree's
`target/issue406-complete-acceptance/string-interner-validation.json`.
This establishes less framing work, not a measured wall-time improvement.

## Borrow pending rows during checkpoint encoding

Commit `0dbbfea5` passes the immutable PendingLog directly to the synchronous
encoder. For each side it builds exact, prepaid selections of borrowed upserts
and tombstones, preserving row-ID ordering and every payload/key/time/charge
field. Generic read-only row accessors feed the existing compact, IPC, payload
frame and index writers. Actual selection vectors retire before their child
reservation; one side's scratch does not remain charged during the next side.

Three runtime REDs are resolved: unprepared 8,192-left/64-right encoding now fits
its unchanged 2,400,000-byte pool; cancellation no longer leaves temporary credit
in a still-live caller workspace; and a cancelled nonempty log takes precedence
over a full-pool encoder admission. Independent geometric-index and StreamWriter
payload oracles cover rich sliced Legacy/Rowed inputs, empty/one-sided/tombstone
histories and removed/reused pending slots. Exact and one-byte-short selection
admission, actual allocation peaks, held segment aliases, every encoder callback
cancellation point and full capture rejection/retry are covered.

All 30 unique focused tests pass: 28 encoder/compaction and existing Base-drain
checks, one full-capture fault/retry test, and one distinct dirty-Delta worker
ownership/drain test. The latter confirms copied pending inputs survive actual
worker execution even after live rows and old history retire. Library Clippy
passes; final test Clippy still fails solely on the generated-libtest
`large_stack_arrays` diagnostic. No new lint suppression or resource limit is
introduced. Source/evidence hashes and exact commands are in the parent checkout
at `target/issue406-ipc-lifetime-followup/pending-validation.json`.

At the fixed 4M E1 geometry, synchronous selection becomes 31,999,976 bytes for
left rows and 512 for right rows at separate phases. It removes the original
960,014,640-byte owned pending copy and 863,999,352-byte materialized-row
reservation. Reusing the prior conservative source/compact/IPC terms yields a
979,287,699-byte leading sum; unresolved owners and overlaps prevent treating that
as a complete capacity certificate. Async preparation still owns and funds its
full pending copy, so the corresponding large async case remains too large.
Fresh preparation, held-E4 replacement, restore and expiration funding retain
their separate requirements. No new release build or performance workload ran.

The supplement is
`target/issue406-ipc-lifetime-followup/pending-capacity-supplement.md` in the parent
checkout, SHA-256
`39db752837f811f592fe73efb8e2230f0121e6e982c96157e34ed3c1d10de3aa`.
The eight-row timing tables remain the last measured runtime; current-runtime
acceptance is still outstanding.

## Share checkpoint payloads and bound expiration storage

Commit `70941d5f` rebuilds only indexes for trusted, completely checkpointed
append-only V2 history. Its sorted merge checks row IDs, payload locators,
complete coverage, retained row/charge totals and next-row bounds; replacement
indexes retain the original whole payload owners. Pending operations, tombstones and migrated or
OFF-mutated history use the existing encoder. An OFF mutation invalidates history
before any partial output or state mutation; ordinary delta/clean captures cannot
silently restore trust. Installation checks the complete Cut and current policy.

A three-array indexed expiration heap replaces the ordered tree. The recorded
1,024-row constructor RED observed 101,696 requested peak bytes; the new exact
arrays use 32,768, with no construction copy. Heap comparisons, dense removals,
append and admission ordinals match the independent tree oracle. Source/V1
accounting keeps its existing logical boundary. V2-restored heaps retain their
own reservations through empty reuse, with old/new growth overlap prepaid before
operator emission and refunded after actual allocation destruction.

Sparse chunk inventory now stores a dense-slot hint in the existing 32-byte
64-bit layout. Lookup validates row identity, exact chunk and source offset;
moved rows and repaired chunks rebind that hint. Tests retain actual Shared
payloads after managed V1 restore and check left/right positions independently.
This avoids adding a second per-row lookup map or a retained-row scan.

There are 41 distinct passing checks across the combined increment, including
all-admission-point rebase refusal/retry, restored interleaved IDs, continuation,
old-index retirement, worker drain, sparse repair and cancellation. Library
Clippy and scoped format/contract/whitespace checks pass. The known pinned
libtest-generated lint remains a separate unresolved CI limitation. Root fixture
failures and their corrections remain in raw logs; they are not performance or
expected behavior REDs.

Source/evidence inventories are in the parent checkout's
`target/issue406-expiration-heap/validation.json` and the feature worktree's
`target/issue406-complete-acceptance/rebase-validation.json`. No release build,
4M retry or timing workload ran for that increment. At that revision, restore
still charged historical maps and cumulative reader/row scratch.
The eight-row tables above still describe runtime `e8a28d99` and do not establish
performance for this implementation.

## Bound restore history and temporary scratch

The new restore path validates each index frame completely, then merges its
independent upsert and tombstone streams by row ID and epoch. Per-identity
lifecycle validation rejects reuse, duplicate introductions, invalid tombstones
and mismatched keys/times. Side/range/frame-ID validation, unique introductions
and complete payload-row counts establish a locator bijection without per-row
historical maps or sets. Gapped epochs and independently interleaved left/right
IDs remain valid.

A first scan reconciles exact live rows and logical bytes against metadata and
limits before admitting live-row storage. A second scan invokes the existing
key/time/charge oracle for every introduced row, including subsequently deleted
rows. Payload-reader child credit outlives the IPC Reader and temporary
schema, dictionary and record controls;
per-row scratch survives validation and resident-owner transfer. Both retire with
their allocations. No checkpoint wire or atomic install contract changes.

The original 4,096-rich-row fixture fails with the old history workspace, then
passes with the new implementation under the same 4 MiB available credit. There
are 21 distinct passing selected tests. All 56 positive admission points reject
atomically under fault injection, and all 611 cancellation points preserve old
state, refund temporary credit and permit successful retry. Controls cover
malformed history, full pre-IPC rejection, dictionary/map payloads, held aliases,
empty/one-sided restore, managed validation, writer continuation and rebase.
Library Clippy passes. Test-target Clippy retains exactly the prior generated
libtest diagnostic, with identical code, message and spans. Frozen source hashes
and exact commands are recorded in the feature worktree's
`target/issue406-bounded-restore/restore-validation.json`.

The pre-build capacity appendix is
`target/issue406-complete-acceptance/restore-capacity-current.md` in that worktree,
SHA-256 `39213c5223fcc5e6854e472ec9bfbd62ec0a6de61382340d9713c14702201c39`.
Its conservative maxima are fresh 997,684,907 bytes, E1 979,353,235, held-E4
rebase 941,251,393, E5 capture 941,244,263, and E4/E5 restore
928,493,775 / 928,455,991, each below the original 1 GiB pool. The bound includes
current small-owner/cursor/heap controls, phased IPC scratch and independent
segment funding. Restore bounds require synchronous execution into an empty
destination with separate source/destination pools. Fresh preparation and rebase
retain their owned asynchronous workers. These bounds certify neither RSS nor
old-state/same-pool restore.

The parent checkout's `target/issue406-bounded-restore-followup/plan.md` preserves
the diagnostic-first sequence, seven workloads/five physical groups, fixed
sample counts and 900-second budget. Controller/provenance checks pass (11).
The following measured follow-up supersedes the earlier capacity estimates
where noted; the earlier eight-row tables retain their original runtime e8.

## Measured bounded-restore follow-up at cffcf57b

The ordinary wheel and enabled native binary were built from the same clean
`cffcf57be9c5cada667c6109156bcbfe5b084f27` source and harness, with pinned compiler,
dependencies and ordinary release flags. Build/audit took 667.792 seconds.
The fixed 4M correctness lifecycle passed, but performance acceptance remains
incomplete and a material terminal regression requires correction.

All values below are medians. Main `588726c8` and earlier e8 columns retain prior
evidence; main has no true OFF mode. Main LOW/ON values marked dagger failed EOF
checkpoint completion and cannot support a complete-lifecycle speedup claim.
Current numbers measure cff only; later source changes require new evidence.

| Row / workload                     | Historical main 588726c8        | Earlier e8a28d99              | Current cffcf57b / target                       |
|------------------------------------|---------------------------------|-------------------------------|-------------------------------------------------|
| 1: Join 1M, 1 / 32 CPU             | LOW 125.6 / 305.8 ms †          | OFF 87.1 / 89.5 ms            | Unmeasured; ≤60 / ≤15 ms                        |
| 2: Join 10M, 1 / 32 CPU            | LOW 3355.7 / 5132.5 ms †        | OFF 933.8 / 916.6 ms          | Unmeasured; ≤600 / ≤150 ms                      |
| 3: Retained Join 4M ÷ 1M           | LOW 4.711× †; ON failed         | OFF 4.697×; ON failed         | OFF 3.486×; observed ≤4.4×                      |
| 4: Capture / restore per total row | Preparation failed              | Preparation failed            | 0.221539 / 0.494471 µs; above ≤0.1 / ≤0.2       |
| 5: Handler, compacting ÷ steady    | Fixture checkpoint failed       | Fixture checkpoint failed     | 0.985× partial; 18/20; inconclusive; target ≤2× |
| 6: ASOF 1M per input, 1 / 32 CPU   | LOW 292.7 / 319.2 ms †          | OFF 300.1 / 320.3 ms          | Unmeasured; ≤100 / ≤30 ms                       |
| 7: ASOF harness, 1,024-row batches | LOW H 361.1; 2×floor 386.0 ms † | OFF H 363.3; 2×floor 344.9 ms | Unmeasured; H ≤2×projection floor               |
| 8: Full 1,024-row Join callback    | Enabled 92.52 µs                | OFF 90.93; enabled 89.55 µs   | Unmeasured; ≤50 µs                              |

The fresh retained-OFF comparator was remeasured e8, rather than the historical
e8 column: 1M processing p50 383.842 to 258.226 ms and 4M 1512.430 to 900.200 ms.
Each size completed two rounds of ten adjacent alternating pairs. Both maintained
processing gates classify improved. The round changes/intervals are -31.53%
[-37.28%, -25.72%] and -30.17% [-36.53%, -14.60%] at 1M; -39.31%
[-43.27%, -33.70%] and -43.27% [-52.05%, -37.93%] at 4M. Each ten-pair interval
has 97.8516% coverage. Ratios of aggregate medians give -32.73% / -40.48%, a
different statistic from paired medians. Fresh scaling is 3.940x e8 / 3.486x cff;
no paired scaling interval is claimed across the separate size groups.

The phase costs reveal a regression that the processing timer excludes:

| Workload / phase | Fresh e8 p50 | cff p50     | Maintained paired classification |
|------------------|--------------|-------------|----------------------------------|
| 1M terminal      | 246.586 ms   | 403.564 ms  | regression                       |
| 4M terminal      | 378.132 ms   | 1475.115 ms | regression                       |
| 4M full cycle    | 1876.961 ms  | 2381.441 ms | regression (+26.88% p50 ratio)   |

Both round lower bounds exceed +5% for these phase comparisons. The 1M full-cycle
comparison is inconclusive. These are measured regressions within this
acquisition, with unverified WSL2 isolation; they are not a machine-independent
latency guarantee. The terminal timer includes the withheld watermark that
retires retained rows before source EOF. That path is the next focused diagnosis.

Capture/restore has 20 complete observations plus two warmups. Its denominator is
500,000 left plus 500,000 right rows, one million total. Snapshot p50 is
221.539 ms, including 210.613 ms preparation and 11.292 ms capture-only medians;
restore is 494.471 ms. Separate phase medians need not add to the aggregate
median. Omitting preparation would incorrectly claim the snapshot target passed.
Both per-row targets remain missed, and previous failed main/e8 components do
not provide a valid speedup denominator.

Compaction retains 18 paired component observations (rounds 10 + 8) and two
warmups. Handler medians are compacting 0.509049 ms and steady 0.516818 ms;
compacting preparation is 217.803 ms and combined time 218.291 ms. The partial
0.984968x ratio is descriptive only. The fixed 240-second group budget cancelled
the next call; the final planned call was never attempted. No retry or additional
samples ran. Every completed observation retains four dirty epochs, changed base,
folded deltas, equal output/state and the restore oracle.

The whole collector stopped at 566.758 / 900 seconds, including setup, warmups,
oracles and settlement. Of 133 planned physical calls, 132 were attempted:
118 completed measured calls, 12 completed warmups, one completed correctness
lifecycle and one cancelled call. The eight metric arms contain 156 timed
observations: six complete arms and two partial arms. All 52 owned PIDs settled,
with no forced kills, survivors or unsettled pipes. Sampled maximum aggregate
RSS was 2,391,285,760 bytes; no resource or competing-work threshold failed.

The unchanged 4M diagnostic observed fresh/history pool peaks of
997,663,225 / 979,293,354 bytes and E4/E5 restore peaks of
932,489,048 / 932,453,624, below the original 1,073,741,824-byte limit. All four
pools had zero failed grows and zero final reservations. Both restored histories
matched the complete left/right row oracles and retained E4 segment hashes.

The earlier source worksheets omitted all-valid Arrow IPC bitmaps: serialized
backing needs 2,000,128 additional bytes, and restored backing needs 4,000,172.
These were already reserved correctly by the implementation. Corrected
conservative rebase/capture bounds are 943,251,521 / 943,244,391 bytes; restore
bounds are 932,493,947 / 932,456,163. Existing fresh/E1 constructor envelopes
already counted validity workspace. The new reconciliation preserves the old
worksheets and exactly explains observed restore peaks after subtracting
already-retired history/row scratch.

Evidence lives in the parent checkout's
`target/issue406-bounded-restore-followup/analysis/report.md` (SHA-256
`bb0b01dda43716784f6cf17840a94d525259e79231764e4a3a8e3a509061c352`) and
`analysis/summary.json` under that same root (SHA-256
`ec2228b18df537acf400fc0f1583ca5c71fe0fbd3459a6b0926c10848e008696`).
The measurement evidence seal is
`d1ff91234143437af061603bcbcb0f225b50565dcf5d868b46d65680004e95c7`.
The feature worktree's capacity correction is
`target/issue406-complete-acceptance/restore-capacity-bitmap-reconciliation.md`,
SHA-256 `62aabdec6bd36bedcb56dba86043001ebc108d2ef9bb0007747a6f49203c80ba`.
Independent review verified all 207 sealed inputs and 413 evidence files. The
ordinary stripped artifact has no type DWARF; its source/revision/flags/sections
were audited, and actual capacity comes from the hard-capped diagnostic.

## Post-measurement ASOF descriptors and complete Join expiry

Commit `3b4ecafb` consumes the owned ASOF input-descriptor vector through a
fallible iterator transformation instead of explicitly allocating a second
output-descriptor vector. Payload partitioning, projection, compact row
references and reservation ownership retain their prior behavior. On the
aligned 4,088-row fixture, the old source's extra allocation peak is 229,440
bytes; the new source records 512 bytes on each ingress. Eleven distinct focused
controls pass, including the separate 4,096-row unaligned layout, discarded
batches, projection, duplicate/workspace precedence, journal/recovery and
cancelled worker waits.

The aligned test checks actual vector capacity divisibility. Rust may still
perform an allocation-layout adjustment for unaligned capacities; this change
does not guarantee pointer identity or zero reallocations. Reused 64-byte input
slots become 56-byte descriptors on the measured test architecture, so retained
capacity can exceed a freshly exact-sized output vector. The evidence establishes
removal of the explicit second backing, not an unconditional retained-memory or
latency reduction. Source and six raw logs are frozen in the feature worktree's
`target/issue406-asof-descriptors/validation.json`.

Commit `5ab1ac09` addresses the measured retained-Join terminal path. The
expiration index tracks its maximum event time inside funded storage. With
checkpoints disabled, a watermark strictly beyond that maximum plus the
retention extension, or opposite-ingress Ended progress, can retire the complete
side without individual heap pops and native-index repairs. Partial expiry and
checkpoint-enabled ordinal-sorted tombstones retain the existing path.

The bulk path checks aggregate metrics before mutation, marks shared payload
offsets dead, then clears rows and lookup inventories. Funded expiration arrays
keep their capacity, reservation and next admission ordinal through empty reuse.
The production final-watermark RED records 65 heap visits; the corrected path
records zero. Ten focused controls pass, including strict equality and i128
boundaries, out-of-order append/growth, Ended/reuse, pre-cancellation, metric
overflow, restored admission refusal/retry, sparse aliases and unchanged ON
journal ordering. Two affected tests were rechecked after a test-only refactor.

Both increments pass library Clippy, scoped formatting, generated-contract and
whitespace checks. Test-target Clippy retains only the known generated-libtest
diagnostic and remains an unresolved gate. The EOF validation manifest is
`target/issue406-eof-bulk/validation-final.json`, SHA-256
`fd5538e5ba90137054e842b2498955edd18e5c2ef11991b568dd6d12f7db7a5c`.
Neither increment has new release timings: all eight measured table rows and
the terminal regression above still describe their named historical runtimes.
The added maximum scalar also changes expiration-control size, so the earlier
cff capacity bounds are not relabeled as exact bounds for this new source.

## Direct ASOF, parallel Join gather and payload digest reuse

Three later source increments remain unmeasured in release performance:
`16af86e4` prepares strictly ordered ASOF left input directly as columnar chunks;
`d4e66037` gathers eligible Join output columns on owned parallel workers;
`a681e7da` constructs each checkpoint payload segment once and reuses its
checksum for its inventory name and index reference. The ASOF path is restricted
to one inline Utf8 key and one UInt64 sequence, with an actual resident-state
ordering proof. Other inputs retain the validated path. Join parallel gathering
requires at least 64,000 output rows and six eligible columns; small or unsupported
outputs keep serial gathering. Neither restriction is a claim of speedup.

Ten ASOF and nine Join controls pass. They cover exact dictionary/order and
error behavior, projected nullable slices, checkpoint bytes and continuation,
funded worker cancellation/abandonment, actual concurrent gathering, early
reservation fallback, failed service admission and output credit across awaited
delivery. The direct ASOF task is eight bytes smaller on this build, charged
once by its wrapper and once by work admission. The intermediate credit differs
by 16 bytes; the later overall peak and successful state-install cutoff remain
unchanged. This does not claim a general 16-byte reduction or lower row funding.

Eleven checkpoint controls pass after a genuine base-path RED observed 3,264
bytes hashed for 1,632 payload bytes; cleanup completed before the assertion.
The final test exercises both base and pending encoders, checks the independent
segment SHA against metadata and decoded index references, and requires one
hash pass. Exact wire, funded allocations, alias retirement, refusal, cancellation
and restore controls pass. The implementation keeps `StateSegment::new` as the
checksum constructor and prepays segment/control ownership before constructing
it; it does not extend the trust contract of `from_validated`.

Library Clippy, scoped formatting, whitespace and generated-contract checks
pass. Test-target Clippy still fails with exactly the prior
`large_stack_arrays` compiler message; no lint is suppressed. Final independent
source and evidence review approves these scoped implementations, not merge
readiness or the performance targets.

The checkpoint diagnosis reused the audited cff native binary: one excluded
warmup and three profiles completed in 35.276 / 180 seconds with full oracles
and all owned processes settled. Of 2,194 preparation-root samples, SHA-256
has 931 self samples (42.4%) and cancellation checking has 810 inclusive samples
(36.9%; its tree check has 707 self samples). The redundant explicit payload
SHA contributes 208 samples, 9.48% of attributed preparation samples. It is not
all SHA work, a statistical bound, or a predicted wall-time improvement.
Restore history validation has 4,358 / 5,674 inclusive samples (76.8%).

The run lost 42 / 37 / 36 events. Whole-process samples total 60,310, with
52,330 (86.8%) outside the selected phase roots; capture-only has just 112.
Release inline information and WSL2 isolation are limited. These CPU observations
justify the narrow digest correction and further investigation of bounded
cancellation checks and per-row restore setup, without changing the timing
verdicts above or removing historical validation.

Profile evidence is under `target/issue406-checkpoint-profile/`. The plan hash
is `a02e08f15f661d728f8bcc8bce1bafa30aac3994fe1168c354acadbe3c6164b7`;
completion is `9e70d04242dc51172e0ae78b34145801d65043f866765600e6d3cefb421adfcd`;
the acquisition evidence manifest is
`3d47c24a242887be4c5e219dbfd00ec8200dcce3fe523421acea5aed5262be3b`.
Independent review checked the native attribution, source hash chain and all
sealed evidence. New release timings remain outstanding.

The implementation validation manifests are in the feature worktree:

- `target/issue406-direct-left/validation-final.json`,
  `afd8395b9ff8ddfe66a323e18d6851d1a22789ddb07571ac399ca180e718f5f1`;
- `target/issue406-column-gather/validation-final.json`,
  `28f11087fc6bf22d02a26a652f7039a79367535e06475ff15facb5c21ea37a39`;
- `target/issue406-payload-digest/validation-final.json`,
  `88bbf02d1f7d1a7625c86663532ed7eb3108dd5072820a1df0d376eb5b8a48bb`.

The single saved CI snapshot for `40b39c70` shows Linux Rust parity and Windows
Clippy failures. Their exact logs both stop at the known large-stack-array
signature on Rust 1.88; neither establishes a runtime test failure or a green
run. No further status snapshot was polled for that head. The diagnosis is
`target/issue406-ci-40b/report.md`, SHA-256
`3dd14977a2862e88e907dc97658f4af7d64d3ce704f93f904d764b0b8d567dd3`.

All eight measured rows and main/OFF/LOW/ON limitations above remain unchanged.
The next acquisition uses one common new runtime for the complete requested
inventory, plus a pinned cff retained-OFF comparison to test the terminal
regression correction. It will use separately recorded fixed sample counts and
wall budgets; it will not extend the sealed incomplete cff compaction attempt.
