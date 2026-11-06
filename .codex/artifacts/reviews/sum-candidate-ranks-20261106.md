# Grouped SUM candidate/rank review

## Scope and checkpoint

**The completed campaign failed performance acceptance; the runtime prototype
is withdrawn. Evidence, documentation and latest-main integration PASS review.
Required final-head CI and final merge approval remain pending.**

- Base: `1fcafde11a247b1be275d3cb6ab41a59476d6832`, branch
  `feature/sum-candidate-ranks`.
- Reviewed the [specification](../specs/sum-candidate-ranks-20261106.md), current
  incremental SQL transaction and pinned hashbrown 0.17.1 allocation behavior.
  The reviewer read repository guidance and `$code-style`; no native build,
  test, fixture, profile or measurement was run.
- Preserve PR #548's empty inline accumulator vector, skipped inline
  finalization and typed integer output, and PR #549's batch SQL policy.
  Earlier profiles only motivate this hypothesis; they do not establish
  current-main costs or improvement.

## Design assessment

The existing slot-keyed candidate HashMap serves partial seed/merge, snapshot
overlay, container preparation and transaction consumption. Replacing only its
insert operation would leave ownership and lookup obligations elsewhere.

The accepted proposed scope is the existing trusted grouped inline integer
SUM binding. A private store selects either the unchanged generic CandidateMap
or a ranked variant with `Vec<Candidate>` and a small global-slot-to-rank map.
Candidate and Group layouts remain unchanged. Vector capacity follows the
actual small map's natural capacity, rather than an independent power-of-two
row/global-group reserve. Storage is O(transaction-touched groups), including
for sparse updates; no O(all historical groups) rank array is permitted.

Newly admitted candidates retain the original group-budget/candidate-payment
and partial-slot sequence. Dense position, touched-key rank and partial rank
must agree. Global slots need not be monotone or equal ranks: reversed old
groups, newly interleaved groups and restored state must retain explicit
mapping. Both ordinary canonical-key and primitive-native admission paths
must establish the same invariant.

Partial merge obtains ranked candidates directly, removing its T slot-map
lookups; row/key hashing and full-snapshot overlay lookups remain. Output uses
the small slot map. Do not describe this as eliminating every lookup, all
hashing or the O(historical groups) full cumulative snapshot work.

After merge completes, move the existing `PartialGroups.slots` vector into
ranked ownership before dropping the partial. It is then zipped with owned
candidate iteration to produce `(global_slot, candidate)` without sorting,
unsafe indexed moves or another Option payload array. The small slot map may
drop when no longer needed. Its output/container/prepared-transaction lifetime
must remain explicit.

Global, float, mixed, ordered and variable-extrema shapes retain the generic
path. Their original charge/admission/error behavior must remain unchanged.
No string-key dictionary, filter optimization, worker change, public knob,
checkpoint descriptor or persisted identity is part of this work.

## Prepayment and lifetime requirements

Do not reduce existing candidate charges. In particular, the old per-candidate
container allowance is
`4*sizeof((usize,Candidate)) + 8*sizeof(usize)`; touched-key, partial/native
states, output, prepared updates and persistent container fees are separate.
The new store cannot borrow their allowances to hide a growth peak.

For the proposed natural one-entry map growth, a large hashbrown table with
B buckets has C=7B/8 usable entries. At T=C+1, its next usable capacity is
bounded by 2C+1. Aligning the vector to these capacities gives old-plus-new vector
capacity below 3T. Old-plus-new table buckets are 3B, below 24T/7; table entry,
control bytes, alignment and small-table constants still require inclusion.
The transferred slot vector has capacity below 2T under the existing policy.
With the expected host widths Candidate=128, tuple=136 and usize=8, this is
approximately 459T plus constants versus the old 608T allowance. These host
sizes and the exact reserve policy must be verified, not assumed portable.

Use checked capacity/layout arithmetic and source-derived bounds before
growth, with small-table/header cases covered by the existing metadata floor.
`allocation_size()` is a useful pinned-library profiling observation, not a
post-allocation substitute for this proof. A single incremental growth trace
through representative capacity transitions can validate the derivation;
building an independent map for every size would add an unnecessary quadratic
test workload.

Before the partial reservation drops, candidate-owned credit must independently
cover the moved slot vector for its remaining lifetime. Existing group
reservations remain with groups through prepared transaction ownership. Include
simultaneously live Vec/map old/new allocations, touched keys, emitted arrays,
slot storage, output borrowing and prepared/new group containers in the full
owned peak. Refusal, cancellation, discarded transactions and output rejection
must release the complete optional transaction without installing state.

An independent power-of-two vector can cost more than the old map around
growth boundaries such as 257 or 513 touched groups. Even the aligned design
must not claim universal memory reduction from a favorable 256-group probe.
Budget coverage at every supported transition is required; actual reductions
and allocation-count increases must be reported at their measured scope.

## Required evidence and remaining gates

Before production changes, observe a meaningful baseline RED around actual
merge slot lookup operations, with output correctness established first. Put
the counter on the actual lookup boundary and retain a generic positive
control. An independent old-equivalent container/allocation control must include
construction and rehashing inside its counter. Record total/current/peak bytes
and counts; retained-only or preallocated maps do not prove a whole-lifetime win.
Also exercise the actual update/scatter/output/teardown path against its newly
paid scratch and compact credits.

Focused semantics must cover signed/unsigned null/wrapping state, first-seen
NULL/empty/Unicode tuple keys, WHERE versus FILTER group creation, sparse and
permuted historical slots, new groups across chunk boundaries, held outputs,
discard/refusal/retry, cancellation, recovery and unchanged generic behavior.
One-byte and finite/custom-pool controls must retain the original limits and
failure ordering. Private wrapper metadata must fit its paid structural floor.

The five-case measurement proposal remains separate: dense 1T/4T, sparse 1T,
all-distinct 1T and moderate two-occurrences-per-key 1T. The last case is not
an admission threshold. Proposed 200 formal observations, twenty warmups and
two five-window profiles share one fresh 600-second acquisition; builds are
separate. No old binary, profile or residual budget is current evidence.

The design has no unresolved structural blocker under these constraints.
Conditional TDD clearance does not grant a native execution window: the parent
coordinates that separately. Frozen source/tests/lint, actual public build and
protocol admission, complete measured acceptance, final documentation review
and green required final-head CI remain necessary before a merge.

## Measurement-tool and protocol-draft review

**Tool/draft PASS** at `b425b595dbce7305c4a10e649a91658d4dcf8345`.
This reviews the new 87-line entry, its focused tests and the shared collector
change, plus CI registration `62dde326`. It is not acquisition or source
approval.

The shared collector has two static frozen contracts and a typed context.
The old default columnar contract retains its cases, recipes, acceptance and
route-record behavior. The new entry passes its candidate-rank contract
explicitly without changing globals or using a dynamic registry. Its fifth
case reuses the old edge IPC recipe as `repeat2`, a low-repetition control;
neither an interner selector nor an admission threshold is implied.

The new phase requires five cases, two rounds of ten alternating pairs,
200 formal observations, twenty warmups and both dense-1T profiles with five
valid windows after excluding the first of six jobs. It requires fresh main
`1fcafde1` and candidate builds with common657; old 22f/P74 products are not
valid substitutes. Missing or inlined frames remain unresolved. The new phase
does not require old columnar symbols, emit a columnar `profile-route.json`
or use that experiment's route evidence to prove rank optimization.

All thirteen declared helpers are sealed at setup and through native
finalization and publication. Publication still uses the original deadline;
late publication or helper drift revokes durable completed/acceptance status.
Invalid required profiles and cancellation retain partial evidence without
claiming success. The owner recorded a genuine old-validator rejection RED,
then 26 unchanged controls plus eleven new tests passing, with Ruff/format
and maximum affected-function CCN seven. The reviewer read source/evidence
instead of rerunning tests or native work. CI registration remains after dev
dependencies are installed.

Independently checked source SHA256 values:

- Shared collector: `401e9b8f510b16cbd240b4eb616f7ca76a4a45e0cc18f36a58be907ca51e9ac1`.
- New entry: `a46e743e831599ac4d451f7c1e012c552ce5c7bf0c8441c570bfda42ce74adf9`.
- New tests: `08690ad2be204b70db753122da8c1a0b1b6f6df489c7abf75e5978e73bbeb611`.

All thirteen actual helper hashes match draft protocol SHA256
`e32c7240d1cf851abb5266705d730e75110906ac5a99dc7a0013539aa1d4bfec`.
The candidate and both build receipts are still null: acquisition is
**inadmissible**. Before final freezing, the draft's coarse pipeline description
should reflect the actual native post-seal/cleanup before publication and
helper checks around publication; the implementation already retains that
ordering. Historical protocol hashes correctly reject the changed current
helper source, so reproduction must use their archived source revision rather
than relabeling the new collector as the old one. Historical sealed data is
unchanged.

## Meaningful RED audit

**PASS: implementation may proceed within the conditionally accepted design.**
This is independent review of existing source, receipt and complete failure log;
the reviewer did not rerun native tests, builds or measurements.

The frozen `grouped_candidate_rank_tests.rs` SHA256 is
`e04d36499fb34870094572189819887f72f69a426392ba62f8386c1b36e9afd4`.
The completed `candidate-ranks-red-20261106.receipt.json` records the focused
`cargo test --locked -p calc-flow --lib --features benchmark-internals
grouped_candidate_ranks -- --nocapture --test-threads=1` command, exit 101 and
695.444006353966 seconds including compilation. Its log records two executed
tests, both failing for their intended assertions, 2571 filtered tests and
0.06 seconds of test execution.

The test-only counter sits immediately before the actual candidate
`get_mut(slot)` in `PartialGroups::merge`. The inline path performs 256 such
lookups against the intended zero, while the mixed SUM/COUNT positive control
also performs 256. Reversed historical slots and successful independent SUM
checks precede the operation assertion. No production behavior has changed.

The second test constructs each legacy map naturally inside the allocation
scope, including insertion and growth; it does not reserve an old map outside
the counter. Both controls execute actual input admission, chunk updates,
partial merge, output, prepared-container/scatter and owned teardown. The
independent oracle verifies exact output schema, values and first-appearance
order across old/new interleaved groups and the chunk boundary before any
allocation comparison. Both branches settle reservations and counted live
allocations to zero.

Baseline and forced legacy are still the same implementation. Each reports
retained bytes 99,744, total lifecycle bytes 700,937 and lifecycle peak bytes
422,209; retained allocation counts are 602 total, 386 current and 402 maximum,
while lifecycle counts are 630 total, zero current and 402 maximum. Both
record prepaid peak 2,689,680. The intended retained-byte reduction assertion
fails first; the logged total and peak are also equal. These are genuine
operation/allocation REDs, not evidence of speed or an already achieved memory
improvement.

Production source review must still establish capacity-based prepayment before
growth, independent payment for moved slots before partial-credit release,
unfavorable growth transitions, all generic-path controls and the required
semantic/lifecycle cases. The fixed acquisition, final source/documentation
review and green required final-head CI remain separate gates.

## Recovered baseline admission

**Baseline provenance PASS. Acquisition remains inadmissible pending the
reviewed public candidate, its build and the frozen final protocol.**

The admitted recovered receipt is
`target/sum-candidate-ranks-20261106/build/baseline-final/recovery/build.json`,
SHA256 `e781654b43bb072036e8e97af46cd732bcde16edc54889e8301c82c9dd4a9e9b`.
Its binary SHA256 is
`da0af0d5db0485b1f7ebeb9ddb935bde831d5e813596343e652f9c61f387d128`.
The recovery script and underlying records were read; independent read-only
hash/Git-blob verification confirmed:

- All 1010 current, before-build, after-recovery and receipt source seals agree.
  The 1009 product/config/vendor/support files match public `1fcafde1`; the
  remaining file is exact common657 instrumentation from public `84186554`.
- The retained compiler stream has 327 records, exactly one core and one
  diagnostic-example artifact with `fresh=false`, matching frozen source paths,
  release profiles and original binary location. Its final record is
  `build-finished` with `success=true`. Arrow buffer/data/schema records are
  version 59.3.2.
- Four original record hashes, original/copied core and example depfiles,
  retained compiler-environment values, Rust compiler cache identity, cached
  symbol-table hash and original/copied binary identity agree with the receipt.
  The environment values came from the live compiler observation, rather than
  being reconstructed from the current shell.

The interrupted builder did not persist its subprocess return code or precise
elapsed time. The recovered receipt keeps all corresponding fields null and
labels its command as reconstructed from the fixed maintained helper and
retained invocation. The retained Cargo log reports `21m 23s` (1283 seconds,
one-second precision); this is recorded separately and is not represented as
precise wrapper wall time. Recovery took 8.131882011017296 seconds and invoked
no native workload. The separate 145.4088840484619-second aborted shared-target
lock-wait attempt and its owned-process cleanup record remain preserved.

The maintained build admission requires actual fresh core/example compilation,
source/configuration/harness equivalence and binary seals; it does not require
a numeric build duration or persisted wrapper return code. The retained final
Cargo success and complete identities satisfy those gates without fabricating
missing values or repeating a completed build. Builds and recovery are separate
from the still-unused 600-second measurement budget. No native build, test,
fixture or profile was run during this independent review.

## Source and focused verification gate

**SOURCE PASS** for the seven Rust files sealed in
`candidate-ranks-final-source.json`, committed as
`721b0f5c74fca27528b0f71abea9ad46fa51ed44`. Independent Git-blob hashing confirms
the commit contains exactly those seven reviewed Rust files and their frozen
bytes. Public candidate/build admission follows separately. This is source
clearance for the fixed study; it does not establish runtime improvement or
merge readiness.

The reviewer read all changed Rust files and the allocation/lifetime proof,
including admission, partial seed/merge, full output overlay, container
preparation, owned iteration and teardown. The final principal source SHA256s are:

- `candidate_groups.rs`:
  `1d93ae6126a9d790daa919d03ac6b25f090f8cbf3e535a0305ac799c00c534fc`.
- `incremental.rs`:
  `d1ae1302938804312ff73117b406e5697726173b87db840e2270ea646650ca51`.
- `grouped_candidate_rank_tests.rs`:
  `c0c693c1eff22b0335109d58cd9b12a98ce3dc10a22bd70094a23572c4df735f`.

Eligibility remains exactly the trusted existing `inline_sum.is_some()` binding.
The ranked Vec, small integer-slot map and moved slots all scale with transaction
touched groups. Both canonical and primitive-native admission preserve partial
rank order; merge uses `values[rank]`, while full snapshots retain slot lookup.
Global, floating, mixed, variable-extrema and ordered aggregates such as
`LAST_VALUE` keep the generic path. Existing inline SUM queries with HAVING or
output ORDER BY remain admitted. No key dictionary, workers, public API, wire
format or checkpoint identity changed; PR #548's inline/finalization/output
optimizations and PR #549's batch policy remain intact.

Every existing fee is preserved. Before ranked growth, checked arithmetic pays
the conservative simultaneous old/new map and payload Vec plus future moved
slots from the existing candidate-only `4*sizeof((usize,Candidate)) +
8*sizeof(usize)` allowance. Actual map capacity drives the exact Vec reserve;
the post-allocation capacity validation supplements the source-derived bound.
Pinned hashbrown control/alignment overhead is bounded by 31 bytes per map.
For touched count T >= 3, combined payload capacity is at most 3T, combined
buckets at most 24T/7 and moved slots below 2T. With S=sizeof(Candidate) and
U=sizeof(usize), the conservative bound is
`T*(3S+(62U+24)/7)+62`, below the unchanged `T*(4S+12U)` at observed
S=128/U=8; T=1/2 use their explicit small-table capacities. Runtime checks use
actual target widths. The bounded real growth trace confirms representative
transitions through 1025; it supplements rather than replaces the general bound.

The moved slots are independently paid before partial credits drop. Owned
`slots.zip(values)` consumption adds no sorting, payload Option array, slot
copy or retained partial lease; Group credits move into prepared state. Review
found the zero-candidate native path could reserve four partial slots and
transfer them without candidate credit. A separate actual one-test RED observed
capacity 4 versus expected 0; the fix discards empty storage before releasing
the partial reservation, and the final regression passes.

### Verification evidence

`candidate-ranks-final-tests-20261106.receipt.json` and all eight referenced logs
record **30/30 passing tests**: eight new rank tests, ten inline controls, eight
integer/UTF8 controls, two seed controls and two codec controls. Final source
before/after seals and the reviewer's current seven-file hashes agree. The
focused Cargo command is the recorded baseline RED command with filter
`grouped_candidate_ranks`; it rebuilt the library in 3m28s and executed eight
tests in 0.11s. The remaining exact controls reused that compiled test binary.
The complete verification took 211.7611091149738 seconds.

Coverage includes native/default signed and unsigned wrapping/null sums,
nonmonotone historical slots, interleaved new groups, chunk boundaries,
WHERE/FILTER distinctions, NULL/empty/NUL/Unicode/composite keys, exact schema,
metadata, first appearance, held output, discard, refusal/retry, recovery and
generic/global behavior. New lifecycle controls deterministically cancel an
already admitted ranked transaction at the existing merge yield and verify
unchanged installed output/export and exact refunds. A full update using the
custom OwnedStatePool refuses at measured peak minus one byte, preserves
installed state and succeeds once at the exact original peak.

Scoped strict `cargo clippy --locked -p calc-flow --lib --tests --features
benchmark-internals -- -D warnings` passed in 99.82467809796799 seconds. Earlier
new-code lint failures were corrected without lint exceptions. The recorded
seven-file Rust 2024 formatter check, schema no-drift and whitespace checks
passed. Scoped complexity covers 67 changed functions with maximum CCN eight.
The reviewer independently checked source hashes, schema drift and whitespace;
no native execution was repeated. Full workspace, Python, connector,
cross-platform, packaging and combined 90% coverage gates were **not run
locally** and remain CI responsibilities.

### Allocation evidence and limits

Final ranked/forced-legacy retained bytes are 97,696/99,744; total lifecycle
bytes are 692,809/701,033 and peaks 411,697/422,209, against identical prepaid
peak 2,689,680. Lifecycle allocation counts increase to 638 versus 630, with
maximum live counts 404 versus 402. Counted live bytes/counts and reservations
settle completely. Allocation counts do not establish speed.

The current forced-legacy total is 96 bytes above its earlier baseline RED.
The current map/store sizes are 64/112 and the preparation future is 3288 bytes;
Tokio boxes that future in this debug configuration. A larger future is
consistent with the constant difference, but the old future size was not
measured, so a causal attribution is not established. Generic payload/map fees
and layouts remain unchanged, and actual complete peaks stay funded.

The trace explicitly records equal retained ranked/legacy bytes at 257 and 513,
70,160 and 140,304. Source algebra likewise predicts equality at T=10,000:
both map bucket count and moved-slot capacity are 16,384 on this host. That is
a capacity derivation, not a measured workload result. The favorable 256-group
probe supports no universal memory-reduction claim. Only the frozen five-case
acquisition can establish the required dense timing improvement.

No unresolved source, style or focused-test blocker remains. There is no PR/CI
snapshot at this branch-stage gate. Public candidate/build/protocol review,
the parent's explicit acquisition GO, complete measured acceptance, final
documentation review and green required final-head CI still precede merge.

## Actual acquisition-packet admission

**PASS at the exact frozen protocol below.** The parent must issue the unique
acquisition GO separately, and the in-budget live global quiet guard must pass.
This gate neither starts acquisition nor establishes a performance result.

- Public candidate P: `ec1a629f3a4d7e12f53407f795666d58cee91bb8`, tree
  `7ae05ed4d3ed0fcb9db4445cb81aa73f6ad70739`, identical to the reviewed source
  tree. Baseline remains public `1fcafde1`, main at study start.
- Candidate build receipt SHA256:
  `2bc036a64425cd19b68f81132cde5d3d069653af9ffa8c22fee9126d10bc59b4`;
  binary SHA256
  `d2ce2a6644ab0934a8cf46e8f3c8d370eea3ba610fe91dbfa084fc96c176dd61`.
  Its ordinary wrapper attempt completed with exit zero and precise build time
  805.830162640952 seconds; the original normal receipt remains preserved.
- Final protocol `target/sum-candidate-ranks-20261106/protocol-final.json`:
  3,089,441 bytes, SHA256
  `75d53e62d98989e28c598bf4b43f402512adb32648363403647be23e3b4cecee`.
  Recovered baseline receipt remains the already admitted immutable `e781654b`.
  Its missing return code and precise duration remain null, with Cargo's
  rounded 1283 seconds and recovery/aborted-attempt costs separately qualified.

The reviewer read actual candidate sealing code, receipts and final protocol.
Independent read-only checks verified all 2022 source entries across both arms
(1010 baseline, 1012 candidate), current/before/after inventories, per-file
public Git identities and SHA256s. Common657 from public `84186554` is the sole
native overlay. The seven product/test source differences are exactly the
reviewed files. Both compiler streams contain actual fresh core/example
artifacts and a final successful build record; all 292 registry compiler
identities, locked dependencies, vendor/config/support files, compiler identity
and effective build flags agree. Captured live compiler environments agree,
including fields beyond the maintained normal receipt's smaller field list.
Copied depfiles include the new candidate module; binary and cached-symbol
seals match. No native build, test, fixture, profile or workload was repeated.

The portable build proof embeds 25 exact UTF8 record payloads, including both
compiler streams/logs, before/after inventories, copied depfiles, original
normal/recovered receipts, environment observations and recovery qualifications.
Each encoded payload matches its original bytes and SHA256. Its per-file Git
audit matches both public revisions. Native binaries, raw perf recordings and
symbol tables remain explicit sealed persistent local artifacts; they are not
claimed to be embedded. The separate source-verification archive is 15,252
bytes, SHA256 `f81bd81f06642f8ad82e584088109ad47c21fde1c51cf88b1544075e7de7b17d`.
Its 20 exact records and seven final public source identities were independently
checked. It explicitly preserves the unavailable original RED test-payload and
local-commit limitations, and is predeclared separately from acquisition.

All thirteen active/helper/test seals match current and public P source. The
maintained build helper's separately embedded source matches its seal. Original
proposal and draft protocol hashes remain unchanged. The final contract keeps
the same five cases, two rounds of ten alternating pairs, 200 formal samples,
twenty warmups and twelve dense-1T profile jobs, excluding each first job and
requiring five accepted windows per arm. Repeat2 is a low-repetition control,
not an admission threshold. No columnar-route or worker-parallelism requirement
is introduced; missing/inlined frames remain uncertainty. Acceptance requires
complete evidence, at least one exact maintained dense improved verdict and
no confirmed regression in the five cases.

The single 600-second acquisition has prior debit zero. Fixtures, preflight,
oracles, profiles/decoding, samples/resources/statistics, native post-seals,
owned-process cleanup, local publication and helper post-publication/deadline
checks share its original deadline. Publication or helper drift revokes a
completed result. Acquisition and publication directories were absent at
admission. No extra cases, samples, retries or deadline renewal are authorized.

Main's later `70e88cc7` changes ASOF-only build/probe helpers, with no direct
SUM admission/candidate/store/harness/dependency overlap found. All measured
claims remain bound to the fixed `1fcafde1`/P cohort. Subsequent integration
must preserve that main change and pass independent review/required CI; the
integrated binary's timing remains unmeasured. A source-only supplementary
manifest may bind the predeclared component archive after the outcome without
modifying the collector's sealed acquisition manifests or raw archive.

## Branch delivery review: feature/sum-candidate-ranks to main

**Delivery base:** `70e88cc70016a24807a8cbe46287070a9a501717`.
There is no PR/check-run or prior GitHub review snapshot at this prepublication
handoff. The final delivery commit/file inventory and required CI are pending.

### Summary

The fixed campaign completed but neither dense target met the maintained
`improved` rule. The parent restored all seven prototype Rust paths, removed
the two added modules and integrated latest main's separate ASOF change.
Independent comparison confirms the entire delivered `crates/calc-flow` tree
equals `70e88cc7`, with no untracked or retained ranked modules. The delivery
therefore preserves the experiment's tools, evidence and documentation while
shipping no candidate-rank runtime change. Measured P remains public and its
frozen source/build trees remain preserved; integrated-product timing is
unmeasured.

### Build and Test Results

- Rust: **Passed** for the recorded experimental-source focused gates
  (30 tests, strict scoped Clippy/formatting and maximum CCN eight); production
  core is **Not touched** relative to latest main. No local full matrix or
  integrated-product benchmark was run.
- Python product: **Not touched**. The unchanged measurement-tool review
  retains its 37 passing helper controls and focused CI-registration evidence.
  Final required CI remains pending, including combined 90% Rust coverage,
  cross-platform and other required repository gates.
- New unresolved failures: **None** in the reviewed evidence. Historical TDD
  failures and corrected lint failures are retained as such.
- Regressions: **None confirmed** among measured cases. Sparse remains
  inconclusive; neither that nor the other verdicts proves equivalence.

### Blocking Issues

No unresolved source, evidence, documentation or integration issue remains.
Green required checks on the final delivery head remain a merge prerequisite;
this handoff does not claim CI success or merge readiness.

### Style Issues

None introduced in the reviewed delivery. The narrative, index/changelog
changes, linked evidence and aligned tables are consistent with repository
conventions. Collector outputs remain immutable.

### Test Coverage and evidence integrity

The reviewer independently matched all eight collector files to manifest
SHA256s and byte lengths. The collector manifest SHA256 is
`1acb6b0129045f66e22411e395fc077d28358fb6e386aec5a143226e6c93b375`;
the complete raw archive SHA256 is
`d6827f49e2baf599df9578f4550e2cd1d895c9636969483f2f9941670aa89338`.
The separately archived protocol remains exact `75d53e62` and its embedded
protocol, helper seals, build identities and portable proofs agree.

All five cases retain forty formal observations in the exact alternating
two-round/ten-pair order. Each observation equals its original stdout and
resource receipt, and its argv binds the correct binary, CPU affinity and
sample count. There are exactly 200 formal jobs, twenty warmups and twelve
profile jobs: **232 benchmark jobs and 4872 complete snapshot checks**.
The profile jobs share two six-job processes; adding 220 timing processes and
one fixture-preparation process gives **223 physical native invocations**,
all with recorded reaped receipts. Every benchmark job reports natural end,
zero live tasks/errors, empty edge occupancy and checkpointing disabled.
The final settlement handoff records no remaining owned native processes.

Both profiles retain five nonempty processing windows after excluding their
first jobs, with 798/840 accepted user-CPU samples and 142/167 unresolved leaf
samples. All 2205/2322 raw timestamps match imported clocks; maximum recorded
error is below 6e-8 milliseconds. No loss/unwind/warning diagnostic appears.
Published stack counts, target records, windows and binary/perf hashes agree
with the raw evidence. CPU shares overlap, generic symbols combine callers,
and missing/inlined frames do not establish zero cost or causal speedup.

The recorded pooled changes and exact maintained verdicts are:

- Dense one CPU: +0.147441426221806%, `no-confirmed-regression`.
- Dense four CPUs: -3.3585205479178204%, `no-confirmed-regression`.
- Sparse one CPU: -1.3085739733837642%, `inconclusive`.
- Distinct one CPU: -5.852208086511068%, `no-confirmed-regression`.
- Repeat2 one CPU: -4.5695083650928%, `no-confirmed-regression`.

The reviewer checked the recorded intervals against the unchanged maintained
5% verdict branches, without rerunning statistical estimation. Both dense
cases have interval upper endpoints above -5%; sparse's second upper endpoint
is +18.247891606228684%. The negative pooled medians are descriptive. The
unchanged acceptance rule correctly returns false, requiring withdrawal
without another campaign, case expansion or favorable retry.

### Documentation Consistency

The final narrative, benchmark indexes and changelog correctly lead with
rejection/withdrawal and preserve narrow allocation, causal, scope and
recovered-build qualifications. The independently verified core restoration
matches their delivery statements. Subsequent research directions are marked
as hypotheses rather than measured improvements. Product guides need no new
API/behavior update because the runtime equals latest main.

Budget accounting retains all original timestamps. The raw acquisition and
summary snapshots record 143.64614346902817 seconds before final publication;
the immutable final collector manifest stops at **143.80610578000778 seconds**.
The separate read-only integrity/protocol-archival/settlement handoff completes
at **235.56555740401382 seconds from the same original start**, below 600.
It performs no new workload or statistical calculation and does not rewrite
the collector stop or sealed files. Independent specialist review occurs after
that sealed handoff; it does not rerun acquisition.

The separately added source-verification manifest SHA256 is
`678408f1cc596117885f2f902890617679945504ff4893ead084086107d7f9d5`.
Its byte/hash bindings for component `f81bd81f`, archival handoff, collector
manifest and exact protocol all match. The component was predeclared before
GO, and the supplementary manifest leaves collector artifacts unchanged.

### Verdict

**Comment Only: evidence/documentation/integration PASS; final required CI
pending.** The rejected runtime is correctly withdrawn. The parent may commit
and publish this reviewable delivery; final reviewer approval and merge still
require green required checks on that actual final head. No new native tests,
profiles, fixture workloads or statistics were run during this outcome review.
