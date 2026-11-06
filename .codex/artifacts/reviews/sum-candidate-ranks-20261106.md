# Grouped SUM candidate/rank review

## Scope and checkpoint

**Source and focused local verification PASS. Public candidate/build admission,
measurement acceptance and final merge approval remain pending.**

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
