## Branch Review: C1 aggregate snapshot cache

**Branch:** `feature/ta-arroyo-c1-snapshot-cache` → main
**Reviewed head:** `8d37708f4b008816046d4eb6b4afd5a44c400807`
**Tree:** `1847453b5791b0e7427e34c67c3994b33f0fb791`
**Base:** `3ca4c019b46d75a58a18c1df6111ddc2f8262da4`
**Scope:** ten changed implementation/test/decision/documentation files.

### Summary

C1 reuses immutable raw aggregate arrays for unchanged full-snapshot chunks.
It retains complete snapshot contents/frequency, postprocessing and original
emit-before-commit ordering. No source correctness, budget or ownership
must-fix was found. This is code/static readiness, not performance or merge
approval; no remote/CI query, native/build/test or measurement was performed.
Only this owned review artifact was edited.

### Build and Test Results

Rust focused results are **agent-reported**, not independently executed or
verified from raw logs. The versioned API note and parent handoff report the
meaningful clean-array-identity RED→GREEN, extra-registration RED→GREEN and
all sixteen C1 tests passing. The combined control selection was 28/29: its
seed fixture retained the new owner beyond the unchanged zero-reservation
assertion. The final fixture explicitly drops that owner; its single repaired
test then passed. This is not claimed as a new full 29-test run.

Reported scoped lib/tests Clippy passed in 3m02s; format/complexity checks are
also reported green. No saved C1 logs/receipts are available under target;
copied P3 logs were not accepted as C1 evidence. Reported execution handles
are 29000/90586, 95327/43066, 31479, 22265 and Clippy39866. Final CI/coverage,
release provenance and formal workload correctness remain outstanding.
Reviewer checks: actual clean head/base diff, schema no-drift and whitespace
pass. Python/runtime/benchmark code is unchanged; no performance run exists.

### Blocking Issues

No source-stage must-fix remains. Paired performance and exact-head required
CI/final review remain merge gates. The harness proposal is not execution-ready
until its invocation/sample/validator unit is explicitly resolved below.

### Implementation and Ownership Review

`CandidateMap` keys invalidate chunks for the current transaction, independently
of checkpoint dirty tracking. Reuse additionally requires the same aggregate
schema identity, first immutable key identity and row count. Append-only group
ordinals retain that proof; growing tails/new groups rebuild the affected
chunk. Schema changes rebuild conservatively. No committed array is mutated.

The cached values are the existing `GroupAggregates::result` raw readouts,
covering generic aggregates as well as inline integer SUM. Projection, HAVING,
ORDER/LIMIT and original unsupported/volatile admission are unchanged. No SUM-
only correctness assumption or prepared-row/delta-output contract is added.

The replacement cache belongs to `Transaction`; the original compact path
still emits before `native.commit`. Failure, cancelled preparation or dropping
a suspended emit drops the candidate. The next accepted update uses committed
values/cache. The test-only work counter correctly records attempted updates;
the committed QuotaLedger is the rollback witness. Those are distinct metrics.

The cache is enabled only for an engine pool reporting `MemoryLimit::Infinite`.
Finite/unknown pools retain the original mandatory allocation/error path and
no acceleration claim. `scratch.new_empty` shares the original registration,
including when disabled; no extra engine consumer is registered. Key-column
caching and compact-state accounting stay on their existing paths.

Metadata is reserved before bounded exact-capacity vectors/owner metadata are
allocated. Raw-array charge comes from the existing paid ENGINE output charge:
`stage` checks remaining transferable credit before `split`, so expected cache
refusal does not panic or become a query error. Reused clean chunks retain their
single original lease; their exact charge is deducted from the next output
reservation. A failed optional allocation or entry that cannot fit remains an
ordinary rebuilt column. The old and prepared cache can share chunk owners
without cloning or double-charging their reservations.

The private8MiB cap includes each cache view's metadata and array charges;
installed and prepared replacements may co-retain old/new versions until
commit. It is not a job-wide8MiB or process-RSS limit. Exported Arrow ArrayRefs
do not carry the private cache lease after its runtime owner disappears; the
API note explicitly preserves this existing engine-accounting boundary rather
than promising last-external-buffer accounting. Restore/reset build cold
caches. Codec, descriptor/fingerprint/profile, original logical-charge formulas
and public schema are unchanged. No unmerged D1/D2 carrier is included.

### Test Coverage and Documentation Consistency

Source tests cover the actual closed10,000-group/native sparse and dense entry,
all full snapshots against an independent SUM model, immutable held outputs,
key ordinals, clean-array identity, generic readouts, postprocessing, tail/
empty/schema changes, emit refusal/drop, cancellation, restore/reset, volatile
fallback, finite/unknown behavior, metadata refusal, single shared charge,
capacity cap and insufficient transferable credit. Decimal/ordered-LAST/codec/
key-output controls are reported separately; their logs were not invented.
The three existing fixture edits pass new private arguments or drop the new
owner with existing outputs; original zero-reservation assertions remain.

The decision note and streaming guide describe the implemented raw-array
scope, engine eligibility, per-view cap/co-retention, ordinary fallback and
cold restore accurately. Full snapshots remain the public output contract.
No new public API/dependency/style exception or schema drift was found.

### Harness Proposal and Recommended Next Role

Route next to **cf-performancer** to freeze the minimal two-case adapter and
its correctness/report unit before execution. Preserve the existing scheduled
SQL inventory and validators. Resolve how one fresh `release_pairs` invocation
relates to the scheduled SQL validator's twenty samples/report; do not weaken
the validator or silently impose an unplanned nested twenty. Pin the inner
count/report semantics, identical baseline/candidate harness, all full-snapshot
oracles, normal resource/provenance guards, two-round/ten-ABBA-pair contract and
one inclusive600s budget, with builds separately estimated. This review neither
requires a full benchmark before source critique nor approves an unresolved
proposal as runnable.

### Verdict

**Comment Only — code/static readiness cleared.** No source must-fix was found
against the accepted known-Infinite-engine scope. Reported-test/raw-log limits
remain explicit. Formal gain, compatible build/fixture/adapter evidence and
green exact-head CI/final specialist review are not yet established. Stop here;
no additional test, build, timing case or remote action was authorized by this
review.
