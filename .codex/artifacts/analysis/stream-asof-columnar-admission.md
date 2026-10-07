# ASOF columnar admission

## Scope and design

Base: `764843e634ae1a1da7a5b010095c3017349a25f4`, including approved A3.
The [approved critique](../critiques/stream-asof-columnar-admission.md) and
[FR16](../specs/stream-join-asof-acceleration.md) authorize a narrow left
chunk-construction change. Existing admission identities, descriptor funding,
integer sequence storage, output ranges, retirement, and checkpoint layout 10
remain the contract.

The remaining cost is the per-payload borrowed identity/position `Vec` in
`PreparedLeftChunk::prepare_checked`, followed by repeated order/position
validation in `ChunkData::prepare`. This phase does not introduce columnar
sequences or shared output arrays; those paths already exist.

Eligibility requires all rows accepted, exact microsecond timestamps, one
native-width integer sequence, one scalar integer or UTF-8 key, strict
canonical identity order across records, and a first identity beyond the
actual committed left maximum. Empty, late, reversed, overlapping, duplicate,
composite, and string-sequence inputs retain the legacy path. The proof is
private and is carried through the existing inline/owned-worker boundary.

Inline left `AdmissionRef.key_index` remains zero. Chunk dictionary IDs are
independently assigned by the legacy first-occurrence key interner. The six
actual chunk capacities and owned time/sequence copies must remain identical.

## Test-first evidence

Two focused tests attach a test-only count at the actual borrowed-vector
construction and inspect completed admission on 1,024-row inline and
64,000-row owned-worker inputs. Actual RED command:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::asof::admission::identity_tests::columnar:: -- --test-threads=1
```

The command compiled successfully, then failed both assertions: 1,024 and
64,000 borrowed rows respectively, against zero (exit 101). Both fixtures
completed cleanup and checked pool reservation zero before those assertions.
Counts travel with the actual chunk, avoiding a caller-only thread-local
inference about workers. The implementation reuses the existing duplicate
probe's sorted/after-state proof, combined with zero duplicates. It adds no
second admission-order scan. The supported schema/all-accepted checks happen
before payload compaction, so mixed late input cannot become eligible merely
because the accepted payload is contiguous afterward.

## Verification and performance

The first GREEN build was interrupted at the parent-requested performance
quiet boundary (exit 130); it did not run tests. After the subsequent grant,
test-fixture compile issues were corrected without changing production state
types. Actual GREEN passed 15 new columnar tests and 23 directly affected
admission/CPU cleanup tests. The latter command skips the already passed new
module and unchanged parallel-right module. The eight ASOF property tests and
three boundary integration tests also passed. No full-workspace or complete
full ASOF module run was performed.

The new matrix compares admission identities, payload records/refcounts,
first-occurrence dictionaries, six actual capacities, integer raw bytes and
encoding-owner inventory. Cases cover 0/1/1,024 rows; 1/3/5/64 keys; short,
long, empty and non-ASCII strings; all eight signed/unsigned integer widths
and extremes; equal-time ties and record cuts. It preserves reversal,
composite and string-sequence fallback, mixed late input, restored/live
out-of-order boundaries and duplicate refusal. Full/projected/zero-column
payloads and oversized slices retain the same payload decision.

Same-epoch metadata and every segment byte/hash match the forced legacy
constructor after admission, repeated capture and restored continuation.
The same test then commits a real 512-row prefix on live and restored state,
checks the output sequence against `0..512` without sorting/deduplication,
and compares state/metadata/segment bytes and hashes after the cut. This
enhanced case passed its focused recheck (one test); the unchanged 15/23/8/3
groups were not repeated. A 4,097-row owned-worker
fixture compares refusal category, status, reservation peak and complete
refund at/below the original identity, payload, worker-descriptor and aggregate
install thresholds, plus a 32 MiB operator pool. Existing duplicate/workspace
precedence and new entry/final-commit retirement cancellation tests passed;
cancelled final retirement retains committed state and funded owners until
job drain.

Actual allocation counters cover caller `prepare_admission` future polls and
the constructor on the actual executing thread. The 64,000-row constructor
thread differs from the caller; the 1,024-row constructor runs inline. The
worker registry is primed before the paired allocation check. Counts below
are constructor allocations; inline constructor counts are already included
in caller counts and must not be added again.

| Input rows | Caller allocations, legacy / fast | Constructor allocations, legacy / fast | Constructor allocated bytes, legacy / fast |
|------------|-----------------------------------|----------------------------------------|--------------------------------------------|
| 1,024      | 104 / 103                         | 27 / 26                                | 47,508 / 31,124                            |
| 64,000     | 95 / 95                           | 27 / 26                                | 2,314,644 / 1,290,644                      |

The removed allocation contains 16 bytes per row: 16,384 bytes inline and
1,024,000 bytes on the detached worker. Retained constructor bytes are equal:
23,192 and 1,282,712 respectively. Allocation count stays bounded for this
64-key fixture; arbitrary unique-key input is not claimed constant.

Clippy initially found three fixture-only style issues; they were corrected
without suppressions. Final `cargo clippy --locked -p calc-flow --lib --tests
-- -D warnings` passed, as did `cargo fmt --all --check`, generated-contract
drift and whitespace checks. All Cargo commands used the shared root debug
target and two build jobs; no full workspace lint or coverage ran.

The raw whole-file Lizard command exits one for unchanged legacy
`ChunkData::prepare` (eleven). A body comparison against the sealed parent
checks 74 added/modified functions, with maximum eight and exit zero;
`target/issue363-a4/complexity.json` contains that inventory. No complexity
configuration or suppression changed. Generated contracts have no drift.
All owned native/build processes have exited and debug cache is returned at
handoff. Specialist review remains required; CI has not run for this local
source. No release build or timing has run. The planning
target of 100 ms per million rows is not an observed result. Sealed timing and
peak process RSS belong to a separately coordinated performance task.
