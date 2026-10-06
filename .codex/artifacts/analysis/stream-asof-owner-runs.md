# ASOF prefix ownership aggregation (A2)

## Approved scope

Implement FR14 from the approved stream join/ASOF acceleration specification.
Matching and cancellation remain per selected row. Prefix bookkeeping uses
the maximal ordered runs already produced by A1. Each selected run adds one
batch count, groups selected key IDs, and groups canonical sequence allocation
IDs. Typed integer sequence columns own no encoded allocation and skip that work.

Retain the existing prefix maps, partial owner counts, inventory, gauges,
post-delivery installation boundary, journal, v3 state, and row-log layout 10.
No schema, binding, public API, or checkpoint format changes are planned.

## Allocation design

Copy only the selected run's key IDs into an exact-capacity `Vec<u32>` and
sort/group in place. Drop that scratch before creating exact-capacity sequence
allocation-ID scratch. Canonical sequences group by actual allocation address;
key and sequence contributions sharing an address still add separately to the
same removal count. Peak new scratch is at most eight bytes per selected row.
Tiny cuts allocate by the selected cut, never by retained chunk cardinality.

The existing prefix reservation remains 640 bytes per selected row plus 2,048
bytes and iterator heap storage. Allocation-counter tests must demonstrate that
prefix maps and scratch fit this unchanged reservation for unique keys and
tiny cuts. Matching/output storage keeps its existing separate reservation.

## TDD and verification

Prepared the first complexity test against all three production candidate
matching paths. Two nonoverlapping 256-row chunks with long string keys and
canonical string sequences must perform two batch, two key, and two sequence
owner updates. The existing rowwise path performs 512 of each. Test-only
counters record actual update sites.

Observed RED with the parent's explicitly granted root debug target:
`cargo test --locked -p calc-flow --lib matching_aggregates_batch_key_and_shared_sequence_owners_by_run`.
Compilation succeeded; the test failed on `(512, 512, 512)` updates versus
required `(2, 2, 2)` in binary matching. All prefix maps matched the independent
rowwise oracle before that failure. The shared target was released immediately
after the process exited.

Before production edits, cherry-picked the corrected A1 gallop implementation
(`aa0c9ece`, local cherry-pick `202d1679`). Source implementation now aggregates
all three production matching paths. Prepared independent rowwise comparisons
at every generated cut, overlapping chunks, partially consumed chunks,
composite identities, string/typed integer sequences, and one allocation shared
between key and sequence. The oracle sorts canonical identities gathered from
unordered storage, independently of A1's run iterator. Each generated cut also
compares exact journal changes, projected inventory, index length, and pool
workspace. A 1,024-row integer-sequence/64-key case requires `(1, 64, 0)` updates.
The legacy test representation retains rowwise bookkeeping.

Actual allocation-counter checks cover unique
keys and cuts of 0, 1, 2, 5, 17, 127, 1,024 and 10,000 selected rows, under the
unchanged prefix reservation. No measured timing improvement is claimed.

## Observed local checks

After the next exclusive shared-cache grant, touched all four edited source
files before compiling. The focused `operator::asof::finalize::owner_runs::`
filter passed all five new tests. They exercise 516 independent generated cuts,
41 cuts sharing key/sequence owner storage, all three candidate matching paths,
the 64-key integer-sequence case, and 16 allocation measurements against the
unchanged reservation.

One broader `operator::asof::` run with default test concurrency returned 291
passes and six failures. Every failure reported the established process-wide
infrastructure-credit ceiling (`ASOF gather: bounded process infrastructure
exhausted`). The parent confirmed this module already requires serial local
execution for its shared 32 MiB ceiling. The diagnostic retry used
`-- --test-threads=1 --quiet` and passed all 297 ASOF tests, including checkpoint,
owner-capacity, cancellation, resource-limit and cold-recovery regressions.
No ceiling, worker budget, unrelated probe test, or cleanup behavior was changed.
The parallel failure remains recorded; the serial result does not establish
the cause of the separate Linux CI cleanup failure observed on the A1 PR.

Initial package lib/tests Clippy identified 15 fixture-only findings: checked
integer conversions, borrowed arguments, and explicit single-range slices.
All were corrected without production changes. After another exclusive cache
grant, `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings` passed
and the focused five-test recheck passed again. The unchanged production ASOF
module was not rerun. All Cargo/test processes exited before releasing the
shared debug cache to the next owner.

Package formatting, whitespace, and generated-contract drift checks passed.
Final specialist review approved the source and artifact. CI and paired
release measurements remain unverified; no performance acceptance is claimed.
The implementer made no remote changes.

## Analysis follow-up

The run-owner visitor is split into batch, key and sequence helpers to address
Codacy's complexity finding. Key scratch finishes before sequence scratch is
allocated, preserving the original allocation inventory and update order.
All five focused ownership tests and package library/test Clippy pass. Final
specialist review approved this decomposition and the A2 changelog entry.
Paired release measurements must identify this revised production source.
