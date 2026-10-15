# Issue 436: ASOF columnar merge completion

Baseline: main `718ab62ff5ad562994a694a20102b89991fc6b2b`, including #435.
The task branch is `feature/dal342-asof-columnar-merge` (DAL-342).

## Implementation

General admission keeps the canonical `(time, key bytes, sequence)` permutation
and builds per-key physical positions from it. Duplicate proofs merge those
positions with resident identities. Right proofs include payload, ordered
identity-only, and general identity-only histories. Left proofs reserve their
chunk-iterator scratch and use the existing scalar proof if that optional
reservation is refused, preserving rejection priority.

Integer-sequence right installation borrows each reserved bucket once and
merges backwards into its existing column capacities. Each displaced resident
slot moves once. Parallel workers borrow the same immutable column/permutation
owner and batch references; they produce detached buckets without allocating
canonical row descriptor vectors. Canonical sequence ownership continues to
use its existing explicit compatibility boundary.

Ordered and normalized unordered left chunks share the monotonic finalization
kernel. Repeated candidates use a per-key latest Arrow payload route. Fixed
width values remain in their funded Arrow columns; variable values keep their
row references. General identity history advances through a borrowed tree
range, rather than seeking that tree for every left row. Dictionaries beyond
64 keys use funded route vectors when at least four selected rows amortize
each slot. Legacy and short runs keep their generic route; cursor funding or
eligibility refusal retains binary-search finalization.

Route and cursor size assertions accompany the workspace bounds. Existing
output source registration, logical charges, prefix ownership, output limits,
successful-emission commit boundaries, and checkpoint layouts remain in place.
No public schema, binding, or generated API type changes are required.

## Contract boundaries

- Arrow Dictionary payload types are rejected by the existing flat ASOF public
  schema validator. This change tests that rejection and does not add a new
  payload type contract. Variable Utf8 payloads use the existing SQL/legacy
  differential and checkpoint/recovery cases.
- Canonical string/composite sequence owner accounting still materializes its
  compatibility rows. Integer sequences use the columnar path for general
  keys and unordered input, including parallel installation.
- Identity-only history remains necessary for duplicate detection after a
  payload is retired. This change neither expires that history early nor
  changes frozen logical state fees.
- Full CI, cross-platform coverage, and the 90% Rust coverage gate belong to
  the existing CI. Formal performance acceptance remains #443.

## Resource observation protocol

Two diagnostic workloads will run through the existing
`caller_and_actual_worker_allocations_scale_with_distinct_keys` unit test:
1,024 and 64,000 rows, each with 64 keys. These cover inline and owned worker
construction after removing the ordered generic descriptor route. The
candidate's Rust 1.88.0 debug executable compares actual caller and chunk
allocations with its forced scalar reference. `/usr/bin/time -v` records the
entire reference-and-candidate test process's peak RSS, without a per-arm RSS
claim. One invocation, 60-second execution budget, compilation excluded.
The separate merge regression asserts zero allocations after column reservation.
No release timing or speed claim is planned; these observations cannot establish
the ASOF 40/30 ms targets.

## Resource observations

The selected invocation completed within the 60-second budget and passed its
allocation, thread, cleanup, and scalar-reference assertions. Counts and bytes
below are allocator `count_total` and `bytes_total`, not retained-state fees.
Inline chunk measurements overlap caller measurements; the large chunk is
measured on its actual owned worker.

| Rows   | Caller count ref/current | Caller bytes ref/current | Chunk count ref/current | Chunk bytes ref/current |
|--------|--------------------------|--------------------------|-------------------------|-------------------------|
| 1,024  | 104 / 159                | 231,276 / 165,552        | 27 / 26                 | 47,508 / 27,124         |
| 64,000 | 95 / 150                 | 11,094,538 / 7,723,724   | 27 / 26                 | 2,314,644 / 1,286,644   |

The new caller has 55 additional allocations in both cases, while cumulative
allocation bytes fall. Actual chunk allocations stay at 26 for both sizes.
The whole reference-and-candidate test process peaks at **92,392 KiB RSS**;
this is neither a per-arm RSS comparison nor production release evidence.
The separate reserved-merge regression passed its zero-allocation assertion
for all supported integer widths and canonical history.

The resource command was:

```bash
timeout 60s /usr/bin/time -v target/debug/deps/calc_flow-b7e69e6a365d09b0 \\
  --exact operator::asof::admission::identity_tests::columnar::caller_and_actual_worker_allocations_scale_with_distinct_keys \\
  --test-threads=1 --nocapture
```

## Verification

All Rust commands used pinned Rust 1.88.0, locked dependencies, two build jobs,
and serial tests. The final affected unit coverage is 69 tests: 67 passed in
the admission/matching selection, followed by two passing focused rechecks
of the descriptor-free boundary expectations. Product code did not change
between those runs; the final changes correct two test classifications of
the new general columnar route.

```bash
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib -- \\
  operator::asof::admission:: matching_aggregates_batch_key_and_shared_sequence_owners_by_run \\
  --test-threads=1 --quiet
# Final selection: 67 passed; the remaining two expectations were corrected below.
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib -- \\
  test_direct_left_dictionary_boundaries_and_fallback_match_legacy \\
  test_direct_left_uses_actual_restored_maximum --test-threads=1 --quiet
# 2 passed.
CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow \\
  --test stream_asof_join_properties --test stream_asof_join_boundaries \\
  -- --test-threads=1 --quiet
# 8 properties and 3 boundaries passed.
```

The initial broader 124-test selection supplied 55 passing, unchanged
right-run, output-plan, probe, and dominated-history checks. Its two failures
were in the admission/matching selection rechecked above. That run reported
122 passed and 2 failed; it is not represented as a fully passing invocation.
The retained passing checks include the reserved-merge allocation assertion,
identity-history cursor parity, cancellation, and retired-prefix preservation.

Formatting, whitespace, schema drift, and Lizard 1.23.0 checks passed. Changed
production functions in admission, merge, cursors, and output planning stay
within the existing complexity limit of 8. No full workspace test, coverage,
cross-platform, or paired release performance result is claimed.

The final `CARGO_BUILD_JOBS=2 cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
and `cargo fmt --all --check` both passed on the delivered product and test source.
