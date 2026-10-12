# Issue 406: ASOF right columnar admission

Implementation base: `8ee5b71b7b862d4e44bd0be390d182cf950d17f7` (PR #409 merged).
Rebased onto release main `8ce5380bd6b6704473cba4ca1083820dd1059e42`
without conflicts; the release changes versions/docs and do not alter this path.
The overall eight-target performance goal is paused at the user's request.
This increment completes the already-started ASOF admission change; the next
explicit task is diagnosing the morning benchmark failure.

## Behavior

Eligible right batches retain typed time/UInt64 sequence columns and remapped
dictionary IDs instead of a canonical identity tuple for every accepted row.
The input must be globally strictly ordered, contain no discarded late rows,
use one Utf8 key with inline canonical encodings, and follow each touched key's
actual retained maximum. The maximum includes payload, ordered identity-only,
and general identity history. Unsupported and overlapping inputs use the
existing validation path.

Installation preserves bucket copy-on-write preparation and capacity reservation,
then appends typed values and payload references. Integer state remains little
endian. State counts, owner accounting, checkpoint index lengths and delta
journals include columnar rows even though the generic descriptor vector is
empty. Delta journals reconstruct canonical identities only when needed.
There is no public API/format change and no new default memory limit.

## Verification

The focused RED on base main built successfully and failed with 4,097 canonical
right-row descriptors against the required zero. The first module attempt used
the default parallel test setting and exhausted existing native-worker process
capacity; it aborted and is not a passing result. Subsequent module checks use
`--test-threads=1` without changing any runtime resource limit.

The serial module run passed 341 tests. Added controls compare forced legacy
with the columnar path for independent dictionaries, empty and sliced record
batches, UInt64 extremes, ties, restored and unordered resident state, late/wide
fallbacks, exact output and checkpoint bytes, accounting, explicit budget refusal,
duplicate priority and cancellation refunds. The actual delta test records three
journal changes, captures epoch 2 after the base at epoch 1, requires two frames,
and verifies exact replay outputs. Its intermediate test failures exposed reuse
of epoch 1 (which correctly rebases) and a retained old snapshot in the test;
both fixture lifetime issues were corrected.

The final eight focused direct-admission tests also pass, including held-bucket
COW and nullable payload projections (left-only, right-only and empty output),
with full and sliced inputs and caller-owner checks. All-feature core and test
Clippy passes in 49.78 seconds; formatting, generated-contract and whitespace
checks pass. Specialist source review approves with no remaining code blockers.
No unit test registrations were added; the generated all-feature harness remains
at 2,048 tests. CI and performance acceptance are separate outstanding gates.

PR #411 Codacy identified two complexity findings. Splitting record preparation,
key remapping and retained-order proof into focused helpers preserves validation
and cancellation order. The differential test now delegates seed/restore and left
probe setup to focused helpers. The final eight direct tests pass again in
8.63 seconds and all-feature core/test Clippy passes in 26.64 seconds; formatting
and generated-contract checks remain clean.

## Performance evidence and complete target inventory

Earlier OFF profiles attributed 44.62% inclusive samples to ASOF admission,
20.79% to identity construction and 21.96% to installation. These regions
overlap. The preceding dictionary-cache increment measured only about 3% gain.
The current structural change has no timing claim before paired measurement.

Main ASOF values below come from the audited `59518ad6` ordinary release whose
production sources equal current main `8ee5b71b`; its preserved source diff
contains test-only changes. Other values are historical main-equivalent evidence,
not fresh measurements of this candidate. They remain separate from acceptance.

| Workload                        | Main evidence                            | Current candidate                       | User OFF target               |
|---------------------------------|------------------------------------------|-----------------------------------------|-------------------------------|
| Sealed Join 1M, 1 / 32 CPU      | 103.33 / 93.10 ms (historical)           | Not remeasured; Join source unchanged   | ≤60 / ≤15 ms                  |
| Sealed Join 10M, 1 / 32 CPU     | 1094.44 / 962.35 ms (historical)         | Not remeasured; Join source unchanged   | ≤600 / ≤150 ms                |
| Retained Join 4M / 1M           | 4.467× (historical)                      | Not remeasured; Join source unchanged   | ≤4.4×                         |
| Join capture / restore per row  | 0.186 / 0.437 µs (historical)            | Not remeasured; Join source unchanged   | ≤0.1 / ≤0.2 µs                |
| Join handler during compaction  | 1.049× steady (historical)               | Not remeasured; Join source unchanged   | ≤2× steady                    |
| ASOF 1M, 1 / 32 CPU             | 243.45 / 277.23 ms (previous paired)     | Pending focused paired measurement      | ≤100 / ≤30 ms                 |
| ASOF harness, 1,024-row batches | 488.66 ms over operator (historical)     | Unmeasured; compatible tracing required | ≤2× measured projection floor |
| Join full 1,024-row callback    | OFF 91.89; enabled 89.83 µs (historical) | Not remeasured; Join source unchanged   | ≤50 µs                        |

Checkpoint OFF, LOW and ON remain separate. OFF ASOF targets are 100 / 30 ms;
the existing relative throughput goals imply LOW 111.11 / 33.33 ms (90% of OFF)
and ON 133.33 / 40 ms (75% of OFF). These are goals, not measured results or
achieved acceptance. No full eight-row acceptance is claimed.
