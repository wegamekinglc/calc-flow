# Issue 406: estimate identity workspace from typed offset ranges

This increment continues the complete issue 406 plan and its eight original
acceptance targets. It builds on reviewed parent `88922e13`, whose ordinary
product is `05c13fd9`. PR #421 separately corrects the compaction target report
and is merged as `1606e516`; its original during-compaction metric remains
unmeasured. Prior ON qualification failures and inconclusive results remain.

## Evidence and scope

The exact ordinary parent release is profiled on one ASOF case: 1M rows per
side, 64k batches, true OFF and 32 CPUs. One warmup and five fixed processing
windows complete within 10.382 of 180 seconds, including preflight, fixtures,
oracles, symbolization and owned cleanup. All six invocations pass their full
output oracle, lifecycle, cleanup and zero-checkpoint-file checks. Five windows
contain 549/574/553/536/561 samples, totaling 2,773; 2,522 have native frames.
All 1,654 frozen input hashes and 23 output hashes verify. The first launch
fails before plan freeze because PYTHONPATH is missing; correcting that import
path starts the sole acquisition. No observations are retried or removed.

Identity workspace estimation has 6.64% inclusive samples, with 2.70% self in
`identity_range_workspace` and 3.14% self in `variable_length`. These shares
overlap and establish no timing improvement. The current string range loop
dispatches the Arrow type and downcasts its array once per row. Exact per-row
alignment still requires visiting value lengths, but one typed offsets slice
can serve the complete accepted range. WSL2 host isolation is unverified.

## Requirements

1. Resolve Utf8/LargeUtf8 offsets once per accepted range. Read adjacent offsets
   directly while retaining every row's fixed charge, 64-byte alignment,
   blocked-string framing and retained sequence-copy multiplier.
2. Preserve exact workspace totals, error categories and failing node names;
   retain empty-range, sliced-array, null-buffer and fixed-width behavior.
   Retain all reservation lifetimes, late-row selection, state limits,
   cancellation boundaries, checkpoint formats and immutable caller buffers.
3. Record a focused complexity test failing before implementation. Reuse the
   existing range-parity test registration, require scalar length dispatches
   bounded by column count, and compare all selected subranges against the
   unchanged scalar historical charge oracle for both string widths.
4. Run focused workspace and directly affected admission checks, scoped Clippy,
   formatting, complexity, schema-drift and whitespace checks, followed by
   specialist review. Full CI and coverage remain separate requirements.
5. Any later timing comparison freezes exact ordinary parent/candidate builds
   and a fixed proportional selection before execution: two rounds of ten
   adjacent AB/BA pairs per case, one warmup per arm per round and an inclusive
   600-second budget. Builds are separate. Preserve every stopped or
   inconclusive result; report allocation and RSS evidence required by issue
   406 for admission work. A subset cannot establish all-eight acceptance.
