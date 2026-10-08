# Funded Join V1 Boolean payload recovery

This slice extends certified managed V1 full-base, base-plus-delta and
delta-only restore payloads to non-null Boolean columns. The baseline is merged
main `376afb86898798c7e301a81f3f6765a24b54bec5`. The existing
[restore constructors and ownership](stream-join-funded-restore-segments.md)
and [Utf8 recovery](stream-join-funded-utf8-restore.md) remain the funding
boundary; existing Work/Decision fields, control fees and logical row charges
are unchanged.

IPC must declare Bool, with packed values occupying the checked ceiling of
rows/8 bytes within the body. High padding bits are valid. The fresh one-row
copy matches Arrow 58.3's original IPC writer: byte-aligned Boolean offsets
preserve the complete selected source byte; other offsets copy the logical bit
and mask unused high bits. It performs no temporary bit-slice allocation.
The fresh MutableBuffer requests 64 bytes for one visible byte and follows the
existing FundedBuffer/PayloadFunding lease. The external Buffer capacity is
not a measurement of that internal allocation.

Boolean key support is unchanged: public Boolean keys remain supported,
while the optional paid key profile continues choosing Original for them.
Actual null values still choose Original; a nullable declaration alone does
not reject non-null data. Existing schema, metadata, compression and budget
refusal behavior remains, including the original reader's diagnostics and
state-installation order.

The single Managed fixture captures a natural first-dirty checkpoint,
reconstructs the job in the same process, verifies retained rows and actual
reopened cursors, and keeps the watermark held through three real continuation
pairs. It compares complete timestamp/Boolean tuples from retained Arrow
output after shutdown. The original test passed those semantic and lifecycle
checks before failing at two reader entries with no independent workspace;
the later resident assertions did not execute in that RED.

Two direct controls cover full original/funded IPC and charge parity at bit
offsets 8/9, source immutability, actual requested allocations, last-value
Buffer/schema lifetime and real refund; plus valid padding, packed 8/9-row
reader boundaries, actual-null and Boolean-key Original behavior, and malformed
buffer error/no-install/input parity. The 8/9-row probes concern IPC packing;
V1 row recovery still requires one row. Exact execution results, current-source
lint and frozen source/raw hashes belong to the handoff receipt. Requested
allocation totals are not RSS or a measurement of the whole restore peak.

Final source review gates publication; required CI, coverage, Codacy and
review resolution gate merge for every published head. The maintained quote
workload has no Boolean payload, so no timing gain is claimed. Generic/nested
types, V2 codecs, migration and terminal recovery remain later work.
