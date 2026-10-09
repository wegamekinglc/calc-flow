# Funded Join V1 nullable Float64 payload recovery

This slice adds actual nulls in nullable Float64 payload columns to the existing
certified managed V1 restore routes. Its merged baseline is
`4e5fd9c12def97cddd931abbf9741a9506c904a7`, with the same tree as the reviewed
Boolean amendment used for the initial RED. The existing
[Utf8 restore ownership](stream-join-funded-utf8-restore.md) and
[Boolean packing rules](stream-join-funded-boolean-payload.md) remain in force.
Work, Decision and input fields, control fees, key support and V1 framing are
unchanged.

The new actual-null certificate requires one row, nullable Float64, a null
count of one, one in-body validity byte with its low bit clear, and the existing
aligned eight-byte values range. Uncertified bitmaps, nulls in other types and
unsupported schemas retain the authoritative Original reader and its error
order. A null count of zero retains the reader's interpretation without adding
an all-valid NullBuffer.

Fresh values preserve the selected raw eight bytes, including hidden null-slot
bytes, signed zero and NaN payloads. Validity uses its own BooleanBuffer bit
offset: aligned copies retain the complete byte and high padding; unaligned
copies match the original one-bit slice. Values and validity each use a fresh
FundedBuffer carrying the same existing row funding and retirement guard.
Admission adds a checked 64 bytes per nullable Float64 field. Actual chunk
backing and selected-byte inventory add one 64-byte bitmap allocation and one
visible byte only when a bitmap exists. Frozen V1 NULL logical charge remains
one; it is separate from physical buffer funding. Existing control bounds cover
the two buffer owners and NullBuffer overlap, with direct requested-allocation
checks. External Buffer capacity does not measure its internal 64-byte request.

One maintained Managed fixture captures the first dirty checkpoint, rebuilds
the job in the same process, checks real retained rows and reopened cursors, and
holds the watermark through three collected continuation pairs. Its independent
Arrow oracle checks null validity, signed zero and a specified NaN bit pattern
after shutdown. The semantic, lifecycle, wire and parse assertions ran before
the RED's missing native workspace failure. The unchanged GREEN passed those
assertions and the later two post-copy owner checks; two reader entries share
one actual Work workspace.

Two direct controls cover raw hidden values, validity offsets 8/9, complete
original/funded IPC and logical charge parity, input immutability, requested
allocations and both last-buffer release orders. They also cover malformed
bitmap Original diagnostics, non-nullable schema-mismatch priority and actual
partial resident funding refusal without installing partially copied state. Exact results and
source/provenance hashes belong to the handoff receipt; requested allocation
totals are conservative request bounds, not RSS or a whole-restore peak.
Locked Arrow panics on empty or out-of-body bitmap slices; the malformed
control catches and compares that exact existing panic separately from returned
errors. It does not convert production panics into structured errors.

The initial local handoff contains three unique passing checks (Managed recovery
and the two direct controls) and strict current-source core lib/tests Clippy. The
earlier zero-test/cache-only runs, malformed-fixture expectation failure and
test-only qualification compile failure are retained separately; they do not
establish additional passes.

A review follow-up adds one scalar-key, two-full-base control through the real
managed metadata dispatcher. It checks actual NULL certification by the scalar
inspector, paid native readers and post-copy rows, literal key/time/validity/raw
Float64 bytes, full Original IPC and logical-charge parity, unchanged snapshot
bytes, and escaped last-buffer funding through final refund. This focused
coverage check passed with two paid native row readers, two paid post-copy
rows and one metadata parse, including its later last-buffer refund assertions.
It complements the existing Utf8-key Managed consumer and brings the handoff
to four unique passing checks; it adds no production behavior or public hook.

Final independent source review gates publication. Required CI, coverage,
Codacy and review resolution gate merge for every published head. No timing
gain is claimed. Other nullable types, generic/nested decoding, V2 migration
and terminal recovery remain later work.
