# Funded streaming Join Utf8 restore

This slice adds non-null Utf8 payloads and keys to normal managed V1 recovery.
The distinct private RestoreUtf8Work uses the existing owned native service.
Independent final source review gates publication; required CI, coverage,
Codacy and review resolution gate merge for every published head.

Declared Utf8 schemas select the new route before the four existing metadata,
schema, scalar-base and scalar-segment Work paths. Their layouts and fee bodies
remain unchanged. Complete two-base, base-plus-delta and nonempty delta-only
inventories reuse the checked outer framing and original fold/order rules.
A one-base incomplete inventory retains the existing fallback.

Input copies, verifier trace, fixed facts and initial reader/fold/install controls
are prepaid before construction. One accepted worker parses metadata once, then
runs all safe FlatBuffers verification and UTF-8 scans. Checked actual key/value
facts grow its independent workspace before reader construction. Each fresh row
reserves its own selected value bytes, native i32 offsets, scalar buffers, schema
and controls before copying. Offset and value Buffer wrappers both retain the
unique resident funding and retirement guard; empty values retain their controls.

The profile certifies uncompressed one-row IPC, zero null count, nonnegative
in-body i32 offsets and valid selected UTF-8. Nonzero source offsets are allowed.
Whole-values validation is a conservative eligibility test: the original reader
can accept invalid bytes outside the selected span, so such input returns to
Original rather than acquiring a new public error. Unsupported types, compression,
nulls and unproved metadata/layouts preserve the original reader and diagnostics.

Workspace separately includes variable framed-key bulk growth, folded key copies,
raw delta headers/seen identities, maximum value scratch, segment ordering and the
complete checked installation tail. Candidate live marks precede the final stop
check and state assignment. This is not a certificate for generic long-lived
key/index residency. No ingress worker, public API, checkpoint format, V2,
migration or terminal restore changes are included.

Healthy optional funding refusal releases partial owners and unused work credit
before actual cleanup and the validated Original handoff. It performs no second
accepted work or successful metadata parse. Cancellation and deadline propagate
through the existing checked decoder/fold callback and do not replay Original.
Prepared rows keep the complete funded workspace through caller installation.

The single Managed quote consumer restores one row per side, continues with three
actual matching pairs and rejects a distinct multibyte symbol within the time
interval. It checks retained state and cursors before continuation, then reads
symbol, sequence, timestamp and price bits after shutdown. Two readers share one
workspace independently paid from wire and schema; each copied row carries its
own resident lease. Four direct controls cover original-reader compatibility,
variable allocation/install overlap, last offset/value owners and refusal/stop
cleanup. Allocation totals are requested bytes for these controls, not RSS.

No performance measurement has run. The maintained interval recovery recipe is
only a proposed later direct-path measurement; its admission still needs actual
confirmation. Generic/nested, nullable, LargeUtf8/Utf8View, V2 and terminal scopes
remain separate implementation and source gates.
