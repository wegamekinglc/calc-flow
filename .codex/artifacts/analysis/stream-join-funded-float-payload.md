# Funded Join V1 Float payload recovery

This slice extends certified managed V1 full-base, base-plus-delta and
delta-only restore payloads to Float32 and Float64. The baseline is merged
main `a07e39d2f4ff35c92d82d2ee7da14e769bce5edc`. Limited Design Approve
`23dded98` applies; the existing
[restore constructors and ownership](stream-join-funded-restore-segments.md)
remain the funding boundary.

Restored payload width is four or eight bytes. IPC FloatingPoint precision
must match SINGLE or DOUBLE exactly; HALF remains unsupported by this optional
route. Key inventory uses the original integer/timestamp width profile, and
public Float keys remain rejected. Ingress payload copying is unchanged.
Existing null, metadata, alignment and compression predicates remain in force;
unsupported rows and optional budget refusals preserve original V1 recovery.
There are no new carrier fields, execution fees or logical row charges.

The single Managed fixture captures two first-dirty delta segments, restores
one row per side and checks actual source cursors before continuing. It waits
for three real Join output pairs before WM120, then reads both Float types'
raw bits from retained Arrow output after shutdown. Pair identities include
both timestamps; exact bits expose persisted NaN payloads and signed zero on
both sides without floating-point equality or arrival-order assumptions.

RED02 ran one actual failure at the missing workspace/native ownership
assertion after all value, cursor, lifecycle, wire and parse checks passed.
The later resident assertion did not execute. RED01 ran zero tests from stale
cache and is excluded. The identical Managed test then passed with two actual
owned-native reader entries, an independent 54,310-byte workspace and two
post-copy payload owners reporting 4,167 paid bytes each. Payload tuples identify
chunks, not reservation nonces. This is same-process durable reconstruction,
not an OS-process restart.

Three direct controls cover exact precision and key rejection, actual resident
allocation with V1 IPC/charge parity and last-Buffer refund, and null-payload
fallback with original reader/state/wire behavior. Their observed results,
current-source lint and frozen source/raw hashes are in the handoff receipt.
Final source review gates publication; required CI, coverage, Codacy and review
resolution gate merge for every published head. No maintained timing harness
directly exercises this profile; no performance or RSS claim is made. Generic
decoded types, V2, migration and terminal recovery remain later work.
