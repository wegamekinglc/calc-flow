# Funded Join V1 delta-only restore

Normal managed Join recovery extends the existing certified scalar paid route
to nonempty CFJDLT1 histories without named bases. It reuses
[the base-plus-delta constructors and ownership](stream-join-funded-restore-segments.md).
The baseline is merged main `2a0f1215e96b5a858db99d987745888f0707208d`, tree
`8c2a1f2df05a6e3c6a7ab9940cf3b68dd4dc07db`. Limited Design Approve `940233a5`
and merged-main interface review `236d4c7b` apply to this narrow extension.

## Input and payment boundary

The borrowed inventory gate accepts either nonempty histories with no named
bases, or the existing two-base history with additional deltas. Empty histories,
one-base mixtures and unsupported framing/schema/IPC still select the original
routes. Both declared sides remain checked; a side with zero decoded rows is
not excluded. There is no production row, segment or operation cap derived from
the test size.

After the complete existing frame scan, let S be its actual segment count and
n the number of named bases in that same immutable snapshot. The private
temporary delta count is checked D=S−n, with eligible n equal to zero or two.
Only the carried-delta tree and checksum-clone installation terms use D.
The conservative two-base installation allowance remains. The old n=2 history
therefore retains its exact S−2 terms; delta-only histories pay for every carried
delta. No saturation or test constant substitutes for inventory.

No carrier fields or Work/Decision types change. Original metadata/schema
fallback types, native control/execution formulas and existing actual size_of
terms remain unchanged. The existing cumulative upsert/raw-key/seen/order/input,
workspace and resident partitions still cover construction and the checked
installation tail. Budget refusal preserves the exact original routes and
actual refund wait. The current checked decoder, cancellation, error priority,
fold order, unique lease transfer and final assignment stop remain unchanged.

## Actual focused validation

One new Managed fixture admits one left row at 95 and one right row at 100,
then naturally captures its first dirty checkpoint: zero CFJOIN1 bases and two
CFJDLT1 segments. After cancellation and shutdown, one same-process durable
runner reconstruction verifies retained (1,1), actual source cursors
[(95,1),(100,1)] and held watermark before releasing WM120. Completion, both
shutdowns, aggregate [1], actual Local wire credit, one successful metadata parse
and two reader entries are required.

RED02 compiled the actual worktree and ran one failure: both readers reported
missing workspace and native ownership. All preceding state/cursor/lifecycle/
output/wire/parse/count assertions passed. The later resident assertion did not
execute. RED01 ran zero tests from stale cache and is excluded; its diagnostics
and generated dependency-order-only lockfile change were preserved separately.

GREEN01 ran the identical test bytes: one pass, runtime 0.04 seconds, 66.69 seconds
including build. Both real native readers held a 44,010-byte workspace distinct
from wire and descriptor reservations. Both post-copy payload owners reported
2,341 paid bytes. Resident tuples identify chunks, not reservation nonces;
independently prepaid lease-to-Buffer ownership is the reused source proof.
This is not an OS-process restart or an individual pair-payload oracle.

The directly changed n=2 resident-refusal helper derives D from its own scanned
snapshot. Its existing refusal/cancellation control ran one pass with all
original funding, cleanup, cancellation and state assertions preserved. The
other unchanged controls were not repeated. Current-worktree lint/static results
and exact sources/raw/provenance are recorded in the separate handoff receipt.

Final connected source review gates publication. Required CI, coverage, Codacy
and review resolution gate merge for every published head. This slice makes no
performance, RSS, generic decoded-state, V2 or terminal-recovery claim.
