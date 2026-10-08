# Funded Join V1 base and delta restore

Normal managed Join recovery optionally prepays certified scalar CFJOIN1 bases
and CFJDLT1 history on the existing owned native worker. The original metadata
parser, checked IPC decoder, fold order and installation tail remain authoritative.
Fresh payload owners retain independent credit through surviving Arrow buffers.
Independent source review and required CI gate every published head. This slice
makes no timing, RSS, generic decoded-state, V2 or terminal-recovery claim.

The baseline is merged main `a4c14e65ebcc6cd04015274b7fd30cdfe5c2be9a`, tree
`0d4f6dc3e282f605b2ea954a74f653c23d372648`. It reuses the certified scalar,
verifier, formatter, key and Buffer constructors from
[the full-base restore slice](stream-join-funded-restore-bases.md).
Limited delta Design Approve is `ca0c50ec`; the new constructor gate is
`d90f06d741f5a98b0bba2768a06bc24954ed38954fc833d3dee02ca640002470`.
The immutable constructor-source-v1 remains separate from connected validation.

## Optional profile and original semantics

The new route requires both full bases and at least one recognized delta. Counts,
wire lengths, IDs and arithmetic are checked against actual input and the configured
pool. The one-upsert test does not impose a production row, operation or segment
ceiling. The already certified scalar schema/IPC profile remains the eligibility
boundary. Unsupported names, framing, metadata, types, compression, nulls or
alignment retain the original reader and its existing acceptance and diagnostics.

The worker inspects every base and upsert IPC before the first reader, including
upserts later removed by tombstones. It calls the original checked decoder with
the same paid stop. Left folds before right; each side reads its base then deltas
in numeric epoch/segment-ID order. Duplicate identity is checked before applying
its tag or reading upsert IPC. Header key bytes, including arbitrary tombstone
keys, follow the original identity semantics. Final rows keep original IDs,
event times, charges and deterministic ordering. Carried segments keep exact V1
bytes and checksum strings; there is no writer, capability or fingerprint change.

Real cancellation remains an error: a decode/fold error is dropped and the same
stop is checked before selecting the original diagnostic route. Legacy wrappers
keep their existing no-op callback. Mark-live completes the owned candidate;
a fresh job cancellation/deadline check immediately precedes the sole state
assignment. No row loop or await follows that final check.

## Ownership and payment

`RestoreSegmentsConstruction` owns fresh partial input first, then unused resident
and workspace reservations, then the original schema construction last. Its
original guards therefore survive all newly constructed data and credits.
Independent input/workspace/resident partitions are reserved before copying.
The borrowed framing scan allocates no new keys, strings or verifier errors;
existing asynchronous boundaries bound it to 64 operation headers or 4096 ID
bytes between yields. Copying keeps partial segments in the explicit input carrier.

`RestoreSegmentsWork` and its decision are distinct types. Original MetadataWork,
SchemaWork, their decisions, construction and execution fees remain unchanged.
Optional pre-acceptance denial drops fresh data before refund and selects those
exact existing routes, with at most one accepted attempt and one successful
metadata parse. Installed healthy-home refusal uses the actual-job cleanup wait
and does not confuse attempt abandonment with job cancellation.

Let B be total base rows, U every decoded upsert (including overwritten/deleted
rows), Q=B+U per side, S all carried segments, W actual wire bytes, H all raw
header key lengths, Hseg the maximum per-segment sum and Hmax the maximum key.
All sum/product/Layout calculations are checked. No surviving-row count or
schema key width substitutes for these input-dependent terms.

| Partition | Concrete constructor terms                                                                                                                 |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Input     | All S exact-capacity wire/ID Vecs, Q leases, key indices, name, fresh map, Arc<Vec>, 64-byte checksums and actual carrier sizes.           |
| Reader    | Reused paid verifier trace and reader/schema/body terms for every Q row; unknown IPC selects the original reader before StreamReader.      |
| Fold      | Both cumulative Q row/key/fold inventories, raw header+identity+seen key overlap, per-segment seen tree and operation counts.              |
| Order     | S inventory/ordered Vecs, stable-sort workspace and simultaneous temporary segment-ID Strings.                                             |
| Tail      | Both expiration/retained candidates, repeated key validation, S carried references/maps and checksum clones, original diagnostic overlaps. |
| Resident  | Independent per-row fresh schema/buffers/chunk/live inventory, credit Arc, actual retirement guard and late registration.                  |
| Protocol  | Actual new Work/output layouts and observed cleanup controls; the existing native execution-fee calculation.                               |

The input map retains the fresh wire credit while the original caller snapshot
retains its own wire owners. No bare bytes_arc escape is introduced. Resident
leases are split and registered before dispatch, using cumulative Q counts;
unused leases refund after final fold. A surviving row receives one uniquely
transferred lease, whose actual Buffer owner retains credit and retirement.

Prepared output keeps both row vectors and descriptors before the complete
installation workspace. ObservedOutput::install retains work credit through
the original synchronous checked tail; the output is consumed/dropped before
awaiting the actual attempt refund. Slot removal alone does not prove refund.
Long-lived encoded-key/index/container resident funding remains a later gate;
complete temporary install workspace is not that certificate.

## Observed production RED and GREEN

The single real Managed fixture naturally produces two compacted bases from
three left and one right row, then adds a fourth left row and captures one
upsert delta. One same-process durable runner reconstruction verifies startup
Join retained counts (4,1) and source cursors (4,1) before releasing WM120,
then checks window aggregate [4], cancellation/completion and both shutdowns.
This is not an OS-process restart or a complete individual pair-payload oracle.

RED02 compiled the actual current WT and ran 1 FAIL, 0 PASS. The Local prepaid
reader observed two CFJOIN1 and one CFJDLT1 segment; parser/count/state/cursor/
lifecycle/output assertions executed before all five readers reported missing
decode workspace and native ownership. The later resident assertion did not
execute; baseline had only four base post-decode observations. RED01's stale
cache ran zero tests and is explicitly excluded, with its dep-info preserved.

GREEN01 compiled the actual current WT and ran the same exact case: 1 PASS,
0 FAIL, runtime 0.15 seconds (87.86 seconds including build). All five real
readers observe the actual owned worker context and a positive workspace
reservation independent of both descriptor and managed Local wire payment.
All five post-copy rows report a positive funded payload owner. That hook tuple
contains the chunk identity, not a reservation identity; the independently split
lease-to-Buffer ownership chain is a separate source fact and last-Buffer control.
The parser
runs once; all earlier startup/cursor/lifecycle/output assertions still execute.
The immutable managed-green-frozen-v1 manifest is `def7d3b3`, receipt `9614e9c4`.

## Direct controls and handoff

The four new direct controls cover numeric delta order/tombstone/raw keys and
V1 carry/error priority; cumulative requested allocation plus the real second
next error; last delta Buffer and abandoned worker retirement; and optional
refusal with real cancellation/final assignment stop. All four ran in direct-controls-01 on the actual current WT: 4 PASS, 0 FAIL,
runtime 0.03 seconds (127.80 seconds including build). The control source stayed
unchanged through the subsequent style fixes.

Allocation counters measure newly requested allocations. Releasing older
objects during a guard can make net-current negative and reduce net-maximum;
bytes_total is the conservative additional bound on all new requests. Inputs
allocated before the guard already have independent constructor funding.
The cumulative-Q5/final4 control includes a 2048-byte raw tombstone key, all
fold/order/carry copying and the full installation tail. It requested 53,539
bytes total, covered by independently reserved 208,455 bytes. The real delta
second-next error guard requested 19,705 bytes total against 187,055 paid bytes;
its net-current was -9,004 and net maximum 5,554. The error equals the original
extra-batch diagnostic, and state stays uninstalled. These observations
supplement the checked source inventory, not whole-process RSS or proof of
every supported schema from one sample.

Clippy01 returned a cached Finished-only result; both selected dependency
inventories omitted the new module and libtest pointed to the old WT. It is
excluded. Its log/dep-info are preserved, and only the two core Clippy success
markers were invalidated; dependency and ordinary test caches stayed intact.
Clippy02 actually Checked the current WT and failed on five production style
findings and the 116-line Managed test. The corrections remove two unnecessary
semicolons, use a direct tree function pointer, make an unused-self helper
associated, derive Copy on the existing scalar Bounds, and extract the unchanged
Managed owner assertions. No fields, layouts, fees or drop order changed.
Clippy03 actually Checked the current WT and passed in 86.71 seconds. Separate
lib and libtest dep-info include all new modules; libtest names the actual WT.
The affected Managed exact test passed after the observer extraction (runtime
0.15 seconds, 258.01 seconds including rebuild). The allocation/full-tail exact
case passed using that same just-built binary, with unchanged requested totals.
The other three unchanged controls were not rerun. Five unique cases are green.
Local Lizard reports 84 new functions, maximum CCN 8; this is not a remote
Codacy verdict. Format, contracts and whitespace checks pass. Full CI and
final connected specialist review remain the publication gates.

No full workspace, full coverage, benchmark, V2 writer, migration, generic IPC
expansion or terminal dispatcher is part of this change.
