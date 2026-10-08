# Funded Join full-base restore

The optional normal-managed V1 restore route now prepays fresh base input, complete
decode/installation workspace, and independently owned scalar row payloads. It uses
the existing owned native worker and original parser/IPC/fold/checking functions.
This is a finite full-base slice, not generic decoded-state funding or V2 support.
No timing or RSS claim is made. Independent source review and required CI gate
every published head.

The source baseline is merged main `3734eb86de32371f152e32fbfdea0899e42192e0`
(tree `1c7b0aad40f407b088730b2016720f46f8cb9031`). The original observed RED was
on the same tree at `3fd7dfbb`; its frozen sources and receipt remain unchanged.
Limited Design Approve is `37552e66`; the trace-only pinned source proof is
`c5d7021f5bb732221b671a7f51a032e006d61c325c39f276964d427c2763b7db`.

## Observed RED and scope

The exact Managed test restores three left rows and one right row from two
naturally compacted CFJOIN1 bases. Four dirty cuts followed by the next prepared
cut create those bases without changing thresholds, flags or metadata. One
same-process durable runner reconstruction reads both bases and emits the
window aggregate 3. This is not an OS-process restart or a complete pair-payload
oracle. The real current-WT test ran one case: 1 FAIL, 0 PASS. All four reader
observations had no decoded workspace. The resident assertion after that first
failure did not execute; only its four recorded payload observations are known.

The raw command/log/receipt and three source files are in
`target/issue363-join-restore-bases-v1/managed-red-02-*`. The compile-error first
attempt and its log are preserved and excluded from behavior RED. No unchanged
RED test was repeated while writing these constructors.

The useful optional route accepts the already certified fixed scalar schema
family and parameterized V1 counts/bytes. It has no new small production row,
column-name or IPC-byte ceiling derived from its four-row test. Only two actual
full bases, with no deltas, select this slice. All other histories retain the
original route. Bool, null masks, unproved types/metadata/compression/alignment
and Some("") timezone normalization remain original-reader cases.

## Private source mapping

- `RestoreBasesConstruction` contains original `SchemaConstruction` unchanged,
  then `OwnedInput` data, unused aggregate resident credit and workspace. It
  secures independent input/workspace/resident partitions before new copying.
- `OwnedInput` contains its fresh snapshot, two byte Vecs, key-index Vecs,
  per-row leases and name before its final input credit. Raw bytes are copied
  in at most 4096-byte parts; indices in at most 64-cell parts. Every partial
  state is inside this carrier across each existing copy-boundary yield.
- Per-row leases are split from the already paid aggregate and registered
  before submission. Partial close/cancel cannot create an unfunded payload.
  The existing metadata registration/bootstrap contract is reused; no new
  service, pool or consumer label is introduced.
- `RestoreBasesWork` contains an original `SchemaWork`, fresh input and complete
  workspace. It invokes the original metadata parser once, builds descriptors,
  then inspects every IPC before the first StreamReader. It reuses the exact
  original left/right decode/fold functions instead of replacing V1 semantics.
- `RestoreBasesDecision` is a distinct output type: UseLegacy, numeric validated
  metadata plus original rows route, or PreparedSides. Original MetadataWork,
  SchemaWork, their decisions, construction and fee expressions are unchanged.
  Their exact admission route remains the pre-acceptance refusal target.
- `PreparedSides` holds both row Vecs, descriptors and numeric metadata before
  its complete installation workspace. The observed-output install callback keeps this workspace through the original
  checked tail.
- `columnar/restored.rs` copies each certified one-row payload into fresh core
  buffers and a fresh empty-map schema. The new constructor reuses the actual
  FundedBuffer/Bytes owner implementation. PayloadFunding includes an independent
  credit Arc and actual retirement guard. Its schema/data precede credit/guard;
  a surviving ArrayData/Buffer retains that lease after chunk/work/wire drop.

## Checked payment terms

Let Nl/Nr be declared V1 counts checked against real segment lengths, Wl/Wr the
actual two input lengths, F the already certified schema field count and K the
framed scalar key bound (9 + timezone bytes + scalar width per key column).
Every sum/product/rounding is checked; arithmetic/Layout refusal is optional
fallback before any reader. Layout and constructor expressions are in
`restore_bases/inventory.rs` and `columnar/restored.rs`.

| Partition         | Concrete terms and surviving payer                                                                              |
|-------------------|-----------------------------------------------------------------------------------------------------------------|
| Original metadata | Original JSON/spec construction and original SchemaConstruction input/descriptor fees; unchanged bodies.        |
| Fresh input       | Wl+Wr, exact lease/index Vecs, name, two Arc<Vec>, fresh two-entry BTreeMap, ids and SHA strings, real types.   |
| Verifier          | Fresh default Message verifier depth64: trace<=128, Vec peak192*sizeof(ErrorTraceDetail), actual alignment.     |
| Decode workspace  | Both body totals plus conservative 63-byte allowance per batch, bulk metadata Vec growth, schema/array overlap. |
| Fold workspace    | Both StoredRow Vecs, original key growth/value temporary/Arc, identity/key copies, both fold and ID trees.      |
| Installation tail | Both expiration collect/sort/node peaks, RetainedRows Arc<Vec>s, row view/sorts, carried map/segment clones.    |
| Resident payload  | 64-byte MutableBuffer constructor per scalar, array/Buffer/Bytes owners, fresh schema, chunk/live inventory.    |
| Resident control  | Separate A(PayloadFunding), A(PayloadChunk), A(MemoryReservation), complete late metadata registration.         |
| Native protocol   | Actual WorkAdapter<RestoreBasesWork> and output size, observed cleanup controls; existing execution fees.       |

A(T) uses the actual sizeof/align and checked Arc header padding. Vec growth
uses pinned RawVec minimum and checked next-power-of-two old/new overlap
only for the byte-at-a-time paths. Bulk key extension and reader metadata resize
separately pay max(3K,8) for their respective checked byte bounds:
the old capacity before growth is below the required length, the new allocation
is at most twice K, and both may coexist. Installation separately pays one
maximum-side repeated-key encoding peak while all retained keys are still live.
MessageReader body from_len_zeroed uses exact Layout size; the extra 63-byte
term is a conservative allowance, not its actual allocation geometry.
Formatter Strings separately pay max(3L,2I), where L bounds complete output and
I is the exact literal-template length. Installation additionally pays the live
column formatter, original InvalidArgument field and fixed overflow message,
and the overlapping OperatorReason inner String, copied message and node name.
BTree node terms include an allocated root, 11 key/value slots, 12 edges and
header/alignment; FromIterator pays the original collect Vec and sort workspace.
Each fresh resident row has truthful physical rowID/time/offset0, one inventory
entry and live=false. Candidate mark_live completes before the fresh job stop check immediately
preceding one state assignment; no N-row loop or await follows that check.

The direct exact flatbuffers dependency names only the existing locked25.12.19
node, enabling actual ErrorTraceDetail sizeof/align rather than guessed words.
The only old helper edits are private visibility on wrap_buffer, column_controls
and fixed_width; their bodies are unchanged.

## Constructor and reader gate

Limited constructor review `be57ccf8` accepted the bulk-key, formatter and complete
installation-tail inventory. The scalar/null-free/aligned certificate excludes
unproved reader branches before StreamReader; their original acceptance and
errors remain available through the compatibility branch. Two diagnostic runs
preserved the missing-workspace failure while identifying an overly strict
validity-buffer inspector condition. Arrow 58.3 writes an all-valid bitmap even
when nulls is None. With node null_count zero, the original primitive reader
ignores that mask. The correction only verifies its nonnegative checked in-body
range; it preserves null_count zero and all value-buffer/type/alignment checks.
Limited leaf review is `afcadcaa`.

Production dispatch selects a separate RestoreBasesWork only after independent
funding. Pre-acceptance refusal drops fresh partial data/credits before the
original exact SchemaWork/MetadataWork route. Installed healthy-home closure
uses the existing actual-job-only cleanup wait, which requires actual attempt
and output credit refund before the original restore. Prepared output stays
paid through the installation callback; only then does the caller await cleanup.

## Observed focused verification

The original Managed RED remains unchanged. Managed GREEN03 ran exactly one
case with four real readers inside RestoreBasesWork on the maintained native
worker and four independent resident payloads. The test checks that each actual
workspace reservation differs from the descriptor and actual managed Local wire
credit identities. Its retained counts and window aggregate remain 3+1 and 3.
The current-WT compilation and binary dep-info are frozen with the observed
sources in managed-green-source-v1. The observed one-case result is 1 PASS,
0 FAIL; no full IPC, pair-payload, OS-process restart or performance claim follows.

Direct requested-allocation controls retain the full allocation-counter results.
Old objects can be released inside a measured guard, so a negative net count can
understate the new requested peak. The conservative proof additionally checks
bytes_total: every newly requested allocation in the guard is bounded by that
independent funded total. Inputs allocated before the guard were already paid
by their constructor partitions; the guard does not measure whole-process RSS.

The both-side copy/decode/complete-install guard requested 39,876 bytes total
(net maximum 13,939), covered by independent funding 165,973. The real second
next error guard requested 11,724 bytes total (net maximum 5,554; net current
-8,083), covered by 168,213; it returns the exact original extra-batch diagnostic.
A separate fresh resident constructor requests 1,703 bytes total, with net
maximum 1,399 and its own guard 2,343. These observations validate the selected
fixture and supplement the checked source inventory, rather than proving every
schema shape from samples.

The actual last Arrow Buffer outlives its rows, descriptors, wire input and work.
The pool then equals actual home + generation + that row's resident reservation.
Managed drain remains Pending until that Buffer drops; the schema Weak also
stays live until then. Abandoned work similarly keeps the actual workspace paid
while a native reader gate is live. Real cleanup, job drain and final owner drop
preserve exact pool-zero assertions.

Controls02 observed three PASS and one fixture failure. All final-cancel and
post-installed/gen0 healthy-home-close semantics passed before its final zero
assertion found 16,384 bytes. Its cfg(test) hook still captured the real home
owner: actual funding reported generation zero, attempt zero and pool equal to
home. The corrected exact control passes after dropping that target/hook before the
job; the original raw failure is retained. This was an ownership error in the fixture,
not evidence of a production refund bug.

No full suite, coverage, benchmark, V2 writer, migration, delta decoder or terminal
dispatch ran. Lasting key/index/container physical resident proof remains a later
gate; temporary installation workspace coverage is not that certificate.

## Local handoff

Seven unique focused cases passed: the real Managed consumer; four new direct
controls; the directly affected existing schema/V1 comparison fixture; and the
frozen V1 checkpoint fixture. The latter checks all five capture bytes and the
final compacted capture's restore/continuation. Its existing current reader,
caller immutability, checkpoint bytes, diagnostic and failed-state assertions
remain unchanged. The new route initially omitted its cfg(test) comparison hook;
the actual reader now receives that original hook. The fixture drops its live
operator before whole-job drain, matching managed actor cleanup order.

The scoped command `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
passed with an actual current-WT Checking line in 76.21 seconds. Separate lib and
libtest dep-info include every new restore module; libtest also retains current-WT
absolute fixture provenance. The preceding real lint failure is preserved, not
relabeled a pass. Its corrections use a private stack-only EOS enum, an extracted
pure reader helper, a Copy scalar Bounds container and cfg(test) local renames;
no suppressed lint, new heap allocation or enlarged old fallback type was added.
The original metadata/schema type/decision/constructor/execution fee bodies are
whole-file byte-identical after removing only the new dispatcher/module lines.

Target logs under `target/issue363-join-restore-bases-v1/` retain the exact command,
exit, duration and hashes for every run. Historical RED/diagnostic/compile/lint
failures and immutable constructor/Managed snapshots remain intact. Formatting,
whitespace, generated contracts and local complexity are separate static checks;
a local Lizard result does not establish the remote Codacy gate. Full regression,
coverage, cross-platform checks and final independent Source Review are CI/handoff
gates, not local pass claims.

## PR381 cancellation and refusal revision

The remote cancellation finding was reproduced with an actual owned native
ticket. Its first paid reader-entry hook cancels the same job; the original
worker nevertheless entered four readers. The new test first reaches the
Cancelled, unchanged-state and exact-refund assertions, then fails on four
entries versus one. It is a direct ticket fixture, not another durable runner
restart, and the entry hook does not prove the first reader completed.

The checked restore route now borrows the original GatherStop check for each
row reader and fold insertion, side/segment switches, and before/after sorts and
collection. The original route supplies a no-op callback. A reader/fold error
is dropped before a fresh stop check, so cancellation cannot select Original
and synchronously decode again. Healthy codec failures retain the original
diagnostic route. No Work/Decision fields, buffer ownership, construction or
execution fee expressions change. Standard library sorts and a single Arrow
reader remain synchronous operations; no whole-callback latency bound is claimed.

The revised observer counts every reader entry before checking only the first
entry's actual native context/workspace and cancelling. Its GREEN count is one;
this includes potential later native or Legacy entries. The ticket never installs
an Original decision dynamically; the no-replay error classification is also
supported by the worker's stop-check branch.

Windows CI's schema-refusal fixture failure was reproduced locally: the generic
attempt probe first denied the newly eligible full-base Work, after which the
original SchemaWork correctly constructed one descriptor. The fixture now has
one valid Utf8 payload column: SchemaWork remains eligible while the certified
scalar full-base constructor refuses that shape. All original copies=2,
parses=1, constructed=0, actual attempt-fee, checkpoint and refund assertions
remain intact; the production refusal route and its exact fees are unchanged.

The targeted revision has six unique observed PASS: cancel-during-decode,
schema refusal, the Managed consumer, both-side/second-next reader inventory,
last Buffer/abandoned worker, and frozen V1 bytes/continuation. Pure helper
extraction follows the initial GREEN to keep changed functions at local CCN<=8;
only directly affected cases are checked again. Current-source lint and static
checks are recorded separately with provenance. Prior raw failures and frozen
source bundles remain unchanged under `review-fix-v2/`; no full suite, coverage
or benchmark ran for this revision. Independent source review and CI gate the
next published head.
