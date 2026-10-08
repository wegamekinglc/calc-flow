# Funded streaming Join metadata validation

## Scope

Normal managed V1 Join startup uses an optional prepaid metadata validator on the
existing owned native CPU service. The caller keeps the original snapshot and
operator. Only fresh metadata, the expected Join specification and operator name
enter the worker. The successful fixed result supplies the original IPC/fold,
identity, key, charge and logical-limit checks without parsing metadata again.
The final assignment checks the actual job cancellation/deadline after building
replacement expiration indexes. Progress restoration and startup acknowledgement
keep their original position after state restoration.

The checkpoint writer, layout 1, semantic capability 1, fingerprint, public API,
terminal path, schema handling and generic decoded-state ownership are unchanged.
This is not V2 codec implementation, generic decode funding, an IPC acceptance
scanner or a performance measurement. It follows the immutable checkpoint
requirements in [the introduction](../../../docs/introduction.md).

## Design and constructor gate

The implementation follows proposal `197d4eab`, Design Review `47b5498a`,
constructor inventory `fa71637c` and limited constructor review `cd0ef2b9`.
Those were design approvals, not proof that arbitrary serde or Arrow constructors
are funded. This smaller profile makes unbounded parser error construction
unreachable before any new payload is copied.

The borrowed eligibility pass accepts one exact key per side, a compatible valid
specification, layout-1 numeric counters, booleans and known metrics. The name
and all expected specification text each have a 1,024-byte bound, checked before
content comparisons. Every top-level copied JSON unit contains fewer than 64
value/key headers and at most 4 KiB of copied text/control slots. An unconditional
yield and actual job stop checks precede each name/spec/JSON clone. Caller
schemas, Arrow buffers and original wire segments are never sent to the worker.

Unknown metric fields, missing fields, other key counts, incompatible specs,
unbounded text, unchecked geometry, missing initialized runtime, existing
pending preparation or denied optional credit select the original reader. They
are eligibility decisions, not public input rejection. Missing default prefixes
and missing/null maximum lateness preserve the old parser's behavior. No extra
IPC/schema preflight can reject a V1 stream that the old reader accepts.

## Checked physical funding

The initial reservation is obtained from the already configured Join runtime and
uses the explicit consumer `sql-incremental:stream-join-metadata`. It does not
borrow the wire-load lease or create another runtime pool. Every bound uses
checked arithmetic; overflow takes the original reader before cloning.

`P = 2J + C8 + 2S + V1 + B` bounds requested heap allocations:

| Component | Concrete constructors and overlap                                                                                                                                                                            |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| J         | Fresh JSON BTree nodes including an allocated empty root, copied key/string bytes and exact Value vectors; both independent input and parser clone coexist.                                                  |
| C8        | The original parser's eight-entry collect Vec and destination BTree; Rust 1.88 sorting eight items uses no heap scratch.                                                                                     |
| S         | Fields/final-spec key vectors, event-time copies, chosen/default-prefix overlap; 3 times all spec text, 18 default bytes and 10 String slots. Twice S conservatively includes the independent expected spec. |
| V1        | Sequential one-key BTreeSet validation, including the generic four-slot pointer Vec; one-item sort has no heap scratch.                                                                                      |
| B         | Shared registration and bounded label growth; caller scope/name/token and stop controls; fresh name and concrete construction/work/result/reservation carriers.                                              |

Pinned Rust 1.88 tree geometry is
`A=max(align(K),align(V),align(usize))`,
`Node=round(sizeof(usize)+2*sizeof(u16)+11*(sizeof(K)+sizeof(V))+12*sizeof(usize)+6*(A-1),A)`
and `Tree(n)=(n+1)*Node`. The extra root covers a cloned deleted-to-empty tree.
JSON map order uses locked serde_json 1.0.150 without `preserve_order`; typed Vec
construction uses locked serde_core 1.0.228's exact owned sequence hint. The one-key
profile's collect/sort bounds include the real Vec even when sorting needs no
heap scratch. The profile does not certify arbitrary parser errors, debug
escaping, nested schema controls or alternative feature/toolchain layouts.

Registration requires the existing small MemoryConsumer/shared-registration
bootstrap before `try_grow`: its bounded literal-label allocations are named in
B and destroyed on refusal. The rest of the payload/control constructors follow
successful prepayment. The caller-control split retains registration/name funding
until the last caller alias dies; it does not use an already refunded work item.
Moving this existing component to the longer-lived split leaves initial P
unchanged. Record aliases additionally declare their complete control footprint
through `OwnedCpuWork::control_bytes`. Existing execution fees pay work/output
boxes, cleanup markers and owned native attempt controls. No new service or
shared gather layout/fee is introduced.

These are logical reservation bounds on requested allocations, not an allocator
bookkeeping, thread-stack or whole-process RSS ceiling. An allocation sample
confirms the listed copy/parser overlap; static constructor geometry remains
part of the source gate.

## Ownership and failure handling

Construction fields place partial snapshot/spec/name before caller-control credit,
work credit and retirement. Every partial copy across a yield stays inside this
owned carrier. Complete work moves into the existing ObservedSubmission before
its first admission await. The existing observed path stores AttemptCleanup in
the operator immediately after successful attempt installation and before
`ensure_pool` can await. The private Gather change only re-exports ObservedTicket;
it does not change the cleanup protocol or native execution structures.

The worker invokes the original parser/compatibility function once. Its input,
specification and any temporary parser diagnostics die while work credit remains
alive. Success returns only numeric counters, metrics, ended and epoch. An
unexpected parser incompatibility/error returns fixed UseLegacy, and the caller
then reproduces the original parser's precise error behavior. No dynamic worker
error or schema descriptor is installed as restored state.

ObservedOutput holds actual work credit while the caller consumes the result and
builds/checks replacement state. The caller drops that output before awaiting
cleanup, avoiding a wait on its own escaped credit. Scope/stop and shared
registration remain prepaid in a separate actual caller split during this late
wait. Cancellation retains the external observer; original managed cleanup
waits for the actual native attempt and retirement. Reset does not replace the
configured pool or discard existing cleanup tracking.

Pool or healthy closed-home admission refusal first performs a fresh actual job
stop check. New work is really destroyed/refunded before the original reader is
used. Actual cancellation/deadline and non-budget runtime errors propagate.
There is no sleep, timeout increase, constant subtraction or silent second
metadata parse on the successful path.

## Local evidence and limitations

The actual Managed production startup test first failed because the original
restore supplied no independent metadata reservation. Its RED ran exactly one
test from this isolated worktree (exit 101). The first implementation GREEN ran
the same test once (exit 0), preserving the existing checkpoint/restart oracle and
observing one actual worker parser call. A private alias compile error is not a
behavioral RED.

The first four direct controls passed after correcting test-only teardown and
valid-empty-inventory fixtures. Measured copied data held 5,150 bytes; the original
parser added a 5,522-byte peak, giving a conservative 10,672-byte overlap under
84,939 bytes of independently observed credit. Their combined net allocation
was zero. This measurement starts at actual owned-future polling, not Tokio
runtime startup, and does not claim all process/control peak allocations.

Partial-copy drop checks hold credit and keep managed drain pending until real
carrier destruction. A gated native cancellation checks actual attempt credit,
noninstallation, drain Pending, home/generation/attempt attribution and exact
zero after true job/service teardown. Unknown nested metric JSON keeps the
original reader's acceptance, and original layout errors remain identical.
The historical failed fixture log is preserved and is not counted as a production
regression: an empty segment inventory is rejected by the original decoder;
TestService shutdown belongs outside Tokio; live job home credit is not a leaked
attempt reservation.

Final local checks:

- The eight metadata controls pass (8/8, 0.01 s): actual copy/parser funding and
  net release; partial-copy managed retirement; controlled native cancellation;
  original nested metrics/error compatibility; pre-copy and native-fee budget
  refusal; healthy admission-close fallback; final-stop nonassignment; actual
  late caller-control credit after work refund.
- The real Managed restart/independent-credit/worker-parse-once test passes after
  final dispatcher extraction (1/1, 0.05 s).
- Changed Legacy restart, original entry-ack order and restored Join status each
  pass as exact one-test checks from the current compiled artifact.
- The frozen V1 oracle passes: all five captured byte inventories match, and the
  final compacted capture restores and continues correctly. It is not a claim
  that each of the five captures was independently restarted by that test.
- The original Clippy report did not establish that the new metadata modules
  were checked. A subsequent cached return's dep-info still referenced another
  worktree. Invalidating only the two calc-flow check fingerprints exposed five
  current-source diagnostics.
- Final formatting, whitespace/contracts and source provenance are recorded
  alongside the frozen source receipt. The original Lizard maximum of 8 did not
  establish that Codacy's gate passed; their counting differs.

Codacy reported three profile functions over its limit: `expected_spec` (16),
`metrics` (9) and `value_shape` (10). The revision extracts their existing checks
and shape accumulation into helpers, preserving check order, default prefixes,
nullable lateness, bounded text inspection and the original fallback behavior.
The local complexity ledger includes `?` early returns and decision branches;
the required Codacy result remains a CI gate.

The current-source Clippy revision uses `map_or` for missing prefix defaults,
an `Option<ValidatedMetadata>` fixed-size result, a fresh `String::from(&str)`
name copy and the original-position caller `_credit` field. Tests borrow that
field into a named local reservation for their actual funding checks. It adds no
boxed payload or allocation. Fresh name construction preserves exact-length copying;
`clone_from` on an empty String could instead use amortized vector growth.
The fee inventory still uses actual `size_of` values and the caller credit stays
after its scope and stop fields. The changed eight controls and Managed restart
are rerun; unchanged V1, Legacy and entry-ack checks retain their previous scope.

Logs are under `target/issue363-join-metadata-validation-v1` in the isolated
worktree. Raw RED, compile errors and incorrect initial fixture results are kept;
none are rewritten as later passes. The one-key successful constructor profile
is intentionally finite. Other key counts, error grammars and shapes keep the
existing reader, so this is not a full metadata/IPC optimization certificate.
Full CI, coverage and cross-platform acceptance are not claimed by local checks.
No performance case was run for this safety tranche.
