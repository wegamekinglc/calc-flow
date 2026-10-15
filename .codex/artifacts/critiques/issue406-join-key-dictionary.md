# Issue 406: Join persistent key dictionary - Critic Critique

## Target

- Spec: [issue406-join-key-dictionary.md](../specs/issue406-join-key-dictionary.md),
  frozen at `1901e45824530ec6683682a712addee1073e79f2`.
- Committed source inspected: `bb142320062995aeb318ad406c901288537fd766`.
- Domain: [Introduction: basic vocabulary](../../../docs/introduction.md#the-basic-vocabulary)
  and [supported boundaries](../../../docs/introduction.md#supported-boundaries).
- Scope: one bounded design critique. No source changes, builds, tests,
  performance measurements, remote writes or implementation approval.

## Verdict

**Proceed with caveats.** No concrete compatibility break or unimplementable
requirement was found in the specified increment. The existing ASOF state
container is not a suitable unchanged implementation of the shared identity
boundary. The reviewed phase 1.1 integration head and independently reviewed
codec prerequisite remain implementation gates; this critique does not approve
either provisional source or a phase 1.2 implementation.

## Findings

### Blocking Issues

None for the frozen proposal. The caveats below must be resolved in the
implementation evidence and final source review, rather than treated as
permission to weaken the requirements.

### Significant Concerns

- **Share identity ownership without sharing wire encoding or rank.** FR1,
  FR7, FR12 and AC12 support a reusable identity/lifetime boundary, while the
  scope explicitly leaves ASOF unchanged. That boundary must not assume Join
  V1 frames are ASOF canonical identities or sortable ranks. Join's
  [V1 encoder](../../../crates/calc-flow/src/operator/join.rs)
  (`encode_join_key_columns_v1`/`append_key_block`, lines 7278-7313) stores
  tags, lengths and little-endian values. ASOF's
  [identity validator](../../../crates/calc-flow/src/operator/asof/identity.rs)
  (lines 4-35) pins non-null Arrow row encoding; its
  [typed comparison](../../../crates/calc-flow/src/operator/asof/identity_compare.rs)
  (`encoded_value`, lines 119-130) uses sign-adjusted big-endian integers.
  Moreover, [ASOF Entries](../../../crates/calc-flow/src/operator/asof/state/key_dictionary/entries.rs)
  (`swap_remove`, lines 186-197) renumbers a moved entry, and
  [RightState](../../../crates/calc-flow/src/operator/asof/state/key_dictionary.rs)
  owns buckets and three expiration heaps. Those facts rule out using that
  container unchanged for stable Join handles observed by workers.
  - **Suggested resolution:** Final review should identify the actually used
    identity/owner/lease boundary separately from the Join codec and time runs.
    It should explicitly record that equality, canonical bytes, byte ordering
    and ASOF checkpoint formats remain operator-specific. No ASOF migration,
    unused adapter or generic public framework is needed for this increment.

- **The optional-grant exception needs a precise allocation and workload
  inventory.** FR9's batched SQL route is implementable without adding a heap
  dictionary: committed
  [retention preparation](../../../crates/calc-flow/src/operator/join.rs)
  (`retained_rows`, lines 5111-5142) already constructs legacy canonical row
  owners when native keys are absent, and
  [FramedKey](../../../crates/calc-flow/src/operator/join/columnar.rs)
  (`From<Vec<u8>>`, lines 243-249) explicitly represents an unfunded legacy
  owner. This is existing storage behavior, not a newly funded dictionary.
  FR14's phrase “every new ... canonical byte owner” must therefore be read
  with the expressly preserved FR9 compatibility exception; it cannot both
  impose a new mandatory grant on that legacy owner and preserve the same
  zero-headroom acceptance. The
  [zero-headroom control](../../../crates/calc-flow/src/operator/join/tests/sql_key_scratch_tests.rs)
  (`paid_key_pressure_case`, lines 727-817) proves SQL matching acceptance and
  release; it does not execute the retained-row commit after that match.
  - **Suggested resolution:** Name the old canonical allocation sites that
    remain exempt, count their framing work, and distinguish optional
    dictionary refusal from ordinary type/nullable/multi-record dispatch.
    Carry the existing SQL pressure control and add a focused production
    retention/refusal control if claiming end-to-end retention parity. Cite
    an actual maintained suite case that exercises each retained row-wise
    exception, as required by issue 406 section 6; naming a normal funded
    workload or a unit test alone does not establish that suite coverage.
    If no such case exists, record that gap and supply a bounded variant in
    the maintained harness before claiming acceptance. This does not require
    timing measurements during implementation. The exception must not grow
    into an unfunded handle table, lease owner or row adapter.

- **Persistent canonical ownership changes credit retirement.** FR6, FR7,
  FR13 and FR14 correctly require reuse of one canonical owner across batches.
  The existing batch interner instead attaches all its keys to a shared
  reservation and splits only aggregate scratch credit in
  [ProbeKeyInterner::finish](../../../crates/calc-flow/src/operator/join/native_lookup.rs)
  (lines 246-288). Its side index can refresh an equal key's owner from a
  surviving row in
  [NativeIndex::refresh_key_owner](../../../crates/calc-flow/src/operator/join/native_dictionary.rs)
  (lines 313-320). Copying that pooled-owner strategy into a persistent
  dictionary can pin every expired batch key's credit behind one surviving
  canonical key. The maintained
  [owner-expiration control](../../../crates/calc-flow/src/operator/join/tests/j2a_dictionary_tests.rs)
  (lines 444-483) already rejects that lifetime coupling.
  - **Suggested resolution:** State which reservation owns each surviving
    canonical allocation and which owns the handle vectors/table backing.
    Adapt the owner-expiration control to a cross-batch shared canonical
    owner, include pending history and actual worker result release, and
    observe intermediate retirement as well as final pool zero. An immutable
    byte-only history owner may keep its own allocation funded after its
    slot is recycled; it must not keep unrelated scratch or the live slot.
    Keep a leased retired identity distinct from a live matchable entry when
    re-retaining the same value. The spec's delayed-reuse option is sufficient;
    generations are necessary only if the implementation chooses earlier reuse.

- **Separate restore cancellation from optional acceleration refusal in the
  acceptance assertions.** The state section requires a valid restored
  candidate to remain usable through SQL when an optional dictionary build is
  refused, while AC11 says to “Cancel/refuse an optional dictionary rebuild
  and preserve prior state.” Those are compatible only when “prior state” is
  tied to the relevant operation. A cancelled/invalid restore must preserve
  the old installed operator; a successfully validated and installed SQL-only
  replacement must contain the restored rows, not the old rows. A later
  refused re-enable attempt must preserve that installed restored state.
  Historical validation must also remain independent of live-key coalescing:
  [checkpoint row validation](../../../crates/calc-flow/src/operator/join/checkpoint_v2/row.rs)
  (`validated_key`/`decoded_key`, lines 85-109) checks the payload against
  historical bytes before any live dictionary reuse.
  - **Suggested resolution:** Give these three paths separate assertions,
    including different old/candidate rows, writer/history identity and pending
    operations. Retain tombstoned-key corruption rejection and full V1 fee
    checks. This is a clarification for the author and implementer, not a
    user-facing algorithm decision or a reason to change checkpoint layouts.

### Minor / Style Notes

- FR4 and the handoff ask the critic to choose concrete table/encoder
  representations. The critic can reject unsafe choices; the implementer
  owns the concrete representation within these constraints. Change that
  attribution without requiring another user decision.
- Reusing the current synchronous hashing/equality helpers does not by itself
  satisfy FR15: [BorrowedKey](../../../crates/calc-flow/src/operator/join/borrowed_key.rs)
  (`visit`/`CanonicalStream::write`, lines 63-80 and 99-109) has no cooperative
  yield inside a wide value. The focused tests must exercise a long single
  value and a forced collision chain, not only many short rows.

## Axis Check

- **Correctness:** OK at the requirement level. Batch immutability,
  admission coordinates, exact canonical equality, full fees and emission
  order are explicit. No new engine, provider or expression evaluation exists.
- **Hidden assumptions:** OK with the two recorded prerequisite heads.
  Multi-record, sparse, sliced and nullable physical input cannot select an
  ordinary legacy route. Schema equality is an existing compile contract.
- **Missing edge cases:** Empty/probe-only input, overflow, collisions,
  cross-side last-reference eviction, active leases and historical corruption
  are covered by the acceptance plan. Add the precise refusal/retention and
  restore assertions described above.
- **Backwards compatibility:** OK under the unchanged wire/schema/fingerprint
  contract and codec prerequisite. The accidental Int8/date failures are
  already assigned to that prerequisite and are not a fresh blocker here.
- **Performance:** The structural counters are testable. This proposal supplies
  no speed, RSS or eight-target acceptance result. Rebuild work and surviving
  leases must be counted separately from ordinary key processing.
- **Surface and ergonomics:** OK. The new handles are private; unknown keys and
  typed exhaustion/refusal paths are explicit. No public API naming decision.
- **Test plan:** Strong and not satisfiable by a no-op dictionary: it requires
  real shared aliases, ordered match parity, actual worker gates, exact fees,
  strict recovery and allocation observations. Suite coverage of the remaining
  legacy exception is still evidence to provide.
- **Risk and scope:** Appropriate split. Preserve phase 1.1 offset-charge work,
  phases 1.3-1.5, later ASOF/runtime work and all eight original targets.

## Counter-Proposals

Use a small, actually consumed crate-private identity/ownership component with
an operator-specific codec boundary, rather than generalizing ASOF RightState.
Keep stable leased identities separate from side-local time runs and byte-only
history owners. Prefer the already permitted delayed slot reuse unless evidence
justifies generation-checked earlier reuse. These are bounded implementation
directions, not an alternate specification or an ASOF migration plan.

## Questions for the Author

- Which exact retained legacy allocation sites and maintained suite cases will
  the FR9 exception inventory name?
- Which canonical-owner and worker/history leases retain which reservations
  after a batch, entry or namespace retires?
- Will AC11 explicitly distinguish cancelled restore, successful SQL-only
  restore and a refused later acceleration rebuild?

Resolve these in the author/implementation handoff and evidence. They need no
additional user permission. Final specialist source review and required green
CI remain separate gates.
