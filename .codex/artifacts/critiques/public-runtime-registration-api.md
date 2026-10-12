# Public Runtime registration API - Critic Critique

## Target

- API note: `.codex/artifacts/api-notes/public-runtime-registration-api.md`
- Scope: "Registration Record Contract" and "Paired Snapshot Consistency"
  only. Names, signatures, alias removal, the private native method, and plain
  `dict` records are decided and were not reviewed.
- Line numbers are from baseline `8f890f28`, read before the rename landed.
  Nothing was executed; every finding comes from reading the cited lines.

## Verdict

**Proceed with caveats.** No blocking finding. The rename can go ahead as
specified. Three passages should be reworded before `cf-doc-writer` copies
them into `docs/` and before the pair docstring is written, because that is
the point where they become cross-repository promises.

## Blocking Issues

None: no wrong implementation, compatibility break, or acceptance failure.

Source check: every key, value type, and presence rule in the record table
matches the code (`pipeline.py:268-278`, `305-306`, `317-330`, `458-469`,
`486-507`, `510-528`; `udf.py:10-18`; native argument types at
`crates/calc-flow-python/src/config.rs:338-346`, `464-473`, `506-512`). The
pair bullets match `pipeline.py:546-554` and `capabilities.py:495-512`,
`563-572`. The two statements that do not hold as worded are in W3.

## Wording To Change Before Publication

### W1. `revision == len(records)` is a new promise the consumer does not use

- True today (`pipeline.py:550`), but not inherited. Published docs promise
  only the step: "advances once for each successful registry entry"
  (`docs/api-reference.md:284-289`) and a helper "normally advances by three"
  (`docs/python-api.md:668-672`). "Rises by exactly one" restates that in
  different words. The equation, "always holds", and "starts at `0`" are new.
- Open question 3 names two fix routes. Advancing the revision at the
  stream-proof step gives `revision > len(records)`, which contradicts the
  equation, item 8's "revision `3`", `docs/python-api.md:671`, and
  `python/tests/test_capabilities.py:447`. Making the helper atomic
  contradicts nothing (`_registration_lock` is an `RLock`, `pipeline.py:149`).
  Publishing the equation picks the atomic route without saying so.
- Studio compares `(session_id, revision)` for equality only
  (`run_manager.py:1290`, `1314-1316`; `models.py:73` requires `ge=0`). Core's
  other user stamps the pair on `AnalysisResult`
  (`docs/symbolic-api.md:112-113`, `133`). Neither reads a count. What the
  cache needs is the converse, equal key implies equal response, which the
  Limit says is not true for lifecycle fields.

Replace the pair docstring body (note lines 71-74) with:

> Return ``capabilities()`` and ``registration_snapshot()`` as one consistent
> pair. The records are the registrations that ``capabilities`` describes:
> each ``provider`` and ``scalar_udf`` record has one capability entry with
> the same identity.

Replace note line 187 with: "`records` are the registrations that
`capabilities` describes at `capabilities.scope.revision`."

Replace note lines 194-197 with:

> Across calls on one `Runtime`, `scope.session_id` is constant and
> `scope.revision` never decreases; a successful registration advances it as
> described under Runtime capabilities. Compare revisions for equality only:
> do not derive a record count or a number of registrations from one.

Item 16: delete "add `revision == len(records)`". The example's line 251
becomes `assert capabilities.scope.revision == 1`. Items 6 and 7 may keep the
equality as a regression pin. Alternative: keep the equation everywhere and
record in open question 3 that the fix is the atomic helper.

### W2. The `stream_lifecycle` paragraph is not a usable consumer rule

- Presence of the key switches `modes`, `finality`, `microbatch_invariant`,
  `deterministic`, `replay_safe`, `supports_static_inputs`,
  `partition_contract`, and `array_rules` (`capabilities.py:515-533`).
- The section promises replay is "equal on every listed key" (note 141-144),
  an equality that excludes the one difference that changes behavior, and it
  forbids reading the only record-level sign of that difference. "Do not
  replay it" is not actionable, because no public method accepts it. The real
  statement is about the record: public replay cannot reproduce it.
- Published docs already supply the vocabulary: public registration is always
  batch-only (`docs/python-api.md:636-647`), helpers attach a process-local
  proof (`649-659`).
- "Ignore keys" is too weak for a consumer that transports records. Studio
  pickles whole records (`run_manager.py:228`, `975-981`), so a private value
  crosses the process boundary unless it is dropped first.
- "Skip kinds they do not handle" is right for listing and wrong for replay:
  a skipped `kafka_decoder` is a decoder missing when the job opens
  (`pipeline.py:481-484`).

Replace the last sentence of the compatibility rule (note 149-150) with:

> Consumers read `kind` first and use only the listed keys. A consumer that
> lists registrations may skip kinds it does not handle. A consumer that
> replays them must treat a skipped kind as a registration missing from the
> target runtime, and fail or report it when the workload references it.

Replace note lines 152-157 with:

> **Replay fidelity.** Replaying a record through the public registration
> methods reproduces its identity, callable, ports, options schema, and
> context opt-in. For scalar UDFs and batch-only providers that is the whole
> registration. It is not for the `expression@1` and `symbolic_matrix@1`
> providers installed by `register_numpy` and `register_jax`: their stream
> capability comes from a process-local proof that no public registration
> method accepts, so a replayed copy is batch-only. To get them with stream
> capability on another runtime, call the same helper there instead of
> replaying their records. Do not present a replayed runtime as equivalent to
> its source unless the `ProviderCapability` entries of the replayed providers
> compare equal on both.
>
> Records can carry unlisted keys holding private process-local values; today
> `stream_lifecycle` on those two helper records is the only one. Unlisted
> keys carry no compatibility promise. Build each registration call, and
> anything you transport, from the listed keys only.

### W3. Two statements the source contradicts while a helper is running

- `pipeline.py:372` and `426` add a key to an existing record without
  appending (called from `array.py:1247-1257`, `1275-1285`). So the bullet at
  note 191-192 fails when a proof step lands between the two reads, and
  "Records are append-only" (note 196) is false as a plain sentence.
- Studio selects by identity and compares keys; it uses neither order nor a
  prefix relation (`run_manager.py:811-821`, `1290`, `1314`). The prefix
  sentence also promises that no future registry operation removes or
  replaces a record.

Replace note 191-192 with: "With no registration call, including a
`register_numpy` or `register_jax` call, running or completing in between,
`capabilities == runtime.capabilities()` and
`records == runtime.registration_snapshot()`."

The W1 replacement already drops the prefix sentence at note 196-197; delete
it even if W1 is not adopted. Keeping "returned in registration order" is
fine: it is cheap, true by construction, and makes replay deterministic.

## Other Caveats

- **Tolerant reader versus Studio (question 3): caveat, not blocking.** Record
  shapes on release day are today's (note line 78), so nothing fails that
  does not already fail, and both defects need an embedder-supplied runtime
  with a helper or Kafka decoder applied (`app.py:136-148`). But Studio's
  range `>=2026.10.12,<2027` (`web-ui/backend/pyproject.toml:10`) admits every
  later 2026 core release, and Studio's tests pin rejection of any unexpected
  key and any unknown kind
  (`web-ui/backend/tests/test_run_manager.py:1277-1330`).
  Until that changes, core cannot use the freedom the rule grants. The
  tracking issue should gate both the split and any additive record change.
  The fix belongs at the parent read sites (`run_manager.py:246-258`,
  `811-821`), so the worker preflight (`866-965`) can stay strict.
- **Item 2 "exactly".** Consistent with the rule only as a producer-side pin.
  As written it makes the contract table a closed schema, which is how Studio
  reads it (`run_manager.py:903`, `951`). Reword the lead to: "Records carry
  every listed key with the listed type and presence rule. As a regression
  pin, not as contract, today's records carry no other key (item 8 covers
  `stream_lifecycle`)." Keep "exactly" out of the published table.
- **Item 7 cannot fail for a locking regression.** Both equalities are
  computed from one local tuple (`pipeline.py:547-552`), so they hold with the
  lock removed; under the GIL so do the monotonic and prefix checks on a list
  that is only appended. Keep it as a smoke test or drop it, and do not cite
  the lock in the docstring as the reason for the guarantee.
- **Replay of helper records is untested.** Item 5 avoids the one case where
  replay is lossy. If W2 is adopted, add to item 8: replaying
  `register_numpy`'s three records through public methods yields records
  equal on every listed key, without `stream_lifecycle`, whose `expression@1`
  and `symbolic_matrix@1` capability entries have `modes == ("batch",)`.
- **Kafka rows may ship unverified.** Item 2's Kafka clause runs only on a
  connector wheel, and the existing test returns early without one
  (`python/tests/test_runtime.py:45-50`). The `--features kafka` step I found
  (`.github/workflows/ci-linux.yml:153`) is a Rust connector test, not pytest.
- **Existing sentence to reconcile.** `docs/python-api.md:642-644` already
  mentions "a registration record carrying forged lifecycle keys". The new
  section and that sentence should describe unlisted keys the same way.

## Questions for the Author

1. Which route does open question 3 take? If the atomic helper is chosen now,
   W1's equation is safe to publish and only "always holds" needs softening.
2. Does any CI job run pytest against a wheel built with `connector-kafka`?

## Effect On The Implementer

Source and Studio changes are unaffected. Two small items: the pair docstring
text (W1), and one optional core test (helper replay, above). Everything else
is wording for `cf-doc-writer`.
