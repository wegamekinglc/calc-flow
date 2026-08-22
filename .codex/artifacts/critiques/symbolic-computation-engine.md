# Symbolic Computation Engine Contract Critique

| Field            | Value                                                                  |
| ---------------- | ---------------------------------------------------------------------- |
| Status           | **APPROVED — B1–B4 closed in the one correction round**                |
| Issue            | GitHub #167 / SCE-00                                                    |
| Baseline         | `feature/symbolic-contract@f6b8a6f90b7a978de1976f5a163ea689b989caee` |
| Controlling spec | `../specs/symbolic-computation-contract.md`                             |
| API note         | `../api-notes/symbolic-computation-engine.md`                           |

## Verdict

**APPROVED.** The one permitted correction round closes B1–B4 without
reopening the already-approved recovery, serialization, finality, or API
decisions. No original blocker remains. The non-blocking caveats below retain
their original status and do not delay implementation.

## Closed blocking findings

### B1 — Capability v2 legacy-provider finality — CLOSED

**Closure evidence:** spec D9 adds `unproven` to the closed finality vocabulary,
defines it as absence of proof, preserves legacy registrations as batch
selectable, and rejects them from stream compilation. API note sections 6.1,
6.2, and 9 repeat the exact serialized/default value and rejection rule.
Acceptance section 16.1 now requires a legacy-registration vector proving both
batch compatibility and pre-callback stream rejection.

### B2 — Late-error row-value exposure — CLOSED

**Closure evidence:** API note section 4 replaces `{identity}` with the exact
zero-based logical `row_index` before internal reorder, retains only the
permitted event-time and watermark coordinates, and expressly forbids
entity/sequence formatting. The same non-payload rule now covers
duplicate-identity and malformed-row diagnostics.

### B3 — `max_lateness_micros` semantics — CLOSED

**Closure evidence:** spec D7 and API note section 3.2 now freeze
`checked_u64(i128(W) - i128(t))`, state that `L` affects only closure, and use
the dropped row's `t` for bucketed groups. They define zero/`None` initial
state, one atomic per-envelope update, checked job sums and maximum, branched
operator counting, checkpointed per-operator state, failed-attempt rollback,
replay-once behavior, and terminal recovery. Spec section 16.3 and the API note
carry the same nonzero-`L` vector through checkpoint/restore and failed replay.

### B4 — `StaticInputSpec` derive conflict — CLOSED

**Closure evidence:** API note section 3.3 removes `Eq` from the exact
`StaticInputSpec` derive and explicitly records why its
`Vec<ArrowFieldSpec>` table member limits it to `PartialEq` on the baseline.
`StaticInputDigest` correctly remains `Eq`.

## Passed adversarial surfaces

- **Existing-lineage recovery:** API note section 8 explicitly computes
  payload digests before opening state, selects `StateLineageKey` using only
  pipeline name plus semantic plan fingerprint, and compares the exact digest
  map inside that lineage. Payload changes cannot silently select a fresh
  lineage.
- **Envelope atomicity:** `late_policy.error.scope = "envelope"`, strict JSON,
  fingerprint participation, and the staged-delta-before-state-or-emit rule
  jointly make no-output/no-state failure implementable. B2 concerns only the
  diagnostic projection, not the transaction boundary.
- **Project-v3 compatibility and omission:** the two new operator variants are
  strict and additive; all stateful defaults are serialized explicitly; an
  empty top-level `static_inputs` list is omitted, preserving old canonical
  project bytes.
- **Static digest determinism:** the magic/version domain separator, tagged
  table/array envelopes, big-endian widths, length-prefixed UTF-8, schema/type
  tags, row/C order, dictionary logical-value rule, signed-zero preservation,
  and per-dtype canonical NaNs are sufficient for independent cross-language
  implementations.
- **Final-only state and replay:** the spec requires retained history, open
  groups, reorder buffers, duplicate evidence, finality frontier, output
  sequence, aligned checkpoint capture, output-before-watermark ordering, and
  post-`on_end` terminal checkpoints. Recovery can be implemented without a
  missing or duplicate final row.
- **Strict JSON:** rolling/cross-section semantic fields have no serde defaults,
  non-applicable variant fields are rejected, and old operator omission rules
  remain isolated from the new variants.

## Non-blocking caveats

- Adding fields to the public Rust `ProjectSpec`, `CheckpointManifestFields`,
  and `ManifestExpectation` structs breaks downstream struct literals even
  though serialized project/manifest compatibility is preserved. The package
  promises source compatibility only for provider registration and
  `StreamingRunner::new`; record this Rust source impact in the eventual
  changelog rather than silently calling the entire change source compatible.
- Several frozen/slotted Python dataclasses that contain `Expr` values retain
  generated dataclass equality. Because `Expr.__eq__` is symbolic and
  `Expr.__bool__` fails, equality of `Program`, `FeatureSet`, or grouping values
  can raise unexpectedly. No acceptance criterion currently relies on that
  equality, so this is not blocking; the implementation should set `eq=False`
  or document that only expression `identical()` and program fingerprints are
  comparison surfaces.
- The tagged digest encoding is sufficiently exact, but implementation tickets
  should publish shared table/array golden vectors for Rust, NumPy, and JAX.
  This is verification evidence, not a missing contract decision.

## One-cycle correction disposition

The correction is accepted. B1–B4 are closed, the original caveats remain
non-blocking, and SCE-00 is **APPROVED** for downstream implementation.
