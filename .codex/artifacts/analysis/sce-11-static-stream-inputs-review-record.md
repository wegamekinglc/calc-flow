# SCE-11 Static Stream Inputs — Review Record and Caveat Ledger

| Field         | Value                                                                                                                                                     |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status        | Accepted with caveats; code review Approve with zero blocking findings                                                                                    |
| Scope         | PR #208 (`feature/static-stream-inputs`), `333e6c0` + `ebb7414`, base `main@079d0ed8`                                                                     |
| Contract      | `.codex/artifacts/api-notes/sce-11-static-stream-inputs.md` (frozen, git blob SHA-256 `da9a9ec1687f8e1487d6b0341b838c69d1d07a54992cf3ee4b59b77b451c85d2`) |
| Adjudications | cf-reviewer final review and cf-orchestrator acceptance, 2026-08-28                                                                                       |

This record persists the accepted deviations and follow-up obligations that
the frozen API note cannot carry (the note file is byte-pinned and must not be
edited). Normative behavior is documented in `docs/streaming-guide.md`
(static inputs), `docs/connectors.md` (static input declarations), and
`docs/api-reference.md`.

## Accepted caveats

- **Python recovery exception class (F3).** A recovery digest mismatch on the
  Python surface raises a structured `StreamingRuntimeError` with
  `category="checkpoint_mismatch"`, not the `CheckpointError` named in the
  API note's Error Cases table. The note's table describes the engine-level
  error class: every streaming start-phase engine error crosses the frozen
  streaming projection (`StreamingRuntimeError` with a fixed category
  vector), and opening a `CheckpointError` escape hatch for static inputs
  would fork the error contract. The normative content — the byte-exact
  message, the `static_inputs.{name}` path, and the before-source-open
  moment — holds on both surfaces and is tested. Non-streaming projections
  still map the engine's checkpoint mismatch to `CheckpointError`.
- **Binding-seam dtype rejection (implementer decision 2).** Supplying an
  array whose dtype is outside the digest-v1 set (for example `complex128`)
  raises `ValueError` at the Python binding seam on the
  `static_inputs.{name}.dtype` path, not the table's `StreamingRuntimeError`.
  The seam cannot extract such a payload at all; publicly declared projects
  reject unsupported dtypes at project validation, and engine-level
  declared-vs-supplied mismatches still raise `StreamingRuntimeError`.
- **Reference hash convention (F6).** The SHA-256 values the API note
  declares for its controlling inputs were computed over CRLF working-tree
  bytes, while git stores LF blobs. The git blob hashes are identical across
  the baseline, the note branch, and the PR, so there is no integrity risk;
  future artifact pinning should cite git blob hashes (or normalize line
  endings first) to avoid the ambiguity.
- **No declaration/port kind cross-validation (F7).** A static declaration's
  `kind` (table or array) is not cross-checked against the kind of the port
  it names; an array declaration attached to a table port passes project
  validation today. No contract currently requires the check; it is folded
  into follow-up obligation 4 below.
- **No-action observation (F5).** Recovery-time missing/extra static names
  are unreachable on public paths because declarations participate in the
  plan fingerprint (a different declaration selects a different lineage);
  the defensive rejection path is unit-tested inside the crate.

## Follow-up obligations bound to the first consuming operators (F4)

These obligations activate when SCE-08 (rolling time windows) or SCE-09
(cross-section) land an operator or provider that consumes static input
values. Implementers of those stages must satisfy all four:

1. Add a read-only value access path for consuming operators and providers.
   Today a latched value has no operator-facing access route; only its
   digest evidence participates in preflight, lineage, and checkpoint
   identity, which is why D10's "visible read-only to every consuming
   operator/provider" is vacuously satisfied with zero consumers.
2. Extend the engine-owned latched-handle lifetime to the complete job
   teardown. `run_job_driver` currently destructures the prepared set
   (`static_inputs: _`), dropping the handles when the driver is consumed
   rather than at full teardown; with no consumer this is unobservable.
3. Preserve exactly-once release of the handles across all four exit paths
   (success, cancellation, startup failure, recovery failure) while making
   the change in obligation 2.
4. Add the declaration-kind versus port-kind cross-validation from caveat F7
   above as part of the consumer-facing contract work.

## Confirmed adjudications (for traceability)

- Studio fail-closed point (API note Open Question 1): run-manager validation
  before worker spawn, `422` naming `static_inputs.{name}`, zero resources
  created; engine-level preflight remains the in-process backstop.
- Engine-owned immutable handle may not alias caller-mutable memory (Open
  Question 2): the explicit reading of D10 that makes the caller-mutation
  acceptance criterion testable for NumPy-backed values; array payloads
  snapshot at the binding seam, strictly narrowing the aliasing window.
- Implementer decisions 1, 3, and 4 confirmed as implemented (engine-only
  latch constructor and the four frozen crate-root exports; the
  sanitizer/projection pass-through gated on error type plus the
  `static_inputs.` prefix; job-driver ownership with Drop-based release and
  digest-only evidence in operator-visible state).
