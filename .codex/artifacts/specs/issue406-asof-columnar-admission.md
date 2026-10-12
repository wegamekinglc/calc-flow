# ASOF right columnar admission

Base: main `8ee5b71b7b862d4e44bd0be390d182cf950d17f7`.

## Problem and scope

The maintained ASOF 1M workload is still about 243 ms with checkpoints off on
one CPU, against the user's 100 ms target (30 ms parallel). The preceding key
cache change measured only about 3% improvement. The earlier OFF profile
attributed 44.62% inclusive samples to admission preparation, 20.79% to identity
construction, and 21.96% to installation; these overlapping shares must not be
added. This increment removes right-side canonical row descriptors and ordered
insertion checks from a provably append-only batch. It is not complete goal
acceptance.

## Contract

- Qualify only nonempty right input with one non-null Utf8 key and one UInt64
  sequence column, inline canonical keys, no discarded late rows, and strictly
  increasing `(time, key, sequence)` across all record batches.
- Compare the first incoming identity for every touched key with that key's
  actual retained maximum, including identity-only histories and restored state.
  Fall back if it does not strictly follow that maximum. Never use accepted-row
  counters as ordering evidence.
- Encode each distinct batch key once, map its dictionary ID to an admission
  handle, and keep typed time/sequence columns and row IDs. Do not construct a
  canonical identity tuple per accepted row on the qualifying OFF path.
- Reserve each right bucket's columns once, then append typed values and compact
  payload references without per-row identity comparisons. Keep existing
  copy-on-write preparation, capacity projection, owner accounting, row limits,
  checkpoint index length, and infallible install boundary.
- When delta checkpointing needs row edits, reconstruct canonical identities
  from the typed columns solely for the journal. Preserve v3 checkpoint bytes,
  compatibility, replay, duplicate diagnostics, and ON/OFF lifecycle behavior.
- Preserve all validation fallbacks, cancellation checks, owned-buffer lifetime,
  caller immutability, and explicit memory budgets. Introduce no default pool
  cap and no public API or schema changes.

## Validation

Extend existing test entries to avoid increasing the generated 2048-test unit
harness. First record failure of the canonical-row complexity assertion. Compare
fast and forced-legacy state, outputs, accounting, checkpoint round trips and
delta recovery. Cover independent dictionary IDs across batches, sliced inputs,
same-time sequence ties, restored maxima, duplicates, unordered/mixed-late/wide
fallbacks, resource refusal, and cancellation. Run focused ASOF checks and core
Clippy; final specialist review remains required.

Before performance execution, freeze maintained harness cases, artifact
provenance, sample counts, and a total measurement budget. Compare current main
with candidate for ASOF 1M OFF one CPU, OFF parallel, LOW and ON. The affected
1024-row overhead case remains explicitly unmeasured in the complete inventory:
the available ordinary baseline lacks the tracing feature, as established in
the preceding run; do not repeat a known incompatible measurement. Build
time is separate. Report the entire eight-row target inventory and distinguish
historical evidence from fresh paired results. Stop at the budget; do not retry
until favorable. Full acceptance and scheduled broad benchmarks remain separate.
