# Symbolic Stream Join Contract

| Field             | Value                                                         |
| ----------------- | ------------------------------------------------------------- |
| Status            | APPROVED - SCE-17 implementation contract                     |
| Issue             | GitHub #223 / SCE-17                                          |
| Artifact slug     | `symbolic-stream-joins`                                       |
| Project format    | `3` (unchanged)                                               |
| Native dependency | existing `stream_join@1` operator and v1 checkpoint state     |

## 1. Scope and authority

SCE-17 adds a declaration-only symbolic entry point for the existing bounded
native stream inner join. It does not add a second join engine, a Python data
path, a project-v3 variant, or a checkpoint layout. The native
`stream_join@1` specification remains authoritative for matching, watermark,
state-limit, row-order, metric, checkpoint, and recovery behavior.

This document freezes only the symbolic analysis and lowering boundary. The
same-slug API note owns the exact Python signature and declaration attributes.

## 2. Semantics

A symbolic stream join MUST:

- accept two table expressions and one non-empty, equal-length ordered key
  list per side;
- perform the native inner equi-join using inclusive event-time bounds;
- require a named, non-null `timestamp[us, UTC]` event-time field on each
  resolved input schema;
- require equal key types at corresponding key positions;
- require explicit native `JoinTimeBounds` and `JoinStateLimits` values;
- preserve each source field's type and nullability;
- emit every left field followed by every right field, named
  `{left_prefix}__{field}` and `{right_prefix}__{field}` respectively; and
- mark both source lineages as temporal so stream input ordering declarations
  are checked before lowering.

The declaration MUST lower to exactly one native `stream_join@1` node. The
lowerer MAY place existing expression, rolling, or cross-section stages before
either side and MAY place stateless row-local projection, derivation, and
filter stages after the join. Multiple outputs descending from the same join
MUST share that one physical join node.

## 3. Modes, capabilities, and failures

The symbolic join is stream-only. Batch analysis and compilation fail with
`unsupported_mode`. The selected capability snapshot MUST expose
`stream_join@1` in stream mode with two required table inputs named `left` and
`right`, one required table output named `output`, watermark support,
checkpointed state with a positive state version, determinism, and replay
safety. A missing fact fails with `capability_mismatch`.

Declaration misuse raises `TypeError` or `ValueError` before a node is built.
Schema, type, lineage, ordering, and capability failures use the existing
symbolic issue vocabulary and deterministic paths.

## 4. Frozen SCE-17 composition boundary

One program may contain one unique join declaration, referenced by one or more
outputs. Every output in that program MUST descend from that join. Nested or
independent joins, unrelated outputs, matrix attachment around a join, event
windows after a join, and rolling or cross-section state after a join are not
supported by SCE-17.

These restrictions keep one native join state owner and avoid inventing an
event-time, entity, sequence, update, or finality declaration for the joined
output. They fail during analysis or symbolic lowering; they MUST NOT fall
back to Python execution. Relational DAGs with multiple joins and explicit
post-join ordering metadata require a separate contract.

## 5. Durability and compatibility

SCE-17 introduces no serialized project fields and no durable state version.
The lowered project uses the existing native join specification verbatim.
Checkpoint publication and restore therefore remain governed by the native v1
join state, including configuration/schema compatibility checks, buffered
left/right rows, watermarks, deterministic replay, and state limits.

Declaration digests and program fingerprints include the complete ordered join
arguments and normalized configuration. Changing keys, event-time fields,
bounds, limits, or prefixes changes identity and the compiled plan fingerprint.

## 6. Acceptance

Acceptance requires tests for declaration validation and immutability,
deterministic identity, schema inference, ordering diagnostics, batch and
composition rejection, capability spoofing, one-node lowering and shared
fan-out, row-local post-join execution, micro-batch segmentation invariance,
watermark finality, state-limit propagation, and mid-checkpoint recovery.

The symbolic execution vectors MUST compare against an independently derived
bounded inner-join reference. Native join tests remain the exhaustive source
for low-level match ordering, null keys/times, late data, state bytes, and
restore corruption behavior; SCE-17 tests prove that the symbolic plan reaches
that same implementation without a Python execution path.
