# Symbolic Relational DAG Contract

| Field             | Value                                                     |
| ----------------- | --------------------------------------------------------- |
| Status            | APPROVED - SCE-18 implementation contract                 |
| Issue             | GitHub #225 / SCE-18                                      |
| Artifact slug     | `symbolic-relational-dag`                                 |
| Project format    | `3` (unchanged)                                           |
| Native dependency | existing `stream_join@1` operator and v1 checkpoint state |

## 1. Scope and authority

SCE-18 removes the SCE-17 single-join composition restriction. One symbolic
program may own an acyclic graph of independent or nested bounded joins,
unrelated table outputs, and executable rolling or cross-section stages after
a join. Every unique symbolic join still lowers to exactly one existing native
`stream_join@1` state owner.

This stage adds no Python execution path, native join implementation,
project-v3 variant, or checkpoint layout. Native matching, watermark,
state-limit, metric, checkpoint, and recovery behavior remains authoritative.
Event-window declarations remain analysis-only until a separately approved
symbolic window-aggregate lowering contract exists.

## 2. Declaration versions and compatibility

The SCE-17 call with no output ordering remains byte-identical
`stream_join@1`. It is sufficient for terminal or row-local join output.

A declaration that supplies complete output ordering uses symbolic primitive
`stream_join@2` and adds exactly these canonical attributes:

```text
output_entity_by     non-empty sequence[string]
output_event_time    non-empty string
output_sequence_by   non-empty sequence[string]
```

The three attributes are all present or all absent. Caller-owned sequences are
copied before the immutable node is built. Symbolic v2 is declaration
metadata only; lowering strips these attributes from the existing native
`stream_join@1` project specification.

## 3. Post-join ordering proof

Output ordering does not sort, buffer, or execute Python code. It names the
joined fields used by downstream native state and is valid only when analysis
can prove all of the following:

- `output_entity_by` is the ordered left join-key list after applying the left
  output prefix;
- `output_event_time` is either prefixed join event-time field and remains a
  non-null `timestamp[us, UTC]` field;
- `output_sequence_by` concatenates the prefixed left and right input sequence
  keys in that order; and
- every named field remains present after intervening table projections.

This canonical tuple gives each emitted pair one deterministic entity,
event-time, and tie-breaking identity. A joined value feeding another join,
rolling stage, or cross-section stage MUST carry the complete proof. Missing,
partial, mismatched, or projected-away metadata fails with
`ordering_required` before native compilation.

## 4. Relational DAG lowering

The lowerer discovers unique joins by declaration digest and assigns each the
existing `cf_stream_join_{digest16}` physical identity. It materializes one
typed source fan-out node per reachable declared table input, lowers unary
row-local or supported stateful segments between relational boundaries, and
wires join outputs directly or through those segments.

The graph MUST support:

- multiple independent joins and outputs in one program;
- a prior ordered join feeding either side of another join;
- one join output fanning out to multiple consumers;
- row-local, rolling, and cross-section stages after an ordered join; and
- outputs unrelated to any join while retaining one source owner per declared
  input.

Graph nodes and edges are emitted deterministically. Duplicate references to
one join digest MUST NOT create duplicate physical state. Matrix attachment
around a join remains unsupported because it requires a separate relational
table/array ownership contract.

## 5. Durability and finality

Every physical join retains the native v1 checkpoint state and stable operator
identity. A checkpoint contains one independent entry per join plus the
existing entries for downstream rolling or cross-section state. Recovery
validates the complete graph fingerprint, every operator configuration and
schema, source cuts, progress, and state segments before any source resumes.

The selected post-join event time is one of the time fields already bounded by
the native join. Downstream watermark finality therefore remains derived from
the native multi-ingress frontier; the symbolic metadata creates no new
watermark and cannot override native lateness handling.

## 6. Acceptance

Acceptance requires RED evidence for the previous API and composition
failures, plus tests for:

- v1 identity compatibility and immutable v2 declarations;
- partial and invalid output-ordering diagnostics;
- independent joins, unrelated outputs, nested joins, and physical sharing;
- deterministic source/sink bindings, graph identity, and explain facts;
- post-join rolling and cross-section compilation;
- independently derived three-stream results across micro-batch segmentations;
- checkpoint recovery with state retained by both nested join owners; and
- all existing SCE-17 declaration, execution, and recovery regressions.
