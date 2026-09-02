# Symbolic Relational DAG API Note

| Field            | Value                                     |
| ---------------- | ----------------------------------------- |
| Status           | APPROVED - exact SCE-18 API freeze        |
| Issue            | GitHub #225 / SCE-18                      |
| Artifact slug    | `symbolic-relational-dag`                 |
| Controlling spec | `specs/symbolic-relational-dag.md`        |
| Project format   | `3` (unchanged)                           |
| Native primitive | `stream_join@1`                           |

## 1. Python surface

`calc_flow.symbolic.table.stream_join` extends its keyword-only surface:

```python
from collections.abc import Sequence
from calc_flow import JoinStateLimits, JoinTimeBounds

def stream_join(
    left: TableExpr,
    right: TableExpr,
    /,
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    left_event_time: str,
    right_event_time: str,
    bounds: JoinTimeBounds,
    limits: JoinStateLimits,
    left_prefix: str = "left",
    right_prefix: str = "right",
    output_entity_by: Sequence[str] = (),
    output_event_time: str | None = None,
    output_sequence_by: Sequence[str] = (),
) -> TableExpr: ...
```

Omitting all three output-ordering arguments preserves the SCE-17
`stream_join@1` declaration exactly. Supplying any one requires all three to
be non-empty and builds `stream_join@2`. Strings and bytes are not accepted as
key sequences, and every caller-owned sequence is defensively copied.

## 2. Canonical ordering example

For inputs ordered by `entity_by=["account_id"]`, event times
`authorized_at`/`paid_at`, and `sequence_by=["sequence"]`, this declaration is
canonical:

```python
matched = table.stream_join(
    authorizations,
    payments,
    left_keys=["account_id"],
    right_keys=["account_id"],
    left_event_time="authorized_at",
    right_event_time="paid_at",
    bounds=bounds,
    limits=limits,
    left_prefix="authorization",
    right_prefix="payment",
    output_entity_by=["authorization__account_id"],
    output_event_time="authorization__authorized_at",
    output_sequence_by=[
        "authorization__sequence",
        "payment__sequence",
    ],
)
```

The entity list must be the prefixed left join keys. The event time may choose
the prefixed left or right join event time. The sequence list must concatenate
all prefixed left and right input sequence keys.

## 3. Nested declaration

The ordered result can feed another join. The next call names fields in the
resolved prefixed schema and consumes the intermediate ordering proof:

```python
enriched = table.stream_join(
    matched,
    settlements,
    left_keys=["authorization__account_id"],
    right_keys=["account_id"],
    left_event_time="authorization__authorized_at",
    right_event_time="settled_at",
    bounds=bounds,
    limits=limits,
    left_prefix="matched",
    right_prefix="settlement",
)
```

The outer join needs output ordering only if another stateful consumer follows
it. Independent terminal joins remain valid v1 declarations.

## 4. Lowered project-v3 shape

No output-ordering field is serialized into the native join spec. Each unique
join still has this operator shape:

```json
{
  "id": "cf_stream_join_<digest16>",
  "operator": {
    "kind": "stream_join",
    "spec": {
      "join_type": "inner",
      "left_keys": ["account_id"],
      "right_keys": ["account_id"],
      "left_event_time": "authorized_at",
      "right_event_time": "paid_at",
      "bounds": {"before_micros": 5000000, "after_micros": 2000000},
      "limits": {
        "max_state_rows_per_side": 1000,
        "max_state_bytes_per_side": 16777216,
        "max_matches_per_input_batch": 10000
      },
      "left_prefix": "authorization",
      "right_prefix": "payment"
    }
  }
}
```

General relational DAGs expose reachable sources through deterministic
`<declared-input>.input` bindings and multiple outputs through the ordinary
`<output-node>.output` graph naming rule. A one-output plan retains the native
single `output` sink binding. If an input name collides with a physical or
output node id, its source node uses deterministic `cf_source_{digest16}`
identity instead of failing or aliasing another node.

## 5. Failure surface

Partial ordering arguments fail at declaration with `ValueError`. Unknown,
nullable, non-timestamp, non-canonical entity, or incomplete sequence fields
produce stable `ordering_required` analysis issues. A nested or post-join
stateful composition without v2 metadata also fails with `ordering_required`.
No failure falls back to Python execution.
