# Symbolic Stream Join API Note

| Field            | Value                                    |
| ---------------- | ---------------------------------------- |
| Status           | APPROVED - exact SCE-17 API freeze       |
| Issue            | GitHub #223 / SCE-17                     |
| Artifact slug    | `symbolic-stream-joins`                  |
| Controlling spec | `specs/symbolic-stream-joins.md`         |
| Project format   | `3` (unchanged)                          |
| Primitive        | `stream_join@1`                          |

## 1. Python surface

`calc_flow.symbolic.table` adds exactly this method; no package-root export is
added:

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
) -> TableExpr: ...
```

The method reuses the exact public configuration classes already accepted by
`PipelineBuilder.stream_join`. It defensively copies both key sequences.
Strings and bytes are not sequences of keys. Keys and event-time field names
are non-empty strings. Key lists are non-empty and equal in length. Prefixes
are distinct ASCII portable identifiers. `bounds` and `limits` require their
exact public classes.

## 2. Declaration encoding

The node has primitive identity `stream_join@1`, ordered arguments `(left,
right)`, and these normalized attributes with no omitted defaults:

```text
left_keys                       sequence[string]
right_keys                      sequence[string]
left_event_time                 string
right_event_time                string
before_micros                   integer
after_micros                    integer
max_state_rows_per_side         integer
max_state_bytes_per_side        integer
max_matches_per_input_batch     integer
left_prefix                     string, default "left"
right_prefix                    string, default "right"
```

Microseconds are the exact non-negative values validated by
`JoinTimeBounds`. Limit values retain the native safe-JSON-integer range.

## 3. Lowered project-v3 shape

The physical node id is `cf_stream_join_{digest16}`. Its two input ports use
the analyzed exact left and right schemas. Its operator document is the
existing shape:

```json
{
  "kind": "stream_join",
  "spec": {
    "join_type": "inner",
    "left_keys": ["account_id"],
    "right_keys": ["account_id"],
    "left_event_time": "authorized_at",
    "right_event_time": "paid_at",
    "bounds": {"before_micros": 5000000, "after_micros": 2000000},
    "limits": {
      "max_state_rows_per_side": 100000,
      "max_state_bytes_per_side": 134217728,
      "max_matches_per_input_batch": 1000000
    },
    "left_prefix": "authorization",
    "right_prefix": "payment"
  }
}
```

No symbolic-only field is serialized. The output port remains derived by the
native compiler. A direct join root therefore exposes source binding ids
`left` and `right` and sink binding id `output`. Synthetic boundary node names
used while composing upstream and downstream symbolic segments are
deterministic implementation details and are not public declaration
attributes.
