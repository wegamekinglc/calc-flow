# Projects and persistence

[Documentation](README.md) / 2.5 Projects and persistence

A project stores a graph and execution configuration as data. Use it to save a
calculation, validate it on another entry point, or load it into Studio. Saving
a project does not save a running job's state; streaming state belongs to
[managed checkpoints](streaming-guide.md#checkpoints-and-recovery).

## Save, load, and execute a graph

Run [14_project_persistence.py](../examples/14_project_persistence.py):

```bash
uv run --no-sync python examples/14_project_persistence.py
```

The example starts with the addition graph from the introduction and performs
these operations:

1. Validate `builder.project` as a `ProjectDocument`.
2. Round-trip its data through canonical JSON and safe YAML.
3. Create the document in a temporary `FileProjectStore` and read it back.
4. Compile the loaded document through `Runtime` and execute Arrow input.
5. Assert that totals remain `[3, 7]` and the original builder is unchanged.

It prints the totals and a round-trip confirmation. Its temporary store is
cleaned up on exit. For persistent application storage, provide a stable
directory to `FileProjectStore` instead.

## Document and store APIs

`ProjectDocument` validates the strict project-v3 model. Use
`calc_flow.store.export_project_json` / `import_project_json` and the matching
YAML functions for serialization. `FileProjectStore` provides async `create`,
`put`, `get`, `list`, and `delete` operations. `create` requires a new project
identity; `put` writes an existing or new identity. The explicitly named
`*_blocking` forms are for callers outside an active event loop.

Unknown fields, duplicate JSON keys, executable objects, and unsafe YAML
content fail validation. Project documents contain exact UDF/provider
references; register their trusted implementations in the receiving runtime
before compilation. Input data passed directly to a plan is supplied again
when executing a reloaded batch graph.

## Stream projects and connector bindings

A connector-backed stream project declares `runtime.mode`, graph nodes,
source/sink bindings, watermark policies, delivery requirements, and state
configuration. Compile it with `compile_stream_project(project)` and start
`StreamingRunner(plan)`. The plan supplies the registered connector factories
and managed state settings.

For application-owned Python sources and sinks, use `builder.compile_stream()`
and supply bindings to the runner, as in
[04_continuous_runtime.py](../examples/04_continuous_runtime.py).
The [connector guide](connectors/README.md) provides transport-specific fragments.
Secret references select a trusted resolver; credential values do not belong
in project options.

The generated [JSON Schema](../schemas/project-v3.schema.json) is the field
reference. See [Python API](python-api.md#projects-and-persistence) for method
usage and [architecture](design.md#project-and-registry-design) for storage
and registry ownership.

## Union and event-time windows

Project v3 represents the built-in same-schema union directly:

```json
{
  "id": "merge",
  "operator": {"kind": "union"},
  "input_ports": [
    {"name": "left", "kind": "table", "required": true},
    {"name": "right", "kind": "table", "required": true}
  ]
}
```

Window nodes require one exact table input schema. Geometry is expressed in
exact microseconds; `slide_micros` distinguishes a hopping window from a
tumbling window.

```json
{
  "id": "minute_totals",
  "operator": {
    "kind": "window",
    "spec": {
      "event_time_column": "event_time",
      "group_by": ["account"],
      "geometry": {"kind": "tumbling", "size_micros": 60000000},
      "aggregates": [
        {"function": "sum", "column": "amount", "output": "total"}
      ]
    }
  },
  "input_ports": [{
    "name": "input",
    "kind": "table",
    "required": true,
    "schema": [
      {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
      {"name": "account", "data_type": "string", "nullable": false},
      {"name": "amount", "data_type": "float64", "nullable": false}
    ]
  }]
}
```

Stream Join nodes are the other two-input table operator. Both inputs carry an
exact schema, the bounds and limits are required with no defaults, and the
output schema is derived from the prefixes rather than declared:

```json
{
  "id": "match",
  "input_ports": [
    {
      "name": "left",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "account_id", "data_type": "int64", "nullable": false},
        {"name": "authorized_at", "data_type": "timestamp[us]",
         "nullable": false}
      ]
    },
    {
      "name": "right",
      "kind": "table",
      "required": true,
      "schema": [
        {"name": "account_id", "data_type": "int64", "nullable": false},
        {"name": "paid_at", "data_type": "timestamp[us]", "nullable": false}
      ]
    }
  ],
  "output_ports": [],
  "operator": {
    "kind": "stream_join",
    "spec": {
      "join_type": "inner",
      "left_keys": ["account_id"],
      "right_keys": ["account_id"],
      "left_event_time": "authorized_at",
      "right_event_time": "paid_at",
      "bounds": {"before_micros": 300000000, "after_micros": 30000000},
      "limits": {
        "max_state_rows_per_side": 100000,
        "max_state_bytes_per_side": 134217728,
        "max_matches_per_input_batch": 1000000
      },
      "left_prefix": "authorization",
      "right_prefix": "payment"
    }
  }
}
```

## Static input declarations

A stream project declares immutable static side inputs as a data-only root
array. Each entry names an unconnected external input port of a graph node —
the same port a source binding would feed, minus the connector:

```json
{
  "graph": {
    "nodes": [
      {
        "id": "merge",
        "operator": {"kind": "union"},
        "input_ports": [
          {"name": "left", "kind": "table", "required": true},
          {"name": "weights", "kind": "table", "required": true}
        ]
      }
    ]
  },
  "static_inputs": [
    {
      "kind": "table",
      "name": "weights",
      "mutability": "static",
      "schema": [
        {"name": "factor", "data_type": "float64", "nullable": false}
      ]
    }
  ]
}
```

An array-valued input declares the provider identity instead of a schema:

```json
{
  "kind": "array",
  "name": "weights",
  "mutability": "static",
  "backend": "numpy",
  "dtype": "float64",
  "shape": [3]
}
```

`mutability` accepts only `static`. Validation is strict and fail-closed:

| Rule                                     | Failure path                                          |
|------------------------------------------|-------------------------------------------------------|
| Unique portable SQL identifier names     | `static_inputs[i].name`                               |
| Name must be a graph external input      | `static_inputs[i].name` (`unknown_binding`)           |
| Name must not be a source binding        | `static_inputs[i].name` (`source_binding_conflict`)   |
| Unique table field names                 | `static_inputs[i].schema[j].name`                     |
| Table fields in the digest-v1 type set   | `static_inputs[i].schema[j].data_type`                |
| Array backend of 1 to 64 bytes           | `static_inputs[i].backend`                            |
| Array dtype in the digest-v1 set         | `static_inputs[i].dtype`                              |
| Array rank at most 16                    | `static_inputs[i].shape`                              |

The declaration joins the compiled plan's semantic fingerprint, so changing it
selects a fresh lineage. Live values never enter the document: the caller
supplies them per job through the runner, and a restart with a different value
is rejected against the recorded digest before sources open. See
[static inputs](streaming-guide.md#static-inputs) for the runner semantics and
the digest contract. An empty declaration array is omitted from canonical
JSON when no static values are declared.

Studio REST cannot carry live values, so submitting a stored project that
declares static inputs fails closed with `422` before any worker is spawned;
the `detail` names the first `static_inputs.{name}` as unresolvable and no
run, handle, or worker is created. Supply the values through the Python
runtime instead.

Next: [connectors](connectors/README.md).
