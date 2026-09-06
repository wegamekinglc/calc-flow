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
The [connector guide](connectors.md) provides transport-specific fragments.
Secret references select a trusted resolver; credential values do not belong
in project options.

The generated [JSON Schema](../schemas/project-v3.schema.json) is the field
reference. See [Python API](python-api.md#projects-and-persistence) for method
usage and [architecture](design.md#project-and-registry-design) for storage
and registry ownership.

Next: [connectors](connectors.md).
