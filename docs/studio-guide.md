# Studio

[Documentation](README.md) / 2.7 Studio

Calc Flow Studio edits project-v3 documents and operates continuous jobs on a
local computer. Install and start it using [getting started](getting-started.md#start-and-stop-studio).
The packaged application opens at `http://127.0.0.1:8765`; the managed
development frontend opens at `http://127.0.0.1:5173`.

## Edit a calculation

Use the order-total calculation in
[01_datafusion_pipeline.py](../examples/01_datafusion_pipeline.py) as a small
graph to understand: an expression calculates `gross`, and a connected
projection/filter keeps the large orders. Studio edits the same node, port,
and expression fields found in the project document. Validate the graph after
changing nodes or connections, then save the project.

For batch input cards, **Edit data** opens a draft editor. **Confirm** applies
valid input; **Cancel**, Escape, or closing the dialog discards the draft.
Confirming a card does not save the entire project. See the
[Studio application guide](../web-ui/README.md#edit-data-sources) for controls.

[Example 14](../examples/14_project_persistence.py) demonstrates the project
serialization used by application code. Executable UDFs and provider callbacks
are registered in a trusted runtime and are not embedded in the saved graph.

## Inspect symbolic calculations

The programs in [examples 09–13](symbolic-workflows.md) compile symbolic
declarations into native project nodes. Selecting a node in Studio shows the
**Lowered project inspection** section: serialized expressions, operator and
provider identity, state and watermark requirements, and recognized matrix
copy boundaries.

This inspection reads the project document. It does not run a symbolic Python
compiler or reconstruct the original expression objects. Reported sizes are
declared limits or estimates; use live metrics to observe memory and latency.

## Start and observe a continuous job

Configure a stream project with the registered sources, sinks, formats,
watermark policy, delivery requirements, and managed state settings described
in the [connector guide](connectors.md). The Job observatory starts the job,
shows status, results, and bounded metrics, and resumes event observation after
a connection interruption.

Use the checkpoint control to await a durable epoch, graceful shutdown to drain
accepted work, and cancellation to stop work and settle resources. These
controls correspond to the lifecycle demonstrated by
[04_continuous_runtime.py](../examples/04_continuous_runtime.py) and
[08_streaming_recovery.py](../examples/08_streaming_recovery.py); their in-memory
Python connector objects are not serialized into Studio projects.

## Service limits

Studio binds to loopback and is a local single-user application. Jobs run in
workers with concurrency, resident-memory, checkpoint-disk, and lifecycle
limits. The REST job API accepts connector-backed stream projects. A project
declaring static inputs can be inspected, but job creation returns `422`
because the REST contract has no field for live static values. Run such a
project through Python with explicit `static_inputs`, as in
[example 11](../examples/11_symbolic_static_matrix.py).

Consult the [HTTP API reference](api-reference.md#local-http-api) for routes and
the separate [Studio architecture](design.md#studio-boundary) for ownership.

Next: [API reference](api-reference.md).
