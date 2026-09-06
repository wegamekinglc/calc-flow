# Calc Flow documentation

Read these pages in order for an introduction to Calc Flow 4.0, or choose the
feature you need. Function guides explain usage with runnable examples. Design
pages explain how the implementation provides those behaviors.

## 1. Overview and first run

1. [Introduction](introduction.md): capabilities, vocabulary, and execution modes.
2. [Getting started](getting-started.md): installation, a first calculation,
   and starting Studio on Linux or Windows.
3. [Example learning paths](examples.md): which programs to run next.

## 2. Function guides

1. [Batch calculations](batch-guide.md): expressions, filters, SQL joins,
   registered UDFs, and async execution; examples 01, 02, 03, and 05.
2. [Arrays and matrices](array-guide.md): NumPy/JAX registration, array
   expressions, table-to-matrix multiplication, and static weights;
   examples 06, 07, and 11.
3. [Continuous streaming](streaming-guide.md): sources, sinks, event time,
   windows, bounded joins, job controls, and recovery; examples 04, 08,
   10, 12, and 13, plus the Rust window example.
4. [Symbolic workflows](symbolic-workflows.md): financial features, analysis,
   composition, and batch/stream execution; examples 09–13.
5. [Projects and persistence](projects-guide.md): validate, serialize, save,
   load, and execute a project; example 14.
6. [Connectors](connectors.md): registered transports, project configuration,
   secrets, and delivery limits.
7. [Studio](studio-guide.md): edit projects, inspect calculations, and operate
   local jobs.

## 3. API references

1. [API reference](api-reference.md): public surfaces, HTTP routes, errors,
   and package/protocol versions.
2. [Python API](python-api.md): builders, execution options, providers,
   persistence, and runner methods.
3. [Symbolic API](symbolic-api.md): declarations, ordering requirements,
   analysis, and compilation.
4. [Rust API](rust-api.md): native types, operators, traits, and examples.

The [project schema](../schemas/project-v3.schema.json) and
[OpenAPI document](../web-ui/openapi.json) define the serialized contracts.

## 4. Design and implementation

1. [Architecture](design.md): component ownership, data paths, stateful
   operators, extension boundaries, and failure handling.
2. [Stream runtime contract](runtime-envelope.md): messages, ordering,
   backpressure, progress, checkpoint publication, and recovery invariants.
3. [Symbolic compiler design](symbolic-design.md): analysis, lowering,
   physical sharing, static values, and compile caching.

## 5. Development and operations

1. [Verification](verification.md): checks for examples, documentation, and
   each implementation surface.
2. [Benchmark suite](benchmark-suite.md): workloads, correctness, timing
   boundaries, reports, and regression gates.
3. [SQL performance controls](sql-datafusion-performance.md): partitioning,
   telemetry, rewrite limits, and measurements.
4. [Warm-stream measurements](warm-stream-performance.md): persistent jobs,
   sparse appends, latency interpretation, and reproduction.
5. [Python release guide](python-release.md): packaging and publication.

## History and maintenance

[CHANGELOG.md](../CHANGELOG.md) is the single change history. Guides and design
pages describe the implementation in this checkout; dated plans, migration
narratives, and results for individual commits belong in history.

Maintain one executable inventory in [examples/README.md](../examples/README.md).
Each function guide links to its programs and explains inputs, operations,
expected results, and limits. Keep implementation details in design pages.
Update this reading order when adding a page.
