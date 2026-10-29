DAL-475 measures only the five directly affected workloads below. The product
baseline is current main `d9e2759fb31bfd0c915d31f0cc534665ece72ee9`.
Research used `eb30d22523ae62315eb87a9d8685bf1c5b8ac20c`; main additionally
contains #514 benchmark artifacts and #512 compilation/recovery cleanup. The
candidate retains that cleanup. Candidate source/native hashes and final SHA
will be sealed in the results.

- Arrow import/export: two int64 columns, 64 rows, two chunks; checks every buffer.
- Short table compute: the same input, projection `total = a + b`, default output.
- NumPy expression: maintained array provider/plan harness, elementwise operation
  over 1,000 float64 values; each timed call includes the input snapshot.
- CPU table matrix: 64 rows, two float64 columns and 2x2 weights; plan execution,
  Arrow import and matrix assembly; weights are captured before timing.
- JAX CPU table matrix: the same shape in float32; completion is awaited in each
  timed call, including explicit NumPy host conversion. First execution/compilation is recorded separately from warm calls.
  No GPU is available for this task, so GPU transfers are unmeasured.

Use two rounds of ten adjacent baseline/candidate pairs per case, alternating
AB/BA. Each invocation is a new process and records seven raw warm observations;
its observation is their median. Use the maintained array harness and paired
statistics/+5% verdict from `scripts.benchmark_suite.report`. Both sides share
one interpreter/dependency environment, fixed single-thread settings, and CPU.
Verify output values, schemas, input immutability, loaded native hash, source
identity and array metadata before accepting observations. Record resource RSS
and tracemalloc peaks separately; these do not count native allocations or GPU
memory. Copy ceilings also use the existing directed ownership tests.

Total measurement wall budget: 600 seconds, including process startup, fixtures,
preflight, first execution, correctness, resource diagnostics and cleanup. Builds
are timed separately. Stop on budget exhaustion or incompatible identities;
report incomplete cases without increasing samples or retrying for a verdict.
