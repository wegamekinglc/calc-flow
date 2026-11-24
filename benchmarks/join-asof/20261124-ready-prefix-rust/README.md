# Rust ASOF ready-prefix settlement

On the registered `balanced_512` workload the direct Rust watermark call
moved from **40.0975 µs to 38.0750 µs P50** (-5.04%).
It missed the prospective 10% pooled-P50 goal, whose upper bound was
36.08775 µs. The repository's independent two-round 5% criterion
returned **`no-confirmed-regression`**. P95 was 68.7585 µs and
40.3552 µs, respectively.

- Round 1 had arm medians of 40.0805 µs and 37.9835 µs, a median paired change of -4.67% and a 97.85156% interval of [-20.34%, +1.90%].
- Round 2 had arm medians of 40.1685 µs and 38.6060 µs, a median paired change of -4.99% and a 97.85156% interval of [-7.86%, -2.05%].

The input has 512 pending left rows, 512 retained right rows and 32 keys.
All 512 left rows become ready on the same frontier. Both arms completed
20 measured processes each plus two excluded warmups each. In all 44 processes
the preliminary settlement validated all eight output fields and their order,
followed by a separately timed native settlement with row, state and budget
checks. All 88 callbacks delivered one 32,768-byte, 512-row batch, evicted
480 right rows, retained 32 right rows and advanced the output watermark to
1,999,999 µs, with zero state, workspace or output limit failures.

Allocation counters cover each instrumented formal callback.
Baseline recorded 136 allocations, 91388 cumulative bytes and 57272 peak bytes. Candidate recorded 136 allocations, 91388 cumulative bytes and 57272 peak bytes.
The direct full-readiness fixture also performs the ASOF lookups,
materialization, delivery and state maintenance. Sparse readiness, restoration,
Python public API and a managed runner have separate paths and clocks.

Baseline `85af0b53e0a38e19cb472aff18c1ccfd84048912`, tree `211269ccfb5b76dc5db3554ec8447edef6610645`, binary SHA-256 `07b98985100c02329829eca1bf7535129898a10e49c1813669467540ae9de201`. The separately budgeted Cargo build took 817 seconds.

Candidate `a21d3eed7088795763f511d9aed2fd81920e8201`, tree `d44249d5436e5bd4929e943f248a73d8b7a8bb68`, binary SHA-256 `127f86c2fe5620c497123e037294ed5a59a6ef7db2b37379f7b90fbd7f70f0c5`. The separately budgeted Cargo build took 498 seconds.

The collector pinned every worker to logical CPU 2 on a
13th Gen Intel(R) Core(TM) i9-13900HX. Resource monitoring covered 4.017 seconds in
2 intervals. Available memory was at least
28.95 GiB, maximum sibling
utilization was 0.00% and maximum whole-host
utilization was 0.96%. All intervals passed.
The collector took 4.044 seconds inside the 240-second budget.
The registration, original process/output records, resources, identities and
build receipts accompany this report.

## Reproduction

Each worker starts a fresh process, seeds the Rust operator, executes a full-value
settlement to qualify the fixture, then times one settlement on a fresh operator.
The direct watermark clock includes output materialization, collector delivery,
commit and eviction. It excludes seeding, checkpoint capture and the full-value
qualification. The maintained release worker instruments allocations while the
clock runs. A separate source check and binary digest for each arm appear in
`arms.json`. The Cargo artifact receipts retain complete release profiles,
features, build times, source roots, revisions and binary paths.

Verify the raw results from a repository checkout at either revision. Supply an
absolute evidence path in the environment and run a Python interpreter with
the repository's dependencies.

```bash
PYTHONPATH=. python3 "$EVIDENCE_DIR/verify.py"
```

For an independent reproduction, use one disposable Git worktree with both
commits available, one unchanged absolute Cargo target directory, Rust 1.94.0
and CPython with psutil. Set `EVIDENCE_DIR`, `SHARED_CARGO_TARGET` and
`PYTHON` to absolute paths. Keep the evidence outside the disposable checkout.
Use the exact revisions reported in `arms.json`. Run from the worktree root.

```bash
mkdir -p target/observations
cp "$EVIDENCE_DIR"/{build.sh,measure.py,verify.py,seal.py,protocol.md,sources.sha256} target/observations/
git switch --detach "$BASELINE_SHA"
bash target/observations/build.sh baseline "$BASELINE_SHA" "$SHARED_CARGO_TARGET"
git switch --detach "$CANDIDATE_SHA"
bash target/observations/build.sh candidate "$CANDIDATE_SHA" "$SHARED_CARGO_TARGET"
"$PYTHON" target/observations/seal.py "$CANDIDATE_SHA"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --kill-after=2 238 "$PYTHON" target/observations/measure.py "$CANDIDATE_SHA"
```

Each `WORKER` header in `workers.log` records the SHA-256 of the following
two original lines. The verifier reproduces the pooled P50, linearly
interpolated P95, both paired intervals and maintained classification, then
reconstructs the original partial observation file's digest. With 20 formal
observations per arm, P95 is descriptive. Each paired interval uses ranks
two and nine of ten pairwise changes, yielding 97.8515625% coverage under
the repository's independent-pair assumptions.
