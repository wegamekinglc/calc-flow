# Rust ASOF candidate timestamp settlement

On the registered `balanced_512` workload the direct Rust watermark call
moved from **42.2040 µs to 41.0070 µs P50** (-2.84%).
It missed the prospective 10% pooled-P50 goal, whose upper bound was
37.98360 µs. The repository's independent two-round 5% criterion
returned **`no-confirmed-regression`**. P95 was 54.6889 µs and
50.4605 µs, respectively.

- Round 1 had arm medians of 42.5440 µs and 41.4730 µs, a median paired change of -2.58% and a 97.85156% interval of [-23.74%, +2.49%].
- Round 2 had arm medians of 42.0635 µs and 40.5575 µs, a median paired change of +0.68% and a 97.85156% interval of [-8.27%, +4.89%].

The exact optimized eligible path is documented in [eligible-tail.md](eligible-tail.md).

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
Each lookup selects the right bucket’s last retained entry. The fixture also performs the ASOF lookups,
materialization, delivery and state maintenance. Sparse readiness, restoration,
Python public API and a managed runner have separate paths and clocks.

Baseline `a21d3eed7088795763f511d9aed2fd81920e8201`, tree `d44249d5436e5bd4929e943f248a73d8b7a8bb68`, binary SHA-256 `127f86c2fe5620c497123e037294ed5a59a6ef7db2b37379f7b90fbd7f70f0c5`. The Cargo build took 498 seconds.

Candidate `f34040ea135004021bbc1f9ca83b186d2edb8d0f`, tree `307532292b498c4bdb247a548cb2a003cd849dec`, binary SHA-256 `a28683cf67ddaf61dc66c83786b45661382bf338ff0d50545972dc96e4d58696`. The Cargo build took 570 seconds.

The official baseline is merged main `dc1fd149032c586aeb5334095f6862d935b1c8fd`. Its complete tree is identical to the sealed ready-prefix candidate binary’s source tree above. Every baseline timing observation here is new.

The collector pinned every worker to logical CPU 8 on a
13th Gen Intel(R) Core(TM) i9-13900HX. Resource monitoring covered 4.013 seconds in
2 intervals. Available memory was at least
28.72 GiB, maximum sibling
utilization was 0.00% and maximum whole-host
utilization was 3.33%. All intervals passed.
The collector took 4.066 seconds inside the 240-second budget.
The registration, original process/output records, resources, identities and
build receipts accompany this report.

## Reproduction

The same archived baseline release binary was reused. All baseline observations in this registration were newly measured.

Each worker starts a fresh process, seeds the Rust operator, executes a full-value
settlement to qualify the fixture, then times one settlement on a fresh operator.
The direct watermark clock includes output materialization, collector delivery,
commit and eviction. It excludes seeding, checkpoint capture and the full-value
qualification. The maintained release worker instruments allocations while the
clock runs. A separate source check and binary digest for each arm appear in
`arms.json`. The Cargo artifact receipts retain complete release profiles,
features, build times, source roots, revisions and binary paths. The build helper reproduces both arms from source independently.
The ASOF lookup assembly analysis and the two original binary fingerprints are
in `eligible-tail.md` and the accompanying code records.

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
mkdir -p target/tail-lookup-on-main
cp "$EVIDENCE_DIR"/{build.sh,measure.py,verify.py,seal.py,protocol.md,sources.sha256} target/tail-lookup-on-main/
git switch --detach "$BASELINE_SHA"
bash target/tail-lookup-on-main/build.sh baseline "$BASELINE_SHA" "$SHARED_CARGO_TARGET"
git switch --detach "$CANDIDATE_SHA"
bash target/tail-lookup-on-main/build.sh candidate "$CANDIDATE_SHA" "$SHARED_CARGO_TARGET"
"$PYTHON" target/tail-lookup-on-main/seal.py "$CANDIDATE_SHA"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --kill-after=2 238 "$PYTHON" target/tail-lookup-on-main/measure.py "$CANDIDATE_SHA"
```

Each `WORKER` header in `workers.log` records the SHA-256 of the following
two original lines. The verifier reproduces the pooled P50, linearly
interpolated P95, both paired intervals and maintained classification, then
reconstructs the original partial observation file's digest. With 20 formal
observations per arm, P95 is descriptive. Each paired interval uses ranks
two and nine of ten pairwise changes, yielding 97.8515625% coverage under
the repository's independent-pair assumptions.
