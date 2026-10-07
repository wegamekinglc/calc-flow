# A3 ASOF output-range performance evidence

## Measured result

This dedicated compatible comparison completed all original eleven cases. Six have no-confirmed-regression and five are inconclusive under the maintained two-round +5% rule. No case establishes a greater-than-5% improvement or regression. The matrix does not establish overall equivalence or a material pipeline gain.

At 1M rows, contiguous full/projected output p50 decreased 2.58%/2.03%. The paired round intervals retain that modest direction for full output, but neither shape passes the greater-than-5% improvement rule. Full/projected fragmented p50 increased 3.90%/2.49%; their positive intervals cross +5%, so regression remains unresolved there. Small full output and both projection controls are also inconclusive. No passing case was rerun to select a faster result.

These custom shapes use the maintained engine-style statistics with the same common fixture against both clean release wheels. They are dedicated measurements, not a completed scheduled-suite CI run. Percentage changes below concern elapsed time (negative is faster), not throughput percentage.

## Timings

Each case has two rounds of ten adjacent AB/BA pairs, 20 measured samples per revision. Each round starts one fresh worker per revision, prepares one warm-up, then measures ten samples; there are four fresh workers per case, not a fresh PID for every sample. All 44 workers and all 440 measured timings plus 44 warm-ups are retained.

| Case rows/variant/batch             | Base p50 ms | Head p50 ms | p50 delta | Round A median [CI]      | Round B median [CI]    | Maintained verdict      |
|-------------------------------------|-------------|-------------|-----------|--------------------------|------------------------|-------------------------|
| 10,000/full/1,024                   | 11.737      | 12.041      | +2.59%    | -0.25% [-3.83, +16.01]   | +6.18% [+0.66, +19.99] | inconclusive            |
| 10,000/projected/1,024              | 10.031      | 9.881       | -1.50%    | -1.33% [-6.79, +3.89]    | +1.79% [-5.03, +4.07]  | no-confirmed-regression |
| 1,000,000/full/64,000               | 384.364     | 374.465     | -2.58%    | -3.42% [-4.93, -0.99]    | -2.03% [-4.95, -0.23]  | no-confirmed-regression |
| 1,000,000/projected/64,000          | 347.360     | 340.293     | -2.03%    | -2.02% [-4.25, +0.85]    | -2.33% [-3.89, -0.47]  | no-confirmed-regression |
| 1,000,000/full-null/64,000          | 354.039     | 352.110     | -0.54%    | -0.33% [-4.13, +2.60]    | +0.04% [-2.87, +3.00]  | no-confirmed-regression |
| 1,000,000/projected-null/64,000     | 309.968     | 301.313     | -2.79%    | -4.05% [-5.27, +2.25]    | -2.51% [-4.74, -1.71]  | no-confirmed-regression |
| 1,000,000/projected-reuse/64,000    | 211.805     | 206.954     | -2.29%    | -2.47% [-6.22, +0.43]    | -1.68% [-5.65, +1.53]  | no-confirmed-regression |
| 1,000,000/full-fragment/64,000      | 588.358     | 611.324     | +3.90%    | +2.16% [-0.40, +5.19]    | +2.84% [+1.13, +6.20]  | inconclusive            |
| 1,000,000/projected-fragment/64,000 | 536.400     | 549.782     | +2.49%    | +2.72% [+0.54, +4.68]    | +3.14% [+1.09, +5.47]  | inconclusive            |
| 10,000/projection-control/1,024     | 3.965       | 3.786       | -4.52%    | -11.50% [-23.57, +21.64] | +0.73% [-7.87, +10.58] | inconclusive            |
| 1,000,000/projection-control/64,000 | 6.348       | 6.505       | +2.48%    | -3.73% [-14.21, +12.02]  | +6.75% [-5.65, +13.42] | inconclusive            |

Intervals concern median per-pair percentage change, rather than the ratio of aggregate medians. They use the maintained deterministic order-statistic calculation with at least 95% coverage (97.8515625% for ten pairs), not Monte Carlo bootstrap. Both lower bounds above 5+1e-12 mean regression; otherwise any upper bound above that threshold means inconclusive; otherwise both upper bounds below negative threshold mean improved. The remaining outcomes mean no-confirmed-regression. Alternating pairs does not prove independence or a common distribution on shared hardware.

| Case rows/variant/batch             | Base p95 ms | Base p99 ms | Head p95 ms | Head p99 ms |
|-------------------------------------|-------------|-------------|-------------|-------------|
| 10,000/full/1,024                   | 12.627      | 12.683      | 14.062      | 14.851      |
| 10,000/projected/1,024              | 10.612      | 11.769      | 11.108      | 11.310      |
| 1,000,000/full/64,000               | 407.289     | 417.943     | 404.446     | 405.025     |
| 1,000,000/projected/64,000          | 365.066     | 365.788     | 350.687     | 357.417     |
| 1,000,000/full-null/64,000          | 372.954     | 373.473     | 368.442     | 369.362     |
| 1,000,000/projected-null/64,000     | 319.185     | 319.588     | 315.650     | 322.459     |
| 1,000,000/projected-reuse/64,000    | 222.809     | 222.850     | 216.820     | 224.381     |
| 1,000,000/full-fragment/64,000      | 613.403     | 619.183     | 629.351     | 630.829     |
| 1,000,000/projected-fragment/64,000 | 558.065     | 562.047     | 581.419     | 596.837     |
| 10,000/projection-control/1,024     | 4.628       | 4.793       | 5.492       | 5.978       |
| 1,000,000/projection-control/64,000 | 7.962       | 8.545       | 7.970       | 8.112       |

P95/P99 are descriptive interpolated quantiles of only 20 samples per revision. They do not establish a tail upper bound or tail improvement.

## Scope and complete oracle

The ASOF primary timer starts after both public ready sources are waiting. It includes event enqueue/backpressure, native matching/output planning/gather, sink receipt and final Arrow concatenation. Input construction, plan compilation, startup, full-column equality, EOF, final managed manifest and awaited job cleanup are outside that timer. Their cost is included in parent-observed preflight/full-case wall clocks.

Each input has six flat Arrow columns: UTC microsecond time, uint64 sequence, 64 symbols, exact eighth-unit float prices, independent 32–95-byte UTF-8 payloads, and nullable tags. Full output has twelve columns; projected output has four. Independent Arrow oracle construction checks every output column and canonical row order without sorting output. Null variants omit every fourth right row; reuse variants retain 64 right candidates per ingress chunk and repeat references; fragment variants split each left chunk into even/odd backing arrays. Workload journals and completed source metrics attest actual 1,024/64,000-row chunks, including 32k left halves in fragment cases.

Independent post-timing checks passed 396 ASOF full-payload/order/status proofs and 88 projection full-vector proofs, including all warm-ups. All 484 outputs have final v3 manifests with ended sources; ASOF status was sampled after awaited natural completion with task_count/task_errors/pending_left_rows/state_rows zero and the exact emitted count. Every worker exited zero and its recorded PID was absent. This establishes output, EOF and terminal-manifest evidence. It does not establish durable checkpoint restart: ordinary ready sources have no replay and the fixture never restarts from a checkpoint. J1 direct raw-V1 restore evidence belongs to the separate compaction report.

## Memory and resource admission

| Case rows/variant/batch             | Base max HWM GiB | Head max HWM GiB | Max pair sum GiB | Sum x1.25 GiB | 70% gate / Swap |
|-------------------------------------|------------------|------------------|------------------|---------------|-----------------|
| 10,000/full/1,024                   | 0.356            | 0.350            | 0.706            | 0.882         | pass / 0        |
| 10,000/projected/1,024              | 0.345            | 0.350            | 0.695            | 0.869         | pass / 0        |
| 1,000,000/full/64,000               | 1.673            | 1.587            | 3.260            | 4.075         | pass / 0        |
| 1,000,000/projected/64,000          | 1.542            | 1.522            | 2.993            | 3.741         | pass / 0        |
| 1,000,000/full-null/64,000          | 1.518            | 1.584            | 3.086            | 3.858         | pass / 0        |
| 1,000,000/projected-null/64,000     | 1.454            | 1.480            | 2.919            | 3.649         | pass / 0        |
| 1,000,000/projected-reuse/64,000    | 1.128            | 1.069            | 2.197            | 2.747         | pass / 0        |
| 1,000,000/full-fragment/64,000      | 1.680            | 1.659            | 3.338            | 4.173         | pass / 0        |
| 1,000,000/projected-fragment/64,000 | 1.550            | 1.544            | 3.079            | 3.849         | pass / 0        |
| 10,000/projection-control/1,024     | 0.357            | 0.348            | 0.704            | 0.880         | pass / 0        |
| 1,000,000/projection-control/64,000 | 0.495            | 0.495            | 0.988            | 1.235         | pass / 0        |

Process VmHWM includes native worker threads, input construction, all warm-ups/samples, full oracle and cleanup observations. Pair sums conservatively add the two workers' individual cumulative maxima, which need not occur simultaneously; they are not a direct synchronized job peak. Every observed VmSwap was zero. Each round's 1.25 times the sum stayed below 70% of the minimum observed MemAvailable in its original IPC records. Original kB lines and 1024-byte conversion remain in the raw logs. Descriptive two-round RSS observations do not establish a memory regression/improvement verdict or actual paid-credit refund.

The original conservative 10k→1M prediction was 91,144,704,000 bytes, above the 22,061,117,030-byte admission threshold. It remains preserved as a prediction refusal, not an observed 1M OOM. A separately authorized 100k/64k resource bridge, using unchanged worker/oracle/lifecycle and no subtraction of an import floor, predicted 1M at 12,557,107,200 bytes below 22,042,500,300 bytes. All original 1M shapes then passed actual preflight. The two bridge cases are excluded from every timing verdict and the original eleven-case inventory is unchanged.

Same-shape full-inventory preflight estimated 278.218 seconds including startup, construction, warm-up, all outputs/oracles/EOF/IPC and exit. Actual full matrix wall was 237.859 seconds. Its largest conservative observed pair sum was 3,584,421,888 bytes (full-fragment), or 4,480,527,360 bytes with the 1.25 factor. No fixture or user-duration limit was used to omit a required case.

## Source and build identity

| Identity                    | Baseline                                                         | Candidate                                                        |
|-----------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| Source commit               | eccb26973811bc476f0944b977ddedf8564b0237                         | 764843e634ae1a1da7a5b010095c3017349a25f4                         |
| Source tree                 | 5d5959091db3bfcb48698b2bb689df0cc5e41096                         | 5b81c4caf3ca0503eac6541d805c23ad97543719                         |
| Source SHA256               | a9cb90664b78e13c4df17fefc5b72ddc3a87f7f93527a4944cc2aff22d1bae2a | fe7fdb4506b9e4e348afc29401e5b6555cb09107ecbce461e3e3d2feb42922bf |
| Actual loaded native SHA256 | 06b152baf350be5d4ac21d9d8d4d5ca9481ef740e8d7d1ef62268098d44b3390 | e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889 |
| Wheel SHA256                | f68f4ac605a288d9ec63c1ffe61bfb1852d7fdbb33f6e18aec668425d27b75a8 | ab38ee224ee306e1ce8e6524419a6f45cd34170612a80d9e7dc149f327b07e69 |
| Cargo.lock SHA256           | 84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840 | 84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840 |
| Build-only seal SHA256      | d437e9ef68980f3e6e7f45a8302a4530326a7a4d4ca530e670fddde9ec5ad7a6 | 124d28b0eaa5ec8f2a1b0e4dbce34fa3bf61076e1249278fabf94cadd8f2e3cf |

Candidate production/build inputs are byte-identical to the independently approved 9983cfd0c3d096da7a60c3a96ae03b785f5ae955 source; measured 764843e6 adds its review document. Both exported snapshots were clean. Native Rust files were touched in owned exported sources before changed builds to prevent stale shared-cache reuse. Builds used Rust 1.88.0, cargo 1.88.0, Maturin 1.15.0, CPython 3.13.9, --release --locked, default connector-file plus pyo3/abi3-py313, empty RUSTFLAGS and jobs=2. Matching Cargo lock, build settings and worker dependency/thread identities are frozen. The warm Cargo directory accelerated compilation and is never treated as immutable evidence.

Baseline reused the freshly sealed ECCB wheel with actual native 06b152b..., rather than the older code-equivalent 22fcb/188d... wheel mentioned in historical static preparation. Both exact wheels, source archives, original build commands/logs and post-build file checks are preserved. Actual loaded native hashes were checked in all fresh workers, not inferred from branch names.

## Machine, fingerprints and quiet window

Environment: 13th Gen Intel(R) Core(TM) i9-13900HX, WSL2 Linux 5.15.167.4, 32 logical CPUs with affinity 0–31; configured Tokio/Polars threads 32, OMP/OpenBLAS/MKL threads 1. CPython 3.13.9, NumPy 2.5.2, PyArrow 24.0.0, DataFusion 54.0.0, Polars 1.44.2. Worker platform/dependency/thread environments matched across all 44 workers. Machine, dependency and per-case workload SHA256 records are embedded below.

Sampling ran 2026-10-06T18:40:05.034436+00:00 to 2026-10-06T18:44:02.893538+00:00 in the root-granted continuous quiet window. Root froze all team native builds/tests/preflights; A4/J2 remained frozen/static and no J1 worker ran concurrently. Observed one-minute Linux load ranged 1.141–2.798. WSL2 host activity cannot be fully excluded from inside the guest. This controlled team window and retained CIs do not turn inconclusive cases into proven equivalence. After every A3 native process exited, root resumed independent debug TDD; all final checking/archiving happened after sampling. J1 native preflight stayed paused.

## Structural benefit and remaining coverage

Independent source correctness tests, documented in [the A3 implementation analysis](stream-asof-output-ranges.md), reduce actual left source/range bookkeeping for a 1,024-row contiguous run from 1,024/1,024 visits to 1/1, remove the unused left positions vector, and allocate left spans by discontinuities. The constructor's original 57,344-byte RED is replaced by its GREEN allocation bound. The 256-byte/row +16 KiB planning reservation, Arrow backing-owner funding and layout10 state encoding stay unchanged. These structural/allocation checks do not imply a confirmed >5% whole-pipeline gain or reduce the logical funding budget.

Priority follow-up coverage is a maintained fragmented-range/permutation case matching the observed unresolved direction, then a matched output-planning microbenchmark to separate source/range work from matching/gather/runner cost. The existing scheduled ASOF engine case covers ordinary contiguous output; these custom full/projected/null/reuse/fragment shapes provide dedicated coverage. Wide flat payload, one-sided/zero column output, cancellation/paid retirement, and durable restart are supported by separate correctness inventories or remain unmeasured here. Nested/dictionary schemas retain their explicit rejection and are not new supported timing workloads. Any targeted remeasurement needs a stated unresolved question and fresh sealed evidence; this report does not justify rerunning passing cases to obtain a gain.

## Evidence, commands and independent recomputation

Raw root: `target/issue363-asof-output-range-perf/matrix-attempt1`. Full raw results SHA256 `8ef9aab5ab7544655d91b63619a449d0274dbf39f610e23e025df8b711e728fe`; independent proof SHA256 `c7deb31b33f29047be92a365b47a3b72e6779937846f45e20e48fdf60674b1d0`. Original v2 checker validates worker logs/oracles/fingerprints and CIs; separate v3 corrects only the maintained verdict rule, preserving original labels and all CI values. The v2 mixed-round classification bug and missing 1e-12 endpoint protection are not used for this final decision. Eleven stdlib synthetic cases matched an independently extracted maintained function. Independent final checking additionally compared every CI with maintained statistics, actual AB/BA UTC request order, batch/source dimensions, full inventory, final manifests, actual loaded native, PID/exit and worker-inclusive RAM/Swap. No engine was imported for postprocessing and no native case was rerun.

| Evidence                    | SHA256                                                           |
|-----------------------------|------------------------------------------------------------------|
| Static v2 seal              | 4f34a6c9fac95a325f1c4d62b2c010e0af87cc3acde625c4aa078c770c76e63c |
| Common maintained harness   | 6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5 |
| Unchanged paired driver     | f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e |
| Unchanged worker/fixture    | 51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f |
| v3 postprocessor            | 00b6b2f91b5b879c127bce25ff0a4af73035ed8da652a64015adf0d58529dfec |
| v3 seal                     | 86eb66656fbf9a29c8c9fe370597de15657664eb10c3453ce2085f4154d97108 |
| Verified v2 summary         | eda777ca9766462e6a24c7e186d34cc38e00acb5142361e9c0a6869231d94fde |
| Verified v3 summary         | 3b8123fe5060fc94472f41b7fce58e7246e4887cdb3a53cb84368c8b67654807 |
| Independent raw checker     | c3e4d597963cba95d45910fe20f4826cb35df7ad1348abede4ba59ebb5ebeb77 |
| Preflight inventory proof   | 0a20a1359f287e20885725e406f7d63bfeab09949b5f244eda81b73bb5430355 |
| Resource bridge helper seal | 9fb720dc4711533d46fb627484dc64ad5bec5feb90ad73618bfc166fd5fe2c09 |

```bash
BENCH_PY=.claude/worktrees/join-polars-analysis/target/benchmark-venv/bin/python
PYTHONPATH=$PWD/target/issue363-asof-output-range-perf/release-v1/source \
$BENCH_PY target/issue363-asof-output-range-perf/paired.py \
  --source target/issue363-asof-output-range-perf/release-v1/source \
  --baseline-release target/issue363-compaction-perf/release-v1/baseline/wheel/release.json \
  --candidate-release target/issue363-asof-output-range-perf/release-v1/wheel/release.json \
  --resource-preflight target/issue363-asof-output-range-perf/preflight-100k-bridge-attempt1/results.json \
  --destination target/issue363-asof-output-range-perf/matrix-attempt1 --quiet-granted
# These original destinations are immutable; use a new destination for any authorized rerun.
$BENCH_PY target/issue363-compaction-perf/check.py \
  target/issue363-asof-output-range-perf/matrix-attempt1 --stage a3
$BENCH_PY target/issue363-compaction-perf/checker_verdict_v3.py \
  target/issue363-asof-output-range-perf/matrix-attempt1/verified-v2-summary.json
$BENCH_PY target/issue363-asof-output-range-perf/check_after_matrix_v1.py \
  target/issue363-asof-output-range-perf/matrix-attempt1
```

Build commands and original argv/environment are retained in `release-v1/build-input.json`, `build-result.json`, both `wheel/build.json`, `wheel/release.json` and their original logs. Full-matrix classification uses only the original eleven paired cases; no preflight/bridge sample is included. There were no failed native attempts in this collection. The historical 10k→1M prediction refusal remains preserved.

### Frozen archive

```json
{
  "contract": "issue363-a3-evidence-archive-v1",
  "created_utc": "2026-10-06T18:54:49.532450+00:00",
  "archive_path": "target/issue363-asof-output-range-perf/evidence-a3-v1.tar.gz",
  "archive_sha256": "f02bba3b7bd3c759b08110e3db3505482da0b9f032db39ad9674184af40d5010",
  "archive_bytes": 109885045,
  "files_manifest_sha256": "d8db6f7af958eeb3761f035212e8b58f37a68505cc386ed7efb9140b218bdbc9",
  "file_count_excluding_manifest": 1491,
  "archive_contents": "all matrix/preflight raw journals/statuses/manifests plus sealed scripts/build records/two exact wheels/two source archives",
  "excluded": [
    "later Markdown report",
    "installed duplicate wheels/native modules",
    "mutable Cargo output cache",
    "Rust owned closure binaries unrelated to A3 wheel timing"
  ]
}
```

This local archive includes all raw matrix/preflight IPC and terminal manifests, exact wheels, clean source archives, original build records and sealed scripts. Its paths remain local until parent publication; the embedded arrays below allow a PR reader to recompute all timing statistics independently. The archive excludes this later Markdown report, duplicate installed modules, mutable Cargo output and unrelated owned Rust-core binaries. Archive SHA256 therefore does not recursively depend on a report that contains it.

### Embedded compatible fingerprints

```json
{
  "contract": "issue363-a3-comparison-fingerprints-v1",
  "machine": {
    "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43",
    "logical_cpus": 32,
    "cpu_affinity": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ],
    "tokio_worker_threads": "32",
    "polars_threads": 32,
    "thread_environment": {
      "MKL_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "POLARS_MAX_THREADS": "32",
      "XLA_FLAGS": null
    },
    "controller_cpu_model": "13th Gen Intel(R) Core(TM) i9-13900HX"
  },
  "machine_sha256": "27c922da1e41880e63a86629c28781c7859db41309991fa8afd0c36921257e1f",
  "dependencies": {
    "python": "3.13.9",
    "numpy": "2.5.2",
    "pyarrow": "24.0.0",
    "packages": {
      "TA-Lib": "0.7.1",
      "datafusion": "54.0.0",
      "jax": "0.11.1",
      "jaxlib": "0.11.1",
      "polars": "1.44.2"
    },
    "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840",
    "build_profile": "release",
    "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)\nbinary: rustc\ncommit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc\ncommit-date: 2025-06-23\nhost: x86_64-unknown-linux-gnu\nrelease: 1.88.0\nLLVM version: 20.1.5",
    "features": [
      "default=connector-file",
      "pyo3/abi3-py313"
    ],
    "RUSTFLAGS": "",
    "build_jobs": 2
  },
  "dependency_sha256": "021a3386de03c2a7d6b49501c74bf4a529cd55add3dec50fb8aca523cc87e378",
  "workloads": [
    {
      "id": "issue363/a3/10000/full/batch-1024",
      "workload": {
        "id": "issue363/a3/10000/full/batch-1024",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 10000,
        "fixture_batch_rows": 1024,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "full",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "214cfa591df9d2ff8a67bb6750abc00d12d926f945398ddca64960de96f53572"
    },
    {
      "id": "issue363/a3/10000/projected/batch-1024",
      "workload": {
        "id": "issue363/a3/10000/projected/batch-1024",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 10000,
        "fixture_batch_rows": 1024,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "projected",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "42bca2b9b462bf219539669ba2b6bb43a20d5fc89d34df7ca50506b86f4bf196"
    },
    {
      "id": "issue363/a3/1000000/full/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/full/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "full",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "64d130f353b8235288cb96bca99eb7fe79fe6d569fb2bd3fc61df49e475348f7"
    },
    {
      "id": "issue363/a3/1000000/projected/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/projected/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "projected",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "84327a8068116ce7980b7e127a0f0e2928aac9133821c30896e22492cbbec1a8"
    },
    {
      "id": "issue363/a3/1000000/full-null/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/full-null/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "full-null",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "9e446e33c43e129953f7ba1630f161983544b7987f3d00731738eaa78fbb63f0"
    },
    {
      "id": "issue363/a3/1000000/projected-null/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/projected-null/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "projected-null",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "3496c72677246c9d1d13d49653b18b0084ba5fbf4d827cf956d4ec23248b3016"
    },
    {
      "id": "issue363/a3/1000000/projected-reuse/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/projected-reuse/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "projected-reuse",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "1a9ab7b797b6f9f4ecb576328390a9b559f647102d40d1f92b4d0404144027ab"
    },
    {
      "id": "issue363/a3/1000000/full-fragment/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/full-fragment/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "full-fragment",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "d9583c6febfbc45cc0d7df4f3f1fc62f45990350b459918f4ccd5a319fb67cc2"
    },
    {
      "id": "issue363/a3/1000000/projected-fragment/batch-64000",
      "workload": {
        "id": "issue363/a3/1000000/projected-fragment/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
        "variant": "projected-fragment",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "ba3500301d2b8df9e5ed78c0e47210c7cdc451db014ed20d1c44feed7be9d1b1"
    },
    {
      "id": "issue363/a3/control/10000/batch-1024",
      "workload": {
        "id": "issue363/a3/control/10000/batch-1024",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "projection",
        "rows": 10000,
        "fixture_batch_rows": 1024,
        "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "513e82e2e597e5305fc40b4d205269a822820c5ad4c9c7f3eea1b1ce2f93b60e"
    },
    {
      "id": "issue363/a3/control/1000000/batch-64000",
      "workload": {
        "id": "issue363/a3/control/1000000/batch-64000",
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "projection",
        "rows": 1000000,
        "fixture_batch_rows": 64000,
        "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
        "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
        "driver_sha256": "f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e",
        "worker_sha256": "51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f"
      },
      "sha256": "ed493303d301b2e6abf9b9c644d081210b50fdf71c976686d2ce8ac780eb428a"
    }
  ],
  "same_worker_environment_all_44": true,
  "original_v2_prepared_files_unchanged": true,
  "identity_historical_static_22_wheel_not_used": true,
  "actual_baseline_native": "06b152baf350be5d4ac21d9d8d4d5ca9481ef740e8d7d1ef62268098d44b3390",
  "actual_candidate_native": "e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889"
}
```

### Embedded original measured arrays (seconds)

These are the exact two ten-pair rounds, preserved in original pair order. Warm-ups, bridge and preflight observations are excluded. Recompute each paired change as `100 * (candidate / baseline - 1)` and use sorted ranks 2 and 9 for the ten-pair interval; apply the maintained v3 rule to both rounds. P50/P95/P99 use all twenty observations per revision. Exact array JSON is copied without rounding.

```json
{
  "cases": [
    {
      "id": "issue363/a3/10000/full/batch-1024",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.012013078,
          0.01189975,
          0.011470555,
          0.012696613,
          0.011745175,
          0.011715685,
          0.011750969,
          0.012623672,
          0.012473459,
          0.012071535
        ],
        [
          0.011728,
          0.01218739,
          0.011006082,
          0.01167619,
          0.011555014,
          0.010974322,
          0.010815214,
          0.011757398,
          0.011443503,
          0.011500903
        ]
      ],
      "candidate": [
        [
          0.013942365,
          0.011570242,
          0.013306455,
          0.012808531,
          0.011582498,
          0.013495602,
          0.010617606,
          0.012986611,
          0.012280408,
          0.011608595
        ],
        [
          0.015047632,
          0.01297925,
          0.0117986,
          0.01401028,
          0.011801828,
          0.011617239,
          0.010985942,
          0.011186861,
          0.011518565,
          0.013085786
        ]
      ]
    },
    {
      "id": "issue363/a3/10000/projected/batch-1024",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.010225541,
          0.009910099,
          0.009400045,
          0.009994263,
          0.010407907,
          0.010369768,
          0.009273127,
          0.012058519,
          0.010029402,
          0.010181977
        ],
        [
          0.009792744,
          0.009811959,
          0.009338218,
          0.010033598,
          0.010278749,
          0.010536132,
          0.009612591,
          0.010126548,
          0.00976847,
          0.010346045
        ]
      ],
      "candidate": [
        [
          0.011095105,
          0.00954066,
          0.009548889,
          0.009995627,
          0.009701248,
          0.009826077,
          0.009633675,
          0.010831431,
          0.009901559,
          0.010040585
        ],
        [
          0.010191797,
          0.009650492,
          0.0095211,
          0.009684784,
          0.010540702,
          0.010005801,
          0.009859828,
          0.009470405,
          0.009926832,
          0.011359896
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/full/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.406587612,
          0.381292786,
          0.37570429,
          0.402926354,
          0.394667242,
          0.385099661,
          0.376133973,
          0.373918837,
          0.420606266,
          0.395071977
        ],
        [
          0.393770612,
          0.389155939,
          0.362008201,
          0.387109943,
          0.383047759,
          0.390014472,
          0.369935577,
          0.37224027,
          0.383628636,
          0.378712869
        ]
      ],
      "candidate": [
        [
          0.37454745,
          0.374382071,
          0.363817462,
          0.383063148,
          0.376156559,
          0.366639753,
          0.372255333,
          0.370217537,
          0.405169783,
          0.404407512
        ],
        [
          0.379492218,
          0.38824958,
          0.361047925,
          0.377558885,
          0.37693643,
          0.359513526,
          0.351619939,
          0.367849594,
          0.373928253,
          0.381050232
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/projected/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.342582101,
          0.346107803,
          0.332381122,
          0.348611377,
          0.365967921,
          0.365018686,
          0.337305569,
          0.351783855,
          0.358301213,
          0.336580819
        ],
        [
          0.354636372,
          0.349043722,
          0.332007237,
          0.353727389,
          0.357633057,
          0.344262771,
          0.340141223,
          0.357023763,
          0.342782645,
          0.328448312
        ]
      ],
      "candidate": [
        [
          0.335158851,
          0.350244481,
          0.322948867,
          0.336430318,
          0.359099923,
          0.33527378,
          0.336371776,
          0.346714786,
          0.34305597,
          0.339448676
        ],
        [
          0.342414408,
          0.347403504,
          0.327179433,
          0.339954326,
          0.34305804,
          0.340631427,
          0.332739828,
          0.343439687,
          0.343840988,
          0.320284377
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/full-null/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.351404232,
          0.342246832,
          0.345103575,
          0.355990355,
          0.354448822,
          0.373603267,
          0.346075701,
          0.353629361,
          0.360732423,
          0.354756289
        ],
        [
          0.36014683,
          0.358294481,
          0.332874937,
          0.364822867,
          0.353223494,
          0.372919679,
          0.340394983,
          0.348526623,
          0.350276855,
          0.371532138
        ]
      ],
      "candidate": [
        [
          0.360557606,
          0.349198638,
          0.365505826,
          0.357791325,
          0.348055452,
          0.351793283,
          0.346066099,
          0.339039805,
          0.356385715,
          0.352427348
        ],
        [
          0.3555827,
          0.34917012,
          0.336806582,
          0.368381313,
          0.363824672,
          0.369591951,
          0.349593372,
          0.362208946,
          0.340233087,
          0.350012785
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/projected-null/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.312530242,
          0.31729588,
          0.302961391,
          0.311713018,
          0.319688978,
          0.295492028,
          0.306856112,
          0.308253287,
          0.30767909,
          0.31213185
        ],
        [
          0.302147503,
          0.306678018,
          0.298527522,
          0.311597243,
          0.319158088,
          0.316162094,
          0.298555097,
          0.318699732,
          0.315507315,
          0.308338871
        ]
      ],
      "candidate": [
        [
          0.298301343,
          0.298950182,
          0.29042136,
          0.298812497,
          0.307031352,
          0.324160585,
          0.306423505,
          0.315202386,
          0.302670265,
          0.295685122
        ],
        [
          0.300188782,
          0.301448731,
          0.29189783,
          0.302865075,
          0.302504418,
          0.301176644,
          0.286575624,
          0.312635676,
          0.30873625,
          0.295578315
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/projected-reuse/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.222806304,
          0.212146375,
          0.208502625,
          0.208194415,
          0.208411164,
          0.209960653,
          0.211805906,
          0.214157333,
          0.214720736,
          0.206870993
        ],
        [
          0.222859819,
          0.216256432,
          0.214737887,
          0.211803207,
          0.216116037,
          0.209138369,
          0.209995851,
          0.214986387,
          0.210716557,
          0.209830915
        ]
      ],
      "candidate": [
        [
          0.211151361,
          0.214315692,
          0.202538543,
          0.206230122,
          0.203449303,
          0.204578008,
          0.198628967,
          0.199000829,
          0.214378509,
          0.207758296
        ],
        [
          0.226271267,
          0.214039308,
          0.202611023,
          0.210450446,
          0.205328343,
          0.216322349,
          0.205111136,
          0.209700932,
          0.195750408,
          0.207678439
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/full-fragment/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.600426641,
          0.620627487,
          0.574097409,
          0.613022403,
          0.585080441,
          0.581980818,
          0.5766373,
          0.593484218,
          0.587818947,
          0.588897825
        ],
        [
          0.601327874,
          0.602790973,
          0.585351954,
          0.584265922,
          0.609580723,
          0.586578693,
          0.568322628,
          0.577935835,
          0.597109974,
          0.610508935
        ]
      ],
      "candidate": [
        [
          0.604231447,
          0.607315582,
          0.589503482,
          0.614214528,
          0.61256303,
          0.612184777,
          0.586082452,
          0.591104847,
          0.610463614,
          0.623692509
        ],
        [
          0.616982696,
          0.603784596,
          0.597986576,
          0.631198993,
          0.616482615,
          0.622930889,
          0.578958301,
          0.607205119,
          0.617688776,
          0.629253457
        ]
      ]
    },
    {
      "id": "issue363/a3/1000000/projected-fragment/batch-64000",
      "scope": "ready-enqueue-to-arrow/asof-output-range-shapes-v1",
      "baseline": [
        [
          0.563042242,
          0.557802667,
          0.532277177,
          0.541390766,
          0.533966838,
          0.525310178,
          0.522976174,
          0.524706016,
          0.542805162,
          0.532036856
        ],
        [
          0.543345107,
          0.525946994,
          0.517246931,
          0.5410184,
          0.542305945,
          0.541102964,
          0.515745915,
          0.538832959,
          0.540758835,
          0.530024947
        ]
      ],
      "candidate": [
        [
          0.580404118,
          0.566827244,
          0.547316111,
          0.553072154,
          0.53513499,
          0.553967211,
          0.544958256,
          0.549242926,
          0.545746453,
          0.545996732
        ],
        [
          0.60069179,
          0.544328774,
          0.531677858,
          0.561807019,
          0.571959397,
          0.561817211,
          0.52137998,
          0.550320348,
          0.551032081,
          0.534306722
        ]
      ]
    },
    {
      "id": "issue363/a3/control/10000/batch-1024",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "baseline": [
        [
          0.004290137,
          0.00383338,
          0.004360089,
          0.003554287,
          0.004125009,
          0.003779883,
          0.004617198,
          0.004834659,
          0.004077515,
          0.00408512
        ],
        [
          0.003990253,
          0.003764524,
          0.004094501,
          0.003625785,
          0.003750384,
          0.003917895,
          0.003773497,
          0.004094821,
          0.003567523,
          0.003940618
        ]
      ],
      "candidate": [
        [
          0.003866276,
          0.004078863,
          0.003788338,
          0.004323305,
          0.00354036,
          0.003423164,
          0.003481735,
          0.003694995,
          0.006099622,
          0.003488498
        ],
        [
          0.004408087,
          0.003673765,
          0.003660309,
          0.003824122,
          0.004147005,
          0.003783753,
          0.003754057,
          0.003772529,
          0.005460304,
          0.004018256
        ]
      ]
    },
    {
      "id": "issue363/a3/control/1000000/batch-64000",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "baseline": [
        [
          0.00584289,
          0.007857229,
          0.005922539,
          0.005682852,
          0.006245258,
          0.007923608,
          0.006389588,
          0.0060891,
          0.00587639,
          0.006350351
        ],
        [
          0.006544119,
          0.005932983,
          0.006543632,
          0.006528651,
          0.005961282,
          0.008691174,
          0.006366985,
          0.006356545,
          0.006344735,
          0.00630026
        ]
      ],
      "candidate": [
        [
          0.006107289,
          0.006740935,
          0.005601861,
          0.006151129,
          0.006996205,
          0.00655168,
          0.006120258,
          0.00571749,
          0.007960732,
          0.006143888
        ],
        [
          0.00731983,
          0.00637454,
          0.006485148,
          0.007404826,
          0.0081475,
          0.006524412,
          0.006808369,
          0.005997371,
          0.00676155,
          0.006059075
        ]
      ]
    }
  ]
}
```

### Embedded independent proof and exact RSS observations

```json
{
  "contract": "issue363-a3-independent-proof-v1",
  "matrix_sha256": "8ef9aab5ab7544655d91b63619a449d0274dbf39f610e23e025df8b711e728fe",
  "v3_sha256": "3b8123fe5060fc94472f41b7fce58e7246e4887cdb3a53cb84368c8b67654807",
  "fresh_workers": 44,
  "worker_pids": [
    1259576,
    1259578,
    1260606,
    1260608,
    1261637,
    1261639,
    1262580,
    1262582,
    1263523,
    1263525,
    1264581,
    1264583,
    1265618,
    1265620,
    1266655,
    1266657,
    1267729,
    1267731,
    1268761,
    1268763,
    1269799,
    1269801,
    1270922,
    1270924,
    1271954,
    1271956,
    1272908,
    1272910,
    1274109,
    1274111,
    1275140,
    1275142,
    1276188,
    1276190,
    1277301,
    1277303,
    1278343,
    1278345,
    1279196,
    1279198,
    1280049,
    1280051,
    1280904,
    1280906
  ],
  "owned_worker_pids_live": 0,
  "all_worker_exit_codes": 0,
  "measured_timings": 440,
  "warmups": 44,
  "asof_full_payload_order_proofs": 396,
  "projection_full_vector_proofs": 88,
  "terminal_manifest_eof_proofs": 484,
  "durable_restart_observed": false,
  "worker_swap_bytes": 0,
  "comparison_environment": {
    "cpu_affinity": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ],
    "logical_cpus": 32,
    "numpy": "2.5.2",
    "packages": {
      "TA-Lib": "0.7.1",
      "datafusion": "54.0.0",
      "jax": "0.11.1",
      "jaxlib": "0.11.1",
      "polars": "1.44.2"
    },
    "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43",
    "polars_threads": 32,
    "pyarrow": "24.0.0",
    "python": "3.13.9",
    "thread_environment": {
      "MKL_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "POLARS_MAX_THREADS": "32",
      "XLA_FLAGS": null
    },
    "tokio_worker_threads": "32"
  },
  "whole_matrix_seconds": 237.859102,
  "load_average_1m_min_max": [
    1.14111328125,
    2.79833984375
  ],
  "cases": [
    {
      "id": "issue363/a3/10000/full/batch-1024",
      "worker_peaks_bytes": {
        "baseline": [
          381784064,
          373563392
        ],
        "candidate": [
          376143872,
          373710848
        ]
      },
      "max_sum_two_worker_hwm_bytes": 757927936,
      "sum_two_worker_hwm_bytes_by_round": [
        757927936,
        747274240
      ],
      "minimum_observed_available_bytes_by_round": [
        30909566976,
        30912962560
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 7.318617145996541
    },
    {
      "id": "issue363/a3/10000/projected/batch-1024",
      "worker_peaks_bytes": {
        "baseline": [
          370409472,
          370651136
        ],
        "candidate": [
          376238080,
          366788608
        ]
      },
      "max_sum_two_worker_hwm_bytes": 746647552,
      "sum_two_worker_hwm_bytes_by_round": [
        746647552,
        737439744
      ],
      "minimum_observed_available_bytes_by_round": [
        30911991808,
        30918406144
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 5.056146260991227
    },
    {
      "id": "issue363/a3/1000000/full/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1625825280,
          1796055040
        ],
        "candidate": [
          1618423808,
          1704251392
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3500306432,
      "sum_two_worker_hwm_bytes_by_round": [
        3244249088,
        3500306432
      ],
      "minimum_observed_available_bytes_by_round": [
        28717711360,
        28476702720
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 30.03558899697964
    },
    {
      "id": "issue363/a3/1000000/projected/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1655672832,
          1579405312
        ],
        "candidate": [
          1524682752,
          1634226176
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3213631488,
      "sum_two_worker_hwm_bytes_by_round": [
        3180355584,
        3213631488
      ],
      "minimum_observed_available_bytes_by_round": [
        28556570624,
        28514779136
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 28.06112889101496
    },
    {
      "id": "issue363/a3/1000000/full-null/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1630093312,
          1584734208
        ],
        "candidate": [
          1683980288,
          1701322752
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3314073600,
      "sum_two_worker_hwm_bytes_by_round": [
        3314073600,
        3286056960
      ],
      "minimum_observed_available_bytes_by_round": [
        28768227328,
        28620824576
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 31.0556564849976
    },
    {
      "id": "issue363/a3/1000000/projected-null/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1529167872,
          1561485312
        ],
        "candidate": [
          1589387264,
          1572982784
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3134468096,
      "sum_two_worker_hwm_bytes_by_round": [
        3118555136,
        3134468096
      ],
      "minimum_observed_available_bytes_by_round": [
        28585070592,
        28551622656
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 28.316175981017295
    },
    {
      "id": "issue363/a3/1000000/projected-reuse/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1153097728,
          1211154432
        ],
        "candidate": [
          1106685952,
          1148305408
        ]
      },
      "max_sum_two_worker_hwm_bytes": 2359459840,
      "sum_two_worker_hwm_bytes_by_round": [
        2259783680,
        2359459840
      ],
      "minimum_observed_available_bytes_by_round": [
        29435564032,
        29423853568
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 19.579617978975875
    },
    {
      "id": "issue363/a3/1000000/full-fragment/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1688137728,
          1803599872
        ],
        "candidate": [
          1769697280,
          1780822016
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3584421888,
      "sum_two_worker_hwm_bytes_by_round": [
        3457835008,
        3584421888
      ],
      "minimum_observed_available_bytes_by_round": [
        28480184320,
        28447453184
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 40.53913050799747
    },
    {
      "id": "issue363/a3/1000000/projected-fragment/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          1607811072,
          1664770048
        ],
        "candidate": [
          1657540608,
          1641811968
        ]
      },
      "max_sum_two_worker_hwm_bytes": 3306582016,
      "sum_two_worker_hwm_bytes_by_round": [
        3265351680,
        3306582016
      ],
      "minimum_observed_available_bytes_by_round": [
        28411404288,
        28419817472
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 37.73758076300146
    },
    {
      "id": "issue363/a3/control/10000/batch-1024",
      "worker_peaks_bytes": {
        "baseline": [
          383332352,
          383062016
        ],
        "candidate": [
          363626496,
          373256192
        ]
      },
      "max_sum_two_worker_hwm_bytes": 756318208,
      "sum_two_worker_hwm_bytes_by_round": [
        746958848,
        756318208
      ],
      "minimum_observed_available_bytes_by_round": [
        30815715328,
        30930288640
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 3.8931212319876067
    },
    {
      "id": "issue363/a3/control/1000000/batch-64000",
      "worker_peaks_bytes": {
        "baseline": [
          525406208,
          531795968
        ],
        "candidate": [
          531324928,
          528842752
        ]
      },
      "max_sum_two_worker_hwm_bytes": 1060638720,
      "sum_two_worker_hwm_bytes_by_round": [
        1056731136,
        1060638720
      ],
      "minimum_observed_available_bytes_by_round": [
        30646697984,
        30643728384
      ],
      "conservative_1_25_sum_hwm_within_70pct_available": true,
      "whole_case_elapsed_seconds": 6.1688518120208755
    }
  ],
  "maintained_statistics_sha256": "c84b6065a7c8ca0ee0d123377df3750c7fd82e2eb0f50443af33161117757bcd"
}
```
