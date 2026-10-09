# J2a current-runtime measurements and CPU profiles

Recorded on 2026-10-09 for issue #363 and draft PR #393. The candidate still
fails performance acceptance: the two previously sealed Lookup regressions
remain. The new steady control improves; the other new comparisons are
inconclusive. Product source was frozen throughout all measurement and sampling.

Baseline main: 9db4cd2563ad0ef26e52fb173cb29acf1c504fc2.
Local candidate: 6cbbcf506f0d90c09f8adc785d62a6ce3fe7c373.
PR source-equivalent commit: fa844f576eeef190d6b6c91b2599e1e3e7d77d1c.
The latter two share Git tree 033a4659531ff9d8985251ced045947242db13bc.

## Sealed paired results

Each case has two rounds of ten adjacent alternating AB/BA pairs: twenty timed
observations per side. The four earlier cases are reused without resampling;
five lifecycle cases were added. Change compares pooled P50 values. Verdicts
are copied from the maintained paired rule, with its +5% regression threshold.
No aggregate acceptance verdict is inferred from this selected set.

| Case                            | Main P50 ms | Candidate P50 ms | Change  | Maintained verdict |
|---------------------------------|-------------|------------------|---------|--------------------|
| Lookup 1M / batch 64000         | 1069.507    | 1281.843         | +19.85% | regression         |
| Lookup 100k / batch 1024        | 108.305     | 149.257          | +37.81% | regression         |
| Retained interval Join 100k     | 756.196     | 700.944          | -7.31%  | inconclusive       |
| Projection 1M                   | 8.904       | 8.174            | -8.19%  | inconclusive       |
| ASOF 1M                         | 406.169     | 404.442          | -0.43%  | inconclusive       |
| Prepare + 500 rows, compact 60k | 120.034     | 115.967          | -3.39%  | inconclusive       |
| Prepare + 500 rows, steady 60k  | 0.676       | 0.551            | -18.54% | improved           |
| Capture dirty 20k               | 42.858      | 42.898           | +0.09%  | inconclusive       |
| Restore full 20k                | 53.086      | 54.129           | +1.96%  | inconclusive       |

The per-round statistic is the median of paired percentage changes, rather
than the pooled P50 ratio. Nominal interval coverage under the iid assumption
is 0.978515625 for these ten-pair rounds. The shared-host measurements do not
verify that assumption or guarantee realized coverage. No new statistic or
threshold was substituted.

| Case                            | Round 1 median [interval] % | Round 2 median [interval] % |
|---------------------------------|-----------------------------|-----------------------------|
| Lookup 1M / batch 64000         | +25.86% [+17.22, +29.78]    | +19.58% [+14.22, +30.65]    |
| Lookup 100k / batch 1024        | +38.14% [+28.96, +50.12]    | +38.33% [+32.69, +45.15]    |
| Retained interval Join 100k     | -5.30% [-11.85, +2.90]      | -9.54% [-11.90, +7.72]      |
| Projection 1M                   | -3.86% [-25.57, +25.53]     | -13.70% [-25.07, +8.24]     |
| ASOF 1M                         | +22.29% [-9.42, +106.93]    | -3.43% [-17.62, +3.22]      |
| Prepare + 500 rows, compact 60k | -2.15% [-4.46, +5.27]       | -0.52% [-7.75, +1.50]       |
| Prepare + 500 rows, steady 60k  | -17.72% [-21.37, -14.86]    | -19.13% [-21.39, -15.11]    |
| Capture dirty 20k               | +1.49% [-3.30, +2.86]       | +0.33% [-4.54, +14.90]      |
| Restore full 20k                | -0.28% [-8.68, +21.88]      | +1.01% [-1.71, +4.55]       |

ASOF and projection are unchanged-path controls. Inconclusive timings establish
neither equivalence nor a confirmed speedup. The compact/steady ratio is only
an informational ratio of independently measured aggregate medians.

## Timer and correctness scope

Python Join and ASOF use the maintained ready-enqueue-to-Arrow boundary,
including native admission, settlement and output materialization. Their
reference output oracles passed. ASOF uses four worker processes: one per side
per round, with ten timed calls per worker.

Each Rust observation comes from a fresh benchmark process and is the median
of thirty Criterion Duration/iteration points. There are forty processes per
case. P95 across these observations describes process medians; it is not a
single-handler P95 or P99. Mandatory probe cardinality, snapshot structure and
restore-success checks passed. No new full-row restore/replay oracle is claimed.

The compaction and steady cases time prepare_checkpoint_async plus a 500-row
handler. Fixture construction is outside this timer. The 60k fixture parameter
counts total rows across both inputs, plus six epoch-arming marker rows.
These cases do not establish the older pure-handler compaction tail-latency
gate. Capture times the maintained initial dirty checkpoint of 10k rows per
side, with no carried base. Restore times the native operator restore of a
20k-total-row base-plus-delta fixture, also with six marker rows.
It does not cover a durable runner restart or state-size scaling.

## Current CPU distributions

All percentages below describe retained user-space CPU samples. Inclusive
counts overlap and must not be added. They are not wall-time or off-CPU shares.
Native probe sample proportions decrease while row slicing and output
materialization remain substantial.

| Workload | Symbol / group                       | Attribution | Main   | Candidate |
|----------|--------------------------------------|-------------|--------|-----------|
| Lookup   | materialize_output_record            | inclusive   | 30.81% | 31.64%    |
| Lookup   | native_matches                       | inclusive   | 23.46% | 12.92%    |
| Lookup   | RecordBatch::slice                   | inclusive   | 20.41% | 19.85%    |
| Lookup   | tokio:: prefix retained leaf samples | self        | 0.17%  | 6.59%     |
| Retained | materialize_output_record            | inclusive   | 36.17% | 39.90%    |
| Retained | native_matches                       | inclusive   | 11.00% | 5.51%     |
| Retained | RecordBatch::slice                   | inclusive   | 6.87%  | 8.41%     |

The Lookup payload uses Float64 columns outside the owned-copy fixed-width
whitelist in [owned_copy.rs](../../../crates/calc-flow/src/operator/join/columnar/owned_copy.rs).
Native key eligibility therefore does not remove the generic row path.
Source inspection also identifies added per-row Quantum::step calls in generic
admission. Together with the tokio:: leaf change, this supports isolating
admission/yield scheduling costs first; it does not quantify the cause of the
entire wall-time regression. The tokio:: prefix is a precisely defined leaf
group, rather than a claim to measure all scheduler work.

Main-only checkpoint profiles isolate the original timed windows:

| Main-only workload | Symbol                              | Attribution | CPU samples |
|--------------------|-------------------------------------|-------------|-------------|
| Compaction         | RowIpcEncoder::encode               | inclusive   | 60.81%      |
| Compaction         | sha256::x86::digest_blocks          | self        | 21.54%      |
| Capture            | RowIpcEncoder::encode               | inclusive   | 59.07%      |
| Capture            | sha256::x86::digest_blocks          | self        | 22.06%      |
| Restore            | restore_sides_from_segments_checked | inclusive   | 88.86%      |
| Restore            | StreamReader<R>::maybe_next         | inclusive   | 41.11%      |
| Restore            | StreamReader<R>::try_new            | inclusive   | 31.47%      |
| Restore            | install_restored_rows               | inclusive   | 10.94%      |

Compaction and capture point toward row IPC encoding and checksumming; restore
points toward IPC parsing/validation and row/index installation. These are
separate from a native dictionary probe optimization.

ASOF's main self samples include encode_rows 10.52%, admission_identities
9.15%, and match_output_prefix 9.53%; candidate values are 10.48%, 9.46% and
8.99%. These diagnostic shapes do not prove ASOF equivalence or historical
A4 gains.

## Sampling, identities and limitations

Six Python profiles each retain two timed samples after one warmup. Main
checkpoint profiles retain exactly one hundred original timer windows for each
of compact, steady, capture and restore. Timed CPU counts are respectively
53,793, 331, 22,842 and 24,050, with no empty windows. The steady profile has
only three samples per window at its median and supports coarse attribution.

The sampler is an owned samply 0.13.1 build using user-space cpu-clock at
4000 Hz. Kernel sampling and context-switch/off-CPU synthesis are disabled.
Python CLOCK_MONOTONIC windows are cross-checked by six outside-window CRC
markers; Rust Instant coordinates are verified against the exact Linux Rust
1.88.0 implementation and launch/exit envelope. Native paths, binary hashes,
profile code IDs, sidecar breakpad IDs, process generations and symbol address
coverage are checked. Retained samples have zero missing stacks or unresolved
leaves/native frames.

The separately built unstripped Python diagnostic modules match the sealed
wheel modules in all allocated ELF sections except the build-id note.
They have distinct file hashes and are not claimed to be the same binary.
Rust profiles use a separately sealed baseline bench overlay whose marker is
emitted after each original timer; sealed paired timings use the unmodified
bench. The overlays do not alter production source.

Lost-event warnings are preserved: Lookup candidate 51, ASOF candidate 24,
Rust compact 6 and Rust steady 71. These refer to the whole record stream;
their event types and locations within timed windows are unknown. They cannot
be divided by timed CPU counts to obtain a loss rate. No sampling was repeated.
Anonymous libc fun_* symbols remain unexplained. Generic Map<I,F>::fold names
do not identify a specific caller or column merely because full names are kept.

After the user explicitly requested proceeding, new sampling and paired
measurements ran on a shared host without waiting for other work. Host
interference is not excluded. Two main Lookup windows from the earlier idle
session were retained unchanged, so cross-side CPU percentages remain
diagnostic. The preserved paired intervals are the basis for timing verdicts.

No allocation counts were collected. Named malloc/realloc samples do not
establish allocation counts, bytes or live-state peaks. Absence of sampled SQL
frames does not prove zero SQL calls or that denied-credit fallback was absent.
Existing funded-native instrumentation tests cover their focused cases only.
This fixed main-to-candidate comparison does not reconstruct historical J2a,
A4 or J1.6 before/after gains, or establish the complete issue's original gates.

## Resource observations

These ranges are process RSS high-water marks. Rust records them through the
final report point, including setup, probes, warmup, native worker threads and
iteration cleanup, before group.finish and runtime/process exit. ASOF records
them through atexit. They are neither timed-window live bytes nor, for Rust,
peaks through process-exit cleanup. ASOF has only two resource observations
per side; Rust has twenty. No equivalence or memory-growth verdict is inferred.
Resource measurements were not added retrospectively to the four reused cases.

| Case                            | Workers / side | Main HWM MiB range | Candidate HWM MiB range |
|---------------------------------|----------------|--------------------|-------------------------|
| ASOF 1M                         | 2              | 609.00–751.83      | 576.43–678.74           |
| Prepare + 500 rows, compact 60k | 20             | 231.30–276.22      | 227.70–272.92           |
| Prepare + 500 rows, steady 60k  | 20             | 275.66–336.18      | 270.79–319.71           |
| Capture dirty 20k               | 20             | 136.51–158.47      | 136.14–152.82           |
| Restore full 20k                | 20             | 135.79–153.04      | 135.21–154.78           |

## Budget and retained evidence

Combined measurement/profile supervisors consumed
1957.469124 seconds
(32.62 minutes),
including the earlier four-case session and failed startup/analysis receipts.
Adding the reserved 90 seconds for preflight/analysis gives
2047.469124 seconds, below the authorized
3600-second limit. New paired measurement supervision took 1656.099460 seconds.
Builds were separate: sealed Rust build drivers 1927.603 seconds, including a
preserved 1.359-second cache-transition interruption; diagnostic no-run build
371.843456 seconds. Previously sealed wheels and profile tooling were reused.

The original main Lookup analysis failure was an exec-generation identity
guard, repaired by reanalyzing the retained raw data. Rust's first launch
omitted required arguments and exited before sampling; its 0.118140-second
receipt is preserved. Reporting corrected two schema assumptions against
literal retained-case and release manifests. None of these recoveries changed
raw timings, source or sample counts.

The [lossless evidence package](stream-join-j2a-dictionary-measurements-evidence.json.gz) contains all nine maintained
statistical rows, twenty paired observations per side, 4800 Rust inner points,
ten complete full-name CPU audits, resource records, source/binary/tool seals,
budget and cleanup receipts, and 99 raw-source file hashes.
It can be read with Python's gzip and json standard-library modules.

- Uncompressed JSON SHA256: c54164c101074978e3b08dcec80546a482bce0f75298d63090b731658375320f.
- Gzip SHA256: c1e7277b34c942b70ef3c5ae87123afbd21f710858406acc4e3cc4017c21d3b8.
- Baseline sealed native SHA256:
  8a1d4f9caf96b26a7bd9bf5878d13a9498b1ed0c61841c305549dbbf7dc59840.
- Candidate sealed native SHA256:
  b00e859849bc29b4dbd648af2cb46e9837739fcb84c8f18c295b6fff242ada92.

Raw profiles, symbol sidecars, run logs and context remain under
target/j2a/profiles, profiles-stream, rust-profile/output,
measurements and lifecycle/measurements in the isolated worktree. The package's
source hashes bind those retained inputs. This report and package were produced
after all clean-source timing validation completed.

## Next implementation priorities

1. Isolate and reduce newly introduced generic admission/yield scheduling work
   without weakening cancellation, credit or error-ordering requirements.
2. Review repeated borrowed hashing, interner work and count/materialization
   probe reuse before changing the dictionary representation again.
3. Remove remaining generic row slices and repeated output materialization;
   owned payload eligibility is directly relevant to the Lookup fixture.
4. Treat checkpoint row IPC/checksum and restore parsing as separate work,
   preserving frozen V1 compatibility and proving any future format migration.

The prior 186 serial Join tests, four property tests, scoped lint/format checks
and specialist correctness review remain the recorded source validation.
This measurement-only task did not rerun them or claim full CI/coverage green.
Required CI and performance acceptance remain unresolved; PR #393 stays draft,
and this evidence does not authorize merging the regressing candidate.
