# Profile-selected grouped SUM followup

This work item first profiles the merged product at
`4f2a6190d20ce18d10750cd87d86da0bec6f778f`, then measures one optimization
selected from that profile. [The frozen protocol](protocol.json) fixes four
workloads, two rounds of ten alternating pairs, and one shared 600-second
measurement budget. Builds and ordinary functional tests are timed separately.

## Current-product evidence (P0)

[The complete current-stage publication](current/README.md) completed in
**35.703777270 seconds**, leaving **564.296222730 seconds** for the candidate
stage. This debit includes preflight, fixtures, capture, decoding, correctness,
finite timing, profile target attribution, publication, and cleanup. The newly
compiled current Release build took **554.210007899 seconds** separately.

The finite comparison fills the previous unmeasured historical control with
actual original `199d70b4` versus merged `4f2a6190` observations. Its maintained
verdict is **inconclusive**: pooled P50 was 54.291 ms versus 56.131 ms (+3.39%).
The paired median changes were +0.657% and −4.732%; their 97.8515625% intervals
were [−17.063%, +11.719%] and [−10.988%, +18.065%]. These observations establish
neither a regression nor equivalence. All 40 formal observations, four warmups,
independent full-result/schema/hash oracles, CPU time, and peak RSS are retained.

The fresh current profile contains 1,455 user-CPU samples over five processing
windows (279, 280, 290, 299, and 307 samples). The first of six jobs is excluded.
The decoder matched all 3,269 imported per-thread timestamps to perf; its largest
clock-alignment error was approximately 0.000000030 ms. Worker-thread samples
account for 99.86% of selected samples.

| Disjoint stack attribution | Samples | Share  |
|----------------------------|---------|--------|
| update_chunk               | 610     | 41.92% |
| finish_transaction         | 396     | 27.22% |
| commit                     | 207     | 14.23% |
| ExpressionOperator         | 203     | 13.95% |
| Other                      | 39      | 2.68%  |

The following inclusive counts overlap and describe sampled CPU, not phase wall
time. They support investigating output/state materialization in the current
source; they do not establish the effect of a proposed implementation.

| Inclusive symbol or family  | Samples | Share  |
|-----------------------------|---------|--------|
| Group drop                  | 209     | 14.36% |
| candidate                   | 191     | 13.13% |
| ScalarValue::iter_to_array  | 155     | 10.65% |
| output_chunk                | 139     | 9.55%  |
| RowConverter / row encoding | 115     | 7.90%  |
| Hash map operations         | 94      | 6.46%  |
| merge_rank                  | 76      | 5.22%  |
| ScalarValue drop            | 59      | 4.05%  |
| Hashing                     | 43      | 2.96%  |

There are 425 unresolved leaf samples (29.21%). Exact ELF symbol ranges provide
no inline or source-line attribution. An absent named frame is not zero cost.
No off-CPU wait, hardware cache, or memory-bandwidth claim is made. The host is
WSL2 with shared scheduling; live process guards found no competing build,
native functional test, or benchmark. The wide finite intervals remain visible.

The worker affinity is CPU 0 for one-CPU cases and CPUs 0–7 for eight-CPU
cases; those eight logical CPUs report four core IDs. The controller uses CPU
31. Tokio workers match each affinity size. Streaming keeps one partition,
finite execution uses eight, and external thread-count variables remain one.
This is WSL2 virtualized hardware; process guards do not prove an idle host
outside the guest.

## Candidate evidence (P1)

The measured public candidate is
`af400b38511bc7f6255fd3549518095b51b20e0c`. It borrows UTF-8 group-key strings
into Arrow output builders, avoiding intermediate owned string clones during
full-snapshot construction. Other output types retain their scalar conversion
path. [The complete candidate publication](candidate/README.md) compares this
product against merged `4f2a6190`, with identical fixtures and instrumentation.

| Case               | Baseline P50 ms | Candidate P50 ms | Pooled change | Maintained verdict      |
|--------------------|-----------------|------------------|---------------|-------------------------|
| stock-stream-1cpu  | 280.088         | 266.919          | −4.70%        | no-confirmed-regression |
| stock-stream-8cpu  | 278.658         | 269.054          | −3.45%        | inconclusive            |
| sparse-stream-1cpu | 34.302          | 28.346           | −17.36%       | improved                |
| stock-batch-8cpu   | 57.158          | 54.550           | −4.56%        | inconclusive            |

Dense input contains 10,000 seed rows plus 2,000,000 business rows. The sparse
fixture retains 10,000 groups and then supplies twenty 100-row update batches,
with 80 trades and 20 non-trades per batch. Across the complete sparse fixture,
100 distinct groups receive updates. Every streaming arm emits and retains 21
full cumulative snapshots; the finite arm returns one final result. This keeps
snapshot construction relevant even when few groups change.

Only the sparse retained-state case establishes a greater-than-5% improvement
under the maintained rule. Its paired median changes were −18.324% and −18.212%,
with 97.8515625% intervals [−20.490%, −14.908%] and [−23.516%, −15.296%].
The other cases do not establish equivalent performance or a confirmed speedup.
All round intervals, every timing observation, correctness receipts, CPU/RSS,
and command/provenance details are retained in the linked publication.

| Case               | CPU ms baseline → candidate | Peak RSS MB baseline → candidate |
|--------------------|-----------------------------|----------------------------------|
| stock-stream-1cpu  | 279.347 → 266.084           | 245.942 → 244.480                |
| stock-stream-8cpu  | 326.642 → 317.410           | 261.411 → 259.367                |
| sparse-stream-1cpu | 34.302 → 28.346             | 75.168 → 73.894                  |
| stock-batch-8cpu   | 153.323 → 148.671           | 247.081 → 251.269                |

CPU is the P50 processing CPU time; RSS is the P50 whole-process peak, in decimal
MB. The harness retains every output snapshot for independent post-timer oracle
validation, which differs from a final-only application sink. RSS includes
fixture, runtime, and retained-output memory; it is not SQL reservation size.

The accepted final candidate profile contains 1,306 selected CPU samples across
five windows (258, 259, 268, 258, and 263). All 3,115 imported timestamps match
perf, with the same approximately 0.000000030 ms maximum alignment error. There
are 385 unresolved leaf samples (29.48%).

| Disjoint stack attribution | Current samples / share | Candidate samples / share |
|----------------------------|-------------------------|---------------------------|
| update_chunk               | 610 / 41.92%            | 592 / 45.33%              |
| finish_transaction         | 396 / 27.22%            | 300 / 22.97%              |
| commit                     | 207 / 14.23%            | 180 / 13.78%              |
| ExpressionOperator         | 203 / 13.95%            | 199 / 15.24%              |
| Other                      | 39 / 2.68%              | 35 / 2.68%                |

The inclusive `output_chunk` observation is 139/1,455 samples (9.55%) in the
current profile and 71/1,306 (5.44%) in the candidate profile. This is consistent
with the selected output-materialization change. The candidate-only P1 profile and the earlier P0 current profile are different
cohorts. These are separate processing windows and sample proportions, not
paired phase timings or an additional speedup estimate. Inclusive candidate hotspots still include Group drop
184/1,306 (14.09%), candidate construction 172 (13.17%), row encoding 135
(10.34%), hash-map operations 106 (8.12%), merge_rank 66 (5.05%), and ScalarValue
drop 46 (3.52%); their shares overlap.

The four cases cover the changed UTF-8 output path in dense and sparse streams,
with finite execution as a control. Performance of other key types, checkpoint
capture/recovery, and alternative partition policies remains unmeasured here.
The streaming cases keep one partition; eight available CPUs do not establish
parallel scaling of this stateful operator. No additional optimization or
parallelism claim is inferred from these observations.

P1 completed all 160 formal observations and 16 warmups in
**137.326696254 seconds**, including profile capture/decoding, automatic analysis,
publication, and cleanup. The complete shared measurement debit is
**173.030473524 / 600 seconds**. There were no retries, added cases, extra
samples, or stopped stages. The new candidate Release build took
**558.397989215 seconds** separately. Both phase publications are immutable.

## Instrumentation and reproduction

Every native arm uses the same audited diagnostic harness, SHA-256
`657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`,
exported from public revision `84186554b078e25e71003a74fedc91c63b2b7f70`.
This selects instrumentation only. Current product source is newly compiled
`4f2a6190`; the old `84186554` product binary is not substituted. The development
branch retains its current diagnostic example for CI. Only separate permanently
frozen native build trees receive the common example overlay.

The original sealed `199d70b4` artifact was reused after full binary, source,
compiler, dependency, support, feature, and build-setting validation. Its binary
SHA-256 is `65c13e041bc26f49492fde6dbda8860ac2e6d07866c1ff3040e786895bf1efda`;
the new current binary is
`17259e26f471722696cbc9c1af67ebc901d1d3c2c6f67b49f78f77c655137ab2`.
Full source maps, build commands, compiler logs' hashes, resource configuration,
fixture schemas, tool/decoder hashes, profile windows, and raw perf paths/hashes
are inside [the complete evidence](current/complete-evidence.json.gz).
Large raw perf/decoded files remain under the recorded local `target/` paths;
all selected stack/count evidence needed to assess the conclusion is committed.

From a checkout containing these tools, prepare separate detached product
worktrees at the revisions above, export the example into a harness-only tree,
and build each arm through the maintained builder. For example:

```bash
TASK_DIR="$PWD/target/sum-followup-columnar-20261104"
HARNESS_DIR="$TASK_DIR/common-harness"
mkdir -p "$HARNESS_DIR/crates/calc-flow/examples"
git show 84186554b078e25e71003a74fedc91c63b2b7f70:crates/calc-flow/examples/grouped_sum_state_benchmark.rs > "$HARNESS_DIR/crates/calc-flow/examples/grouped_sum_state_benchmark.rs"
uv run python -m scripts.benchmark_suite.sum_state_build \
  --source "$CURRENT_FROZEN_SOURCE" --harness-source "$HARNESS_DIR" \
  --output "$TASK_DIR/build/current" --target "$PWD/target"
```

`CURRENT_FROZEN_SOURCE` must be a new detached `4f2a6190` worktree; apply the same
builder to detached `199d70b4` if a compatible historical receipt is unavailable.
Keep each frozen source tree unchanged after building. Run P0 once on an idle
host, using absolute paths for the verified perf and samply installations:

```bash
uv run python -m scripts.benchmark_suite.sum_followup \
  --phase current --protocol benchmarks/sum-state/sum-followup-columnar-20261104/protocol.json \
  --baseline-build "$HISTORICAL_BUILD_RECEIPT" \
  --candidate-build "$TASK_DIR/build/current/build.json" \
  --output "$TASK_DIR/current" \
  --publish-directory "$TASK_DIR/publication/current" \
  --perf "$PERF_EXECUTABLE" --perf-library "$PERF_LIBRARY_DIRECTORY" \
  --samply "$SAMPLY_EXECUTABLE"
```

The candidate stage must consume the emitted completed P0 receipt with
`--phase candidate --prior-receipt "$TASK_DIR/current/phase-stopped.json"`.
It receives only the remaining shared budget and must use a newly sealed public
candidate product. Reproduction creates new observations; it does not replace
this publication or alter its frozen observations.

Build the public candidate through the same builder in another permanently
frozen detached worktree at `af400b38511bc7f6255fd3549518095b51b20e0c`, with
`--output "$TASK_DIR/build/candidate"`. Then run the fixed candidate stage once:

```bash
uv run python -m scripts.benchmark_suite.sum_followup \
  --phase candidate --protocol benchmarks/sum-state/sum-followup-columnar-20261104/protocol.json \
  --prior-receipt "$TASK_DIR/current/phase-stopped.json" \
  --baseline-build "$TASK_DIR/build/current/build.json" \
  --candidate-build "$TASK_DIR/build/candidate/build.json" \
  --output "$TASK_DIR/candidate" \
  --publish-directory "$TASK_DIR/publication/candidate" \
  --perf "$PERF_EXECUTABLE" --perf-library "$PERF_LIBRARY_DIRECTORY" \
  --samply "$SAMPLY_EXECUTABLE"
```

The sealed measured candidate binary SHA-256 is
`be89d670e89c7142fdf89849ae86240d3df3b86b2bfa508b52979a525f4c96bd`.
The current/candidate stages use the same driver SHA-256
`847d6346e91cc5c8a8505cbf405840d861d2242c9087b637ef89767a6e0369a6`.
The frozen protocol SHA-256 is
`85bb35d17b948cc78967cf84044f5778cca9889e5bc4bd4094621d48a077127a`.
