# Sign CASE retest against current main: complete and inconclusive

The fixed experiment completed successfully against baseline `5a66b6d5cb3f9fa2f6c5d6c8d7ce03db0cea565f` and public candidate `9a3c45174be6aac02bbd405a5205e76238826b62`. All five maintained timing verdicts are **inconclusive**. The candidate therefore fails the predeclared performance acceptance condition: neither dense target has the exact maintained `improved` verdict. No selected case establishes a confirmed regression. Inconclusive results establish neither speedup nor equivalence. The runtime prototype is withdrawn from the deliverable under the predeclared acceptance policy; this is a complete, valid experiment with a negative acceptance result.

This was a new execution explicitly authorized after the earlier experiment stopped at its quiet guard. The earlier stopped campaign and its evidence remain unchanged. Both products in this experiment were freshly built from the current public baseline and public candidate. Historical dc/P binaries were not used or relabelled. The three restored candidate Rust files match the original public prototype byte for byte; this retest introduces no algorithm tuning. The frozen source worktrees and binaries used below remain intact when the implementation is withdrawn from the development branch. Timings apply to these two frozen products, not to a later documentation or withdrawal commit.

The [frozen protocol](sum-case-retest-20261106.protocol.json), [collector summary](sum-case-retest-20261106/summary.json), [all formal observations](sum-case-retest-20261106/samples.csv), [complete raw evidence](sum-case-retest-20261106/complete-evidence.json.gz), and [immutable collector manifest](sum-case-retest-20261106/evidence-manifest.json) preserve the acquisition and its provenance. The protocol SHA-256 is `567df88938c7aa72b33f2c657d01fef7bf72dd47bd7db217ba351d67d15e48fb`.

The five cases were selected before execution. Dense1 and dense4 exercise the changed compiled expression path on the actual stock stream: a 10,000-row zero seed and twenty 100,000-row business batches, with 80,000 trades selected from each business batch. Sparse preserves 10,000 retained groups with twenty 100-row business batches and 80 selected updates per batch. Its business batches alternate uniform sides. The separate pure-buy and pure-sell controls each contain the seed plus twenty 10,000-row all-trade business batches and force side on every seed and business row. They preserve the original side Field, schema metadata, quantity, keys, other column buffers and caller input. Each pure branch is measured separately so an alternating workload cannot hide its cost. Every job validates all 21 cumulative snapshots against independent arithmetic, schema, input and snapshot hashes.

There were two rounds of ten alternating baseline/candidate pairs per case, plus one warmup per arm per round. The complete count is **200 formal observations, 20 warmups, 12 profile jobs, 232 benchmark jobs, 4,872 snapshot checks and 223 physical invocations**. Each product profile has six jobs and excludes the first, retaining five sampled processing windows. All five correctness records are true; both profiles are accepted; helper seals are verified. There was one attempt, zero retries, no case or sample expansion, and no use of remaining budget for another run.

The following values are copied from the frozen collector summary. Overall P50 change is the ratio of the two overall P50 values. The paired medians and intervals below summarize aligned per-pair changes and are a different statistic.

| Case                     | Baseline P50 ms | Candidate P50 ms | Overall P50 change | Maintained verdict |
|--------------------------|-----------------|------------------|--------------------|--------------------|
| stock-stream-1cpu        | 171.881         | 167.045          | -2.81%             | inconclusive       |
| stock-stream-4cpu        | 155.178         | 164.850          | +6.23%             | inconclusive       |
| sparse-stream-1cpu       | 20.382          | 20.528           | +0.72%             | inconclusive       |
| uniform-buy-stream-1cpu  | 88.653          | 89.242           | +0.66%             | inconclusive       |
| uniform-sell-stream-1cpu | 86.366          | 86.868           | +0.58%             | inconclusive       |

| Case                     | Round | Original paired median change | Original confidence interval | Coverage  |
|--------------------------|-------|-------------------------------|------------------------------|-----------|
| stock-stream-1cpu        | 1     | -3.34%                        | [-8.28%, +1.78%]             | 97.85156% |
| stock-stream-1cpu        | 2     | -3.55%                        | [-6.59%, +10.67%]            | 97.85156% |
| stock-stream-4cpu        | 1     | +7.87%                        | [-0.42%, +13.61%]            | 97.85156% |
| stock-stream-4cpu        | 2     | +1.41%                        | [-10.22%, +22.23%]           | 97.85156% |
| sparse-stream-1cpu       | 1     | -0.13%                        | [-4.79%, +2.72%]             | 97.85156% |
| sparse-stream-1cpu       | 2     | +3.04%                        | [+0.34%, +5.11%]             | 97.85156% |
| uniform-buy-stream-1cpu  | 1     | +0.55%                        | [-1.06%, +5.61%]             | 97.85156% |
| uniform-buy-stream-1cpu  | 2     | -0.80%                        | [-7.88%, +3.40%]             | 97.85156% |
| uniform-sell-stream-1cpu | 1     | -0.94%                        | [-5.97%, +5.68%]             | 97.85156% |
| uniform-sell-stream-1cpu | 2     | -1.19%                        | [-4.27%, +3.08%]             | 97.85156% |

The maintained classifier issues one joint verdict over the two rounds; it does not store an independent verdict for each round. Regression requires both confidence lower bounds to exceed +5%; `improved` requires both upper bounds below −5%, after the existing +5% inconclusive branch. None of the five joint verdicts is `regression` or `improved`. Dense1 has negative paired medians in both rounds, but its intervals reach +1.78% and +10.67%. Dense4 has an overall +6.23% P50 change, but neither round has a lower confidence bound above +5%; its second-round interval is broad. Calling dense1 a proven win or dense4 a confirmed regression would contradict the stored verdicts. No extra confidence gate or single-run/minimum rule was substituted.

The environment was WSL2 (`Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43`) on a 13th Gen Intel Core i9-13900HX. The in-budget global quiet guard passed. Host load and power outside the WSL guest remain unobservable. The four-core case used CPUs 0/2/4/6 and four Tokio workers; the other cases used CPU 0 and one worker. The controller used CPU 31 and external helper pools each used one thread. DataFusion target/input partitions remained 1, batch size 8,192, min_rows_per_partition=1, and checkpoints off. Guest quietness does not remove shared-host noise or prove independent paired observations. The broad intervals retain that uncertainty.

The processing timer starts after IPC preload and includes graph planning, execution, retained outputs, natural end and cleanup. Snapshot oracles run outside that processing timer and inside the total acquisition budget. Whole-process peak RSS includes IPC preload and retained outputs; it is not operator-only allocation or state reservation. The measured median cores for dense4 were 1.325 baseline and 1.252 candidate, while dense1 was approximately 1.0 for both. These observations do not demonstrate four-core saturation or scalability of other workloads.

The [profile summary](sum-case-retest-20261106/profile-summary.json) and full [baseline](sum-case-retest-20261106/baseline-stacks.json.gz) / [candidate](sum-case-retest-20261106/candidate-stacks.json.gz) stack records describe user-mode CPU samples for dense1 only. Both captures use the same common657 instrumentation, CPU policy and native input. They are qualitative profiles collected separately from the paired timing samples, not another latency comparison.

| Profile evidence                         | Baseline                  | Candidate                 |
|------------------------------------------|---------------------------|---------------------------|
| Accepted processing windows              | 5                         | 5                         |
| Selected-window samples                  | 804                       | 788                       |
| Samples by selected window               | [153, 162, 165, 161, 163] | [157, 156, 158, 157, 160] |
| All capture timestamps matched           | 2242                      | 2220                      |
| Maximum clock alignment error, ms        | 5.9604644775390625e-08    | 5.9604644775390625e-08    |
| Unresolved leaf samples                  | 157                       | 145                       |
| Reported loss/unwind/warning diagnostics | None reported             | None reported             |

The capture stderr files contain ordinary perf capture completion lines and no lost, unwind, warning, corrupt or truncation diagnostic. This means no such diagnostic was reported; it is not a proof of loss-free sampling or complete unwinding. Clock alignment matched the recorded timestamp/thread inventory and stayed below the maintained 10 ns tolerance. Symbolization uses exact nonzero-sized GNU nm ELF code ranges with dynamic-symbol fallback, without source-line or inline attribution. Unresolved leaf samples remain visible above. Missing, unsampled or inlined frames cannot establish zero work or a rewrite hit.

The stored disjoint phase attribution is:

| Phase              | Baseline samples / share | Candidate samples / share |
|--------------------|--------------------------|---------------------------|
| update_chunk       | 395 / 49.13%             | 411 / 52.16%              |
| ExpressionOperator | 189 / 23.51%             | 148 / 18.78%              |
| finish_transaction | 161 / 20.02%             | 169 / 21.45%              |
| commit             | 30 / 3.73%               | 39 / 4.95%                |
| other              | 29 / 3.61%               | 21 / 2.66%                |

Incremental SUM update and transaction finishing remain the largest classified CPU regions. ExpressionOperator has 189/804 samples (23.51%) in baseline and 148/788 (18.78%) in candidate. The original inclusive Projector::project_batch attribution is 61/804 (7.59%) versus 23/788 (2.92%). Baseline CaseExpr::evaluate has 60 samples (7.46%) and CaseBody::expr_or_expr 52 (6.47%); these frames do not appear in the candidate’s stored top 40 inclusive list. The candidate pattern agrees qualitatively with less CASE projection work, but the profiles neither isolate the rewrite’s elapsed contribution nor establish a dense end-to-end win.

Outer filtering remains visible: compiled::filter_record has 124/804 (15.42%) and 120/788 (15.23%) samples; Arrow filter_record_batch has 108/804 (13.43%) and 107/788 (13.58%). These are inclusive shares and overlap their descendants. They must not be added to phase shares or interpreted as wall-clock percentages. Baseline’s largest stored self entry is FilterBytes::extend_idx 70 samples (8.71%); candidate’s first two are incremental update_chunk_input 60 (7.61%) and arrow_row::variable::encode_one 57 (7.23%). These observations identify filtering, key encoding, candidate update and output construction as remaining CPU work. They do not justify a newly implemented optimization or another undeclared benchmark campaign.

The experiment does not measure mixed CASE with an all-trade identity filter, finite batch mode, nullable/UInt64/generic/UDF CASE shapes, checkpoint/recovery latency, tight/custom memory pools, off-CPU waits, hardware counters or memory bandwidth. Source-level semantics and allocation evidence are separate from timing evidence. The narrow sign-CASE rewrite adds value buffers in uniform masks; the two uniform controls are therefore essential, and their inconclusive outcomes establish no latency equivalence. Future proposals need their own explicit source hypothesis, scope, baseline and budget; no follow-up measurements were appended to this campaign.

The build pipeline used two separate permanent detached source worktrees, two separate new Cargo target directories and ordinary copies of the old compatible caches with distinct inodes. Baseline and candidate Cargo durations were 1199.261133602 s and 1194.335207009 s; helper durations 1200.823800177 s and 1196.182688516 s. The builds overlapped and these durations are ordinary build costs, not controlled build-speed measurements. Actual Cargo and both rustc processes share all 14 whitelisted environment values except each declared CARGO_TARGET_DIR, with jobs=2 and incremental=0. Both core and example were freshly compiled; both 327-record compiler streams end in success; all 292 registry semantic records agree. Public Git/pre/post proofs cover 1,010 baseline and 1,012 candidate native inputs. The sole instrumentation overlay is the identical common657 harness from public `84186554`. Candidate core depfiles include the exact compiled/sign_case.rs path.

| Identity                                   | Baseline                                                           | Candidate                                                          |
|--------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| Build receipt SHA-256                      | `e662b0f6300d30ebc510f6c1d2475b4833f90c66400bbd161f5ef263864fc29c` | `8b5f1d49597a290bfb63fa03d0c91ef3486586da72b13b681e690c863e6d3f89` |
| Copied binary SHA-256                      | `3acab7dab6daf6cc184d60ddaf7bcf3d316a43a536dda2a331c80b2e1cb30aa1` | `c1af9c28913d14810cd25a43784f7cebc11616910604a6020722cd6955bcc0c0` |
| Portable build packet SHA-256 (28 records) | `5c1bc0560182c48e41b402400fad4dd6b0410aa2335b6268659b723e12906ade` | `4857ed0cd3da674a59ba172ea35feea208211dae7d2e7101c09064b86beb4e48` |

An inherited `cost_scope` sentence in the build invocation metadata mentions the earlier study's debug TDD overlap. That description is stale: the two controlled release builds overlapped each other, with no debug TDD build. The original executed scripts and invocation records are retained unchanged. An earlier local style command briefly triggered an implicit package build, which was stopped before these controlled builds and before any measurement; it is outside the two product build durations above.

The new [exact-prototype-restoration proof](sum-case-retest-20261106/exact-prototype-restoration.json) records source restoration and the unrelated latest-main argmax change. Historical RED/GREEN/Clippy/allocation material remains separately identified as historical source verification; it does not replace the new products’ public/build proof. The historical Python RED test’s original source payload was not retained, and its preserved hash/logs are explicitly qualified. No source payload was reconstructed.

The entire acquisition and automatic sealing used one original deadline. Immutable collector final debit is 139.487799316 s; its earlier raw/summary snapshot is 139.305372560 s. The earlier outer clock records 139.981176622 s before owned settlement and 140.003096866 s after it. Automatic protocol archive, original manifest verification, source/binary/helper postseals and owned settlement reached 139.818390378 s from the original collector start, or 140.224041065 s from the earlier outer start. These are separate clock phases. The original collector receipt was not rewritten to include the tail. All phases completed before 600 s; the fixed 585 s interrupt was not needed. Controller exit code 0 was reaped, no signal was sent, and remaining owned processes are empty.

The [terminal handoff](sum-case-retest-20261106/acquisition-handoff.json) is retained unchanged, SHA-256 `5e7cc4f0cce36f22ee1feac0a9a52e0f285cceff47b5fd5de7b9517eebddac38` (47,253 bytes). It embeds the exact executed orchestration source/argv, true outcomes and 14 helper and two binary postseals. Source files in both frozen build trees were also checked against their full native inventories. The protocol archives both new build proof packets and the actual pure-mock launcher RED/GREEN evidence. Earlier unexecuted protocol/launcher versions and their genuine test records remain retained. Report writing is later delivery work and does not add observations or recompute the frozen statistics.

| Frozen published artifact | Bytes   | SHA-256                                                            |
|---------------------------|---------|--------------------------------------------------------------------|
| evidence-manifest.json    | 1807    | `4fe552053ab9520e8091b8f8ce711248bd7de736e28991a4f4b71cbf45093cf1` |
| complete-evidence.json.gz | 1241558 | `f10fcbe77949f85efeb02165887b5423cb9d0069bee975dde9e00b2edc46e3af` |
| summary.json              | 127369  | `f4156b01a601627d723ea984ccb8e7f43a3c96546d5e56972f2484ebade43fe0` |
| samples.csv               | 29993   | `cc85e8e23e67c82d7615230a11adb7bbfaf58796e6a3fea61b81b7f911f35d78` |
| profile-summary.json      | 33141   | `b8e8047ffa72a395a8386ab542f4c56e8c4b8e4e35f2e360e89ca5cdee788cce` |
| profile-targets.json      | 9736    | `f3547c5b5252428c55b756d0bb948ebfc81e24938bfa0fe84544bb7c7b6a09ac` |
| baseline-stacks.json.gz   | 4642    | `d0efd91fc88e738cdc379c4a8ea75945a4e0513623ec943930890382ea2de039` |
| candidate-stacks.json.gz  | 4341    | `8432641cd1949740303fa39b7165828d3a668bbe9c66570dbd0d77756a6a6043` |

The formal CSV header is `case,round,pair,side,wall_ms,process_cpu_ms,average_cores,whole_process_peak_rss_bytes,final_snapshot_sha256`. The original CSV and all collector outputs are preserved byte for byte. The maintained acceptance result is **false because the completed evidence does not establish the required dense improvement**.

The [runtime withdrawal proof](sum-case-retest-20261106/runtime-withdrawal.json) records all 868 core files matching study baseline `5a66b6d5`. The [delivery verification manifest](sum-case-retest-20261106/source-verification-manifest.json) binds the original collector manifest, protocol, terminal handoff and static source proofs. These separately added delivery files do not alter the collector manifest or its eight original outputs.
