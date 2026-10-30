# 重现 Arrow 包装比较

基线 `1041bb7b6d9bd3a16686b8753e856e9f12f2529f`，候选产品
`3c66dfefba618ace56cb3430f244f27d37453b96`；后续证据/说明提交不改产品实现。
依赖沿用 `../sql-prepare-reuse/requirements.txt`：CPython 3.13.9、PyArrow 24.0.0，
Rust 1.94.0，默认 connector-file feature，release profile。

基线复用第二阶段候选 wheel，原源码为 `cfa2a2a2`；该提交与指定 main 的产品 Rust/Python
代码、manifests、Cargo.lock 和 toolchain 完全相同。保留原始 wheel/native/source 身份，
没有改写为 main 的重新构建产物。候选在本任务既有专用构建源路径复制两个已提交 Rust
文件，与候选核对 813 个产品/配置文件及锁文件；未复用 DAL-475 的代码或产物。
候选构建实际 222.576 秒，构建前预估 4–10 分钟。新机器可分别从两提交构建：

```bash
CARGO_BUILD_JOBS=4 maturin build --release --locked --interpreter <python> --out <wheels>
uv pip install --python <python> -r .codex/artifacts/sql-prepare-reuse/requirements.txt
uv pip install --python <python> --no-deps <wheel>
```

两个独立环境的实际 `python`、native/wheel SHA256、source_ref 写入 `seals.json` 的 base/head。
另建 `budget.json` 初值 `{"limit_s":600,"spent_s":0,"stages":[]}`。身份/环境准备、诊断、
fixture、暖身、预检、进程启动/退出、失败与原始审计都记入累计预算；构建另外记录。
源码身份检查在启动 worker 前拒绝 staged/unstaged/untracked 产品变化，支持干净固定源。
交付版同时独立检查 index；封存后补充了“index 改动被工作树反向覆盖”的反例。
原始包 `sampling/collect.py` 保留实际采样版本，`harness/collect.py` 为加固后的交付版，
两者哈希见 manifest。加固仅影响 worker 启动前的拒绝条件，没有重新采样或改写观测。
采集期间持续检查本机 cargo/rustc/maturin，发现即结束 owned collector 并清理其 worker。

从 checkout 根目录运行。原实验 `task` 为 `target/arrow-wrapping`，脚本使用相同路径。
先各 side 两场景一次诊断，再四次预检、两轮每场景十对 AB/BA，共 80 次正式观测；
不能根据结果扩大次数或重试到通过。每个 worker 暖身首批后才释放正式 gate。

```bash
taskset -c 0-1 <base-python> .codex/artifacts/arrow-wrapping/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/diagnostic-base --diagnostic-side base
taskset -c 0-1 <base-python> .codex/artifacts/arrow-wrapping/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/diagnostic-head --diagnostic-side head
taskset -c 0-1 <base-python> .codex/artifacts/arrow-wrapping/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/preflight --preflight
taskset -c 0-1 <base-python> .codex/artifacts/arrow-wrapping/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/results
<head-python> .codex/artifacts/arrow-wrapping/summarize.py <task>/results/samples.json > <task>/summary.json
<head-python> .codex/artifacts/arrow-wrapping/audit.py --directory <task>/results --seals <task>/seals.json --summary <task>/summary.json
<head-python> .codex/artifacts/arrow-wrapping/validate.py <task>
```

原实验的 `run-measurements.py` 包含连续构建监测，原始包内保留其源码和实际命令日志。
正常 `Batch.from_pyarrow`、execute_async、RunResult.outputs、to_pyarrow、concat 和输出 IPC
均在对应时钟内；主指标为已备输入 gate 到父进程 read_all，输入未预绑定。
诊断启用 cProfile 会扰动时间，不参与正式判定。最后一次查询的详细指标在输出 IPC 后读取，
不得把末次指标乘以批数当作整段。表行只经 Arrow/Batch/IPC，JSON 只存元数据和计数。

实际 Python 检查：

```bash
target/arrow-wrapping/env-head/bin/python -m pytest python/tests/test_batch.py -k 'table or metadata or producer or stream' -q
target/arrow-wrapping/env-head/bin/python -m pytest python/tests/test_pipeline.py::test_run_result_properties_return_defensive_values python/tests/test_pipeline.py::test_runtime_plan_and_result_are_visible_to_cyclic_gc python/tests/test_batch_query_templates.py::test_batch_query_templates_match_exact_schema_and_metadata python/tests/test_batch_query_templates.py::test_warm_batch_templates_isolate_concurrent_calls_and_cancellation python/tests/test_compute.py::test_compute_preserves_batch_identity_metadata_and_sync_options python/tests/test_stream_results.py::test_stream_full_output_queue_applies_backpressure_and_cancels -q
```

Rust 检查在已核对相同产品源码的任务构建源目录，使用独立 target 缓存、同一解释器及
其 `sysconfig.get_config_var("LIBDIR")` 作为 LD_LIBRARY_PATH。完整 argv/cwd/settings
记录在原始包 `rust-checks.json`，按顺序执行：

```bash
cargo test --release --locked -p calc-flow-python --lib batch::tests:: -- --nocapture
cargo test --release --locked -p calc-flow-python --lib pipeline::tests::sync_execution_and_result_getters_expose_defensive_values -- --nocapture
cargo test --release --locked -p calc-flow-python --lib pipeline::tests::run_result_gc_collects_cycles_hidden_in_external_outputs -- --nocapture
cargo test --release --locked -p calc-flow-python --lib pipeline::tests::cleared_plan_and_result_reject_access -- --nocapture
cargo clippy --release --locked -p calc-flow-python --lib --tests -- -D warnings
```

另执行 Rust fmt、相关 Ruff/format 和原 complexity/schema/diff 检查，门禁及依赖未调整。
全量、跨平台和覆盖率由 CI 提供；已有 SQL/streaming/Arroyo/窗口测量不在本批重跑。
