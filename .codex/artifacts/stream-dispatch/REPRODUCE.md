# 复现第二阶段 streaming 比较

基线 `1b8ff9cf360b5c05d62cf97e48d12d19f652c17e`，候选产品提交
`cfa2a2a2f2d5a84cc5171c11261a3c8f1ce7dece`。只运行 `PLAN.md` 的两个转换场景。
依赖固定在相邻的 `../sql-prepare-reuse/requirements.txt`，CPython 3.13.9、Rust 1.94.0。
准备独立 base/head venv，安装对应 release wheel；不要覆盖历史环境和 seals。

原基线复用第一阶段候选 wheel，源码为 `9db6d10b`；该提交与新基线的 crates、
Python 产品代码、manifest/toolchain 完全相同，差异仅是可选 Kafka 的 libz-sys 锁项及
文档/测量产物。默认 wheel 依赖图不含 libz-sys，保留历史 native/wheel 哈希，
没有把它标为重新构建的新 main wheel。候选在同任务已有构建源目录中替换两个
binding Rust 文件和新锁文件，核对 813 个产品文件；native 构建单列 207.371 秒。
新机器应从上述两个提交构建，各预估 10–15 分钟，构建时间不计入 600 秒采集预算。

在相应源码目录构建（路径使用对应环境/输出绝对路径）：

```bash
CARGO_BUILD_JOBS=4 maturin build --release --locked --interpreter <python> --out <wheels>
uv pip install --python <python> --no-deps <wheel>
```

生成 `seals.json`：base/head 各含 `python`、`native_sha256`、`wheel_sha256`、`source_ref`。
另建 `budget.json` 初值 `{"limit_s":600,"spent_s":0,"stages":[]}`，环境准备和身份核对
亦记入预算。collector 在启动 worker 前拒绝已暂存、未暂存或未跟踪的产品源码变化。
表数据只经 Arrow IPC；JSON 文件只放配置、哈希、阶段时钟和计数，不放表行。

从含脚本的 checkout 根目录执行，以下 `task` 指本次独立的采集目录。采集期间禁止
本机任务构建；原实验持续监测 cargo/rustc/maturin，记录见原始包 `build-quiet.json`。

```bash
taskset -c 0-1 <base-python> .codex/artifacts/stream-dispatch/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/preflight --preflight
taskset -c 0-1 <base-python> .codex/artifacts/stream-dispatch/collect.py --seals <task>/seals.json --budget <task>/budget.json --output <task>/results
<base-python> .codex/artifacts/stream-dispatch/summarize.py <task>/results/samples.json > <task>/summary.json
<base-python> .codex/artifacts/stream-dispatch/audit.py --directory <task>/results --seals <task>/seals.json --summary <task>/summary.json
<base-python> .codex/artifacts/stream-dispatch/validate.py <task>
```

每场景每轮十对 AB/BA，两轮共 80 次，另四次预检。新进程先消费一批暖身，随后 gate
释放完整既定输入；完整消费计时包含正常 source 校验、Batch/Arrow 转换、回调/调度、
逐批输出、concat、IPC 写出和父进程 read_all。准备、首次批执行及取消清理均独立记录。
源在得到所有输出前保持未 EOF，不以 EOF 强制 flush。固定 64×512 没有合并批次。

四次实现前诊断使用相同接口和 `--diagnostic`（当时两 side 均指向基线）；全部保留，
不作 base/head 性能比较。诊断 cProfile 与 callback trace 会扰动时间，不能代替正式结果。
所有正式判断由仓库 `paired_round` / `_paired_verdict` 重算，不补样选择有利结果。

Python 定向命令见 `validation.json`；绑定层调度单元检查在同任务构建源目录执行：

```bash
CARGO_BUILD_JOBS=4 PYO3_PYTHON=<base-python> CARGO_TARGET_DIR=<cached-target> cargo test --release --locked -p calc-flow-python --lib runtime:: -- --nocapture
```

本机测试二进制需 `LD_LIBRARY_PATH` 包含该解释器的 `sysconfig.get_config_var("LIBDIR")`。
初次启动失败后，实际用该设置运行已编译二进制 `runtime:: --nocapture`，18 项通过。
相关静态命令为 `cargo clippy --release --locked -p calc-flow-python --lib --tests -- -D warnings`、
`cargo fmt --all --check`、相关 Ruff/format，以及原 `scripts/verify_complexity_gates.py`。

原 Arroyo、窗口、有限 SQL 测量及审计证据复用，不在本批重跑。全量、跨平台和覆盖率交 CI。
