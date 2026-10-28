# DAL-417：当前两引擎补测与归因

[中文报告](REPORT.md) 直接回答原实验的引擎类型、另一引擎结果、差距来源和分阶段提升
方案；[预记约定](PLAN.md) 固定本轮 workload、样本数、计时和预算。
原 #494 证据位于 [arroyo-comparison](../arroyo-comparison/README.md)，本轮未修改它。

本轮固定启动时 `main@b9a10d4fa4b109b810d2f5839951912b92e597f8`（2026.10.27），
两种 Program.engine 使用相同的融合表达式图。Arroyo 保持 v0.15.0 / `4ef15f6`。
五个 workload、每组五次，共 70 次正式观测与八次独立诊断；全部 988,160 条正式输出
精确核对，采集及诊断用时 104.68 秒。release 构建另计 9m46s。

旧结论来自持续 StreamingRunner 的集成路径，不能解释成纯批引擎比较。当前有限 SQL
每帧 collect 和整表有限计算分别列出；原生 event-time window 在 SQL engine 上不等价，
报告保留了拒绝/最终值对齐/输出时机的定向证据。原始数据不证明方案已经带来产品优化。

## 文件与证据

- `run.py`、`common.py`、`capture.py`、`engines.py`：固定 workload 和进程/计时边界。
- `profiling.py`：诊断阶段计时、cProfile 和已有 native callback profiling。
- `probe.py`：当前图、实际 DataFusion 计划、SQL/window 语义检查。
- `historical_probe.py`：使用保留的历史 native seal 检查原 TableExpr.stream 调用链和图，
  不重新测量历史性能；复算本轮不要求保留该旧 wheel。
- `analyze.py`、`audit.py`：离线复算描述性统计、callback 阶段差值和样本审计。
- `reproduce.sh`：固定 source、检查 staged/unstaged tracked diff、构建和采集入口。
- `data/`：可读 summary、paired ratios、诊断摘要、identity、audit、校验和及完整原始压缩包。

原始包内 `raw-data/results/` 包含所有 70 次 command、output、arrival/send、RSS、加载的
native seal 和结果；`profile-*/` 保存八次 profile/阶段 clocks、callback、DF metrics。
另含历史/当前执行图、语义 probe、构建日志与 wheel/native 关系、固定 lock、主机信息、
预检失败和修正记录。保留正式测量脚本的原始 hash，未在采样后改写这些六个驱动文件。
包内绝对路径仅作 provenance，不需要访问原机器，不包含 native library、wheel 或可执行文件。

## 离线复核

从 repository root 执行，使用 Python 3.13，依赖 psutil 7.2.2 和 websockets 15.0.1；
不需要加载 calc-flow native 或重新构建产品。可以使用已有 benchmark venv，或用 uv
准备只含这两个依赖的轻量环境。

```bash
sha256sum --check .codex/artifacts/arroyo-attribution/data/SHA256SUMS
mkdir -p target/arroyo-attribution-evidence
tar xzf .codex/artifacts/arroyo-attribution/data/DAL-417-current-raw.tar.gz \
  -C target/arroyo-attribution-evidence
uv run --no-project --python 3.13 \
  --with psutil==7.2.2 --with websockets==15.0.1 \
  python .codex/artifacts/arroyo-attribution/audit.py \
  target/arroyo-attribution-evidence/raw-data
uv run --no-project --python 3.13 \
  --with psutil==7.2.2 --with websockets==15.0.1 \
  python .codex/artifacts/arroyo-attribution/analyze.py \
  target/arroyo-attribution-evidence/raw-data
```

审计成功输出 `samples=70`、`profiles=8`、`outputs=988160` 和 `audit=passed`。
审计直接核对原始值、发送/接收边界与延迟分位数、吞吐、顺序、资源、native seal 及测量
脚本哈希。结果只能用于报告所声明的自定义描述性契约，不是仓库版本回归门禁。
诊断阶段存在开销和重叠，不能把累计回调时间直接相加视为端到端独占墙钟。
RSS 摘要分别保存停止前峰值与包含退出/最终 checkpoint 的全程采样峰值。
交付时另外验证了压缩包解包后重新审计、三个摘要复算一致，以及篡改输出计数被拒绝。

## 重新采集

仅在需要新测量证据时运行。使用 x86_64 Linux/WSL、CPU 0–7、uv、Python 3.13、GitHub
CLI、网络及仓库固定 Rust toolchain；硬件和构建不同须记录新的身份，不冒用原 seal。
本轮实际使用 CPython 3.13.9 和历史 requirements 中的相同 Python 版本集。
原始结果已完整，无需为了审阅或交接重跑。

从后续 PR 的 repository root 建立固定源码 worktree，复制本目录，然后采集：

```bash
git worktree add --detach target/arroyo-attribution-baseline \
  b9a10d4fa4b109b810d2f5839951912b92e597f8
cp -R .codex/artifacts/arroyo-attribution \
  target/arroyo-attribution-baseline/.codex/artifacts/
cd target/arroyo-attribution-baseline
bash .codex/artifacts/arroyo-attribution/reproduce.sh
```

脚本拒绝不同 HEAD、tracked diff（包括暂存的修改）和已有输出目录；所有输出保存在该
worktree 的 `target/arroyo-attribution-reproduction/`。预算为总共 600 秒，其中正式及
诊断 collector 固定 550 秒，留出预检/语义与审计余量；构建单列。

实际正式采样命令（当前证据的原调用）：

```bash
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arroyo-attribution/run.py collect \
  --arroyo target/arroyo-comparison/bin/arroyo-linux-x86_64/arroyo \
  --output target/arroyo-attribution/results --pairs 5 --profiles --budget 550
```

原生产品代码没有修改。原始包及离线数据核对、Ruff、shell 语法与实际 complexity ratchet
为本轮检查范围；全量回归、覆盖率和跨平台留给现有 CI，CI 状态在交付评论另报。
