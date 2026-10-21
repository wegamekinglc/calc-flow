# DAL-417：Arroyo streaming 对比证据

[中文报告](REPORT.md) 包含功能矩阵、优劣和适用场景、计时/I/O/检查点边界、
吞吐/延迟/RSS、重复测量范围和官方/源码来源。[测量前约定](PLAN.md) 保留预定范围。

固定 calc-flow `dab98a559314bd46d866ed7a6489abcdd3810afe`（2026.10.21）和 Arroyo
`4ef15f6336a596651f10cf27c1ee29c29bf81577`（v0.15.0），在同机相同 CPU affinity
和输入下采集四个场景、五组交替 AB/BA，共 40 次正式运行。全部输出在 EOF 前可见，
并通过全量正确性检查；原始数据含 327,680 条无状态输出和 10,240 条窗口输出。

这是包含不同适配层的 WebSocket JSON → 计算 → stdout JSON 集成比较。
它不能提供纯算子排名、最大持续吞吐、版本回归门禁或恢复/分布式性能结论。
本 PR 仅提交既有测量及必要的复核/重现产物，不重新采样，不改产品行为。

## 数据位置

- `data/summary.json`：八个引擎/场景组合的描述性统计。
- `data/samples.json`：全部 40 次样本，含吞吐、分位数、RSS、启动诊断和输入哈希。
- `data/identity.json`：固定 source/native/binary/harness 与环境身份。
- `data/audit.json`：40 次样本的离线复核结果和成对吞吐比。
- `data/manifest.json`、`data/SHA256SUMS`：压缩包及可读副本的内容哈希。
- `data/DAL-417-raw-data.tar.gz`：完整原始证据，6,713,650 字节；与初次 issue 附件一致。

压缩包 SHA-256 为
`c8bf1ea160b16f54bf0744f6ca1383a10017107630b0cd63aaf95a816b318cfc`。
其中 `raw-data/results/` 保存每次 command/query、原始 stdout、expected、arrivals、
send_times、RSS 时间序列及结果；根目录还包含构建日志、双方 Cargo.lock、版本/哈希、
主机信息、失败预检及修正记录。输入是固定生成的合成数据。绝对路径只是当时 provenance，
无需访问原机器。数据压缩包不含 native library、wheel 或 Arroyo 可执行文件。

`raw-data/measure-measured.py` 是正式采样的精确脚本，哈希与采样 identity 一致。
初次交付的 `measure.py` 只整理过一处列表推导式的 Ruff 换行，AST 与采样脚本相同；
双哈希见压缩包内 `format-provenance.json`。当前 `measure.py` 随后从 `acquisition`
抽出 worker 准备、输出观察器、等待、指标计算及子进程停止函数，以通过原有复杂度门禁。
`refactor-provenance.json` 单独记录采样版、初次交付版和重构版的哈希及定向等价性证据：
展开 helper 并对齐观察器的状态与闭包声明后，采集流程的 AST、观察器函数体及其余
原函数均与采样版一致；无状态和窗口各一份既有样本的指标精确相同。
当前完整脚本的 AST 和哈希已改变，未用于原 40 次采样；
采样身份、原始脚本和数据包保持原样，新采样仍由 controller 记录实际脚本哈希。

`audit.py` 拒绝 Python `-O/-OO`，防止离线复核中的断言被优化关闭；审计仍优先使用
数据包内的原始采样脚本核对历史身份。

## 离线复核

在 repository root 使用 Python 3.13 执行；不启动引擎、不构建 native、不进行性能测量：

```bash
sha256sum --check .codex/artifacts/arroyo-comparison/data/SHA256SUMS
mkdir -p target/arroyo-comparison-evidence
tar xzf .codex/artifacts/arroyo-comparison/data/DAL-417-raw-data.tar.gz \
  -C target/arroyo-comparison-evidence
python3.13 .codex/artifacts/arroyo-comparison/audit.py \
  target/arroyo-comparison-evidence/raw-data
```

成功输出 `"samples": 40`、`"audit": "passed"` 及四组完整成对比值。
该脚本只复核此固定样本集，不是通用 regression gate；完整实验的正确性和采样边界
仍由 `measure.py` 的采集检查及报告约定定义。

## 重现采样

仅在需要新的测量证据时运行。现有数据已经完成，审阅 PR 不要求重建或重测。
采集依赖 x86_64 Linux/WSL 的 `taskset`、CPU 0–7、Python 3.13、uv、GitHub CLI、网络和
仓库固定 Rust toolchain。原主机及软件配置见报告；新机器的结果应记录自己的身份。
原 release 构建约 11m19s，正式采集约 76.22s；脚本的 acquisition 预算为 600 秒。

从本 PR checkout 的 repository root 创建固定产品基线 worktree，并复制本目录：

```bash
git worktree add --detach target/arroyo-comparison-baseline \
  dab98a559314bd46d866ed7a6489abcdd3810afe
mkdir -p target/arroyo-comparison-baseline/.codex/artifacts
cp -R .codex/artifacts/arroyo-comparison \
  target/arroyo-comparison-baseline/.codex/artifacts/
cd target/arroyo-comparison-baseline
bash .codex/artifacts/arroyo-comparison/reproduce.sh
```

`reproduce.sh` 在下载或构建前拒绝不同 HEAD 及相对 HEAD 的 tracked diff（含已暂存和
未暂存改动），并拒绝已存在的输出目录，检查 Arroyo 下载件哈希，
构建 release wheel、预检正确性，再采集固定的四个场景。所有新输出在该 worktree 的
`target/arroyo-comparison-reproduction/`，不改调用者已有证据。

新构建的 binary/native/harness 哈希应单独记录，不应假称是原 sealed binary 的重复。
特别是 Python/依赖版本、CPU 资源、适配层或输入变动时，不能套用原比较结论。

前置检查的定向回归验证使用临时本地 checkout，覆盖干净基线、已暂存和未暂存的
产品源码改动；以占位 `uv` 命令终止依赖准备，不下载、构建或采样：

```bash
python3.13 .codex/artifacts/arroyo-comparison/test_reproduce.py -v
```
