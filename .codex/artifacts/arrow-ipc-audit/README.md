# DAL-417：Arroyo 与 calc-flow 对比摘要

**在这次过滤/投影转换测试中，Arroyo 更快：整表吞吐为 calc-flow（CF）的约 1.36 倍，
64×512 行小批约 8.82 倍。** 相同行数拆成小批后差距明显放大，下一步优先定位每批
调用、规划和调度的固定成本；现有证据尚未量出各项独占份额，倍数不是可实现的优化收益。

## 最新 Arrow 转换对照

同机、同一份 32,768 行输入：过滤 `id>=0 AND k<8`，输出 `id,y=x*2+1`。
CF 固定 2026.10.27 / `b5ae227e`，使用 **streaming 引擎的融合表达式图**，不含 SQL 节点；
Arroyo 固定 v0.15.0 / `4ef15f6`，执行自己的持续 SQL 图。单位为**万输入行/秒**。

计时从**放行已备原生 Arrow 批，到完整原生结果进入 sink 集合**，包含正常源/算子/回调
和调度；启动、输入 IPC/绑定及输出绑定/IPC 另计。准备位置和回调实现仍有差异，不能将
这个应用执行边界称为裸算术内核比较，也不对 observer、IPC 或启动作跨引擎速度排名。

| 输入形状           | CF streaming | Arroyo  | Arroyo/CF 成对吞吐比 |
|--------------------|--------------|---------|----------------------|
| 整表一次输入       | 1,906.9      | 2,556.3 | 1.36×                |
| 64 批、每批 512 行 | 301.6        | 2,696.1 | 8.82×                |

吞吐列是各组五次中位数；倍数是相邻 AB/BA **成对比值的中位数**，不一定等于两列直接相除。
成对范围为整表 1.162–1.489×、小批 8.406–9.680×。这是短 burst 的描述性比较，不能外推
长期稳态吞吐。[完整结果与计时契约](ARROYO-RESULTS.md#新成对结果)、
[原始阶段摘要](data/arroyo-extension/phase-summary.json)及[成对数据](data/arroyo-extension/pairs.json)。

## 原端到端吞吐：历史口径保留

以下为 #510 同机测量（CF 2026.10.27 / `b9a10d4`，Arroyo 仍为 v0.15.0）的历史结果，
单位同为**万输入行/秒**。计时从首次发送前到观察器收到最后输出行，包含 WebSocket、
JSON 编解码、计算、最终 stdout 和观察器解析/记录；不含 fixture、进程启动、完整 oracle
及停止。它衡量这条集成路径，不能代表引擎最大持续吞吐。

| 场景             | CF streaming | CF SQL 逐帧有限调用 | Arroyo |
|------------------|--------------|---------------------|--------|
| 512 行帧转换     | 29.86        | 19.49               | 58.09  |
| 8,192 行帧转换   | 31.96        | 31.19               | 65.95  |
| 512 行帧窗口聚合 | 35.08        | 无等价窗口执行      | 204.60 |

CF SQL 列是每帧有限 `collect_async`，不具备该窗口的持续状态/watermark 语义；CF streaming
列是持续作业，两者都使用表达式图。[原端到端报告](../arroyo-attribution/REPORT.md#正式结果)
及[原始摘要](../arroyo-attribution/data/summary.json)保留全部数字。两表的计时范围不同，
不能混合统计，也不能将数值上涨解释为产品优化；本次研究没有修改 CF 产品代码。

## 实际验证范围与证据

**Arroyo 已实际运行。** 基准适配只增加 Arrow 接入、输出和观察，批数据经过真实
SourceCollector/SQL 执行图，没有换成另写的计算算法；这不表示 v0.15.0 发布包已有公共
Arrow source/sink。两个转换场景各五对、共 20 次运行，327,680 行输出的列、值、顺序均核对。
窗口单次探针通过 512 行结果、65 个实际 watermark 和源开放/EOF 前完整输出检查，
**最新 Arrow 口径没有窗口成对吞吐结论**。[调用链与适配边界](ARROYO-RESULTS.md#实际调用链与计时契约)、
[窗口证据](ARROYO-RESULTS.md#窗口已验证范围与局限)、[复现脚本与原始包入口](ARROYO-REPRODUCE.md)。

首阶段“release 未完成/无 Arrow 运行证据”现已补齐，详见[成功构建与资源记录](ARROYO-RESULTS.md#版本构建与资源)。
[首阶段 CF 两引擎报告](REPORT.md)及下文只保留当时边界与数字；最新归因以
[补测方案](ARROYO-RESULTS.md#更新归因与分阶段方案)为准。
[benchmark 计时审计](BENCHMARK-AUDIT.md)逐组说明正常 API/规划等成本；维护主矩阵的
控制 JSON 和统计写出在核心计时之外，未发现旧集成路径那种逐行 JSON 出口混入核心 seconds。
更早 #494 的[报告、脚本和原始数据](../arroyo-comparison/README.md)、
[预记约定](PLAN.md)及[计时修正记录](MEASUREMENT-CORRECTION.md)也完整保留。

## 首阶段 CF 记录与离线复核

本节及以下命令属于首阶段：四个场景、每 arm 五次，最终 40 次、496,640 条输出逐值校验。
原始数据包保留所有 Arrow IPC 输入/输出、单次时钟/命令/RSS、graph/native seal，
并保留排除的首轮全部 40 次及其精确脚本，不能将两轮合并统计。
数据通路为 Arrow/Batch；JSON 文件只保存配置、图和统计元数据。

从 repo root 使用 Python 3.13、numpy 2.5.4、pyarrow 24.0.0 运行；不需要 native：

```bash
sha256sum --check .codex/artifacts/arrow-ipc-audit/data/SHA256SUMS
mkdir -p target/arrow-ipc-evidence
tar xzf .codex/artifacts/arrow-ipc-audit/data/DAL-417-arrow-raw.tar.gz \
  -C target/arrow-ipc-evidence
uv run --no-project --python 3.13 --with numpy==2.5.4 --with pyarrow==24.0.0 \
  python .codex/artifacts/arrow-ipc-audit/audit_results.py \
  target/arrow-ipc-evidence/raw-data/results
uv run --no-project --python 3.13 --with numpy==2.5.4 --with pyarrow==24.0.0 \
  python .codex/artifacts/arrow-ipc-audit/verify_columns.py \
  target/arrow-ipc-evidence/raw-data/results
```

成功输出 `audit=passed, samples=40, output_rows=496640`；summary/paired 也会离线重算。
核对 IPC 输入分批、完整输出数学值、安全整型规范化、校验和、sample inventory、worker
seal、时钟顺序和资源。审计不要求 native 已安装；读取当前脚本哈希以确保没有采样后改写。
补充 verifier 核对全部列名和所有 window_end−window_start=4 秒，覆盖原生窗口额外列。
首轮的历史审计可使用包内 `first-pass/harness/audit_results.py` 及 `first-pass` 路径；
其 audit passed 只证明原始记录自洽，`exclusion.json` 仍禁止将它纳入最终统计。

## 首阶段 CF 重新测量

仅需要新证据时运行；不为交接重复采样。要求 x86_64 Linux/WSL、CPU 0–7、uv、
Python 3.13、仓库 Rust toolchain。先从 PR checkout 复制本目录到固定基线 worktree：

```bash
git worktree add --detach target/arrow-ipc-baseline \
  b5ae227e48ac9bb8a81583a23485e2c6064cee52
cp -R .codex/artifacts/arrow-ipc-audit \
  target/arrow-ipc-baseline/.codex/artifacts/
cd target/arrow-ipc-baseline
bash .codex/artifacts/arrow-ipc-audit/reproduce.sh
```

脚本先拒绝错误 HEAD 和 tracked staged/unstaged diff，使用新目录。
默认 locked release 构建另计；若已有本轮兼容 wheel，可设置 `DAL417_WHEEL`，脚本会
检查固定 wheel SHA。新建构建的 native hash 会作为明确参数传给每个 worker，并记录到
新的 identity；不能冒用本次 native seal 或将不同主机的新数据混入此报告。

本次实际正式命令：

```bash
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/collect.py \
  --output target/arrow-ipc-audit/results --pairs 5 --budget 500
```

## 历史：Arroyo 首阶段接口与构建尝试

本节保留首阶段的补丁和失败记录；可运行 release、完整 Arrow 结果及窗口探针现已在
[最新结果](ARROYO-RESULTS.md)补齐，当前完整观察补丁和重现步骤见
[Arroyo 复现说明](ARROYO-REPRODUCE.md)。

`probe_arroyo.py` 对固定 release binary 尝试 arrow/arrow_ipc/ipc 三种格式，SQL
声明、退出码和错误日志均保留。`arroyo-ipc-adapter.patch` 是固定 v0.15.0 源码上的
最小 single_file source/sink 基准适配；真正调用 SourceCollector/真实图，未替换计算算法。
只对 `.arrow` 路径生效；SQL 的 format=json 仅为声明占位，适配分支不会调用 JSON codec。
该首阶段补丁通过 cargo check，但 release 构建当时未在预算内完成，尚无运行时/窗口语义证据。

```bash
git clone --depth 1 --branch v0.15.0 https://github.com/ArroyoSystems/arroyo.git \
  target/arroyo-ipc-reproduction
test "$(git -C target/arroyo-ipc-reproduction rev-parse HEAD)" = \
  4ef15f6336a596651f10cf27c1ee29c29bf81577
git -C target/arroyo-ipc-reproduction apply --unidiff-zero \
  ../../.codex/artifacts/arrow-ipc-audit/arroyo-ipc-adapter.patch
CARGO_BUILD_JOBS=4 CFLAGS=-std=gnu17 \
  CMAKE="$PWD/.codex/artifacts/arrow-ipc-audit/cmake-compat.sh" \
  cargo check --locked --manifest-path target/arroyo-ipc-reproduction/Cargo.toml \
  -p arroyo-connectors
```

上面相对补丁路径由 git 在目标仓库内解释。`cmake-compat.sh` 只给旧 librdkafka 的 CMake
配置补 policy minimum；不修改依赖源码/lock。实际受限 release 构建命令和 exit=124 在
原始包 `arroyo/adapter-release.json`，没有从未完成构建给出速度结论。
