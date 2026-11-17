# 流式核心重构：架构修改与实现方案

日期：2026-11-15。状态：执行中；全部阶段在单一集成 PR #619 中交付，P1 已并入，≤20 µs 门槛已移到 P2，见执行记录。

本文记录对 Calc Flow 流式引擎的一次整体 review、由此确定的目标架构，以及分阶段的实现方案。
它是后续各阶段规格和逐任务实施计划的统一入口；各阶段的逐任务计划另行编写，本文不包含。
当前实现契约仍以 [runtime envelope](../runtime-envelope.md)、
[streaming guide](../streaming-guide.md) 和源代码为准；本文描述目标状态，已实施的部分见执行记录。

review 基线为 `main` 的 `39cab2fb`，文中的 `路径:行号` 均对应该提交。
本次 review 只阅读了源码和仓库已有的测量产物，没有构建，也没有新增测量。
文中所有“目标”和“验收门槛”都是待各阶段验收的估计值，不是已取得的结果。

## 执行记录

- 交付方式（2026-11-16 用户决定）：整个方案只保留一个[集成 PR #619](https://github.com/wegamekinglc/calc-flow/pull/619)，
  所有阶段在同一分支上累积，最终按整体重构效果评估。原 #617、#618 已作为被集成 PR 取代而关闭。
- P1 `StreamSession`：原 PR #618 的提交已并入 #619，交付公开的同步 push API，
  包括 Rust `StreamSession`、`SessionWatermark`、`SessionOutput` 和 Python `TableExpr.session`、`Program.session`。
  现有 handler 由适配器在调用线程上按拓扑序跑完，没有 source、sink 和进度任务。
  准入范围是单个动态输入，rolling（不含 RSI/ATR/MACD 指标 layout）、表达式和 union 节点，无 checkpoint。
  规格为 `.codex/artifacts/specs/stream-session.md` 第 2 版，critique 和 review 是
  `.codex/artifacts/critiques/` 与 `.codex/artifacts/reviews/` 下的同名文件。
- P1 测量（`benchmarks/stream-session/p1-20261115-summary.md`，每个用例 30 对 AB/BA）：
  预登记门槛（1 行 append 的 session p50 ≤20 µs）未达到。强制 GC 下为 152.9 µs，
  常规 GC 的诊断运行为 90.5 µs，进程内热循环为 18.6 µs。
  相对 runner 的配对加速：1 行和 99 行 append 为 3.5–5.4 倍，6,400 行 64 实体为 1.9 倍；
  runner 对照的置信区间跨 1.0，没有回退。
  强制 GC 下的阶段归因是 Python 参数处理 18.9 µs、原生调用 126.9 µs、结果装配 7.0 µs；
  剩余成本主要在原生算子 dispatch 内，并随缓存状态大幅变化。
- 按 7.2 的停止规则定位成本后，用户于 2026-11-15 决定：P1 作为严格改进并入整体方案；
  ≤20 µs 门槛移到 P2，与每次调用的分配预算、rolling 和 projection 的同步算子 trait（4.1）一起验收。
- 决定 5：原 PR #617 的提交已并入 #619，让便捷流默认关闭 checkpoint，
  有序事件时间输入在每个推进的数据批之后发出 watermark，不再等 100 ms 定时器；
  显式 `StreamingRunner` 仍默认开启 checkpoint。配对测量已完成并归档在
  `benchmarks/stream-convenience-defaults/`：2 批用例按维护规则 improved（−39.24%），
  32 批用例为 inconclusive（−9.33%）；两项默认变化合并测量，不能分别归因。
- P1 其余项尚未开始：NumPy 路径，以及单源链路在 runner source task 内联（取代草稿 PR #616）。
- P2.1（完成；基线登记 `bb201632`）：五个 rolling dispatch 证据用例和 native 配对控制器已并入；
  10 轮基线在 14.8 秒内完成，原始样本、二进制身份和协议归档于 `benchmarks/rolling-kernel-v2/`。绝对基线为
  G1 6.264 µs、G2 19.577 ns/window-row、G3 452.568 µs；两臂使用同一二进制，1.00x 不是性能结论。
- P2.2（完成；数值实现 `16ae0d03` 经 `8434b862` 集成，精确 oracle `a7511438`）：
  rolling sum/mean、离散度和 pair 输出改用唯一的 compensated sliding 算术，Rust/Python/IR 的数值选项及
  project-v3 对应 enum/property 已删除；整数 sum 保持精确检查，未改递推保持不变。旧 profile marker 和
  先前 kernel identity 的 checkpoint 拒绝恢复。`python/tests/test_rolling_numeric_contract.py` 已加入
  exact `Fraction` oracle、误差界与特殊值分类检查，以及 batch/session 逐位一致性断言。
  本项不包括 P2.3 的 slab、undo log 或 layout 7。
  P2.2 的 10 轮 release 证据保存在 `benchmarks/rolling-kernel-v2/p2-2-08b17812.*` 和
  `p2-2-be094d3f.*`：初版大批量 dual-mean 为 0.83x，按输出关闭无用 dispersion 后仍为 0.89–0.90x。
  本步骤规格没有性能 Gate；该回退未隐藏，留作 P2.8/P2.9 的固定成本、run/full-tick 优化输入，最终 G1–G3 仍须验收。
- P2–P7 规格已全部并入：rolling 数值与内核、checkpoint v4、keyed state 与 join、并行执行与 window pane。
  实现按单一 PR 内的提交序列推进，不再建立分阶段 PR。
- P2.4 部分交付：Float64/Float32/Int64/UInt64 的 lag/delta 已进入 typed plan，保留 null/NaN、periods=0、
  checked arithmetic、溢出错误与 checkpoint 历史恢复；rank/quantile 也有 typed window state 和 ties/invalid-row
  parity，kernel registry 已同步；Utf8 lag 使用 typed `Option<String>` 历史和 `StringBuilder`；Duration
  argmax/argmin/unique_count 使用 ordinal、event-time expiry 和 typed candidates/counts；TimestampMicrosecond lag
  使用 signed `i64` 历史并恢复 UTC/naive 类型。rolling 703 项、symbolic optimizer 11 项和 clippy 通过。
  其余 temporal/nested lag、剩余 typed groups 与 general `ScalarValue` 路径删除仍未完成。
- P2.4 继续补齐 Date32/Date64/Time64Microsecond/DurationMicrosecond lag，复用 signed history 和输出类型恢复；
  rolling 707 项和 clippy 通过。其他 temporal/nested 类型仍需单独验收。
- P2.5 部分交付：MACD 三组件通过 typed group/readout 复用原生 `MacdKernel`/`MacdParameters`，不再创建
  indicator resource pool；RSI/ATR 已加入 typed plan 并通过 parity，但仍保留原 resource/lifecycle 路径。
  rolling 705 项和 clippy 通过。layout 5/6 删除与 RSI/ATR slab persistence 尚未完成。
- P3.1 fake-table 协议/故障 harness 已交付；P3.2 COW pages、dirty lineage、精确 undo 与 layout 7 已通过验证。
  P3.3 v4 object/manifest/partition persistence 已验证；P3.4 coordinator 与 opaque adapter 已有状态机和
  epoch roundtrip。Rust 与 Python `StreamSession` 的 durable construction、restore 和同步 checkpoint 已交付，
  使用 managed storage、`job_kind=session`、canonical opaque fingerprint 与 input/watermark progress；Rust session 25 项、
  Python focused 53 项通过，包含 epoch 1/2 continuation 和 running-loop 拒绝。runner 的实际
  source/operator/sink 接线、完整故障矩阵和性能门槛尚未完成。
- P4.1 shared keyed core 与单整数 SUM 已交付；P4.2 typed aggregate state、temporal extrema、8192-group dirty
  snapshot cache 和 rollback/recovery 已通过 327 个 SQL 测试。P4.3 opt-in changelog 已集成，
  SQL 336 项与 Python 5 项测试通过。旧 compact 删除和 P4 测量尚未完成。
- P2.3 已完成 Float64 Rows slab、共享多窗口 history、layout 7 和 pending restore，rolling 694 项测试通过。
  后续同一基线的 paired 测量最初发现 large-batch 回退扩大为 0.29x；批量 counter、readout 定位、逐窗口
  rebuild history、O(1) retained count、checkpoint-off COW elision 和 numeric family 修复后，中间证据
  `p2-2-slab-family-selection` 为 0.68x，随后 first-touch undo 和 checkpoint-off direct recurrent 将
  `p2-2-slab-direct-recurrent` 提升到 1.09x/1.11x（faster），消除了相对基线的回退；绝对 G1–G3 仍未达标。
  独立审查发现的 freeze rollback、mean-only rebuild restore 和 abort dirty-loss 问题已各有回归测试并修复。
  其余 typed groups、P2.4–P2.9、join/ASOF、并行/SIMD 与 window pane 仍需实现和验收。
- P2.7 已为 admitted ordered graphs 接入同步 Rolling：serial prepare/update/emission、sync adapter 双类型分派、
  checkpoint/restore/reset 和空输出处理，StreamSession 24 项、rolling 705 项和 clippy 通过；其他图继续使用
  async runner。`p2-2-slab-sync-rolling` 的 paired large-batch 为 1.12x/1.13x，未超过 full-tick 的 1.15x；
  G1 5.394 µs、G2 20.882 ns/window-row、G3 399.043 µs，绝对门槛仍未通过。
- P2.9 已交付首个 full-tick tier：严格验证 entity 顺序后，首 tick 完整 first-touch，后续 tick 复用同一
  transaction/partition，非 full-tick 自动回退。rolling 701 和 clippy 通过；`p2-2-slab-full-tick` 的 paired
  large-batch 为 1.15x，G2 19.291 ns/window-row、G3 371.773 µs，绝对门槛仍未通过。
- P6 SIMD 打包（`311d248c`）已交付：独立 `calc-flow-python-v3` 与 portable 版本使用相同版本号和
  `calc_flow` import package，要求在干净环境显式安装且二者互斥；Linux、macOS、Windows 三个 x86-64 target
  各有六个 Python tag 和两个 abi3 tier，不发布 v3 sdist。只有 v3 job 设置 `target-cpu=x86-64-v3`，portable
  与 AArch64 保持不变；artifact/hash、v3 vector inspection、clean-install smoke 和独立发布链路已接入。
  静态测试通过：`python/tests/test_distribution.py` 4 项、`scripts/test_release_workflow.py` 4 项、
  `scripts/test_inspect_v3_wheel.py` 2 项、`scripts/test_release_config.py` 42 项；实际 wheel 与跨平台发布仍由
  release workflow 验收。P6 的分区执行器、统一 morsel 池、I/O 与计算分离及性能门槛仍未完成。

## 1. 目标

1. 尽可能提高小 batch 场景下流式引擎的执行性能，与 Arroyo 和 TA-Lib 可比。
2. 重构 checkpoint 机制，使其清晰，并尽量不影响执行性能和并行计算。
3. 调整架构，提升对多线程执行和 SIMD 的支持。
4. 提升 join、group by 的性能。
5. 为 window sum 一类算子单独给出性能提升方案。

整个方案允许对架构和内部实现做较大调整。

## 2. 已批准的决定

| 编号 | 决定                                                                           | 说明                                                                                     |
|------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1    | 引入公开的同步 push API `StreamSession`                                        | 推翻此前对 `StreamingRunner::step` 的移除；保留 runner 只换入口的方案估计停在 200–300 µs |
| 2    | 删除 `stable_v1` 和 `stable_v2`，新算术是唯一数值契约                          | 见第 5 节                                                                                |
| 3    | 旧 checkpoint 不需要可恢复，状态格式和协议完全重做                             | 新格式自身的恢复仍保证逐位续算                                                           |
| 4    | 保持 `unsafe_code = "forbid"`，SIMD 依靠可自动向量化的循环和 `x86-64-v3` wheel | 不引入带 `unsafe` 的 kernels crate                                                       |
| 5    | 便捷流默认关闭 checkpoint，有序输入按批推进 watermark                          | 两者都是可见的行为变化                                                                   |
| 6    | 聚合的 changelog 输出为显式选项，完整快照仍是默认                              | 该项未单独答复，按建议执行；P4 规格前可再确认                                            |

## 3. 现状诊断

### 3.1 成本分层

负载为 warm 状态的 dual SMA(5)−SMA(20)。数据来自 PR #616 在 `b9a46e12` 上的本地测量产物
（`target/compiled-region-performance/`，未入库），主机为 WSL2。

| 层                                                              | 1 行 append | 99 行 append | 6,400 行，64 实体 |
|-----------------------------------------------------------------|-------------|--------------|-------------------|
| Python 流式全链路                                               | 599 µs      | 615 µs       | 908 µs            |
| 原生算子 dispatch（rolling 加 projection，不含 source 和 sink） | 6.1 µs      | 9.3 µs       | 329 µs            |
| TA-Lib 适配器（参照）                                           | 5.8 µs      | 5.9 µs       | 229 µs            |

由此得到三层成本：

- **异步外壳**：约 590 µs，与行数无关，位于算子之外。
- **算子 dispatch 固定成本**：约 6 µs。
- **每行内核**：33–51 ns。

P1a、P1b 融合和 #616 region 优化的是算子之间的那一段，所以没有收益：
#616 的配对比值为 0.96–1.11x，置信区间全部跨 1.0。

### 3.2 没有同步执行核心（目标 1）

- 算子全部是 `#[async_trait]` handler（`crates/calc-flow/src/operator/stream.rs:753`），只能由 Tokio 任务图驱动。
  同步 push 入口是被刻意移除的（`crates/calc-flow/src/lib.rs:19`）。
- 边是带互斥锁的 `VecDeque` 加 `Notify`（`crates/calc-flow/src/runtime/streaming/edge_queue.rs:12`）。
- 基准拓扑下，一次 append 在计时窗口内有 4 次 Tokio↔asyncio 跨线程唤醒、2 次 asyncio 事件循环轮转、
  约 5 次 Tokio task 跳转和 3 次 GIL attach。Data 和 Watermark 各是一次完整的 Python 回调往返。
- 1 行 append 的分阶段中位数：入队到 source 41 µs，data 到 watermark 250 µs，watermark 到 sink 215 µs，
  sink 到接收 81 µs，其中 `to_pyarrow` 占 48 µs。
- 单次回调的桥接成本（`benchmarks/rolling/callback-sparse-818d69c.json`，开启 profiling）：
  排队到分发 48–62 µs，完成到 Rust 恢复 56–80 µs。
- `to_pyarrow` 走 `RecordBatchReader` 再 `pa.table`（`crates/calc-flow-python/src/batch.rs:417`）。
- 任务拓扑不是主因：Tokio worker 从 32 减到 1 只有 1.02–1.11x
  （`benchmarks/rolling/scheduler-32-to-1-818d69c.json`）。
- 便捷 API 的默认 watermark 是 `max_seen − 1 µs`，由 100 ms 定时器发出
  （`python/calc_flow/_stream_inputs.py:124`）。这是一个与 CPU 无关的延迟下限。
- 与 Arroyo 的集成路径对比（2026-10-21 基线，WebSocket JSON 到计算到 stdout JSON，calc-flow 侧含 Python 适配；
  `.codex/artifacts/arroyo-comparison/REPORT.md`）：burst 吞吐 Arroyo 是 calc-flow 的 2.26 倍（投影与过滤）
  和 4.50 倍（四秒窗口）。

### 3.3 Checkpoint 不是可插拔的一层（目标 2）

- 开启 checkpoint 会选择另一条更慢的执行路径：非测试代码里有 24 处 `.checkpointing()` 分支，
  分布在 join、window、rolling、asof 和 sql。
- Rolling 开启后失去 in-place 更新和延迟历史（`crates/calc-flow/src/operator/rolling/ordered_stream.rs:337`），
  并为每个实体维护第二份状态 `RollingHistories`。
- Python 默认 `checkpointing=True`（`python/calc_flow/runtime.py:773`）。
- 触发时停流：barrier 要等 prepare、capture、encode 和落盘全部完成才转发
  （`crates/calc-flow/src/runtime/streaming/operator_task.rs:1722`）。
  落盘持有全 job 一把锁（`crates/calc-flow/src/state/transaction.rs:164`），
  每个新 segment 做三次 SHA-256、至少两次 fsync。每次 checkpoint 停顿 40–240 ms。
- capture 有四种机制：SQL 整算子出借、join 的 owned capture、ASOF 的 managed capture，
  其余算子走同步的 `checkpoint(&mut self)`。
- 复杂度：9 个算子族、约 20 个 layout 或版本常量、约 12 种二进制 magic、4 套增量方案和 3 个全量重写族。
  checkpoint 相关的非测试代码估算约 5 万行，其中 join 约 1.39 万、ASOF 约 1.06 万、rolling 约 0.97 万、
  `state/` 约 0.9 万。
- `benchmark-checkpoint-off` feature 没有从产品路径编译掉任何东西。

开启但未触发就已经有成本：

| 负载                        | 关闭     | 开启未触发          | 周期触发                                      |
|-----------------------------|----------|---------------------|-----------------------------------------------|
| `window_sum`，10M 行        | 1,010 ms | 1,616 ms            | 1 s 间隔 1,930 ms；200 ms 间隔 2,719–3,034 ms |
| Interval join，每侧 200k 行 | 263 ms   | 404 ms（24 h 间隔） | 100 ms 间隔 880 ms                            |

来源分别是 `.codex/artifacts/specs/window-operator-async-checkpoint-flush.md` 和
`.codex/artifacts/analysis/issue406-checkpoint-modes.md`。

### 3.4 并行是单 owner 加临时外包（目标 3）

- 只有一个共享 Tokio 池：Python 用 `tokio::runtime::Runtime::new()` 创建
  （`crates/calc-flow-python/src/config.rs:325`），worker 数等于逻辑核数。
  I/O 和计算没有分离，CPU 工作直接在 Tokio worker 上内联执行。
- 每个算子或融合链一个 task，并行度恒为 1。没有 keyed exchange、shuffle 或算子实例的概念。
- 有三套互不相干的外包机制：`gather_work` 自建线程池（join、ASOF、排序、全局 SQL）、
  rolling lanes 的 `spawn_blocking`、checkpoint 编码的 `spawn_blocking`。
- 准入门槛很高：rolling lanes 要求不少于 64,000 行且每 lane 不少于 16,000 行；
  join 探测要求不少于 8,192 行且不少于 320,000 次访问。
  空载分发的中位数是 45 µs（1 个单元）和 88 µs（4 个单元）
  （`crates/calc-flow/src/operator/join/data_work.rs:20`）。
- 实测不扩展：join 在 1 核是 87.1 ms，32 核是 89.5 ms；grouped SUM 在 4 核以上停在约 1.2 s；
  ASOF 1M 行在 32 核只改善 38.5%。
- 三条并行 grouped SUM 路线（行区间 lanes、完整键分片、委托 DataFusion 的 delta 聚合）都被撤回，
  原因是每批的路由、拷贝和合并是串行的（`benchmarks/sum-state/`）。
- DataFusion 在流路径上默认单分区（`crates/calc-flow/src/datafusion.rs:104`）。
- SIMD 无从谈起：
  - rolling 状态是 enum 后面的 AoS，每行都有实体查找、enum 匹配、`VecDeque<TimedSample>` 的 push 和 pop、
    checked 计数、`Result` 和 builder 的 `append_option`；
  - 聚合是每组一个 enum 或 `Box<dyn Accumulator>`；
  - 几乎没有可自动向量化的数值循环；
  - 仓库没有 `.cargo/config`、`RUSTFLAGS` 或 `target-cpu` 设置，x86-64 wheel 只有 SSE2。

### 3.5 Join 与 group by 是行对象式状态（目标 4）

| 负载（单核）                    | calc-flow 流式 | 参照                               | 来源                                                                                            |
|---------------------------------|----------------|------------------------------------|-------------------------------------------------------------------------------------------------|
| Join，1M × 64                   | 78.5 ms        | DataFusion 13.0 ms，Polars 18.0 ms | `.codex/artifacts/analysis/join-single-thread-performance.md`；参照值来自 2026-10-20 的单核矩阵 |
| Interval join，每侧 1M          | 672 ms         | DataFusion 346 ms                  | 同上                                                                                            |
| ASOF，1M                        | 84.0 ms        | Polars 63.8 ms                     | 同上                                                                                            |
| Grouped SUM，10M 行，100×100 组 | 1.67 s         | calc-flow 批 SQL 0.43 s            | `benchmarks/stock-positions/engine-core-sweep-20261107.md`；两边计时边界不同                    |

- **Join**
  - key 逐行经过三层身份：批内的 `LocalIntern`、持久的 `Dictionary`、每个 key 的 `NativeIndex`。
  - 最内层循环每 64 次访问或 4 KiB 做一次 `check_cancelled` 和 `consume_budget().await`
    （`crates/calc-flow/src/operator/join/columnar/owned_copy.rs:65`）。
  - 每个准入行都走事务式 prepare，包括不保留的行。
  - 输出物化和逐对字节计量占 15–29%。
  - 非原生 key 走一条对物化状态跑 DataFusion 等值 SQL 的旧探测层，成本与状态大小成正比。
- **Group by**
  - 状态是 `Vec<Group>` 加 `HashMap<Arc<[u8]>, usize>`。每组带 `Arc<[ScalarValue]>`、聚合 enum
    和独立的 `MemoryReservation`（`crates/calc-flow/src/operator/sql/incremental.rs:139`）。
  - 每批为每个被触及的组建一个写时复制的候选。
  - 每批输出全部分组的完整快照。
  - 不符合准入的查询形状保留全部输入，每批重跑整个查询。
  - inline 整数 SUM 内核本身约 10 ms 每 1M 行，差距在表达式节点、快照输出和进度协调。
- **Window 聚合**
  - 状态是 `BTreeMap<WindowKey, AccumulatorRow>`（`crates/calc-flow/src/operator/window.rs:335`），
    key 含窗口起止和 `Arc<[u8]>` 分组键。每批排序加两次建树。
  - hopping 把每行展开到所有开放窗口，最多 1,024 个。
- **ASOF**：已经最接近列式设计（每个 key 持有排序的列式 run），准入阶段占 69%（含子调用）。

### 3.6 Rolling 内核的固定成本与布局（目标 5）

- Data 回调只缓冲，Watermark 回调才最终化并输出。
- 约 6 µs 固定成本的来源（读代码推断，没有分阶段实测）：
  - 每次 dispatch 新建三个 `RowConverter`；
  - 约 12 个阶段计时 guard，每个两次互斥锁加两次时钟读取；
  - 端口校验里的 `format!`；
  - 每批深拷贝所有被触及的实体（`crates/calc-flow/src/operator/rolling/kernel.rs:1740`）；
  - 输出装配和两次 `EnvelopeCost`。
- SMA 和 dual SMA 永远走不到已有的 in-place 路径，因为它排除了 `MeanKernel`
  （`crates/calc-flow/src/operator/rolling/kernel/in_place.rs:420`）。
- 每行成本（2026-11-08 的阶段计数，旧源码）：West 递推每窗口每行 16–20 ns（见第 5 节），
  实体路由约 12 ns，顺序证明约 7 ns。
- 状态布局是 `HashMap<Vec<u8>, usize>` 到 `Vec<Arc<TypedEntityState>>` 到 `Vec<TypedWindowState>`
  再到 `VecDeque` 或 `Arc<RowsTape>`，每行三到四次指针追逐，每个样本 24 字节。
- 通用路径悬崖：只要有一个 `lag` 或 `delta` 输出（`crates/calc-flow/rolling-kernels.json` 中标为 general），
  整个算子就退到 `ScalarValue` 和 `BTreeMap` 的行路径。
- 有限（批）路径使用同一个逐行内核。
- 累加器有四套独立实现：rolling、`array/rolling_mean.rs`、window 和 SQL compact。

### 3.7 共性

核心 crate 约 46 万行 Rust，其中约 16 万行是测试。join 约 8.6 万行，rolling 约 7 万行，asof 约 5.4 万行，
流式运行时约 8.6 万行。每个算子自带状态模型、准入计费、checkpoint 编码和并行设施，
没有共享的 keyed-state 层或状态快照层，所以每项优化都要逐算子重做。

### 3.8 证据边界

- 小 batch 的数字是 2026-11-15 的本地测量；分阶段归因结合了一个较旧的带 instrumentation 的构建，阶段之间有重叠。
- join、group by 和 checkpoint 的数字来自 10 月到 11 月初的产物，早于若干修复，未在 `39cab2fb` 上重测。
- 6 µs 固定成本的拆分和所有预期收益都是估计。
- 所有测量都在 WSL2 主机上，唤醒延迟在裸机上可能更低。

## 4. 目标架构

```text
异步 I/O 面 (Tokio):   sources ── checkpoint 持久化 ── sinks
                          │            ▲                ▲
                          ▼            │ 冻结的状态页    │
同步计算核心:   router → shard 0..N-1（每个 shard 是一条 run-to-completion 的 Pipeline）
                共享: keyed-state 核心 · 内核库 · 状态表
驱动方式:   StreamSession.push()（调用线程）| StreamingRunner（连接器）| 分片执行器
```

### 4.1 同步计算核心

算子实现同步 trait。下面是示意，最终签名由 P1 规格确定。

```rust
trait Operator {   // 无 async、无 Box<dyn Future>
    fn on_data(&mut self, port: PortId, batch: &RecordBatch, out: &mut Emitter) -> Result<()>;
    fn on_watermark(&mut self, wm: EventTime, out: &mut Emitter) -> Result<()>;
    fn on_end(&mut self, out: &mut Emitter) -> Result<()>;
}
```

- Pipeline 内部的边是函数调用。发射的批进入该边的预分配出箱，按拓扑序处理。
- watermark、barrier 和 EOF 在同一调用栈内按序处理，保持“先发射已关闭的结果，再转发 watermark”和 FIFO 控制顺序。
- 内核里没有 `.await`。取消检查按 4–8K 行的 morsel 做。
- 指标是 Pipeline 私有的普通计数器，按需或周期性快照发布到 status。
- 预算在入口对整个 envelope 做一次准入（超大消息拒绝保持不变），在出口对输出做一次检查。
  内部边不排队，所以不逐边计费。

同一个核心有三种驱动方式。

**`StreamSession`（调用线程）。** 下面是示意。

```python
session = program.session(config=...)                 # 编译一次，持有原生状态
out = session.push({"ticks": batch}, watermark=ts)    # 调用线程内跑完，直接返回结果
session.checkpoint()
session.close()
```

- 没有线程跳转；Data 和 Watermark 合并成一次调用。
- `watermark=` 让调用方声明“到此为止已完整”，结果在同一次调用里返回。
  不传时沿用 `max_seen − 1 µs` 的生成规则，并按批推进。
- Python 侧缓存 schema 后直接导出 `RecordBatch`，另提供 NumPy 路径。
- checkpoint 是两次 push 之间的一个点，不需要 barrier 对齐。

**`StreamingRunner`（连接器）。** 保留，变成 source task、Pipeline、sink task 三段。
单源链路直接在 source task 内联执行；多源时入口有一个带预算的邮箱，一次唤醒排空全部可用消息。

**分片执行器。** 见 4.5。

迁移方式：现有 handler 在直调 harness 里已经能约 6 µs 跑完，先用一个“poll 到完成”的适配器包住它们，
`StreamSession` 就能在重写算子之前交付。真正会挂起的算子（走 DataFusion `execute_stream` 或 `gather_work` 的）
先留在旧路径。每个算子迁移到同步 trait 时删除其旧实现。

### 4.2 运行时拥有的状态表

- 算子不再实现 `checkpoint()` 和 `restore()`，而是在运行时拥有的状态表里声明状态：
  按 key 的列、定长环形缓冲、追加日志。
- 状态按 key group 分区，一个 shard 拥有若干 key group。
- 页带 epoch 标记，写时复制，配页级脏位图。快照就是冻结页引用，成本与脏页数成正比，算子不做编码。
  SQL 的 `DirtyGroups` 位图是现成范例。
- 批内原子性用 undo log（推广 `kernel/in_place.rs` 的稀疏 undo），或者“先预检，再不可失败地应用”。
  不再为每批克隆被触及的状态。
- 关闭 checkpoint 时脏位图是空实现。热路径只有一条，不再有 `.checkpointing()` 分支。
- 逻辑状态计费按列缓冲区的算术计算，不逐行，也不逐实体持有 `MemoryReservation`。

### 4.3 共享 keyed-state 核心

- 每批对 key 列做向量化哈希，经一张哈希表映射到稠密 group id。
- key 存在列式 append-only 存储里，常见类型有专用快路径。
- 规范 key 字节只在 key 首次出现和 checkpoint 时计算。
- group by、window、join、ASOF、rolling 实体路由和 distinct 共用这一套。

### 4.4 内核库

- 实体状态是 SoA slab：稠密 `EntityId`、定长 f64 环形缓冲、累加器列，全部是连续向量。
- 内核分两级，在计划期选定：
  - 标量级处理极小 append：id 到 slab 到累加器到输出槽；
  - run 级处理大批：按 `EntityId` 稳定分组，对每个实体的 run 把累加器放在寄存器里，
    一次预扫描证明没有 null、NaN 和 Inf 后走无分支循环；
  - 完整 tick（每实体一行）时跨实体并排更新。
- 按 (dtype, frame, 无空值证明) 的特化在计划期完成，不在每行做 enum 分发。
- 输出直接写入 `&mut [f64]` 加位图，不逐行调用 builder。
- rolling、window pane、array 和 SQL 共用一个累加器模块。
- SIMD 交付靠可自动向量化的循环加 `x86-64-v3` wheel。

### 4.5 按 key 分片

- N 条 Pipeline 各自独占状态分区。入口按 key 哈希路由一次，出口做确定性有序合并。
- barrier 和 watermark 按 shard 广播，在合并处对齐。
- 小批时 N=1，在调用线程内联；阈值由测量决定。
- `gather_work`、`entity_work` 和散落的 `spawn_blocking` 合并成一个 morsel 池，只服务无状态的大批工作。
- I/O 面和计算面分离，计算不再占用 Tokio worker。

### 4.6 Changelog 输出协议

聚合只输出变化的分组；需要完整快照时在边界处物化。默认仍是完整快照。

## 5. 新数值契约

### 5.1 `stable_v1` 的除法依赖

`stable_v1` 用 West/Welford 递推算 rolling mean：每进一个样本、每出一个样本，都直接更新 `mean` 本身
（`crates/calc-flow/src/array/rolling_mean_math.rs:63`）。

```rust
// 进窗                         // 出窗
count += 1;                     count -= 1;
delta = x - mean;               delta = y - mean;
mean += delta / count;          mean -= delta / count;
m2   += delta * (x - mean);     m2   -= delta * (y - mean);
```

- 新的 `mean` 要用上一步的 `mean` 经过一次除法才能得到，CPU 必须等这次除法完成才能开始下一行。
- 一次 f64 除法延迟约 13–20 个周期，每个窗口每行两次，dual SMA 是四次。
  这给出每窗口每行约 10 ns 的下限，与其他优化无关；实测是 16–20 ns。
- 同一实体内的行因此无法向量化。
- 这个累加器由 sum、mean 和 variance 共用，所以即使只要 window sum，也在为 `mean` 和 `m2` 的除法付费。
- `stable_v2` 预览版每行仍是这套递推，只是多了周期性重建。

TA-Lib 的写法是 `total += x_new; out = total / n; total -= x_old`。
依赖链上只有加减法，除法结果不回写状态，相邻行可以重叠执行，也能向量化。

### 5.2 新算术

- **sum 和 mean**：带补偿的滑动和，mean 在读出时才做除法。
  补偿采用“误差项单独累加、不回写主和”的形式，依赖链上每行只有一次加法。
  现有的 `CompensatedSum` 是 Kahan 形式，补偿项回写会把链拉长，只用在周期性重建里。
- **variance、stddev、covariance、correlation**：以每实体基准值平移后的补偿和与平方和，
  即 `stable_v2` 重建时已在用的方法。
- **周期性精确重建**：每实体每 max(W, 64) 个位置从环形缓冲重建一次，界定累计误差。
  节奏按实体位置计数，所以结果与批次切分无关。
- **不变的部分**：null、NaN 和 ±Inf 的语义分类（有效样本计数、`min_periods`、同号无穷得无穷、异号无穷得 NaN）；
  整数 sum 的精确检查；EWMA、decay、RSI 等递推。

### 5.3 可见后果

1. 所有 rolling mean、variance 和 covariance 的输出末位会变，批和流两条路径都变。
   现有逐位对照的测试和 fixture 要重建。
2. 窗口内的和超过 f64 上限时得到 ±inf，而 `stable_v1` 在这种输入下 mean 是有限值。
3. project-v3 schema 里 `numerical_profile` 的 `stable_v1` 和 `stable_v2` 两个枚举值被删除，
   显式写了它们的 project 文档会被拒绝；`schemas/project-v3.schema.json` 有预期的 diff。

### 5.4 验收方式

验收从“与旧输出逐位一致”改为三条：

- 相对精确参考值在声明的误差界内；
- 相同输入在任意批次切分下逐位一致；
- 从新格式 checkpoint 恢复后逐位续算。

精确参考 oracle 和误差界的定义在 P0 建立。

## 6. Checkpoint 协议

### 6.1 协议

1. 源切点和 barrier 注入不变。`StreamSession` 驱动下不需要 barrier。
2. 算子在 barrier 处冻结状态页引用，随即转发 barrier 并继续处理。
3. 持久化服务在后台写出冻结的页。每页哈希一次，每个 epoch 合并 fsync。
4. sink pre-commit 后等待 manifest。
5. 所有句柄持久化完成后发布 manifest v4。它仍是最后发布的唯一持久真相，然后 sink commit、source ack。
6. 最多一个在途持久化 epoch；下一个 barrier 到来时若上一个未完成则背压。

### 6.2 兼容性

- manifest 升到 v4，含按分区的状态句柄。旧 lineage 启动时给出明确错误，需要新建 lineage。
- 每个算子迁移时删除其旧 layout 和旧编码，不保留只读恢复。
- `tests/fixtures/v1/` 是历史 v1 对照证据，不受影响。

### 6.3 P3 规格必须定义的内容

- 失败模型的变化。现在的规则是 stage 失败不转发 barrier（`docs/runtime-envelope.md:571`）；
  新协议下 barrier 已经转发，在途 epoch 失败时该 epoch 作废，不发布 manifest。
- 取消、磁盘满、写失败、校验失败、worker panic 和 manifest 发布不确定时如何收敛。
- 在途 epoch 钉住的内存上限。
- 恢复时的完整性校验不得后移到对外确认之后。

### 6.4 未迁移的算子

尚未迁移的算子，其旧快照先当作不透明状态块接入新协议，落盘移到后台。
它们在各自重建的阶段落到状态表上。

## 7. 阶段与验收门槛

| 阶段 | 内容                                                                                                                | 验收门槛（估计值）                                                                                                                      |
|------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| P0   | 固定基线和验收用例；建立新算术的精确参考 oracle                                                                     | 每个后续阶段有基线、目标用例和停止规则                                                                                                  |
| P1   | 同步核心和 `StreamSession`，先覆盖 rolling 后接 projection；Python push 与直接导出；便捷流默认值；并入 #616         | Python 1 行 append 从 599 µs 降到 ≤20 µs（Arrow，已移到 P2）、≤5 µs（NumPy）；大批对照无回退                                            |
| P2   | Rolling 内核重写：新算术、SoA slab、undo log、所有输出进 typed plan、两级内核；rolling 和 projection 迁到同步 trait | 原生 1 行 ≤1 µs；每窗口每行 2–4 ns；6,400 行 64 实体从 329 µs 降到 ≤100 µs；Python 1 行 append ≤20 µs（自 P1 移入）和每次调用的分配预算 |
| P3   | Checkpoint 重做：写时复制页和脏位图、后台持久化、manifest v4                                                        | 已迁移算子开启未触发开销 ≤3%；1 s 间隔 ≥90%、100 ms 间隔 ≥75% 的关闭态吞吐                                                              |
| P4   | keyed-state 核心、列式分组状态、changelog 输出                                                                      | Grouped SUM 10M 行单核从 1.67 s 降到 ≤0.5 s                                                                                             |
| P5   | Join、interval join 和 ASOF 在 keyed 核心上重建                                                                     | Join 1M 单核从 78.5 ms 降到 ≤30 ms；interval 从 672 ms 降到 ≤350 ms                                                                     |
| P6   | 分片执行器、统一 morsel 池、`x86-64-v3` wheel                                                                       | Keyed 大批负载 4 核 ≥3 倍；小批路径不变                                                                                                 |
| P7   | Window 改 pane；其余算子迁移；删除旧路径                                                                            | 高重叠 hopping 用例有改善；核心只剩一条执行路径和一种状态格式                                                                           |

Arroyo 对齐作为独立门槛：在同一集成 harness 上 burst 吞吐不低于 Arroyo 的 0.8 倍，
在 P1 加原生连接器链路之后验收。

### 7.1 P0：基线与验收用例

- 在当前 main 重新测量并固定以下用例的基线，复用现有 harness：
  - 小批 warm rolling：1 行、99 行单实体，6,400 行 64 实体；分别经 Python 全链路和原生 dispatch；
  - TA-Lib 适配器参照；
  - grouped SUM：10M 行，以及 640 行小批；
  - join、interval join 和 ASOF：1M 行，单核；
  - checkpoint 模式：关闭、开启未触发、1 s 间隔；
  - Arroyo 集成路径。
- 建立新算术的精确参考 oracle，写下误差界的定义。
- 产出每个用例的命令、构建身份、基线数值、目标值和停止规则。每个后续阶段从中选不超过五个用例。

### 7.2 P1：同步核心与 `StreamSession`

- 同步驱动器和 `StreamSession`，Rust 与 Python 两侧。
- 首批准入的图：单源、rolling 后接精确投影、无 checkpoint。现有 handler 通过适配器运行。
- Python `push`、缓存 schema 的直接 `RecordBatch` 导出、NumPy 路径。
- 便捷流默认关闭 checkpoint；有序输入按批推进 watermark。
- #616 的 region 准入逻辑和测试并入，不单独合并。
- 停止规则：适配器方式达不到 ≤20 µs 时先停下重新定位成本，不继续后续阶段。
- 执行结果：门槛未达到；定位成本后，用户决定把 ≤20 µs 门槛移到 P2（见执行记录）。

### 7.3 P2：Rolling 内核重写

- 第 5 节的新算术；删除 `RollingNumericalProfile` 的两个旧值。
- 实体状态改成 SoA slab，直接建在状态表接口上，先用最简单的整表快照实现。
- 事务改成 undo log，对所有 group 种类和开启 checkpoint 的 job 生效。
- 去固定成本：计划期缓存 `RowConverter`、持久的 key 到 `EntityId` 路由、类型化的 `(ts, seq)` 末位身份、
  预计算的端点字符串、无锁的阶段计数。
- 所有输出种类进入 typed plan，包括 `lag` 和 `delta`。
- 两级内核，加完整 tick 的跨实体更新。
- 2026-11-08 计划里的 decay 增量内核和多窗口共享历史在这里落地。
- 有限（批）路径和流路径共用同一套内核。
- 删除：每批克隆、`RollingHistories` 双份状态、`ScalarValue` 通用路径、layout 1–6 及其编码。
- rolling 和 projection 迁到同步算子 trait（4.1），`StreamSession` 不再经适配器驱动它们。
- 验收另含自 P1 移入的 Python 1 行 append ≤20 µs（Arrow）和每次调用的分配与内存占用预算，
  同时报告进程内热循环和 harness 两种延迟。

### 7.4 P3：Checkpoint 重做

- 状态表的写时复制页和脏位图实现；第 6 节的协议；持久化服务；manifest v4。
- rolling 已在状态表接口上，切到新实现，内核不用改。
- 其余算子按 6.4 接入。
- 删除全 job 的 stage 锁和重复哈希。
- 门槛针对已迁移的算子（此时是 rolling）。group by、join 和 window 在各自迁移的阶段继承同一组门槛。

### 7.5 P4：keyed-state 核心与 group by

- keyed-state 核心（4.3）。
- SQL compact 聚合改成按 group id 索引的类型化状态列；undo log 取代每组候选；
  取消检查按 morsel 做，去掉每个 chunk 的 `yield_now`。
- 完整快照输出改为切片状态列，只重建脏块。
- changelog 输出作为显式选项。规格需定义初始基线、group 身份、更新顺序、flush 时机和恢复后如何重建基线。
- 扩大 compact 准入，减少“保留全部输入并重跑”的查询形状。
- 状态落在状态表上；删除 SQL compact 的旧 layout。

### 7.6 P5：Join、interval join 和 ASOF

- build 侧改成列式 run：批切片加 key-id、time 和 row-id 列。
- 非保留行通过 retain 位图跳过；定宽 schema 按批计费。
- 探测产生 selection vector，输出用 `take` 和 `interleave` 一次物化。
- 淘汰按 run 的前缀进行。
- 删除 SQL 探测层、三层 key 身份和内层循环的 async 计量。
- interval join 和 ASOF 迁到同一核心。
- 删除 join v1、v2 和 ASOF 的旧编码。

### 7.7 P6：分片与并行

- 分片执行器（4.5）。
- 统一 morsel 池，取代 `gather_work`、`entity_work` 和散落的 `spawn_blocking`。
- I/O 面和计算面分离。
- `x86-64-v3` wheel。

### 7.8 P7：Window 与收尾

- Window 聚合改成按 slide 的 pane 列式状态，建在 keyed 核心上；关闭窗口时合并 pane。
  先接入 COUNT、MIN 和 MAX；SUM 和 AVG 的 pane 合并次序由规格定义后接入。
- cross-section、distinct、order 和 snapshot SQL 迁移到同步 trait 和状态表。
- 删除旧的 async `StreamOperator`、已被替代的 runner 任务图部分、`gather_work`、`entity_work` 和所有剩余旧编码。

### 7.9 顺序与衔接

- P1 最先：它对应最大的那一层成本，其余阶段都建在它之上。它也最便宜证伪。
- P2 和 P3 的协议部分可以并行。
- P2 的 slab 建在状态表接口上，P3 只替换接口后面的实现。
- P4 先于 P5，因为 keyed 核心是共享的。
- P6 必须在状态可分区之后：此前的分片尝试败在逐算子的路由拷贝上。

## 8. 交付与测量规则

- 每个阶段是独立的规格、逐任务计划和实现周期。
  公共契约变化（P1 的 API、P2 的数值、P3 的协议、P4 的输出协议）先经过规格、API 和 critique。
- 分支、提交和 PR 遵守 [AGENTS.md](../../AGENTS.md)。每个阶段可以拆成多个可独立撤回的 PR。
- 每个行为变化先写失败测试。
- 测量遵守 [AGENTS.md 的 Verification](../../AGENTS.md#verification) 和
  [benchmark suite](../benchmark-suite.md) 的配对规则：
  每个实现项最多五个用例、合计 600 秒；执行前登记基线、候选、样本数和预算；
  证据不足即停止，不追加样本，不更换用例。
- 迁移即删除：一个算子迁移到新核心的同一批 PR 里删除它的旧实现、旧 layout 和旧编码，不长期维护两条路径。
- 没有证明真实收益的阶段不进入默认路径。

## 9. 可见契约变化

| 变化                                                                   | 阶段  | 影响                                  |
|------------------------------------------------------------------------|-------|---------------------------------------|
| 新增公开的 `StreamSession`                                             | P1    | 新 API                                |
| 便捷流默认关闭 checkpoint，按批推进 watermark                          | P1    | 结果更早可见；不再创建临时 checkpoint |
| rolling mean、variance 和 covariance 的输出末位变化；溢出窗口得到 ±inf | P2    | 批和流都变；fixture 重建              |
| project-v3 的 `numerical_profile` 删除两个旧值                         | P2    | 显式写旧值的 project 被拒绝           |
| manifest v4，旧 checkpoint 不可恢复                                    | P3 起 | 升级后需要新建 lineage                |
| 聚合可选 changelog 输出                                                | P4    | 新选项，默认不变                      |
| `x86-64-v3` wheel                                                      | P6    | 见第 10 节                            |

## 10. 未决项与风险

1. P1 是证伪点。适配器方式的 `StreamSession` 达不到门槛时，后续阶段的前提需要重新评估。
   P1 未达到 ≤20 µs，处理方式见执行记录。
2. `StreamSession` 的准入范围从单源链路逐步扩大。多输入算子的 watermark 对齐语义在扩大时定义。
3. 新算术的误差界数值，以及是否保留与 DataFusion 滑动 AVG 的 NaN 粘性兼容，由 P2 规格确定。
4. 异步持久化的失败模型由 P3 规格确定（6.3）。
5. `x86-64-v3` wheel 的打包方式由 P6 规格确定。单一 v3 wheel 会放弃不支持 AVX2 的 x86 CPU；
   一个 wheel 内带两份 native 模块并在 import 时选择，则与“核心 wheel 只含一个 native 模块”的发布不变量冲突。
6. 分片后的规范输出顺序需要确定性有序合并，其成本在 P6 实测。
7. 决定 6（changelog 默认值）未单独答复。
8. 所有收益数字是估计；多数现状数字来自 10 月到 11 月初的产物（3.8）。

## 11. 与既有计划和 PR 的关系

- [2026-11-08 计划](2026-11-08-ta-lib-arroyo-performance.md)：
  - 它的 P2（decay 增量内核）和 P3（多窗口共享历史）落在本文的 P2；
  - 它的 D1（持久化与处理重叠）是本文 P3 的协议；
  - 它的 D2（changelog 输出）是本文 P4 的一部分；
  - 它的 C1（快照分块复用）并入本文 P4，C2（hopping pane）并入本文 P7；
  - 它的 C3（批量 MIN/MAX）只在长 run 有热点证据时进入内核库；
  - 它要求保持 `stable_v1`、manifest v3 和既有状态格式，这些约束被本文的决定 2 和 3 取代。
- PR #616（compiled stream regions）：region 的准入逻辑和测试可复用，但 region 必须同时拥有入口和出口。
  它自己的验收门槛没有通过，并入本文 P1，不单独合并。

## 12. 证据索引

- 小 batch 分层与 TA-Lib 参照：PR #616 的本地产物 `target/compiled-region-performance/`（未入库）。
- 回调桥接成本：`benchmarks/rolling/callback-sparse-818d69c.json`、`.codex/artifacts/stream-dispatch/README.md`。
- Tokio worker 数对照：`benchmarks/rolling/scheduler-32-to-1-818d69c.json`。
- Arroyo 对比：`.codex/artifacts/arroyo-comparison/REPORT.md`、`.codex/artifacts/arroyo-attribution/REPORT.md`。
- Checkpoint 成本：`.codex/artifacts/specs/window-operator-async-checkpoint-flush.md`、
  `.codex/artifacts/analysis/issue406-checkpoint-modes.md`。
- Join：`.codex/artifacts/analysis/join-single-thread-performance.md`、
  `.codex/artifacts/analysis/join-remaining-gaps-2026-10-17.md`、
  `.codex/artifacts/analysis/issue406-eight-workload-diagnosis.md`。
- Grouped SUM 与并行尝试：`benchmarks/stock-positions/engine-core-sweep-20261107.md`、`benchmarks/sum-state/`。
- Rolling 阶段计数：`benchmarks/runtime-fusion/p3-20261108/native-counters.json`。
