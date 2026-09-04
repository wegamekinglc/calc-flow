# Calc Flow SQL 与原生 DataFusion 性能提升实施计划

> **状态：** 已实施（2026-09-04）；证据门控项按下述 Go/No-Go 结论处理
>
> **关联工作：** DAL-184；基准与实现证据来自
> [calc-flow PR #236](https://github.com/wegamekinglc/calc-flow/pull/236)，证据快照为
> `6f8d92d0c3304f44a90c001ab98a8505b8b4a773`。
>
> **计划周期：** 8 周实施，另留第 9 周风险缓冲

## 目标和范围

本计划用于缩小 Calc Flow SQL 执行路径与原生 Apache DataFusion 在相同工作负载下的
性能差距。执行顺序遵循三个原则：

1. 先建立参数、输入和物理计划一致的公平基准，避免把配置差异归为框架开销。
2. 优先交付已经实测可达的并行化收益，再用同一二进制逐层定位剩余差距。
3. 分区保留、SQL rolling rewrite 和缓存仅在性能证据满足门槛时实施。

本计划不修改 DataFusion 的结果语义，不用 benchmark 专用捷径替代生产执行路径，也不把
rolling 单点状态更新的 `O(1)` 复杂度描述为整体查询的 `O(1)` 复杂度。整体查询仍可能包含
输入扫描、分组、排序和结果物化等 `O(N)` 工作。

## 当前证据和问题判断

以下数据来自 PR #236 快照上的诊断性公平测试：DataFusion 54、100 万行、64 个 symbol、
`batch_size=8192`，每个案例预热 1 次并采样 5 次，单位为 wall-clock 中位数毫秒。Raw
DataFusion 数据采用包含 context、注册、planning、执行和 collect 的完整包络。

| Workload          | Raw DF p32 | Raw DF p1 | Calc Flow p1 | Calc Flow p16 | Raw DF p16 |
| ----------------- | ----------: | --------: | -----------: | ------------: | ----------: |
| SMA(20)           |    57.61 ms | 226.55 ms |    236.95 ms |      81.64 ms |    65.33 ms |
| SMA(5) - SMA(20)  |    72.58 ms | 354.55 ms |    372.94 ms |      98.34 ms |    85.74 ms |

基线表明：

- Calc Flow 默认 `target_partitions=1`，而原生 DataFusion 的对照默认使用 p32。
- 对齐为 p1 后，Calc Flow 分别只慢 4.59% 和 5.19%；原始绝对差距约 94% 可由并行度
  不一致解释。
- Calc Flow 请求 p32 时，当前每分区至少 65,536 行的限制使 100 万行输入实际使用 p16。
- Calc Flow 从 p1 切换到 p16 后，SMA(20) 延迟降低 65.5%，双 SMA 延迟降低 73.6%，
  相当于 2.90 倍和 3.79 倍加速。
- 在相同 p16 下仍有约 25.0% 和 14.7% 的差距，需要通过同一二进制 A/B 才能归为
  Calc Flow 框架或执行开销。
- 已观察到的 session、表注册、planning 和 wrapper 固定开销约为 1--2 ms，不是百万行
  workload 的首要瓶颈。

这些结果是优先级依据，不是发布级性能承诺。P0 必须用 20 轮成对采样重新建立可复现基线。

## 成功指标

### 已实测可达的短期目标

| Workload，1M/64，effective p16 | Calc Flow wall time | 相对 Raw DF p16 |
| ------------------------------ | ------------------: | ---------------: |
| SMA(20)                        |          `<= 90 ms` |        `<= 1.30x` |
| SMA(5) - SMA(20)               |         `<= 110 ms` |        `<= 1.20x` |

### 中长期目标和通用护栏

- 相同二进制、输入、配置和物理计划下，Calc Flow SQL 达到 Raw DataFusion 的
  `<= 1.10x`。
- `N <= 10,000` 时 auto 模式保持 p1，延迟回归不超过 `max(5%, 0.05 ms)`。
- 发布候选的核心 benchmark 变异系数 `CV <= 10%`。
- 同配置优化的峰值 RSS 回归 `<= 15%`；并行路径相对 p1 的峰值 RSS `<= 1.5x`。
- schema、行数、key、顺序、null/NaN mask 完全一致；浮点结果满足
  `rtol=1e-10, atol=1e-10`。

`<= 1.10x` 是探索性工程目标，不是 P1 发布的前置条件。数量级提升只有在 warm-state
增量 rolling 避免全量 SQL 重算时才可能出现；普通 session、planning 或 wrapper 优化不承诺
10 倍收益。

## 里程碑和依赖关系

| 里程碑         | 周期      | 覆盖阶段        | 主要目标                                     |
| -------------- | --------- | --------------- | -------------------------------------------- |
| M0 基线可信    | 第 1 周   | P0              | 建立公平、可复现、可自动拒绝无效样本的基准   |
| M1 快速收益    | 第 2 周   | P1              | 交付显式并行，兑现已实测的 2.90--3.79 倍收益 |
| M2 自动并行    | 第 2--4 周 | P2、P3          | 完成 auto 策略、调参和默认值 Go/No-Go        |
| M3 差距归因    | 第 2--4 周 | P4              | 解释同配置下至少 90% 的剩余性能差距          |
| M4 深层优化    | 第 5--7 周 | P5、P6、P7      | 仅实施被 P4 证据门控通过的优化               |
| M5 稳定发布    | 第 8 周   | 集成、灰度、文档 | 完成回归、发布条件、监控和回滚准备           |
| 风险缓冲       | 第 9 周   | 未决项          | 处理性能波动、跨平台或语义问题，不扩展范围   |

依赖关系如下：

```text
P0 ──→ P1 ──→ P2 ──→ P3 ──→ auto 默认值决策 ──┐
 │                                               │
 └────────────→ P4 ──→ P5 / P6 / P7 ───────────┤
                                                 ↓
                                              M5 发布
```

P4 与 P2/P3 可以并行。P5、P6、P7 必须分别满足 P4 的证据门槛；auto 是否成为默认值
必须由 P3 的完整矩阵决定，而不能仅依据 100 万行单一 workload。

## 按周执行计划

| 周次   | 主线工作                               | 决策点或交付物                              |
| ------ | -------------------------------------- | ------------------------------------------- |
| 第 1 周 | P0 公平 benchmark                     | 冻结基线合同和首份 20 轮报告                |
| 第 2 周 | P1 显式并行；启动 P2 和 P4            | 验收 p16 快速收益；auto API 初审             |
| 第 3 周 | P2 auto 原型；P3 筛选矩阵；P4 归因    | 淘汰非 Pareto 参数；确认主要剩余成本         |
| 第 4 周 | P3 完整采样；完成 P4                  | auto 默认值 Go/No-Go；P5/P6/P7 分项 Go/No-Go |
| 第 5 周 | 开始通过门控的 P5/P6/P7               | 每项独立 feature flag 和 RED/GREEN 验证      |
| 第 6 周 | 深层优化及逐项 benchmark              | 保留有统计显著收益的改动，停止低收益方向     |
| 第 7 周 | 深层优化收口及组合回归                | 锁定发布候选和 residual-gap 清单             |
| 第 8 周 | 集成、灰度、CI、文档和回滚演练        | M5 发布 Go/No-Go                            |
| 第 9 周 | 仅处理已知风险和阻断项                | 缓冲结束；未解决项进入后续 backlog          |

## P0：公平 benchmark 合同

- **周期：** 第 1 周
- **依赖：** 无
- **主责：** 性能负责人
- **协作：** 核心引擎负责人、测试负责人、评审负责人

### 负责人产出

- 新增独立的公平对比入口，提供 `serial-control`、`matched-adaptive` 和
  `p32-saturation` 三个 profile。
- 固定 DataFusion 版本、release profile、allocator、输入 Arrow batch 边界和
  `batch_size=8192`。
- 记录 requested/effective partitions、分区限制原因、plan hash、环境、原始样本和
  正确性结果。
- 每个正式案例预热 1 次，执行 20 轮 AB/BA 成对采样；median 为主指标，同时记录
  p25、p75、MAD、CV 和 paired ratio。
- 建立 benchmark JSON schema 和自动可比性 gate。

### 验证和退出条件

- Calc Flow 与 Raw DataFusion 的 target/effective partitions、batch size、输入逻辑分区
  完全一致。
- physical plan 结构一致，双 SMA 位于同一个 `BoundedWindowAggExec`，公平测试强制
  关闭 rolling rewrite。
- 正确性满足全局成功指标，核心 workload `CV <= 10%`。
- 两次独立执行的中位数差异 `<= 10%`。
- 任一可比性字段不一致时非零退出，且不得输出 speedup 结论。

### Go/No-Go 和回滚

- **Go：** 全部可比性、正确性和稳定性条件通过，基线 JSON 可由固定命令复现。
- **No-Go：** 计划形态或 effective partitions 不一致，或样本波动无法稳定到阈值内。
- **回滚：** 这是新增 benchmark，不改变运行时；失败时保留现有 informational benchmark，
  新报告不得作为回归门槛。

## P1：显式并行快速收益

- **周期：** 第 2 周
- **依赖：** P0 通过
- **主责：** 核心引擎负责人
- **协作：** 性能负责人、测试负责人、评审负责人

### 负责人产出

- benchmark 和明确的大数据 workload 可以显式设置 `target_partitions`。
- 执行结果或性能记录暴露 configured/effective partitions 及限制原因。
- 保留 fixed p1 路径和一键回退配置。
- 生成 100 万行、64 symbols、matched p16 的正式报告。

### 验证和退出条件

- SMA(20) `<= 90 ms` 且相对 Raw DF p16 `<= 1.30x`。
- SMA(5) - SMA(20) `<= 110 ms` 且相对 Raw DF p16 `<= 1.20x`。
- `CV <= 10%`，正确性和确定性通过。
- 同配置峰值 RSS 回归 `<= 15%`，p16 相对 p1 峰值 RSS `<= 1.5x`。
- p1 fallback 的结果和配置语义保持不变。

### Go/No-Go 和回滚

- **Go：** 两个核心 workload 同时通过性能、内存和正确性条件。
- **No-Go：** 只有单一 workload 获益，或收益依赖不可复现的 plan/机器差异。
- **回滚：** 设置 `target_partitions=1`，并保留 telemetry 以定位回退原因。

## P2：保守的 auto parallelism

- **周期：** 第 2--3 周
- **依赖：** P0；复用 P1 telemetry
- **主责：** 核心引擎负责人
- **协作：** API 负责人、性能负责人、测试负责人

### 负责人产出

- 增加 opt-in 的 `fixed | auto` 模式，以及 `max_partitions`、
  `min_rows_per_partition`、`small_rows_threshold` 配置。
- 采用可解释的保守算法：

  ```text
  requested = min(available_parallelism, configured_max_partitions)

  if rows < small_rows_threshold or active_entities < 2:
      effective = 1
  else:
      work_cap = ceil(rows / min_rows_per_partition)
      effective = min(requested, work_cap, active_entities)
  ```

- 记录决策输入、统计来源、effective partitions 和 fallback 原因。
- 若配置进入公共 Project schema 或 Python API，提供兼容性设计和迁移说明。

### 验证和退出条件

- `N <= 10,000` 和单实体 workload 保持 p1。
- 小数据回归不超过 `max(5%, 0.05 ms)`。
- 缺少或不可信的实体统计时安全降级，不为选择分区数额外扫描全表。
- 相同输入和配置的分区决策稳定、确定且可解释。
- fixed p1 与显式 fixed pN 行为保持兼容。

### Go/No-Go 和回滚

- **Go：** auto 作为 opt-in 交付，所有降级和观测字段完整。
- **No-Go：** 决策依赖额外全表扫描、不稳定统计，或使小数据路径明显回归。
- **回滚：** 关闭 auto feature flag，使用 fixed p1；本阶段不改变默认模式。

## P3：adaptive cap 和 batch size 调优

- **周期：** 第 3--4 周
- **依赖：** P0、P2 原型
- **主责：** 性能负责人
- **协作：** 核心引擎负责人、测试负责人、评审负责人

### 负责人产出

- 执行以下筛选矩阵，并发布 Pareto 前沿：

  | 维度           | 取值                              |
  | -------------- | --------------------------------- |
  | rows           | 100k、1m、2.1m                    |
  | active entities | 1、4、16、64                      |
  | partitions     | 1、2、4、8、16、32                |
  | batch size     | 4096、8192、16384、32768          |
  | workload       | SMA(20)、SMA(5) - SMA(20)         |

- 首轮每个点采样 5 次，只对 Pareto 候选执行完整 20 轮成对采样。
- 记录 wall time、CPU、峰值 RSS、spill、空分区数、每分区行数、偏斜度以及
  Repartition/Sort/Window 阶段时间。
- 给出默认 batch size、每分区最小行数、最大并行度和 auto 默认值的 Go/No-Go 结论。

### 验证和退出条件

- 1M/64 持续满足 P1 的性能、正确性和内存门槛。
- `N <= 10,000` 满足小数据回归限制。
- 候选配置 `CV <= 10%`，没有未解释的 spill 或异常空分区。
- 候选在连续两个 nightly 基线上稳定。
- auto feature flag 和 fixed p1 fallback 均通过验证。

### Go/No-Go 和回滚

- **Go：** 只有以上全部条件通过，才允许单独评审 auto 成为默认值。
- **No-Go：** 未达门槛时 auto 继续保持 opt-in，不因 1M 单一 workload 获益而切默认值。
- **回滚：** 保留原 `batch_size=8192`、fixed p1 默认和上一组稳定阈值。

## P4：same-binary 性能归因

- **周期：** 第 2--4 周，与 P2/P3 并行
- **依赖：** P0
- **主责：** 性能负责人
- **协作：** 核心引擎负责人、测试负责人

### 负责人产出

- 在同一个 Rust binary、allocator、Tokio runtime、SessionConfig、输入和 physical plan
  中逐层执行 A/B：
  1. 直接执行同一个 `ExecutionPlan`；
  2. 加入 SessionContext factory；
  3. 加入 MemTable/Batch adapter；
  4. 加入 SQL parse、logical optimization 和 physical planning；
  5. 加入 run/session/transaction envelope；
  6. 分别加入 rewrite audit、metrics traversal、plan string 和 output wrapper。
- 记录 `runtime_acquire`、`session_state_create`、`input_adapter`、`table_register`、
  `sql_parse`、`logical_optimize`、`physical_plan`、`execution_to_first_batch`、
  `execution_remaining`、`collect_or_coalesce`、`output_arrow_wrap`、`audit`、
  `metrics_traversal`、`physical_plan_string`、`batch_envelope` 和 `run_result`。
- 对 P5、P6、P7 分别提交证据和 Go/No-Go 建议。

### 验证和退出条件

- 解释 Calc Flow 与 same-binary Raw baseline 差值的至少 90%。
- 每个主要成本项都能通过独立开关复核。
- A/B 的 normalized physical-plan hash 一致；计划变化的样本不归类为框架开销。
- 明确剩余瓶颈属于 execution、materialization、WindowAgg 还是固定包络。

### 深层优化门槛

- **P5 Go：** 重复 repartition/sort、collect/coalesce 或跨 DAG 物化占剩余差距
  `>= 20%`。
- **P6 Go：** WindowAgg 占 execution wall time `>= 50%`，且支持形态可以证明
  语义等价。
- **P7 Go：** session/planning/wrapper 占总耗时 `>= 10%`，或重复小查询中稳定超过
  1 ms。
- 未满足对应门槛的方向不进入第 5--7 周实施，仅保留调查记录。

## P5：保留分区并减少物化

- **周期：** 第 5--7 周
- **依赖：** P4 的 P5 Go 结论
- **主责：** 核心引擎负责人
- **协作：** 性能负责人、测试负责人、评审负责人

### 负责人产出

- 在 SQL/DAG 节点间保留 partitioned relation，并传递 partitioning/ordering metadata。
- 消除不必要的 flatten、collect、merge、repartition 和重复 sort。
- 仅在 Python/API 边界、全局排序或不支持分区输入的算子前 merge。
- 使用 Arrow RecordBatch 零拷贝转交，并提供 `preserve_partitioning` feature flag。

### 验证和退出条件

- P4 识别的重复物化或 shuffle 从计划和 phase metrics 中消失。
- 对应 phase wall time 至少降低 30%，且端到端收益具有统计显著性。
- 峰值 RSS 增长 `<= 15%`。
- 输出顺序、重复 timestamp 的 tie-breaker、backpressure、取消和错误恢复测试通过。

### Go/No-Go 和回滚

- **Go：** 独立启用 P5 时同时满足 phase、端到端、内存和语义门槛。
- **No-Go：** 只移动成本、增加内存，或收益小于采样噪声。
- **回滚：** 关闭 `preserve_partitioning`，恢复既有 materialization 边界。

## P6：SQL rolling rewrite

- **周期：** 第 5--7 周
- **依赖：** P4 的 P6 Go 结论和语义矩阵评审通过
- **主责：** 核心引擎负责人
- **协作：** API 负责人、性能负责人、测试负责人、评审负责人

### 负责人产出

- 首批只重写可证明等价的 `AVG(numeric_expr) ROWS BETWEEN k PRECEDING AND
  CURRENT ROW`。
- 支持按实体分区、按标准事件序排序的单 SMA 和同一 WindowAgg 内双 SMA。
- 对 `FILTER`、`DISTINCT`、不支持 cast 或不确定排序等形态 fail closed，并记录结构化
  fallback 原因。
- 将 full-history rewrite 和 warm-state append 分开测量和报告。

### 验证和退出条件

- null、全 null 窗口、NaN/Inf、`W=1`、`W>N`、重复 timestamp、乱序、多实体、
  空分区、alias、输出类型和 partial-window 语义与 DataFusion 完全一致。
- 不支持形态 100% fallback，不允许静默改变结果。
- rewrite on/off 使用同一正确性数据集并独立报告。
- 结果明确区分单点状态更新复杂度和整体查询复杂度。

### Go/No-Go 和回滚

- **Go：** 语义矩阵全部通过，目标 workload 的收益超过噪声且无内存护栏回归。
- **No-Go：** 任一边界语义无法证明，或收益来自输入/计划不一致。
- **回滚：** 关闭 SQL rolling rewrite；所有输入继续使用 DataFusion WindowAgg fallback。

## P7：Session、plan 和 input reuse

- **周期：** 第 5--7 周，默认最低优先级
- **依赖：** P4 的 P7 Go 结论
- **主责：** 核心引擎负责人
- **协作：** 性能负责人、测试负责人、评审负责人

### 负责人产出

- 仅在证据支持时提供可选的 SessionState/SessionContext、SQL parse 或 plan reuse。
- 定义包含 schema、配置、UDF、provider 和 SQL fingerprint 的完整 cache key。
- 实现失效、容量、并发隔离、hit/miss/eviction telemetry 和禁用开关。
- 将 plan string 与高成本 metrics traversal 改为可按需收集，前提是不破坏诊断合同。

### 验证和退出条件

- 目标重复查询的端到端延迟至少改善 5%，否则停止该方向。
- 不产生 stale registration、跨 run/tenant 污染或配置/UDF 失效遗漏。
- 峰值 RSS 增长 `<= 15%`，缓存容量和 eviction 可观测。
- 关闭缓存时恢复现有 run-scoped 隔离语义。

### Go/No-Go 和回滚

- **Go：** P4 证据和至少 5% 的端到端收益同时满足。
- **No-Go：** 百万行收益仍接近已知 1--2 ms 固定开销，或隔离风险高于收益。
- **回滚：** 关闭复用开关，清空缓存，恢复每次 run 的既有生命周期。

## M5：第 8 周集成、灰度和发布

- **依赖：** P2/P3 完成；所有被选择的 P5/P6/P7 已独立验收
- **主责：** 发布负责人
- **协作：** 性能负责人、核心引擎负责人、测试负责人、评审负责人、文档负责人

### 负责人产出

- 端到端正确性、性能和资源回归报告。
- feature flag、监控字段、5% -> 25% -> 100% canary 和回滚手册。
- 配置/API/行为文档、changelog 和 residual-gap backlog。
- 一份明确区分实测收益、探索性收益与不可比案例的最终报告。

### 发布退出条件

- P1 的 p16 性能目标持续满足。
- auto 仅在 P3 的默认值门槛全部满足后才可默认开启。
- 小 N、CV、正确性、确定性和 RSS 护栏全部满足。
- required CI 通过，nightly 没有未解释回归，回滚演练通过。
- 高风险优化具备 feature flag 或明确版本回退路径。

未达到长期 `<= 1.10x` 不阻塞已经验证的 P1/P2 并行收益发布，但必须保留剩余差距
任务，且不得宣称 Calc Flow 与原生 DataFusion 已经等速。

## 2026-09-04 实施收口

本计划的生产代码、证据合同和自动化已实施。性能数字仍由绑定具体 SHA、机器和依赖指纹的
报告决定；仓库不提交本机诊断样本，也不把 hosted runner 的单次结果写成发布承诺。

| 阶段 | 实施状态 | 当前决策 |
| ---- | -------- | -------- |
| P0   | 同二进制 AB/BA benchmark、严格 JSON schema 和 fail-closed verifier 已落地 | Go；正式结论仍要求 20 对样本和独立重复 |
| P1   | 显式 fixed pN、effective partition/原因/资源 telemetry 和 p1 回退已落地 | Go；nightly 持续执行绝对延迟、ratio 和 RSS 门槛 |
| P2   | `fixed \| auto`、三个调节参数、可信实体统计和无扫描降级已落地 | Go（仅 opt-in）；默认仍为 fixed p1 |
| P3   | 完整筛选矩阵、Pareto 选择和候选双 20 对复测已进入 weekly | Auto 默认值 No-Go，直到完整矩阵和连续基线通过 |
| P4   | phase、算子级 compute/count 和同二进制归因器已落地 | 少于 90% 可解释差距时 fail closed |
| P5   | 未引入新的 partitioned `Batch` 公共语义或 DAG 物化路径 | No-Go；只有出现 Calc Flow 独有重复 shuffle/merge 且达到 20% 才重开 |
| P6   | 受限 AVG/ROWS rewrite、独立关闭开关、fallback audit 和边界语义矩阵已落地 | Go；整体查询仍为 `O(N)`，不宣称全历史 `O(1)` |
| P7   | run 内 SessionContext 复用沿用现有隔离；诊断收集可关闭 | 跨 run cache No-Go；未证明 5% 收益，不扩大污染面 |
| M5   | PR smoke、nightly、weekly、配置/监控/灰度/回滚文档已落地 | 灰度推进继续受两次 nightly 和全局护栏约束 |

运行命令、字段解释、canary 和回滚步骤见
[`docs/sql-datafusion-performance.md`](../../sql-datafusion-performance.md)。

## RACI

角色说明：A 为最终负责，R 为执行负责，C 为协作或评审，I 为知会。具体姓名由项目负责人
在每个里程碑启动时填写，避免计划文档与排期系统的人员变更发生冲突。

| 工作                  | 性能负责人 | 核心引擎负责人 | API 负责人 | 测试负责人 | 评审负责人 | 发布负责人 | 文档负责人 |
| --------------------- | ---------- | -------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| P0 公平基准           | A/R        | C              | I          | C          | C          | I          | I          |
| P1 显式并行           | C          | A/R            | I          | C          | C          | I          | I          |
| P2 auto 策略          | C          | A/R            | C          | C          | C          | I          | I          |
| P3 参数调优           | A/R        | C              | I          | C          | C          | I          | I          |
| P4 same-binary 归因   | A/R        | C              | I          | C          | C          | I          | I          |
| P5 分区保留           | C          | A/R            | I          | C          | C          | I          | I          |
| P6 rolling rewrite    | C          | A/R            | C          | C          | C          | I          | I          |
| P7 session/plan reuse | C          | A/R            | C          | C          | C          | I          | I          |
| M5 灰度和发布         | C          | R              | C          | R          | C          | A          | R          |

## CI、nightly 和 weekly 安排

| 频率       | 范围                                                               | 样本与门槛                                                        | 处置                                            |
| ---------- | ------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------- |
| 每个 PR    | 正确性、可比性 contract、核心 workload smoke                       | 3--5 个样本；性能只做异常预警，不用 hosted runner 的绝对值硬 gate | contract/正确性失败阻断；性能异常交性能负责人复核 |
| 每晚       | 1M/64 的 SMA、双 SMA，p1/p16 matched profile                       | 1 次预热 + 20 轮 AB/BA；`CV <= 10%`；记录 CPU/RSS/plan hash       | 回归超过 10% 且 paired CI 不跨 1 时进入复核      |
| 每周       | P3 全矩阵、p32 saturation、已启用 feature 的 on/off 组合           | 保留完整原始样本、环境和资源指标                                  | 回归超过 15% 阻断发布                            |
| 发布候选   | 连续两个 nightly、正确性矩阵、回滚演练、required CI                | 全局成功指标和阶段退出条件                                        | 任一硬门槛失败则 No-Go                           |

CI 输出至少保留下列字段：

- Git SHA、PR、DataFusion/Arrow 版本、编译 profile、allocator、CPU 和 OS；
- rows、active entities、window、SQL shape、输入 partition/batch 边界；
- configured/effective partitions、batch size、限制原因和 normalized plan hash；
- warm-up、采样顺序、原始样本、median、p25/p75、MAD、CV 和 paired interval；
- 正确性摘要、wall time、CPU、峰值 RSS、spill 和 phase timings；
- rewrite/cache/partition-preserving 开关以及 fallback 原因。

## 风险和控制

| 风险                         | 影响                                         | 控制措施                                                       |
| ---------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| 不公平配置造成虚假差距       | 错误安排优化优先级                           | P0 硬 gate target/effective partitions、batch 和 plan hash      |
| 托管 CI 噪声                 | 虚假回归或掩盖真实回归                       | PR 仅 smoke；nightly 成对采样；记录原始样本和置信区间           |
| symbol 分布偏斜              | 分区数增加但 wall time 不降                   | 记录每分区行数和 skew；effective partitions 不超过 active entities |
| 并行化增加内存或 spill       | 延迟抖动、OOM                                | RSS/spill 门槛、保守 cap、fixed p1 回退                         |
| auto 统计成本                | 小查询变慢                                   | 不允许额外全表扫描；缺失统计时安全降级                          |
| physical plan 漂移           | 把算法变化误判为框架开销                     | normalized plan hash 不一致即拒绝 A/B                           |
| 分区保留改变输出顺序         | 结果不确定或下游不兼容                       | ordering metadata、tie-breaker、显式 merge 边界和 feature flag  |
| rolling rewrite 语义偏差     | null、排序或边界结果错误                     | 严格形态 allowlist、完整语义矩阵、fail-closed fallback          |
| cache 污染                   | 跨 run/tenant 泄漏或 stale result            | 完整 cache key、容量/失效/隔离验证、默认关闭                    |
| 组合优化无法归因             | 无法判断某项是否值得保留                     | P5/P6/P7 独立开关、逐项合并和 on/off benchmark                  |

## 收益边界和停止条件

- **可以承诺的近期方向：** 显式 matched p16 已在诊断样本中达到 2.90--3.79 倍加速；
  P0 重复验证后可作为 P1 的交付目标。
- **需要探索的方向：** adaptive cap、batch size、分区保留和同二进制框架开销，只有满足
  各阶段门槛后才能给出收益数字。
- **算法级机会：** warm-state append rolling 可以把每个新增点的状态更新降为 `O(1)`，
  但前提是保留状态并避免全量输入重算；这与公平的 full-history DataFusion 对比必须分开。
- **低优先级方向：** session、planning 和 wrapper 当前仅占约 1--2 ms。P7 若无法提供
  至少 5% 的目标场景端到端收益，应停止实施。
- **总体停止条件：** 相同二进制下 Calc Flow 达到 Raw DataFusion 的 `<= 1.10x`，或
  剩余差距已被上游 DataFusion 执行、硬件噪声或不应破坏的隔离/语义合同解释。

## 建议的交付拆分

为使评审、回滚和收益归因保持独立，建议至少拆成以下工作项：

1. P0：公平 benchmark 合同、JSON schema 和可比性 gate；
2. P1：effective partitions telemetry 和显式并行；
3. P2/P3：auto partitioning 配置、策略和参数校准；
4. P4：same-binary phase profiler 与归因报告；
5. P5：partition-preserving SQL DAG；
6. P6：受限 AVG/ROWS rolling rewrite；
7. P7：仅在门控通过时实现 session/plan/input reuse；
8. M5：组合回归、灰度、文档和回滚演练。

每个实现 PR 应只覆盖一个可独立关闭的优化方向，并在正文中列出对应的阶段退出条件、
实际采样结果和当前 CI 状态。P0/P1 之外的工作不应以预估收益替代证据门槛。
