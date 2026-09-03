# TA-Lib Streaming 启发的 Rolling 与 DataFusion 引擎升级计划

> **状态：** 实施中；按用户要求，P0–P5 以分阶段提交集中在 PR #236
>
> **Calc Flow 基线：** `main@25dd973bc1575bf0ecc4210cca76664e924acebe`
>
> **性能证据提交：** `0052da76e114277d533b09c15bb64be4bc18af28`
> （clean worktree；该提交只新增 benchmark、测试和文档，不修改引擎）
>
> **TA-Lib 研究快照：**
> [`main@df0c6bebbf2a39d49d06193206e2af608cb96624`](https://github.com/TA-Lib/ta-lib/tree/df0c6bebbf2a39d49d06193206e2af608cb96624)，
> 2026-09-04 获取；其 streaming API 尚未发布，官方页面标注计划进入
> v0.8.x
>
> **性质：** 本文是点时架构与执行计划，不改变当前公共 API、rolling 语义、
> checkpoint 契约或 DataFusion 54 的唯一表引擎地位

## 1. 结论

Calc Flow 不应把 TA-Lib 直接替换成默认 rolling 后端。TA-Lib 的突出优势是
窄而专用的数值内核、从批量 IR 自动派生 streaming 状态机、固定参数的有界
状态、无分配的逐 bar 更新，以及批量填充接口对调用开销的摊薄。Calc Flow
同时必须承担 Arrow 类型和 null 语义、多实体排序、event-time finality、迟到
数据、checkpoint/recovery、确定性输出和 DataFusion 集成；这些都不是 TA-Lib
handle 的职责。

建议把当前实现拆成两个明确层次：

1. **finality/order 层**继续负责 watermark、迟到策略、canonical row identity、
   checkpoint 原子性和最终输出顺序。
2. **Arrow-native rolling kernel 层**只接收已经确定顺序与提交边界的列式数据，
   用同一个不可变 `RollingKernelPlan` 驱动 batch、stream 和 DataFusion physical
   execution。

第一优先级不是扩充指标目录，而是消除当前 batch rolling 热路径中的逐单元格
`ScalarValue` 转换、无条件全量排序、`BTreeMap` 分组和行对象重建。当前结果
已经表明：增量算法本身存在，但其外层行式搬运使大数据量下的 native rolling
反而慢于 DataFusion SQL window。

## 2. PR 内的两个可执行性能例子

本计划配套的
[`rolling_indicator_comparison.py`](../../../benchmarks/rolling_indicator_comparison.py)
通过一个统一 harness 提供两个可执行例子，使用同一份确定性 64-symbol 行情、
相同计时边界和相同数据量矩阵：

- **例子 A：** 单指标 `SMA(20)`；
- **例子 B：** 复合指标 `SMA(5) - SMA(20)`。

两者都比较四条路径：Calc Flow 最新 native incremental rolling、早期
DataFusion SQL window、Finance-Python 0.9.10 公共指标算子，以及
TA-Lib Python 0.7.1 对 bundled C library 的 `SMA` 调用。安装、完整命令和
计时边界见 [`benchmarks/README.md`](../../../benchmarks/README.md)。
完整 provenance 与 20 轮原始样本保存在
[`SMA(20)` 报告](../../../benchmarks/rolling/rolling-mean-0052da7.json)和
[`SMA(5) - SMA(20)` 报告](../../../benchmarks/rolling/dual-sma-spread-0052da7.json)。

### 2.1 方法与语义边界

- 输入构造、plan 编译、Finance-Python worker 启动、warm-up 和正确性归一化
  均在计时区间外；每个计时调用完整执行并物化结果。
- 每个规模执行 20 轮，四种方法轮换起始顺序；小规模通过多次执行摊薄计时器
  噪声，垃圾回收发生在计时区间外。
- native、SQL 和 Finance-Python 都按 `min_periods=1` 输出部分窗口，并与独立
  direct-window oracle 以 `rtol=1e-10`、`atol=1e-10` 复核。
- TA-Lib 0.7.1 只在完整窗口后输出；每个 symbol 的前 19 个结果是 `NaN`，因此
  使用单独的 full-window oracle。表中的 `TA-Lib valid rows` 明确记录有效值数。
- TA-Lib 接受单序列连续 `double` 数组；计时包含逐 symbol 切片和拷贝、一次或
  两次 `SMA` 调用，以及恢复 timestamp-major 顺序，但不包含 Arrow envelope、
  watermark 或 checkpoint 成本。
- Finance-Python 必须运行在隔离的 Python 3.9 worker，其他路径运行在 Calc Flow
  的 Python 3.13 环境。因此以下结果是跨运行时诊断，不是回归门或生产容量承诺。

### 2.2 例子 A：`SMA(20)`

|      Rows | Native incremental | SQL window | Finance-Python |   TA-Lib | TA-Lib valid rows | SQL/native | Finance/native | TA-Lib/native |
| --------: | -----------------: | ---------: | -------------: | -------: | ----------------: | ---------: | -------------: | ------------: |
|        10 |           0.262 ms |   0.896 ms |       0.473 ms | 0.007 ms |                 0 |     3.417x |         1.805x |        0.027x |
|       100 |           0.349 ms |   1.063 ms |       1.435 ms | 0.046 ms |                 0 |     3.051x |         4.116x |        0.132x |
|     1,000 |           0.883 ms |   1.219 ms |       1.832 ms | 0.047 ms |                 0 |     1.382x |         2.075x |        0.053x |
|    10,000 |           6.897 ms |   4.107 ms |       6.663 ms | 0.075 ms |             8,784 |     0.596x |         0.966x |        0.011x |
|   100,000 |          80.819 ms |  24.623 ms |      57.937 ms | 0.348 ms |            98,784 |     0.305x |         0.717x |        0.004x |
| 1,000,000 |         899.012 ms | 248.236 ms |     574.629 ms | 7.736 ms |           998,784 |     0.276x |         0.639x |        0.009x |

### 2.3 例子 B：`SMA(5) - SMA(20)`

|      Rows | Native incremental | SQL window | Finance-Python |   TA-Lib | TA-Lib valid rows | SQL/native | Finance/native | TA-Lib/native |
| --------: | -----------------: | ---------: | -------------: | -------: | ----------------: | ---------: | -------------: | ------------: |
|        10 |           0.510 ms |   1.191 ms |       0.787 ms | 0.012 ms |                 0 |     2.333x |         1.542x |        0.024x |
|       100 |           0.589 ms |   1.347 ms |       2.669 ms | 0.080 ms |                 0 |     2.286x |         4.530x |        0.135x |
|     1,000 |           1.167 ms |   1.713 ms |       3.515 ms | 0.085 ms |                 0 |     1.467x |         3.011x |        0.073x |
|    10,000 |           7.729 ms |   5.951 ms |      11.555 ms | 0.125 ms |             8,784 |     0.770x |         1.495x |        0.016x |
|   100,000 |          89.814 ms |  40.128 ms |      94.523 ms | 0.514 ms |            98,784 |     0.447x |         1.052x |        0.006x |
| 1,000,000 |         998.158 ms | 401.156 ms |     946.789 ms | 9.309 ms |           998,784 |     0.402x |         0.949x |        0.009x |

### 2.4 结果解释

结果来自 WSL2 主机上的本地诊断。两个报告均记录 clean
`calc-flow@0052da76e114277d533b09c15bb64be4bc18af28`、依赖版本、workload 和每轮
原始样本；该提交相对基线只增加 benchmark、测试和文档，未改变引擎。由于主机
是虚拟化环境且没有 paired baseline/candidate confidence interval，这些结果仍是
诊断证据，不能当成回归门或生产容量承诺。

可以确认的结构性信号如下：

- 10 至 1,000 行时，native incremental 的固定开销低于 SQL window；
- 10,000 行开始，SQL window 更快；在 1,000,000 行时，单 SMA 的 SQL 用时约为
  native 的 27.6%，复合指标约为 40.2%；
- 复合指标没有使 native 时间翻倍，说明现有 compatible rolling group sharing
  已经有效，但行式转换、排序、分组和重建仍然主导总成本；
- TA-Lib 的量级差距证明紧凑列式数值内核仍有很大优化空间，但不证明把 TA-Lib
  直接嵌入 Calc Flow 就能保留该差距；
- 10、100 和 1,000 行下 TA-Lib 的有效结果数为 0，这些点只反映全 warm-up
  调用成本，不能与另外三条 partial-window 路径比较有效输出吞吐。

## 3. TA-Lib 最新 streaming 实现的优势

### 3.1 生命周期与调用边界足够小

最新设计为每个指标生成 `Open`、`Update`、`Peek`、`Close`、`OpenAndFill`、
`UpdateAndFill`、`Value`、`Clone` 和 `OutRange`：

- `Open` 用历史数据一次性完成 warm-up，并捕获可继续计算的 live state；
- `Update` 每次提交一个 closed bar，handle 在 open 时已完成容量分配；
- `Peek` 用同一 transition 的 shadow frame 计算 forming bar，不提交状态；
- `OpenAndFill` 避免“先 batch 回填、再 open 预热”的两次历史扫描；
- `UpdateAndFill` 在一次调用中提交多个 bar，摊薄 FFI 和参数检查成本；
- `Value` 和 `OutRange` 让调用方不必重算就能观察最后值与输出范围。

这个 API 把稳定的 plan 参数、持久状态和一次调用的输入清晰分离。对 Calc Flow
最有价值的不是 C handle 形式，而是 `open_and_fill` 与 `update_and_fill` 共用
同一 transition 的边界设计。

### 3.2 streaming 状态不是手写的第二套算法

TA-Lib 的 `ta_codegen` 从 batch IR 分析 steady loop、loop-carried scalar、
trailing read、ring buffer 和函数组合，再派生五种 `StreamPlan`：`Loop`、
`DualMode`、`Composed`、`Dispatch`、`PeriodBank`。标记为 streamable 的函数一旦
无法分析或生成 transition，代码生成会失败，而不是静默回退到慢路径。

这种做法解决了两类长期维护风险：

- batch 与 stream 各写一套算法后发生语义漂移；
- 新增或重写指标后，stream 支持悄悄缺失但 CI 仍然通过。

Calc Flow 已经由同一 `CompiledRollingSpec` 驱动 batch/stream kernel，并具备
更强的 state/recovery 契约；下一步应把编译结果提升为更窄的、可解释的
`RollingKernelPlan`，而不是再引入一套独立 TA 指标运行时。

### 3.3 热路径状态紧凑且分配时机可控

生成的 SMA stream 保留固定参数、running total、ring position 和一个周期长度
的 `double` ring；逐 bar transition 只做加、减、除和 ring 旋转。分配在 open
发生，update 不分配。相比之下，Calc Flow 当前 batch rolling 会：

1. 将每个 Arrow 单元格转换成 `ScalarValue`；
2. 为每行构造包含全部输入列的 `BufferedRow`；
3. 无条件排序，再用 `BTreeMap` 建立 entity 到 row-index 的分组；
4. clone retained rows，并把结果重新编码回 Arrow arrays。

因此 Calc Flow 的首要优化对象是数据布局与调度，不是把 running sum 再写一遍。

### 3.4 组合、特化与数值稳定性进入生成计划

TA-Lib 的实现显示了三个值得吸收的方向：

- `Composed` plan 让多输入、多输出指标复用 public sub-stream，再以每 bar map
  组合结果；
- BBANDS 的 SMA 快路径把 moving average 与 variance 融合成一次窗口扫描，
  避免中间数组和重复读取；
- VAR 使用 shifted sums，并在相消风险、离群值退出窗口或周期上限触发时重新
  锚定和重建窗口，数值稳定策略和状态 transition 一起生成。

Calc Flow 当前已有同 input/frame 的 `CompiledWindowGroup` 共享、West/Welford
增删状态和 min/max monotonic queue。升级重点应是把共享扩展到输出 liveness、
中间列消除和多输出融合，同时把数值策略版本化，不能直接改变既有结果。

### 3.5 验证强调非空覆盖与独立路径

TA-Lib 对 batch/stream bit identity、不同 warm-up 前缀、peek 不提交、clone
独立性、array-fill、输出范围、错误路径和 sanitizer 都设有门；benchmark 工具
还会在解析不到足够函数时 fail closed，避免“测了零个函数却成功”。这类
non-vacuity 检查应进入 Calc Flow 的 kernel census、benchmark 和 recovery
matrix。

## 4. 不能直接照搬的边界

### 4.1 streaming API 还不是稳定发布面

本文研究的是固定的 `df0c6beb...` 开发快照。TA-Lib 官方页面明确标记该能力
尚未发布并计划进入 v0.8.x；当前性能例子使用的是 ta-lib-python 0.7.1 的 batch
`SMA`，不是最新 streaming handle。任何对 streaming API 的集成实验都必须固定
源码 SHA，并与发布版 benchmark 分开报告。

### 4.2 handle 不可序列化

TA-Lib 明确规定 handle 与库版本绑定、不可跨进程序列化，推荐保留历史后重新
open。Calc Flow 的 managed continuous runtime 以 checkpoint manifest v3 和
`StateBackend` 为 durable 真相，不能把不透明 C handle 写入 checkpoint，也不
能依赖全历史无限回放。

### 4.3 类型、null 与多实体语义不同

TA-Lib 的核心输入是连续 `double` 数组，单序列调用不负责 Arrow null bitmap、
整数精确性、dictionary/string entity、event-time、sequence tie-break、迟到数据
或多 partition 合并。Calc Flow 必须保留现有 exact integer、null、NaN、正负
无穷、row/duration window 和 deterministic ordering 契约。

### 4.4 `UpdateAndFill` 的部分提交不适合 Calc Flow envelope

TA-Lib 遇到 batch 中的 invalid bar 时允许前缀已提交、错误 bar 计数、后缀未
处理。Calc Flow operator 当前在完整 kernel 成功后才安装 touched histories，
与 batch/stream envelope 的原子失败语义一致。升级后的 `update_and_fill` 必须
先验证可提交范围，或写入 scratch state 后一次交换，不能复制部分提交行为。

### 4.5 复杂度不能统一宣传为 O(1)

TA-Lib 设计文档本身说明部分 window recomputer 为 O(period)，某些 extrema 在
边界触发 O(period) rescan，并非所有 stream 都是摊销 O(1)。Calc Flow 编译计划
应为每个 kernel 标记 `constant`、`amortized_constant`、`periodic_rebuild` 或
`linear_in_window`，并把该事实暴露给 `Program.explain()` 和性能门。

### 4.6 同源 parity 不能替代独立 oracle

从同一个 IR 生成 batch 与 stream 能显著降低漂移，但也可能让共同错误在 parity
测试中保持一致。Calc Flow 必须继续使用 direct-window、高精度或已冻结 fixture
作为独立 oracle，并额外验证 segment、checkpoint/restore 与不同 micro-batch
切分下的一致性。

## 5. 目标架构

```text
Symbolic Program / RollingOperator / supported SQL window
                         |
                         v
              immutable RollingKernelPlan
                         |
          +--------------+---------------+
          |                              |
          v                              v
 finality + ordering layer       Arrow-native series kernel
 watermark / late rows           typed arrays / dense state
 canonical identity              fused outputs / no hot alloc
 checkpoint transaction          open/update/preview/fill
          |                              |
          +--------------+---------------+
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
 batch open_and_fill  stream update_fill  CalcFlowRollingExec
```

公共 Python/Rust graph API 仍然声明指标语义，而不选择后端。编译器根据类型、
frame、排序证明和 semantic profile 选择 fast kernel；任何不受支持的组合回退到
现有通用路径，不能静默改变结果。

## 6. `RollingKernelPlan`

新增 crate-private、不可变且可 fingerprint 的 kernel IR，至少包含：

- 输入列索引与 Arrow data type；
- partition key、event-time 和 sequence order；
- row/duration frame、`min_periods`、`ddof`；
- null、NaN、无穷和 overflow policy；
- 可共享的 state group、输出表达式 DAG 与 materialization liveness；
- `kernel_version`、`state_layout_version`、`numerical_profile`；
- 排序前置条件、复杂度类别、预计 state bytes/entity；
- batch、stream 和 DataFusion physical execution 的 capability 标记。

`CompiledRollingSpec` 继续负责用户声明到语义确定的编译；
`RollingKernelPlan` 只表达已经验证过的执行形状。生成 fast plan 失败必须返回明确
原因并选择通用 kernel，`Program.explain()` 同时报告所选 kernel、共享组、排序
要求、复杂度和 fallback 原因。

## 7. Arrow-native kernel 升级

### 7.1 有序 `Float64` fast path

第一条完整垂直路径只支持最常见且最可测的组合：

- `Float64` 输入；
- bounded row window；
- `sum`、`mean`、`count`、`var`、`std`；
- 输入已经由可信 metadata 或单次线性扫描证明满足 canonical order；
- 现有 partial-window、null/NaN/Inf 和 final-only 语义不变。

执行时直接读取 Arrow value buffer 与 validity bitmap，输出直接写 Arrow builder；
enum dispatch 位于 row loop 外。fast path 中禁止 `ScalarValue::try_from_array`、
`Vec<ScalarValue>` 和逐结果 `ScalarValue::iter_to_array`。

### 7.2 排序与 entity 编码

- 已排序输入只做 O(n) ordering/duplicate proof，随后跳过 sort；
- 未排序输入只构建一次稳定 permutation，并只 gather kernel 所需列；
- partition key 在一个 batch/run 内 dictionary-encode 为 dense entity ID；
- entity state 使用按 dense ID 索引的 `Vec`，序列化时再恢复确定性的 key order；
- 若无法安全编码、证明排序或支持类型，回退通用行式路径。

### 7.3 填充接口

内部 kernel 提供概念上对应 TA-Lib 的三个入口：

- `open_and_fill(columns) -> (outputs, state)`：batch 或 stream bootstrap 一次完成；
- `update_and_fill(state, columns) -> (outputs, next_state)`：一个 micro-batch 一次
  验证、一次 transition loop、一次原子 state 交换；
- `preview(state, forming_row) -> outputs`：只在将来存在明确 forming-bar 产品需求
  时启用，不进入当前 public API。

`update_and_fill` 必须保持 Calc Flow 的失败原子性，不能部分安装 state。

### 7.4 当前实现状态

P1 已在 `operator/rolling/kernel.rs` 落地第一条 typed vertical path：编译期生成
不可变、带 fingerprint 的 kernel plan；运行期用 Arrow row encoding 线性证明
canonical order 和重复 identity，用 dense entity ID 路由预分配的 per-entity ring，
并直接写入 `UInt64Builder`/`Float64Builder`。支持有序 `Float64` bounded-row
`count`、`sum`、`mean`、`variance` 和 `stddev`，共享相同 frame 的 accumulator；
fast path 内不创建逐单元格 `ScalarValue`，也不排序或重建输入列。乱序输入和所有
不在 allowlist 内的类型、frame、primitive 均保留通用 kernel fallback。

在 clean `calc-flow@b6bfeedd993cae9d400f162ddb8669c5fcf1d64e` 上执行的阶段诊断
使用相同机器、依赖与 64-symbol workload，每点 4 轮；它用于确认工程方向，不是
最终 60-pair gate：

| Example               |      Rows | Native typed | SQL window | Baseline/native speedup | SQL/native |
| --------------------- | --------: | -----------: | ---------: | ----------------------: | ---------: |
| `SMA(20)`             |       100 |     0.246 ms |   0.817 ms |                   1.42x |     3.319x |
| `SMA(20)`             |   100,000 |     8.206 ms |  25.129 ms |                   9.85x |     3.062x |
| `SMA(20)`             | 1,000,000 |    73.121 ms | 254.134 ms |                  12.29x |     3.476x |
| `SMA(5) - SMA(20)`    |       100 |     0.406 ms |   0.997 ms |                   1.45x |     2.453x |
| `SMA(5) - SMA(20)`    |   100,000 |     9.555 ms |  40.662 ms |                   9.40x |     4.256x |
| `SMA(5) - SMA(20)`    | 1,000,000 |    90.499 ms | 414.649 ms |                  11.03x |     4.582x |

阶段原始样本见
[`SMA(20)`](../../../benchmarks/rolling/p1-rolling-mean-b6bfeed.json)和
[`SMA(5) - SMA(20)`](../../../benchmarks/rolling/p1-dual-sma-spread-b6bfeed.json)。

## 8. finality、stream 与 state layout

### 8.1 finality 与数值 transition 分离

watermark、allowed lateness、duplicate identity 和 canonical emission 仍由
`RollingStreamState` 外层处理。只有已经 final 的连续列式块进入 kernel；kernel
不自行猜测 closed bar，也不接收 public control message。

这条边界必须保留 SCE-14 的安全规则：rolling sharing 和表达式融合不能跨越
temporal finality 或 cross-section finality 边界。

### 8.2 rolling state layout v3

新的 rolling operator state layout 与 checkpoint manifest v3 是两个不同版本
域。建议 layout v3 保存：

- entity dictionary 一份，而不是在每行重复 `KeyValue`；
- 只包含未 final 数据的列式 reorder buffer；
- 只保留窗口未来会读取的 projected history columns；
- 每个 state group 的 ring/deque/scalar recurrence；
- order frontier、watermark、next output sequence；
- kernel、numeric profile 和 schema fingerprint。

reader 必须继续识别现有 layout v1/v2；新 writer 只写 v3。当前实现已经把
entity dictionary、projected history、reorder buffer 与 EWMA recurrence 分区编码，
并把 kernel fingerprint 和 numerical profile 写入 inline metadata 与 Arrow IPC
schema metadata。声明层仍接受 v1/v2，以保持 project/config fingerprint 稳定；
编译后的 operator capability 与新 checkpoint descriptor 则报告 writer v3。

后续恢复测试必须证明旧 checkpoint 可升级读取、v3 round-trip 稳定、损坏或
fingerprint 不匹配时 fail
closed，并保持 manifest v3 的 checksum 与 lineage 规则。

## 9. 复合指标与多输出融合

以 PR 中的 `SMA(5) - SMA(20)` 为第一条 liveness 用例：

1. 同一次 entity scan 更新 5-row 与 20-row state；
2. 若两个 SMA 不是 graph 的可观察输出，不物化中间 Arrow columns；
3. 直接把差值写入最终 output builder；
4. checkpoint 只保存两个必要窗口状态，不保存派生差值。

随后扩展到 BBANDS、MACD、covariance/correlation 等多输出形状：共享 input、
frame 和数值 recurrence 的输出进入同一 state group；中间结果只在多个下游确实
观察时物化。优化器必须保留 exact operation order、null propagation 和输出
类型，不允许为了 fusion 穿越 finality 或 array/materialization 边界。

## 10. DataFusion 54 升级路径

### 10.1 先消除重复 physical planning

当前 [`datafusion.rs`](../../../crates/calc-flow/src/datafusion.rs) 先调用
`DataFrame::create_physical_plan()` 生成 explain/metric 文本，随后调用
`DataFrame::collect()`；DataFusion 54 的 `collect()` 会再次调用
`create_physical_plan()`。应直接执行已经创建的 `ExecutionPlan`，同时保留同一份
logical/physical plan 文本、metric 和 zero-row schema 行为。

这项修复主要降低 SQL 路径规划开销，不解决 native rolling 热路径，但它是后续
对比可信度的前置条件。测试必须断言每个 SQL node 只 physical-plan 一次。

### 10.2 `CalcFlowRollingExec`

在 Arrow-native kernel 稳定后实现 crate-private DataFusion physical node：

- 声明 required distribution：按 entity hash partition；
- 声明 required ordering：partition 内 entity/event-time/sequence；
- 对每个输入 `RecordBatch` 调用同一个 `update_and_fill` kernel；
- 输出 partition 合并时恢复 Calc Flow canonical observable order；
- metrics 暴露 rows、entities、sort/gather、kernel、output build、state bytes；
- cancellation、memory pool 和 batch size 服从 run-scoped DataFusion context。

不要创建第二个长期共享 SessionContext。当前
[`BatchExecutionPlan`](../../../crates/calc-flow/src/pipeline/batch.rs) 每次执行创建
run-scoped `DataFusionRuntime` 的隔离边界应保留；可缓存的是不可变 logical plan、
kernel plan 或 SessionState template，不是带注册表和运行状态的 context。

### 10.3 SQL window 安全改写

只有同时满足以下条件时，physical planner 才把 SQL window 改写到
`CalcFlowRollingExec`：

- 单一受支持 aggregate；
- bounded `ROWS BETWEEN n PRECEDING AND CURRENT ROW`；
- partition/order 与 Calc Flow 的可证明顺序一致；
- frame、warm-up、null、NaN、type coercion 和 output name 完全等价；
- 无 peer-sensitive、`RANGE`、`GROUPS`、future row 或 unsupported UDF 语义。

其余 SQL 保持 DataFusion 标准计划。rewrite 必须是 allowlist、可解释并 fail
closed；不能因为表达式“看起来像 rolling”就改变 SQL 语义。

### 10.4 并行与批大小

当前默认 `target_partitions=1`、`batch_size=8192`。后续可根据 row count、entity
cardinality、平均 rows/entity 和 state bytes/entity 自适应决定 partition 数；
小数据量保持单 partition，避免调度成本。并行结果必须经过稳定 merge，在不同
线程数、batch size 和输入 partitioning 下产生相同 observable order 与数值。

## 11. 数值策略

第一阶段必须逐位或按当前测试契约保持既有 operation order；不能把 TA-Lib 的
shifted-sum/reseed 直接替换现有 West/Welford 状态。稳定性升级使用显式的
`numerical_profile`：

- `stable_v1`：当前发布语义和 checkpoint 兼容行为；
- `stable_v2`：经过独立高精度 oracle、长序列漂移、极值和性能验证后，才可引入
  shifted sums、周期或风险触发的 deterministic rebase。

profile 进入 plan fingerprint 与 rolling state，恢复时禁止混用。若未来调整默认
profile，需要单独的版本化语义决策、迁移说明和全量 fixture 复核。

## 12. 可观测性

当前 rolling node timing 是一个总值，不足以解释瓶颈。新增内部 stage metrics：

- `input_validation_ns`；
- `order_proof_ns`、`sort_permutation_ns`、`gather_ns`；
- `entity_encode_ns`；
- `kernel_ns`；
- `output_build_ns`；
- `checkpoint_encode_ns`；
- input/output/state bytes、allocations、copied bytes；
- selected kernel、fallback reason、complexity class。

DataFusion metrics 同时拆分 SQL parse/logical plan、physical plan、execution 和
collect。所有 metric 都保持 deterministic schema，不把地址、线程调度细节或
secret 写入 `RunResult`。

## 13. 分阶段实施

| Phase | Scope                     | Main delivery                                            | Exit gate                                                | Status      |
| ----- | ------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | ----------- |
| P0    | semantics and evidence    | freeze two examples; metrics; remove duplicate planning  | equivalent output; one physical plan; exact-SHA baseline | implemented |
| P1    | ordered Float64 fast path | typed buffers; order proof; dense state; direct builders | no per-cell ScalarValue; no sort on proven order         | implemented |
| P2    | state and Arrow types     | layout v3; null/integer/extrema/pair/duration kernels    | batch/stream/restore matrix; old-state reads             | in progress |
| P3    | composed output fusion    | DAG liveness; dual-SMA; BBANDS/MACD-class fusion         | no hidden materialization or finality crossing           | pending     |
| P4    | DataFusion integration    | CalcFlowRollingExec; safe rewrite; adaptive partitions   | deterministic fallback, partitions, and memory           | pending     |
| P5    | generation and numerics   | kernel census; fail-closed generation; stable_v2/preview | oracle, non-vacuity, sanitizer, migration, perf          | pending     |

每个 phase 在当前 PR 内形成独立提交，保留旧通用 kernel 作为 fallback。P1 至 P4
的每个性能阶段都先落 focused RED test，再落实现与 paired evidence；没有对应
正确性门和 rollback 开关时不得删除旧路径。

实施期间按用户要求调整为一个 PR 内的独立 phase commits，而不是拆分多个 PR；
每个阶段仍保留独立 RED test、验证证据、fallback 和可回滚提交边界。

P2 当前已完成 columnar writer v3、v1/v2 reader dispatch、checkpoint kernel identity
校验和 runtime capability 升级。`RollingKernelPlan` 也已提供 failure-atomic typed
`update_and_fill`：跨 micro-batch 保留 dense entity state，并在 stream 首次运行或
restore 后从受限 projected history bootstrap。`Float64` 数值 kernel 已从 bounded
rows 扩展到严格 `(t - duration, t]` 事件时间窗口，并在同一次 transition scan
中支持 min/max monotonic queue 与 covariance/correlation co-moment state。其余
Arrow 类型扩展与
batch/stream/restore 完整矩阵仍是本阶段未完成项，因此状态保持 `in progress`。

## 14. 验证矩阵

### 14.1 正确性

至少覆盖以下正交维度：

- rows：0、1、`window-1`、`window`、`window+1`、10、100、1,000、10,000、
  100,000、1,000,000；
- entities：1、64、高基数、entities 大于 rows、极端倾斜；
- order：已排序、跨 batch 连续、乱序、重复 identity、相同 event-time 的 sequence
  tie-break；
- window：5、20、252、10,000；row 与 duration；不同 `min_periods`、`ddof`；
- values：null、NaN、正负无穷、正负零、exact integer 边界、大幅值与近常数；
- lifecycle：batch、任意 micro-batch 切分、watermark finality、late drop/error、
  checkpoint/restore、failure retry；
- indicators：SMA、dual-SMA spread、sum/count、variance/stddev、min/max、
  covariance/correlation、EWMA/MACD、一个多输出组合。

batch/stream parity、旧 kernel/new kernel parity 和 SQL rewrite/fallback parity 必须
之外，再保留至少一个独立 direct-window 或高精度 oracle，避免对称错误。

### 14.2 性能方法

正式 gate 与本 PR 的跨库诊断分离。正式 gate 要求：

1. baseline/candidate exact SHA、clean status、依赖锁、编译 profile、CPU、OS、
   虚拟化、电源模式和 workload fingerprint 全部入报告；
2. 计时前先断言输入、输出、plan 和 semantic profile 等价；
3. 同进程执行 60 个 alternating pairs；
4. 对 paired ratio 做 20,000 次 bootstrap，报告 95% confidence interval；
5. interval 上界超过 `+5%` 时 fail closed；机器漂移、样本不足或 workload 不可比
   时结论为 inconclusive，而不是 pass；
6. small、throughput、memory、checkpoint/recovery 分开设门，不用一个总分掩盖
   回归。

TA-Lib streaming research benchmark 另设两种边界：

- kernel-only：TA-Lib `UpdateAndFill` 对 Calc Flow typed `update_and_fill`；
- end-to-end：包含多实体路由、Arrow 输入输出、排序证明和 state transaction。

两者都必须固定 TA-Lib source SHA，使用相同 full-window 语义；未发布 API 的结果
不得进入发布 gate。

### 14.3 阶段目标

- correctness 和 state compatibility 是硬门；
- 所有新 fast path 在 10、100、1,000 行不得突破 `+5%` regression interval；
- P1 完成后，100,000 与 1,000,000 行的有序 `Float64 SMA(20)` native 路径应不
  慢于同进程 SQL window；当前 native 大约慢 3.3 至 3.6 倍，因此把约 2 倍以上
  的提升作为工程目标，不在首个实验 PR 中伪造硬承诺；
- 复合 dual-SMA 的 kernel 增量成本应来自第二个 ring，而不是第二次输入扫描或
  两个中间 Arrow arrays；
- fast path 中 `ScalarValue` 转换次数为 0，可信有序输入的 sort 次数为 0，SQL
  physical planning 次数为 1；
- peak memory 满足 `O(output + entities * retained projected state)`，不再创建
  `O(rows * all input columns)` 的行对象。

## 15. 采纳、改造与拒绝

- **采纳：** fixed-at-open immutable parameters、预分配 state、无分配 update、
  `open_and_fill`/`update_and_fill`、stream capability census、生成失败 fail
  closed、batch/stream exactness 和 benchmark non-vacuity。
- **改造：** 从语义 IR 派生 kernel plan，但保留独立 emit/runtime 路径与独立
  oracle；把 opaque handle 改成版本化可序列化 state；把 `Peek` 改成受 finality
  控制的内部 preview；把统一 O(1) 宣称改成逐 kernel complexity metadata。
- **拒绝：** f64-only 公共模型、全局 mutable settings、不可序列化 checkpoint、
  invalid batch 的部分提交、默认 deep clone、绕开 Arrow/DataFusion 的第二引擎、
  以及由用户公开选择内部 backend。

## 16. 风险与回滚

- **语义漂移：** 每个 fast path 先与当前 kernel、SQL 适用子集和独立 oracle
  对照；不等价立即 fallback。
- **状态迁移：** v3 writer 上线前完成 v1/v2 reader fixture；state fingerprint
  不匹配时拒绝恢复，不猜测转换。
- **ordering proof 误判：** proof 必须来自可信 metadata 或本次执行的线性验证；
  debug/CI 可抽样与 full sort 对照。
- **并行非确定性：** partition local state 与 final merge 都有稳定 key；线程数和
  batch size 纳入 parity matrix。
- **内存放大：** plan 编译时估计 state bytes/entity；超过 run limit 时拒绝 fast
  path 或回退，不在运行中无限扩容。
- **数值变更：** `stable_v2` 永不静默替代 `stable_v1`；profile 进入 fingerprint、
  checkpoint 和 evidence。
- **实现复杂度：** 每阶段保留现有 kernel，只有在 correctness、recovery、memory
  和 paired performance 同时通过后才扩大 allowlist。

## 17. 完成定义

本计划在以下条件全部满足后才算完成：

1. batch、stream 与受支持 SQL window 使用同一 `RollingKernelPlan` 和 typed
   transition kernel；
2. 常见 ordered `Float64` rolling 热路径不再逐单元格构造 `ScalarValue`，也不
   无条件排序；
3. rolling state layout v3 可恢复、可校验，并兼容读取 v1/v2；
4. dual-SMA 和至少一个多输出指标完成 liveness/fusion；
5. DataFusion SQL 每 node 只 physical-plan 一次，受支持 window 可安全进入
   `CalcFlowRollingExec`，其他查询可靠 fallback；
6. correctness、finality、checkpoint、determinism、memory 和 paired performance
   gates 在 exact candidate SHA 上通过；
7. 两个跨库性能例子仍可运行，明确区分 partial/full-window 和 kernel/end-to-end
   边界；
8. `Program.explain()` 和 metrics 能说明 kernel、共享、复杂度、排序与 fallback
   决策。

## 18. 参考资料

### TA-Lib

- [C/C++ streaming API](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/website/src/api/stream/README.md)
- [Streaming API design](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/docs/streaming-api-design.md)
- [`StreamPlan` generator](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/ta_codegen/generator/src/streaming.rs)
- [Generated SMA C implementation](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/src/ta_func/ta_SMA.c)
- [BBANDS fusion](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/src/ta_func/ta_BBANDS.c)
- [VAR shifted sums and reseed](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/src/ta_func/ta_VAR.c)
- [Streaming A/B harness](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/scripts/stream_ab.py)
- [Streaming sanitizer harness](https://github.com/TA-Lib/ta-lib/blob/df0c6bebbf2a39d49d06193206e2af608cb96624/scripts/stream_sanitize.py)
- [ta-lib-python v0.7.1](https://github.com/TA-Lib/ta-lib-python/tree/a9ff1b47b3ddbd57274116645d688c0ed677338b)

### Calc Flow 与 DataFusion

- [Current rolling operator](../../../crates/calc-flow/src/operator/rolling.rs)
- [Current DataFusion runtime](../../../crates/calc-flow/src/datafusion.rs)
- [Batch run-scoped runtime](../../../crates/calc-flow/src/pipeline/batch.rs)
- [Symbolic computation engine design](../specs/2026-08-22-symbolic-computation-engine-design.md)
- [DataFusion 54 `DataFrame::collect`](https://github.com/apache/datafusion/blob/54.0.0/datafusion/core/src/dataframe/mod.rs#L1463-L1466)
- [DataFusion 54 physical planner](https://github.com/apache/datafusion/blob/54.0.0/datafusion/core/src/physical_planner.rs)
- [DataFusion 54 `ExecutionPlan`](https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-plan/src/execution_plan.rs)
- [Finance-Python comparison commit](https://github.com/alpha-miner/Finance-Python/tree/3e33d3e70c3458b4c6dcf76b88df6148229b402c)
