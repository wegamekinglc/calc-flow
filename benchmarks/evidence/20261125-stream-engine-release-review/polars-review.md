# Polars 流式实现对照

源码固定在定时任务锁定的 Polars 2.0.0（标签 py-2.0.0）对应提交 `22a147de3d2bb2e44b97338a2510816c7105c9f2`。以下是代码机制及对 calc-flow 的工程判断。本文的耗时来自 calc-flow 自身的发布版配对测量。

### 投影与过滤

SimpleProjectionNode 按预先解析的列位置克隆列句柄，保留 morsel，平行收发。FilterNode 对各 morsel 求谓词掩码并过滤。

对比 10M projection/filter 的 Rust 与 Python 两层耗时，分出数据转换、表达式执行、队列交付。只做投影时检查是否分配了与输入行数成比例的缓冲。

### 滚动与时间窗口

固定窗口的 next_batch 按前后邻域保留跨批数据，裁剪输出。时间 rolling 的 next_windows 用排序键确定已闭合窗口，只取参与这些窗口的数据范围。PlacementTracker 会先看首末值，整批落在范围内可直接通过。

把单调事件时间、键交错、跨批边界和乱序归并分别覆盖。有界均值检查新增值和离开值的更新次数。累计 average 则检查每个实体的计数、递推值和批间状态搬移次数。保持水位等号、迟到、检查点和恢复语义。

### GroupBy

每个输入维护热组哈希表。局部预聚合被移出后，add_pre_agg 按分区记录组索引和偏移，使后续合并可以只读取相应分区的条目。

先定位 hash、聚合更新、快照输出的各自成本。低基数的全量输出和每批少量变化的 changelog 使用各自的 oracle；10M 输入、64 组输出单列。

### 等值 Join

探测端保存匹配位置，对 build/probe 两侧分别 gather。探测侧可延迟 rechunk 并限制所取列。

测静态右表与双活动输入。按匹配数计 gather，按交付批次计物化，保留重复键乘积和顺序。

### 区间 Join

RangeJoinNode 在有序点键上以 search_sorted 求区间起止位置，然后有界地构建／发送输出。

对每组有序时间索引先求端点，再批量输出；候选数和输出数分别记录。Polars 的点对区间 inner join 与双活动水位 Join 约束不同，须另验等号、迟到与淘汰。

### ASOF

调度端按左批需要补足右侧缓冲，prune_right_side 逐步退休历史。计算端保留左表，用匹配索引取右列，最后拼接。分组路径按已排序组的范围处理。

已实测 Arrow 输出是水位最大差项。先识别整段、顺序、有效的定位结果，直接复用完整列或拼接连续输入批；重复匹配、乱序或缺失走索引。双侧水位、未匹配空值与最新 tie 按原约定校验。

固定源码位置见 [投影](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/simple_projection.rs#L47)、[过滤](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/filter.rs#L37)、[固定窗口](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/rolling_fixed_window.rs#L90)、[时间滚动](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/rolling_group_by.rs#L132)、[GroupBy](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/group_by.rs#L133)、[等值 Join](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/joins/equi_join.rs#L1159)、[区间 Join](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/joins/range_join.rs#L356)、[ASOF](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-stream/src/nodes/joins/asof_join.rs#L547)。

## 定时任务中的 Polars 参考耗时

在相同任务的 1,000 万行标准输入上，Polars 线程池设为 4 时 ASOF 为 473.75 ms，GroupBy 为 58.35 ms，静态 Join 为 119.49 ms，累计 average 为 763.37 ms，SMA20 为 881.55 ms。单线程 ASOF 为 965.79 ms。该表记录 Polars LazyFrame 的 streaming collect 到 Arrow 的时间；calc-flow 的公开流式计时还包括作业内水位推进、交付与末端结果收集。源码审阅用于提出可验证的实现选择，11.14 在同入口、同线程数的结果用于 calc-flow 的性能验收。[逐例原始结果与口径](https://github.com/wegamekinglc/calc-flow/blob/e0c73c35a13b8f7b7dccc9be7bd0f102c1b87f2d/benchmarks/scheduled/20261125-45496822000-streaming-audit/all-scheduled-cases.csv)

定时任务的区间查询把右表的时间移动 −5 到 +5 个键内步长，执行 11 次等值连接并拼接，耗时反映这组查询。上文的 RangeJoinNode 属于可借鉴的另一条执行路径。ASOF 查询在 Polars 端使用 backward 且关闭排序检查，未设容差；calc-flow 使用零容差。此输入的每个左键和时间都有精确右匹配，因此两者的完整输出相同。1T 限定 Polars 线程池或 calc-flow Tokio 工作者为 1，任务 CPU 亲和性仍允许四个逻辑核。[查询代码](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/benchmarks/engine_comparison.py#L418) · [calc-flow 容差](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/benchmarks/engine_stream.py#L231) · [依赖锁](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/benchmarks/requirements.lock#L333)
