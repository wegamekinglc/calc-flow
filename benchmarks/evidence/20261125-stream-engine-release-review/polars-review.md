# Polars 流式实现对照

源码固定在 Polars `08b2814ce6386c0efe0fea512912353f42a8b7b6`。以下是代码机制及对 calc-flow 的工程判断。本文的耗时来自 calc-flow 自身的发布版配对测量。

### 投影与过滤

SimpleProjectionNode 按预先解析的列位置克隆列句柄，保留 morsel，平行收发。FilterNode 对各 morsel 求谓词掩码并过滤。

对比 10M projection/filter 的 Rust 与 Python 两层耗时，分出数据转换、表达式执行、队列交付。只做投影时检查是否分配了与输入行数成比例的缓冲。

### 滚动与时间窗口

固定窗口的 next_batch 按前后邻域保留跨批数据，裁剪输出。时间 rolling 的 next_windows 用排序键确定已闭合窗口，只取参与这些窗口的数据范围。PlacementTracker 会先看首末值，整批落在范围内可直接通过。

把单调事件时间、键交错、跨批边界和乱序归并分别覆盖。有界均值检查新增值和离开值的更新次数。累计 average 则检查每个实体的计数、递推值和批间状态搬移次数。保持水位等号、迟到、检查点和恢复语义。

### GroupBy

Sink 保存局部预聚合；SplitPreAggs 在有多个分区时拆分分组，通过 in_partition 只传递对应的组，再分区合并。

先定位 hash、聚合更新、快照输出的各自成本。低基数的全量输出和每批少量变化的 changelog 使用各自的 oracle；10M 输入、64 组输出单列。

### 等值 Join

ProbeOutput 缓存命中位置，区分 build/probe 侧 gather。探测侧可延迟 rechunk 并限制所取列。

测静态右表与双活动输入。按匹配数计 gather，按交付批次计物化，保留重复键乘积和顺序。

### 区间 Join

RangeJoinNode 在有序点键上以 search_sorted 求区间起止位置，然后有界地构建／发送输出。

对每组有序时间索引先求端点，再批量输出；候选数和输出数分别记录。Polars 的点对区间 inner join 与双活动水位 Join 约束不同，须另验等号、迟到与淘汰。

### ASOF

调度端按左批需要补足右侧缓冲，prune_right_side 逐步退休历史。计算端保留左表，用匹配索引取右列，最后拼接。分组路径按已排序组的范围处理。

已实测 Arrow 输出是水位最大差项。先识别整段、顺序、有效的定位结果，直接复用完整列或拼接连续输入批；重复匹配、乱序或缺失走索引。双侧水位、未匹配空值与最新 tie 按原约定校验。

固定源码位置见 [投影](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/simple_projection.rs#L47)、[过滤](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/filter.rs#L37)、[固定窗口](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/rolling_fixed_window.rs#L90)、[时间滚动](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/rolling_group_by.rs#L132)、[GroupBy](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/group_by/pre_agg.rs#L56)、[等值 Join](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/joins/equi_join/probe.rs#L351)、[区间 Join](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/joins/range_join.rs#L331)、[ASOF](https://github.com/pola-rs/polars/blob/08b2814ce6386c0efe0fea512912353f42a8b7b6/crates/polars-stream/src/nodes/joins/asof_join.rs#L749)。

## 定时任务中的 Polars 参考耗时

在相同任务的 1,000 万行标准输入上，Polars 默认线程配置下 ASOF 为 473.75 ms，GroupBy 为 58.35 ms，静态 Join 为 119.49 ms，累计 average 为 763.37 ms，SMA20 为 881.55 ms。单线程 ASOF 为 965.79 ms。该表记录 Polars LazyFrame 的 streaming collect 到 Arrow 的时间；calc-flow 的公开流式计时还包括作业内水位推进、交付与末端结果收集。源码审阅用于提出可验证的实现选择，11.14 在同入口、同线程数的结果用于 calc-flow 的性能验收。[逐例原始结果与口径](https://github.com/wegamekinglc/calc-flow/blob/e0c73c35a13b8f7b7dccc9be7bd0f102c1b87f2d/benchmarks/scheduled/20261125-45496822000-streaming-audit/all-scheduled-cases.csv)
