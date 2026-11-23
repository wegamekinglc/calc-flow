# 增量计算性能实施记录

本记录覆盖截至 2026-11-23 合入 `main` 的小阶段，截止提交为
`0df04cd6`。同一项业务计算在 Rust 运行器、Rust 同步会话和 Python 公开 API
有不同的计时边界。下列每一组前后值都来自同一项实验，跨 PR 的耗时不能直接相减。

## 增量分组 SUM

对于已经存在的一万个组，后续小批次只改变少量键。Rust 的 `KeyedSumState`
保留组槽位与累计值，事务记录本批次触及的组，输出端可以只为这些槽位构造
Arrow 变更行。测量逐批重建累计快照并校验，覆盖 21 批共 12,000 行。
首个阶段的 Rust 单核完整计算 p50 从 352.51 降至 22.07 ms。
独立的 Python 公开流从 472.779 降至 40.734 ms，包含声明、启动、收齐输出和关闭。
[本地状态实现](../../crates/calc-flow/src/operator/sql/keyed_sum.rs) ·
[Rust 原始记录](../../benchmarks/sum-state/20261123-sparse-changelog/README.md) ·
[Python 配对结果](https://github.com/wegamekinglc/calc-flow/pull/633#issuecomment-7092182682)

改用行哈希直接索引组后，Rust p50 为 9.865 → 9.306 ms，两端都达到
10 ms 目标。Python p50 为 24.201 → 23.460 ms，距 15 ms 目标仍差 8.460 ms。
两项比较按仓库的 5% 规则均为 `no-confirmed-regression`。
Arroyo 在 `flush` 时取出 `updated_keys`，逐个计算新值，跳过值未变的组，
按需生成撤回和追加。对 Calc Flow 的启示是让变更键集合贯穿状态提交与输出构造。
下一小阶段应单独计时预热后的一次稀疏更新，保留完整图和快照校验作为对照。
[Rust 配对结果](https://github.com/wegamekinglc/calc-flow/pull/649#issuecomment-7093526848) ·
[Python 配对结果](https://github.com/wegamekinglc/calc-flow/pull/649#issuecomment-7093482797) ·
[Arroyo 增量聚合源码](https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-worker/src/arrow/incremental_aggregator.rs#L638-L677)

## 基于时间戳的滚动窗口

Rust 的 `DurationSamples` 为同一输入列的多个窗口保存一份带时间戳的样本队列，
每个窗口单独推进过期位置和累加器。30、60、90 秒三个窗口的单实体存储对照中，
物理样本从 180 降为 90。另一次千实体、十万行、每批千行的完整 Rust 运行器比较，
p50 从 34.98 降至 28.34 ms。TA-Lib 的 WMA 同时维护普通和加权总和，
利用离开窗口的值递推，并定期重算以控制浮点漂移。Calc Flow 可借鉴递推与数值校验的方法，
事件时间决定样本属于哪些窗口，水位控制窗口何时关闭、如何判定迟到和回收状态。
[窗口实现](../../crates/calc-flow/src/operator/rolling/kernel/duration.rs) ·
[Rust 对照与原始样本](../../benchmarks/rolling/20261123-shared-duration/README.md) ·
[TA-Lib WMA 源码](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/input/wma/wma.c#L118-L207)

对单个 60 秒 SUM 的三次 Rust 诊断中，预热任务的互斥分段中位数为状态准备
4.757 ms、数值更新 3.670 ms、实体定位 3.595 ms。独立主任务分别耗时
14.498、14.777、17.767 ms。下一小阶段先考察已有实体的状态准备，目标 ≤4 ms，
同时复测完整运行器，目标 p50 ≤8 ms。Python 公开流在导入后的 Arrow 缓冲区上
检查事件时间，完整十万行 p50 从 53.292 降至 43.724 ms，p95 从 55.255
降至 48.766 ms。公开流的 p50 目标为 12 ms，应单独追踪批次导入、输出与关闭。
[Rust 分段样本](https://github.com/wegamekinglc/calc-flow/pull/670#issuecomment-7095420880) ·
[Python 配对结果](https://github.com/wegamekinglc/calc-flow/pull/668#issuecomment-7095891562)

## Interval Join

融合投影只读取输出所需的序列列和价格列后，Python 公开路径在两侧各一百万行的区间连接上，
p50 从 1,233.385 降至 423.858 ms。计时从已准备好的输入入队到最终 Arrow 拼接，钟外验证全部 10,998,080 对及其乘积。
后续把输入侧重复坐标表示为连续匹配段，Rust 的 88 对匹配只需读取 110 个坐标项，
原为 176 个。两侧仍分别写出 88 个 Arrow 索引。该阶段 Rust 耗时实验因宿主机
主缺页超过预登记阈值，只完成 8/20 对，无法判断延迟变化。对应 Python 实验完成
20 对，p50 为 443.376 → 417.080 ms，两轮区间对可重复提速的判断仍不充分。
[投影实验](https://github.com/wegamekinglc/calc-flow/pull/648#issuecomment-7092105938) ·
[Rust 坐标与中断记录](https://github.com/wegamekinglc/calc-flow/pull/671#issuecomment-7096670299) ·
[Python 匹配段实验](https://github.com/wegamekinglc/calc-flow/pull/671#issuecomment-7095981916)

RisingWave 的哈希连接根据匹配数决定是否转发未匹配行，或仅转发一次已匹配行，
并用水位清理状态。Calc Flow 后续可将追加流的左半连接和左反连接拆成两个 PR。
前者每个左行在首次匹配时输出一次，目标是避免为重复命中构造结果对。
后者需等右侧水位严格越过闭区间上界，或收到右侧 EOF，才能输出仍无匹配的左行。
Idle 保留未决状态。右侧 EOF 后若仍有左行到达，先探测尚可匹配的右侧历史，有匹配则抑制输出。
两阶段各自验证迟到数据、重复时间戳、空批和恢复，然后分别测首次命中的探测量及
水位推进时的输出延迟。
[RisingWave 匹配与回收源码](https://github.com/risingwavelabs/risingwave/blob/bc563ec0330a6db01788959d249aee2126061d0d/src/stream/src/executor/hash_join.rs#L1215-L1244)

## 单行同步会话

Rust 已把有序输入的固定绑定准备移至会话构造。对已经推进水位的一次冷批次，
从输入校验到边预算检查的临时分配从 10 次、322 字节降至 8 次、240 字节。
Python 普通单输入映射的独立实验先填入 4,096 行，再测双均线单行
`push`，强制 GC 下 p50 为 208.342 → 209.823 µs，p95 为 253.471 → 237.751 µs。
两轮配对区间均跨零，结论为 `inconclusive`。下一小阶段直接测 Rust 公开
`StreamSession::push`，p50/p95 目标为 10/25 µs。同一业务计算的 Python 公开入口继续按
80/120 µs 评估。分配数、Rust 调用耗时和 Python 强制 GC 测量分别报告。
[Rust 分配对照](https://github.com/wegamekinglc/calc-flow/pull/674#issuecomment-7096913764) ·
[Python 单行样本与调用边界](https://github.com/wegamekinglc/calc-flow/pull/673#issuecomment-7097264284)
