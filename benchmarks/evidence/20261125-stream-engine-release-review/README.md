# 流执行引擎对照 2026.11.14

本报告使用 2026-11-25 完成的全套定时结果。发布版为 `7012b5b4`，候选为 `72072c1d`，两者之间已包含此前 Join / ASOF 优化。1,295 条结果中有 49 项通过正确性校验并满足两轮回退判据。这 49 项由公开流式引擎四线程 22 项、单线程 26 项、热态流式用例 1 项组成。其余项含不可比、信息性指标和第三方参考，不应据此宣告引擎达标。[完整运行与原始附件](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000)

## 1,000 万行决定的优先级

以下百分比是该次四线程默认公开流式用例的 p50 变化。输入规模、输出基数及线程数影响排序。全表保留每轮区间、1T、非默认批次、检查点和预热变体。[完整数据、49 项逐例清单及复算脚本](https://github.com/wegamekinglc/calc-flow/blob/e0c73c35a13b8f7b7dccc9be7bd0f102c1b87f2d/benchmarks/scheduled/20261125-45496822000-streaming-audit/README.md)

- 累计 `ts.average` 为 378.58 → 1,208.96 ms，慢 219.34%。它是按实体的累计递推。现实现的快速分派会进入逐行读写状态并构造临时向量的路径。首个小 PR 以实体数约束临时分配，验证空值、NaN、跨批和恢复后的递推。
- ASOF 为 885.02 → 2,553.77 ms，慢 188.56%，每侧各 1,000 万行。小规模阶段归因及百万行公开作业均把 Arrow 输出排在水位处理耗时增量首位。先减少整批有序命中时的物化，再分解接纳、匹配、左提交以及公开作业中尚未归属的时间。各阶段的测量负载单列。[Rust 阶段实测](https://github.com/wegamekinglc/calc-flow/blob/d0fee918f7277557bd068d3da285dfaa44d2a152/benchmarks/evidence/20261125-asof-watermark-phases/native-phase-summary.md) · [Python 作业阶段实测](https://github.com/wegamekinglc/calc-flow/blob/91c75c4d921581f2687f1b5f90fba1bc53c21cfd/benchmarks/join-asof/20261125-asof-public-watermark-phases/README.md)
- SMA20 慢 21.55%，argmax256 慢 18.56%，argmax64 慢 18.42%，双 SMA 慢 12.07%。分别追踪进入的实体并行或串行分支、每个输入批的状态读取/写回次数、滚动队列更新和输出拷贝。多窗口复用需要保持独立窗口边界。
- 静态 Join 为 693.57 → 632.43 ms，GroupBy 为 294.64 → 294.71 ms，四线程默认配置均无确认回退。定时目录的公开流式 `group_by` 把全量输入放入一个按 symbol 分组的固定时间窗（tumbling window），在最终水位产生 64 组。增量 SQL GroupBy 的持续快照和 changelog 需要各自的直接发布版对照。其他线程、批次和投影按照各自行判定。Interval Join 的目录上限是每侧 100 万行，千万行需另设有界输出、完整结果 oracle 的实验。

## 两层验收及后续小 PR

1. 每个运行时改动同时保留 Rust 算子或作业证据、安装 wheel 后的 Python 公开作业证据。比较对象固定为 11.14，并用变更前 main 隔离本 PR 的收益。维持现有结果校验、峰值资源检查和成对统计。Rust 定时分片当前使用候选第一父提交，采用了新接口的微基准不能直接作为发布版对照。需要从两版公共入口执行同一个工作负载，单独标记新增能力。
2. 在 ASOF 输出和累计 average 两个独立改动后，按最新 10M 的绝对超时及相对差距选择下一项。滚动算子覆盖 SMA、argmax 与 unique，按行窗口与按事件时间窗口分开；后者单独验证有序/乱序和水位闭合。在变更涉及的每个家族，保留小批次与低基数控制组。
3. 对静态、区间 Join、GroupBy、跨截面、投影、SQL filter 和固定时间窗口，持续核对 1T / 4T 的各自结果。常规负载已接近发布版时先保留为控制组。计划与算子结构见[两版源码审阅](release-review.md)和[Polars 实现对照](polars-review.md)。
4. 交付目标是同工作量下 p50 至少与 11.14 持平，并保留 p95、样本数及两轮区间。仓库已确认回退需要消除，统计不确定的行单列。相邻提交结果、一个有利样本、外部引擎成绩以及不在目录中的 10M 用例均不能证明整引擎达到发布水平。
