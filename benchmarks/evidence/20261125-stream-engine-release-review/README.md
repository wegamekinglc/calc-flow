# 流执行引擎对照 2026.11.14

本报告使用 2026-11-25 完成的全套定时结果。发布版为 `7012b5b4`，候选为 `72072c1d`，两者之间已包含此前 Join / ASOF 优化。1,295 条结果中有 49 项通过正确性校验并满足两轮回退判据。这 49 项由公开流式引擎四线程 22 项、单线程 26 项、热态流式用例 1 项组成。其余项含不可比、信息性指标和第三方参考，不应据此宣告引擎达标。[完整运行与原始附件](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000)

## 1,000 万行决定的优先级

以下百分比是该次四线程默认公开流式用例的 p50 变化。输入规模、输出基数及线程数影响排序。全表保留每轮区间、1T、非默认批次、检查点和预热变体。[完整数据、49 项逐例清单及复算脚本](https://github.com/wegamekinglc/calc-flow/blob/e0c73c35a13b8f7b7dccc9be7bd0f102c1b87f2d/benchmarks/scheduled/20261125-45496822000-streaming-audit/README.md)

- 累计 `ts.average` 为 378.58 → 1,208.96 ms，慢 219.34%。它是按实体的累计递推。现实现的快速分派会进入逐行读写状态并构造临时向量的路径。首个小 PR 以实体数约束临时分配，验证空值、NaN、跨批和恢复后的递推。
- ASOF 为 885.02 → 2,553.77 ms，慢 188.56%，每侧各 1,000 万行。小规模阶段归因及百万行公开作业均把 Arrow 输出排在水位处理耗时增量首位。先减少整批有序命中时的物化，再分解接纳、匹配、左提交以及公开作业中尚未归属的时间。各阶段的测量负载单列。[Rust 阶段实测](https://github.com/wegamekinglc/calc-flow/blob/d0fee918f7277557bd068d3da285dfaa44d2a152/benchmarks/evidence/20261125-asof-watermark-phases/native-phase-summary.md) · [Python 作业阶段实测](https://github.com/wegamekinglc/calc-flow/blob/91c75c4d921581f2687f1b5f90fba1bc53c21cfd/benchmarks/join-asof/20261125-asof-public-watermark-phases/README.md)
- SMA20 慢 21.55%，argmax256 慢 18.56%，argmax64 慢 18.42%，双 SMA 慢 12.07%。分别追踪进入的实体并行或串行分支、每个输入批的状态读取/写回次数、滚动队列更新和输出拷贝。多窗口复用需要保持独立窗口边界。
- 静态 Join 为 693.57 → 632.43 ms，GroupBy 为 294.64 → 294.71 ms，四线程默认配置均无确认回退。定时目录的公开流式 `group_by` 把全量输入放入一个按 symbol 分组的固定时间窗（tumbling window），在最终水位产生 64 组。增量 SQL GroupBy 的持续快照和 changelog 需要各自的直接发布版对照。其他线程、批次和投影按照各自行判定。Interval Join 的目录上限是每侧 100 万行，千万行需另设有界输出、完整结果 oracle 的实验。

全部 32 个千万行公开作业对照均列出 p50、p95、线程数、批大小及输出行数。[逐项查看千万行结果](10m-results.md)

## 两层验收及后续小 PR

1. 每个运行时改动同时保留 Rust 算子或作业证据、安装 wheel 后的 Python 公开作业证据。比较对象固定为 11.14，并用变更前 main 隔离本 PR 的收益。维持现有结果校验、峰值资源检查和成对统计。Rust 定时分片当前使用候选第一父提交，采用了新接口的微基准不能直接作为发布版对照。需要从两版公共入口执行同一个工作负载，单独标记新增能力。
2. 在 ASOF 输出和累计 average 两个独立改动后，按最新 10M 的绝对超时及相对差距选择下一项。滚动算子覆盖 SMA、argmax 与 unique，按行窗口与按事件时间窗口分开；后者单独验证有序/乱序和水位闭合。在变更涉及的每个家族，保留小批次与低基数控制组。
3. 对静态、区间 Join、GroupBy、跨截面、投影、SQL filter 和固定时间窗口，持续核对 1T / 4T 的各自结果。常规负载已接近发布版时先保留为控制组。计划与算子结构见[两版源码审阅](release-review.md)和[Polars 实现对照](polars-review.md)。
4. 交付目标是同工作量下 p50 至少与 11.14 持平，并保留 p95、样本数及两轮区间。仓库已确认回退需要消除，统计不确定的行单列。相邻提交结果、一个有利样本、外部引擎成绩以及不在目录中的 10M 用例均不能证明整引擎达到发布水平。

## 首轮修复和仍待闭合的差距

实测值使用各自的固定提交、负载和时钟。原生 Rust 定时分片采用第一父提交作基线，下面直接与 11.14 配对的 Rust 实验补充了这一缺口。

- **ASOF Arrow 输出。** [#731](https://github.com/wegamekinglc/calc-flow/pull/731) 已合并。十万行的原生水位回调 p50 从修改前 12.697 降到 8.117 ms，单独对发布版的比较仍慢约 13%。完整有序命中可以共享 Arrow 列或拼接连续批，恢复自同一大块 IPC 缓冲区的旧 run 继续走有界取列。[Rust 原始数据](https://github.com/wegamekinglc/calc-flow/blob/479338da4d73a42fc1dd7bae97c806bc0baa5a60/benchmarks/evidence/20261125-asof-watermark-phases/README.md)
- **累计 average。** [#732](https://github.com/wegamekinglc/calc-flow/pull/732) 已合并。Rust 百万行、1T 作业从修改前 90.745 降至 25.041 ms。独立发布版对照的 p50 为 29.110 → 25.534 ms，判定为无确认回退。Python 千万行、4T 的直接发布版对照为 289.90 → 242.81 ms，两个配对轮次均确认改善。[Rust 数据](https://github.com/wegamekinglc/calc-flow/tree/dcd6b7fb4980576c089147d13b33982eb9a795b0/benchmarks/evidence/20261125-average-native) · [Python 数据](https://github.com/wegamekinglc/calc-flow/tree/25a8ee625eb3c3a86aa6133ad57f8596f34142a0/benchmarks/join-asof/20261125-average-public-10m)
- **ASOF 状态释放和接纳。** [#733](https://github.com/wegamekinglc/calc-flow/pull/733) 已合并。Rust 十万行水位回调与发布版的 p50 为 7.280 → 7.413 ms，区间不确定。Python 每侧千万行 4T 的 p50 为 0.6193 → 1.5710 秒，仍确认回退。后续接纳插桩保留了第一轮十对、第二轮六对完整记录，主机换页超限后中止。候选的已完成样本中，索引安装、键路由、Arrow 紧凑化的中位耗时依次为 402、299、193 ms。首个索引改动应以未插桩公开作业验收。[Rust 数据](https://github.com/wegamekinglc/calc-flow/tree/b9e14450e887c5e06018f44d735452d9baaf07ec/benchmarks/evidence/20261125-asof-watermark-phases/left-retirement) · [Python 数据](https://github.com/wegamekinglc/calc-flow/tree/fac198978dc1e25311870d78a6c8b256f1cbca01/benchmarks/join-asof/20261125-asof-left-retirement-public-10m) · [接纳诊断](https://github.com/wegamekinglc/calc-flow/tree/f7e6a09cacd1c4e57649253cd6fb8532136c5594/benchmarks/join-asof/20261125-asof-public-admission-phases-10m)
- **SMA 与 argmax。** 千万行 1T 的 Python SMA20 在历史主线复测中为 329.41 → 518.33 ms。每轮十对均校验全部输出；水位回调的两轮配对中位增量分别为 184.73、185.31 ms。源码对照发现，64 键的 slab 从按批编码不同字符串键变为逐行编码。独立改动先恢复按不同键编码，再评估数值环和每行实体 ID 的成本。对于 argmax256，Rust 千万行在 [#734](https://github.com/wegamekinglc/calc-flow/pull/734) 的候选上为 485.43 → 469.06 ms，两轮排除 5% 回退，尚未达到确认改善门槛。[SMA 两版调用路径与原始样本](https://github.com/wegamekinglc/calc-flow/tree/760f9d2856fff2840729bc8409d5a3d138da5c27/benchmarks/evidence/20261125-sma20-watermark-phases) · [argmax 原生数据](https://github.com/wegamekinglc/calc-flow/tree/74a0921939f5cc67b6b0ce0398ccdfa0bd0b1bbf/benchmarks/evidence/20261125-argmax256-native)

后续按单个可测环节合并。先处理每侧千万行 ASOF 接纳与按行滚动。再以 100 ms 切点分别处理静态、区间、ASOF Join 恢复。随后逐项复核其余算子、热态和小批次。各阶段目标均为同工作量下 p50 回到 11.14 或更快，保留 p95、误差、资源、初始与末端状态校验；全引擎结论需要发布版与合并后的 main 再跑完整目录。
