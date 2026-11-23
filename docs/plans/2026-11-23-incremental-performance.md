# 增量计算性能实施记录

本记录汇总 2026-11-23 基于 `main` 实施的增量、连接与滚动计算实验。
每个小阶段的代码、测量记录与合并状态见对应 PR。
同一项业务计算在 Rust 运行器、Rust 同步会话和 Python 公开 API
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
后续的单次稀疏更新实验沿用了相同的逐批快照校验。
[Rust 配对结果](https://github.com/wegamekinglc/calc-flow/pull/649#issuecomment-7093526848) ·
[Python 配对结果](https://github.com/wegamekinglc/calc-flow/pull/649#issuecomment-7093482797) ·
[Arroyo 增量聚合源码](https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-worker/src/arrow/incremental_aggregator.rs#L638-L677)

Meilisearch 的 Arroy 向量索引提供了另一个可借鉴的实现。`Writer` 写入、替换或删除
向量时记录修改项的 ID，构建时集中收取这些 ID。常规多节点树更新分支先移除涉及的条目，
重新插入的集合取「当前仍存在的 ID」与「修改过的 ID」的交集。对应到 Calc Flow，
可将同一批中多次出现的组折叠成一个候选输出槽位，净值未变时跳过输出，将新出现的组与需要保留旧值
以便回滚的组区分开。SQL 稀疏更新实验用逐批快照重建校验了这种输出方式。
[Arroy 写入记录](https://github.com/meilisearch/arroy/blob/f1b732701f9de96736cf0c6daa9162d779ea7db3/src/writer.rs#L380-L434) ·
[Arroy 更新集合与建树](https://github.com/meilisearch/arroy/blob/f1b732701f9de96736cf0c6daa9162d779ea7db3/src/writer.rs#L496-L542)

预热后的单次更新采用同一版本的两种输出模式。先建立 10,000 组，再连续提交
20 个百行批次，每批改动 80 组。每种模式正式计时 400 次。Rust 单核运行图从提交到收到
输出，完整快照的 p50/p95 为 251.889/335.532 µs，变更输出为 28.426/101.138 µs。
Python 公开流由同一官方 wheel 执行两种模式，从提交到收到 Arrow 表分别为
728.382/814.463 µs 和 397.923/463.417 µs。变更输出达到 Rust 250 µs 与
Python 1 ms 的独立 p50 目标。每个作业累计输出 210,000 行或 13,200 行，
每个批次均校验完整累计值。
[Rust 样本和运行边界](../../benchmarks/sum-state/20261123-warm-update/rust-native.json) ·
[Python 样本和配对区间](https://github.com/wegamekinglc/calc-flow/pull/684#issuecomment-7101166282)

## 基于时间戳的滚动窗口

在关闭检查点、初期按序提交并进入延迟保留历史路径的作业中，Rust 的
`DurationSamples` 为同列、不同宽度的浮点数值时间窗口保存一份带时间戳的样本队列。
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
14.498、14.777、17.767 ms。随后对已有实体的状态准备做了单独优化，目标 ≤4 ms，
同时复测完整运行器，目标 p50 ≤8 ms。Python 公开流在导入后的 Arrow 缓冲区上
检查事件时间，完整十万行 p50 从 53.292 降至 43.724 ms，p95 从 55.255
降至 48.766 ms。公开流的 p50 目标为 12 ms，应单独追踪批次导入、输出与关闭。
[Rust 分段样本](https://github.com/wegamekinglc/calc-flow/pull/670#issuecomment-7095420880) ·
[Python 配对结果](https://github.com/wegamekinglc/calc-flow/pull/668#issuecomment-7095891562)

单个 60 秒 SUM 的后续实测中，状态准备从 4.988 降至 4.550 ms，
完整 Rust 运行器从 15.580 降至 14.581 ms，两个目标仍有差距。
同时维护 30、60、90 秒三个 SUM 时，把已有实体的撤销记录放入一块批次缓冲区，
128 个预热实体的准备分配从 131 次降至 4 次，请求字节数均为 116,736。
十万行、千实体、每批千行的独立运行器观测为 29.831 → 27.562 ms（p50）。
降幅 7.61%，低于预设的 10%。第二轮配对区间跨零。
[单窗口分段和全程](https://github.com/wegamekinglc/calc-flow/pull/672) ·
[三窗口样本、运行器与原始记录](https://github.com/wegamekinglc/calc-flow/pull/687#issuecomment-7101966562)

新出现的实体在批次被拒绝时可以整体撤销。让这些实体直接走普通数值更新，
只为已有实体保留旧窗口。在 128 个新实体、每个实体三个时间窗口的状态准备中，
保存的窗口从 384 个降到 0 个，请求字节从 290,236 降到 208,316，
临时堆峰值从 262,468 降到 180,548 字节。共享时间窗口和混合行数/时间窗口
均验证过拒绝后重试的完整输出。
[集成后的分配和重试验证](https://github.com/wegamekinglc/calc-flow/pull/690#issuecomment-7102333282)

继续缩减撤销记录后，在相同的 128 个新实体、三个时间窗口上，撤销槽位容量从
128 变为 0，请求字节从 208,316 降到 175,548，准备阶段的临时堆峰值从
180,548 降到 147,780 字节。128 个已有实体的对照组在两版上都是 4 次分配、
114,688 请求字节。混合批次的测试在已有实体前后插入新键，拒绝一次输出后重试，
逐行核对结果。十万行滚动任务的 8 ms 目标仍需以完整运行器复测。
[分配对照及失败重试](https://github.com/wegamekinglc/calc-flow/pull/701)

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
并用水位清理状态。Calc Flow 分两个阶段实现追加流的左半连接和左反连接。
前者每个左行在首次匹配时输出一次，目标是避免为重复命中构造结果对。
左反连接对具备有效键与时间的左行，等右侧水位严格越过闭区间上界或收到右侧 EOF，
再输出仍无匹配的行。已通过迟到等准入检查的左行若连接键或时间为空，可立即输出。
Idle 保留未决状态。右侧 EOF 后若仍有左行到达，先探测尚可匹配的右侧历史，有匹配则抑制输出。
高重复测试先让 8,193 个左行各自得到一个匹配，再提交包含 4,096 个同键同时间右行的批次。
该批次的匹配资格读取从 33,558,528 次降至 8,193 次，两版均输出零行。
另一个用例在左侧仍活动时处理右侧 EOF，期间按序交付先前保留的 8,193 个
无匹配左行，分 257 批，每批最多 32 行。这两组观测分别记录重复探测工作量和 EOF 输出边界。
[左半连接及恢复](https://github.com/wegamekinglc/calc-flow/pull/676) ·
[左反连接、严格水位边界及验证](https://github.com/wegamekinglc/calc-flow/pull/686) ·
[RisingWave 匹配与回收源码](https://github.com/risingwavelabs/risingwave/blob/bc563ec0330a6db01788959d249aee2126061d0d/src/stream/src/executor/hash_join.rs#L1215-L1244)

活动水位的独立延迟实验让左右源始终保持活动。水位等于闭区间上界时目标行仍未输出，
下一次推进严格越界后按序交付 257 个未匹配左行，形成九个输出块。原生 Rust 从进入
水位回调，经过内存收集器接收全部输出、提交算子状态，计至回调返回，p50/p95 为
54.594/66.568 µs。公开 Python 从投递
水位到收到最后一张 Arrow 表为 1.406/1.474 ms。两种入口分别运行在各自的单核环境，
各有 20 次正式观测，达到各自 1 ms 和 5 ms 的 p50 目标。p95 仅作小样本尾部参考。
[原生与公开流的独立样本、时钟及校验](https://github.com/wegamekinglc/calc-flow/pull/694)

融合 Join 的微基准让 `HotProjection::gather` 接收已收集到连续 Arrow 切片中的
4,096 对 Float64 操作数，计至返回两列。每进程调用 256 次，进程均值的中位数
从 13.780 降至 2.190 µs。百万行事实表与 64 行维表的 Python
公开流独立计时，从输入就绪到最终 Arrow 拼接，p50 为 93.331 → 90.724 ms，
p95 为 101.817 → 92.597 ms。两轮配对区间均指向改善，未达到仓库的 5% 改善标准，
公开入口距 30 ms 目标仍差 60.724 ms。
[Rust 投影样本](https://github.com/wegamekinglc/calc-flow/pull/683#issuecomment-7100097085) ·
[Python 公开输出和原始样本](https://github.com/wegamekinglc/calc-flow/pull/683#issuecomment-7100882755)

融合乘积投影现在检查已解析的表达式、精确输出 schema 和全部消费边。
带价格筛选的实测用例曾返回两行，修订后只返回符合条件的一行。引号别名、
字段元数据及另有消费者时的完整输出也有独立回归。另一个恢复修订在直接结束
Inner/LeftSemi 时保留物理输入位置并清空两侧已缓冲的行，使最终检查点可以恢复。
[融合回归](https://github.com/wegamekinglc/calc-flow/pull/689#issuecomment-7102333306) ·
[Join 结束和检查点恢复](https://github.com/wegamekinglc/calc-flow/pull/691)

等值和区间连接共用的驻留数据段现在直接持有组号、事件时间、行号与过期顺序四列。
检查点与活动连接状态共享整个数据段。单键、相同时间戳的 1,024 行构造实验中，
分配从 20 次降至 16 次，累计请求从 82,144 降至 53,408 字节（−35.0%）。
该段临时堆峰值为 45,152 → 45,088 字节。此处的收益集中在构造期间的分配流量；
完整连接的延迟仍须单独测量。等值与区间连接的检查点恢复均核对了后续输出。
[驻留状态与恢复实测](https://github.com/wegamekinglc/calc-flow/pull/700)

## 单行同步会话

Rust 已把有序输入的固定绑定准备移至会话构造。对已经推进水位的一次冷批次，
从输入校验到边预算检查的临时分配从 10 次、322 字节降至 8 次、240 字节。
Python 普通单输入映射的独立实验先填入 4,096 行，再测双均线单行
`push`，强制 GC 下 p50 为 208.342 → 209.823 µs，p95 为 253.471 → 237.751 µs。
两轮配对区间均跨零，结论为 `inconclusive`。后续直接测量 Rust 公开
`StreamSession::push`，p50/p95 目标为 10/25 µs。同一业务计算的 Python 公开入口继续按
强制 GC 模式的 80/120 µs 评估。分配数、Rust 调用耗时和 Python 强制 GC 测量分别报告。
[Rust 分配对照](https://github.com/wegamekinglc/calc-flow/pull/674#issuecomment-7096913764) ·
[Python 单行样本与调用边界](https://github.com/wegamekinglc/calc-flow/pull/673#issuecomment-7097264284)

在固定输入、预装 4,096 行历史、每进程连续推送 64 个单行批次的场景里，
Rust 公开 `StreamSession::push` 的 p50/p95 从 5.692/12.370 变为 5.185/11.914 µs，
两端均达到目标。Python 公开 `Program.session().push` 使用正常 GC 的独立实验，
p50/p95 从 63.285/84.384 变为 62.334/82.315 µs，未达到该场景的 20/40 µs
目标。每侧正式计时 1,280 次，验证水位、输出行及五列内容。两项实验均未证实
超过 5% 的改善。该 Python 负载传入已封装的 Batch。
[Rust 配对记录](../../benchmarks/stream-session/20261123-native-push/README.md) ·
[Python 正常 GC 和公开调用记录](https://github.com/wegamekinglc/calc-flow/pull/688)

后续 Rust 将实体的行数和首行位置合并保存。对传入单行、预计有 4,096 个
驻留实体的 UTF-8 路由，临时分配从 4 次、40 字节降到 2 次、24 字节。
在独立的一组公开会话测量中，p50 从 4.681 变为 4.679 µs，p95 从 5.803
变为 5.752 µs。两轮配对区间分别为 [−2.17%, +3.62%] 和 [−2.43%, +3.84%]，
均跨零。路由分配达到目标，公开调用尚无可确认的延迟改善。
[路由分配与正确性](https://github.com/wegamekinglc/calc-flow/pull/692#issuecomment-7102614957) ·
[公开 Rust 调用样本与原始记录](https://github.com/wegamekinglc/calc-flow/pull/692#issuecomment-7102924755)

对同一官方 wheel 单独做 Python 分段诊断，未插桩的单行公开调用 p50/p95 为
59.621/80.661 µs，插桩后为 62.326/81.992 µs。插桩时各段 p50 分别是输入名称、
批次和水位处理 8.646 µs，PyO3 调用（含 Rust 执行及 Arrow 输出导出）48.230 µs，
Python 返回表封装 5.450 µs。每个进程先取 64 次公开调用的中位数，同轮十对进程的相对差再取中位数。
两轮的观察器开销分别为 4.32% 和 4.70%。判定产品优化收益使用另行采集的未插桩样本。
[三段诊断与观察器对照](https://github.com/wegamekinglc/calc-flow/actions/runs/45187056499)

在 PyO3 绑定中，读取 `Batch.kind` 只需查看 payload 类型。保留读锁中的类型匹配后，
64 次公开属性读取触发的临时批次克隆从 64 次降至 0 次，真正提交输入时仍取得所需的
所有权。两个独立构建的 wheel 使用未插桩公开入口，各计时 1,280 次单行推送。
p50 为 60.243 → 60.032 µs，p95 为 79.9105 → 79.6515 µs，两轮配对区间均跨零，
尚未达到 20/40 µs 目标。此阶段证实了临时克隆的消除，整次公开调用仍需在 PyO3
执行和输出导出内进一步定位。
[绑定修订和两组原始样本](https://github.com/wegamekinglc/calc-flow/pull/695) ·
[未插桩公开调用的独立配对](https://github.com/wegamekinglc/calc-flow/actions/runs/45194346131)

Rust 批次封装将记录、行数和字节估算缓存放入同一个共享对象，可接收已填满的记录向量。
封装一条已有记录时，额外分配从 2 次、104 字节降至 1 次、72 字节。对同一单行双均线
场景，Rust 公开调用的 p50/p95 从 4.911/6.359 变为 4.876/6.101 µs，两端均在
10/25 µs 目标内。两轮配对变化为 −2.11% 和 +1.90%，区间均跨零。用独立官方 wheel
测量 Python 单行公开调用时，同核另一硬件线程忙碌率达到 67.3%，实验按预登记规则
停止。两轮预定的 44 个进程仅完成 24 个（含预热），本次无法判定 Python 的延迟变化。
[Rust 分配与公开调用](https://github.com/wegamekinglc/calc-flow/pull/699) ·
[Python 已完成样本与资源记录](https://github.com/wegamekinglc/calc-flow/actions/runs/45238492928/artifacts/14850617924)

## Python 多输出同步会话

当水位推进未产生新数据时，`Program.session().advance` 仍需为每个声明名称返回
带正确 schema 的空表。现在可复制准备好的名称到空表的映射，每次返回独立的字典，
省去逐名称查找物理输出。两个官方 Linux wheel 的预热均核对了 32 个名称、字段元数据、
行数据和字典的独立性。正式采样的第一个一秒区间记录到 302 次主缺页，超过上限
255 次，已完成的正式配对数为 0。空水位 p50 降低 10% 的目标及有数据推送的对照延迟
均待验证。
[实现与回归测试](https://github.com/wegamekinglc/calc-flow/pull/702) ·
[预登记、构建身份和原始记录](https://github.com/wegamekinglc/calc-flow/tree/e62f3cc9b8dc8596135da0494fc0cf79bc0a9990/benchmarks/stream-session/20261124-quiet-program-results)

## 后续小阶段与验收目标

1. **Python 单行推送。** 在既有双均线负载中，把 PyO3 调用继续划分为 Rust 执行、
   Arrow C Data 导出与 PyArrow 导入。先用相同 wheel 比较有无插桩，再用两个独立
   wheel 复测未插桩的公开入口。正常 GC 下的目标仍是 p50/p95 ≤20/40 µs。
   [PyO3 会话与输出导出](../../crates/calc-flow-python/src/session.rs)
2. **Rust 单时间窗口。** 对已有实体的 60 秒 SUM 分别测量实体定位、状态准备与
   数值更新。每次只改变一段，复测十万行完整运行器。状态准备目标 ≤4 ms，
   全程 p50 目标 ≤8 ms。密集新键与已有键分别保留分配对照。
   [有序窗口更新路径](../../crates/calc-flow/src/operator/rolling/kernel/in_place.rs)
3. **Join 从探测到交付。** 对百万行事实表与 64 行维表的公开 Python 用例，
   分别定位探测、输出投影和最终 Arrow 拼接。Rust 侧独立测量构造与探测，
   再判断驻留列的分配优化能否降低全程延迟。Python 目标仍为 p50 ≤30 ms。
   区间连接单独校验匹配对数与乘积，沿用活动水位释放的 Rust 1 ms、
   Python 5 ms p50 目标。
   [Join 处理与输出](../../crates/calc-flow/src/operator/join/production.rs)
