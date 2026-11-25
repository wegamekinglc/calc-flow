# ASOF 水位回调与作业交付

发布基线为 2026.11.14 的 `7012b5b4`。按输入规模、线程配置和计时起止点分别比较各组。下表是相同十万行原生负载中，每个阶段候选减发布版的配对平均值。两次实验独立运行。单位为毫秒。

| Rust 产出水位阶段 | 历史候选 72072c1d | 输出改动 9aed8b4b |
|-------------------|-------------------|-------------------|
| 外层水位回调      | +5.343            | −2.236            |
| 匹配与就绪判断    | −0.949            | −1.382            |
| Arrow 输出        | +4.772            | −0.530            |
| 左侧提交          | +1.012            | +0.241            |
| 右侧淘汰          | +0.320            | −0.779            |

历史 Arrow 增量占外层回调增量 89.3%。在 9aed8b4b 上，无插桩回调 p50 为 8.467 → 6.127 ms，p95 为 9.028 → 6.528 ms。插桩显示输出预算仍多用 0.156 ms，列定位与取值少用 0.679 ms。发布版每次有一次获准的 gather 请求，候选为零，阶段墙钟包含等待。输出预算判定的整段布局现在可以直接用于取列。单独和此改动的前序版本比较，Rust p50 为 6.201 → 6.111 ms，两轮判定不确定。累计改善不能全部归给 #750。[原生分段与原始样本](https://github.com/wegamekinglc/calc-flow/tree/c98c074656a46436d90d6a16837c8df074f52351/benchmarks/evidence/20261125-asof-watermark-phases/direct-output-layout) · [改动前对照](https://github.com/wegamekinglc/calc-flow/tree/48034359d9f9b6b596d5409e4c75f582c9b33f39/benchmarks/evidence/20261125-asof-watermark-phases/reuse-output-layout-products)

仅推进右侧且无输出时，时间用于释放去重身份，行载荷仍可供后续匹配。11.14 用按键排序的身份前缀收缩索引。新版状态还维护每个 run 的存活标记。#749 免除了复制过期定位符前缀的分配；#751 在所有键的身份全部过期且足够稠密时按 run 批量清除身份标记，稀疏情形仍逐项处理。十万行无输出回调的 p50 从 0.557 降到 0.017 ms；另与 11.14 比较为 0.019 → 0.016 ms，两轮差异不确定。随后推进左侧验证了所有十列和输出顺序。单核 Python 百万行表交付相对改动前为 102.630 → 86.400 ms，相对发布版为 95.885 → 86.481 ms，两组均未确认回退。[源码改动](https://github.com/wegamekinglc/calc-flow/pull/751/files) · [两套产品的四组实测](https://github.com/wegamekinglc/calc-flow/tree/229f87d28b0fb228ea2b2b17e42d44ad32eb83b5/benchmarks/evidence/20261125-asof-watermark-phases/dense-right-identity-products)

## Python 千万行与默认调度

在 de20f821 的安装包上，固定一个物理核时，千万行端到端 p50 为 1,166.645 → 838.007 ms。每作业 23 次回调累计，匹配少用 344.506 ms，右侧淘汰少用 62.558 ms，Arrow 多用 17.591 ms（其中预算 11.759 ms），左侧提交多用 15.563 ms。此处的分段包含 EOF 回调，独立端到端时钟在交付表时结束。原始时间戳与普通、插桩两组均独立保留。[Python 千万行分段](https://github.com/wegamekinglc/calc-flow/tree/c40d6ec0bc1a24b607162d398b3f1a554a93f3d9/benchmarks/evidence/20261125-asof-watermark-phases/hosted-python-10m)

同一 de20f821 的正式全量对照在默认托管环境确认 ASOF 千万行 p50 回退，四线程为 634.16 → 821.03 ms，单线程为 618.38 → 780.28 ms。单线程配置与将进程固定到单个物理核是两个独立变量。此前同机改变 CPU 可见范围的实验有一组在资源预检后停止，尚未定量分离线程调度、数据移动与回调处理这三个因素。[完整 1,295 项及全体回退](https://github.com/wegamekinglc/calc-flow/tree/2bdb08b055f3f83043554966df1139932bf1febd/benchmarks/scheduled/20261125-45623973262-streaming-audit) · [亲和性控制记录](https://github.com/wegamekinglc/calc-flow/tree/0dc0201b282cb644dba8ce8a3864684aff108361/benchmarks/evidence/20261125-asof-watermark-phases/hosted-worker-affinity)

Polars 的分组 ASOF 在同一键上推进已排序右侧的查找位置，再按索引取列；calc-flow 的有序探测游标已用于匹配。当前实现在选择阶段复制待输出的左侧定位符前缀，交付成功后在提交阶段用 `starts_with` 逐项核对该前缀。复制与核验分别计入匹配和提交。额外 20 对分段样本中，左侧提交比发布版平均多用 0.1951 ms，前缀核验占 0.1195 ms，状态变动多用 0.0850 ms，发布版的准备及其余差异抵消了 0.0095 ms。此次插桩的两臂外层 p50 都比此前普通组低约 15%，两组运行时间不同。后续若消除前缀重复遍历，需要保留对同一输出前缀的证明、交付失败不提交及重复提交被拒的语义。[左侧提交互斥分段及原始记录](https://github.com/wegamekinglc/calc-flow/tree/47c6ae9c9be76fa3b933a347d5fde9175bfb971c/benchmarks/evidence/20261125-asof-watermark-phases/left-commit-split) ·[Polars 源码对照](https://github.com/wegamekinglc/calc-flow/blob/8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae/benchmarks/evidence/20261125-stream-engine-release-review/polars-review.md) · [定位符前缀构造](https://github.com/wegamekinglc/calc-flow/blob/9aed8b4b20156a460369d0765c4bc9e42e0925e5/crates/calc-flow/src/operator/asof/columnar/state.rs#L1372) · [确认后提交](https://github.com/wegamekinglc/calc-flow/blob/9aed8b4b20156a460369d0765c4bc9e42e0925e5/crates/calc-flow/src/operator/asof/columnar/state.rs#L1534)
