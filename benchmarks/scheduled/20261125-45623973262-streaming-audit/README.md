# 全量流执行基准复核

本次运行候选为 `de20f8216c8dcb1bfce3e3811c3884a3080157df`。公开安装包的发布基线为 `7012b5b409553df04e29e6d7f9a11abf919cbf93`。计划中的原生 Rust 分片以候选的第一父提交 `723df09bcb6d90e5c577d4aa1b1309221a1351e3` 为基线。运行的最终状态为 `failure`。[GitHub Actions 完整运行](https://github.com/wegamekinglc/calc-flow/actions/runs/45623973262)

完整报告有 1,295 条结果。对 500 个配对用例核对了逐次正确性与两轮判定。共有 21 项确认回退，93 项确认提升，279 项结论不确定，107 项未确认回退。单次指标、外部引擎和信息性比较保留在全表。[全量 CSV](all-scheduled-cases.csv) · [全部回退](confirmed-regressions.csv)

## 每侧或主输入 1,000 万行的默认流式用例

公开时钟从已就绪数据入队计至结果 Arrow 表。表中时长单位为毫秒。静态 Join 的维表最多 64 行，ASOF 两侧各 1,000 万行。每个线程配置独立与同入口的 11.14 对照。变化百分比来自合并样本 p50，两轮配对区间决定回退或改善判定。

| 算子       | 4T 发布 → 候选 p50 | 4T 配对判定       | 1T 发布 → 候选 p50 | 1T 配对判定       |
|------------|--------------------|-------------------|--------------------|-------------------|
| argmax256  | 327.05 → 309.52    | -5.36% 未确认回退 | 322.76 → 308.87    | -4.30% 未确认回退 |
| argmax64   | 310.12 → 282.11    | -9.03% 未确认回退 | 323.28 → 308.02    | -4.72% 未确认回退 |
| asof_join  | 634.16 → 821.03    | +29.47% 回退      | 618.38 → 780.28    | +26.18% 回退      |
| average    | 220.73 → 147.53    | -33.16% 提升      | 214.37 → 135.34    | -36.87% 提升      |
| cs_mean    | 105.44 → 108.87    | +3.25% 不确定     | 101.15 → 103.83    | +2.65% 不确定     |
| dual_sma   | 348.52 → 407.17    | +16.83% 回退      | 330.68 → 389.00    | +17.64% 回退      |
| filter     | 40.24 → 43.67      | +8.51% 不确定     | 43.36 → 44.18      | +1.89% 不确定     |
| group_by   | 201.33 → 198.71    | -1.30% 未确认回退 | 198.64 → 201.55    | +1.47% 未确认回退 |
| join       | 498.68 → 388.58    | -22.08% 提升      | 399.93 → 379.68    | -5.06% 未确认回退 |
| projection | 38.41 → 38.93      | +1.36% 不确定     | 17.57 → 16.15      | -8.08% 未确认回退 |
| sma20      | 270.07 → 316.13    | +17.05% 回退      | 264.31 → 295.70    | +11.88% 不确定    |
| unique64   | 609.06 → 656.33    | +7.76% 不确定     | 603.83 → 654.45    | +8.38% 回退       |
| window_sum | 780.88 → 773.24    | -0.98% 不确定     | 798.25 → 769.91    | -3.55% 未确认回退 |

完整 CSV 另列两臂 p95、两轮区间、输出基数、批大小与 checkpoint 变体。SMA20 和双均线采用行数窗口。window_sum 是 10 秒事件时间分桶窗口（tumbling window），group_by 使用跨越整个输入的分桶求每键总和。逐行时间戳滑窗另有原生与安装包的千万行 20 秒窗口对照。[时间戳滑窗实验](https://github.com/wegamekinglc/calc-flow/tree/ba0a913e9c751598acf094586723168587f59be3/benchmarks/evidence/20261125-rolling-window-followup/timestamp-20s)

区间 Join 的目录上限为 1,000,000 行，应用其最大实际规模的结果。该四线程用例的判定为 `improved`。[算子覆盖清单](stream-operator-coverage.csv)

## 千万行回退的绝对耗时差

表中按 p50 耗时增量对千万行已确认回退的流式用例降序排列。

| 用例                                           | 增加毫秒 | 相对变化 |
|------------------------------------------------|----------|----------|
| engines/10000000/calc-flow-stream/asof_join    | 186.87   | +29.47%  |
| engines/10000000/calc-flow-stream-1t/asof_join | 161.90   | +26.18%  |
| engines/10000000/calc-flow-stream/dual_sma     | 58.65    | +16.83%  |
| engines/10000000/calc-flow-stream-1t/dual_sma  | 58.32    | +17.64%  |
| engines/10000000/calc-flow-stream-1t/unique64  | 50.62    | +8.38%   |
| engines/10000000/calc-flow-stream/sma20        | 46.06    | +17.05%  |

## Rust 核心与小批延迟

原生 Rust 分片采用第一父提交作对照。单独与 11.14 比较时，后续候选 9aed8b4 的十万行产出水位已有 20 对无插桩和 20 对分段样本；de20f821 的千万行原生复测在资源门槛处停止。两组覆盖不同规模与时钟。[原生单水位分段](https://github.com/wegamekinglc/calc-flow/tree/c98c074656a46436d90d6a16837c8df074f52351/benchmarks/evidence/20261125-asof-watermark-phases/direct-output-layout) · [千万行原生复测记录](https://github.com/wegamekinglc/calc-flow/tree/ba6be38a2dc85e305396bb4c2008fcf246147565/benchmarks/evidence/20261125-asof-watermark-phases/hosted-rust-10m)

本轮全量候选早于 #748 的 Unique64 和 #750 的 ASOF 输出改动。公开 Python 的单核固定亲和性实验与本表默认调度分开判读。[千万行 Python 分段](https://github.com/wegamekinglc/calc-flow/tree/c40d6ec0bc1a24b607162d398b3f1a554a93f3d9/benchmarks/evidence/20261125-asof-watermark-phases/hosted-python-10m) · [Unique64 独立复测](https://github.com/wegamekinglc/calc-flow/tree/a6e5e97bcd9ef90b5a3bb761fd97a964f122aafb/benchmarks/evidence/20261125-rolling-window-followup/unique64-products)

| 热态用例 | 追加行数 | 发布 p50 | 候选 p50 | 配对判定   |
|----------|----------|----------|----------|------------|
| dual_sma | 64       | 0.51     | 0.50     | 未确认回退 |
| sma20    | 64       | 0.49     | 0.48     | 未确认回退 |

原始完整报告、千万行分片、绑定的提交和构建摘要保存在同一目录。复核脚本以原始 Actions 附件的 SHA-256 校验来源。[运行与附件标识](sources.json) · [发布版源码与 Polars 对照](https://github.com/wegamekinglc/calc-flow/tree/8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae/benchmarks/evidence/20261125-stream-engine-release-review)
