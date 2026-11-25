# 全量流执行基准复核

本次运行候选为 `ea7a21cead99c424c1d4a6ce5dcf46abfc449aab`。公开安装包的发布基线为 `7012b5b409553df04e29e6d7f9a11abf919cbf93`。计划中的原生 Rust 分片以候选的第一父提交 `3f905765d802635edb455c4c40fd57685666aeb4` 为基线。运行的最终状态为 `failure`。[GitHub Actions 完整运行](https://github.com/wegamekinglc/calc-flow/actions/runs/45596944021)

完整报告有 1,295 条结果。对 500 个配对用例核对了逐次正确性与两轮判定。共有 30 项确认回退，80 项确认提升，281 项结论不确定，109 项未确认回退。单次指标、外部引擎和信息性比较保留在全表。[全量 CSV](all-scheduled-cases.csv) · [全部回退](confirmed-regressions.csv)

## 每侧或主输入 1,000 万行的默认流式用例

公开时钟从已就绪数据入队计至结果 Arrow 表。表中时长单位为毫秒。静态 Join 的维表最多 64 行，ASOF 两侧各 1,000 万行。每个线程配置独立与同入口的 11.14 对照。变化百分比来自合并样本 p50，两轮配对区间决定回退或改善判定。

| 算子       | 4T 发布 → 候选 p50  | 4T 配对判定       | 1T 发布 → 候选 p50  | 1T 配对判定        |
|------------|---------------------|-------------------|---------------------|--------------------|
| argmax256  | 529.85 → 510.05     | -3.74% 未确认回退 | 529.81 → 516.56     | -2.50% 未确认回退  |
| argmax64   | 532.39 → 499.31     | -6.21% 未确认回退 | 528.98 → 502.36     | -5.03% 未确认回退  |
| asof_join  | 1,066.43 → 1,720.48 | +61.33% 回退      | 1,009.68 → 1,873.80 | +85.58% 回退       |
| average    | 374.25 → 251.16     | -32.89% 提升      | 367.79 → 237.23     | -35.50% 提升       |
| cs_mean    | 167.04 → 172.96     | +3.55% 不确定     | 149.97 → 151.62     | +1.10% 不确定      |
| dual_sma   | 621.16 → 724.44     | +16.63% 回退      | 611.75 → 724.58     | +18.44% 回退       |
| filter     | 55.52 → 58.51       | +5.38% 不确定     | 61.34 → 60.79       | -0.89% 未确认回退  |
| group_by   | 376.16 → 382.58     | +1.71% 未确认回退 | 378.98 → 383.84     | +1.28% 未确认回退  |
| join       | 824.32 → 680.63     | -17.43% 提升      | 685.53 → 696.38     | +1.58% 不确定      |
| projection | 57.86 → 56.48       | -2.39% 不确定     | 38.27 → 25.21       | -34.13% 未确认回退 |
| sma20      | 473.13 → 576.41     | +21.83% 回退      | 451.21 → 583.49     | +29.31% 回退       |
| unique64   | 1,103.01 → 1,348.29 | +22.24% 回退      | 1,121.57 → 1,397.05 | +24.56% 回退       |
| window_sum | 981.58 → 978.86     | -0.28% 未确认回退 | 997.26 → 1,008.09   | +1.09% 未确认回退  |

完整 CSV 另列两臂 p95、两轮区间、输出基数、批大小与 checkpoint 变体。SMA20 和双均线采用行数窗口。window_sum 是 10 秒事件时间分桶窗口（tumbling window），group_by 使用跨越整个输入的分桶求每键总和。逐行时间戳滑窗另有原生与安装包的千万行 20 秒窗口对照。[时间戳滑窗实验](https://github.com/wegamekinglc/calc-flow/tree/ba0a913e9c751598acf094586723168587f59be3/benchmarks/evidence/20261125-rolling-window-followup/timestamp-20s)

区间 Join 的目录上限为 1,000,000 行，应用其最大实际规模的结果。该四线程用例的判定为 `improved`。[算子覆盖清单](stream-operator-coverage.csv)

## 千万行回退的绝对耗时差

对所有配置按 p50 绝对差降序排列，列出增加最多的用例。

| 用例                                           | 增加毫秒 | 相对变化 |
|------------------------------------------------|----------|----------|
| engines/10000000/calc-flow-stream-1t/asof_join | 864.12   | +85.58%  |
| engines/10000000/calc-flow-stream/asof_join    | 654.05   | +61.33%  |
| engines/10000000/calc-flow-stream-1t/unique64  | 275.49   | +24.56%  |
| engines/10000000/calc-flow-stream/unique64     | 245.28   | +22.24%  |
| engines/10000000/calc-flow-stream-1t/sma20     | 132.27   | +29.31%  |
| engines/10000000/calc-flow-stream-1t/dual_sma  | 112.83   | +18.44%  |
| engines/10000000/calc-flow-stream/dual_sma     | 103.28   | +16.63%  |
| engines/10000000/calc-flow-stream/sma20        | 103.28   | +21.83%  |

## Rust 核心与小批延迟

原生 Rust 分片中的父提交比较用于查找近期变化。直接对照 11.14 的原生 Rust 单水位与千万行整作业另有固定适配器和逐行校验。[ASOF 单水位与输出分段](https://github.com/wegamekinglc/calc-flow/tree/d26860e433df4d33bbcb92ba23cfc5849bc671a7/benchmarks/evidence/20261125-asof-watermark-phases/output-charge-followup) · [原生四线程千万行对照](https://github.com/wegamekinglc/calc-flow/tree/f1728a86cf24c83b25c1d6ff4188a4b58b3fef0a/benchmarks/evidence/20261125-asof-watermark-phases/dense-projection-native-4t)

| 热态用例 | 追加行数 | 发布 p50 | 候选 p50 | 配对判定 |
|----------|----------|----------|----------|----------|
| dual_sma | 64       | 0.61     | 0.60     | 不确定   |
| sma20    | 64       | 0.65     | 0.60     | 不确定   |

原始完整报告、千万行分片、绑定的提交和构建摘要保存在同一目录。复核脚本以原始 Actions 附件的 SHA-256 校验来源。[运行与附件标识](sources.json) · [发布版源码与 Polars 对照](https://github.com/wegamekinglc/calc-flow/tree/8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae/benchmarks/evidence/20261125-stream-engine-release-review)
