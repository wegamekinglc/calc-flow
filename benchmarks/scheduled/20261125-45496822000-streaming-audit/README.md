# 2026-11-25 定时 benchmark 全算子审计

本报告对应 GitHub Actions 定时任务 45496822000 的第 1 次尝试。候选是 main 的 `72072c1dc6c0874b79ccd8ce78103c97caec623d`，Python 公开引擎和 warm 对照是 11.14 发布版 `7012b5b409553df04e29e6d7f9a11abf919cbf93`。1295 个报告行均返回成功状态。500 个历史配对用例均重新核对了两轮所有正式样本、逐次正确性结果和统计区间，49 个用例的两轮配对中位数置信下界均超过 +5%。其中公开 streaming 4 线程 22 个、单线程 26 个，另有 1 个 warm 增量用例。构建产物、可信摘要与原始报告可复算。 [定时任务](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000) · [GitHub 完整报告](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000/artifacts/14984186563) · [归档原件](original-complete-report.zip)

## 真实 1000 万行

下表是默认批次的公开 managed streaming 入口。每个样本覆盖 1000 万行主输入，默认每批 100 万行，时钟覆盖 ready 后入队到 Arrow 结果交付。ASOF 的两侧各有 1000 万行、64 个键和零容差，静态等值 Join 的右维表有 64 行，预加载和封存发生在计时前。GroupBy 返回 64 组，过滤返回 250 万行，时间窗口求和返回 100 万行。输出条数不同于输入规模。表中延迟为毫秒，Δ 是两组各 20 个样本池化 P50 的变化百分比。 [1000 万行分片原件](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000/artifacts/14984581171) · [数据集和 oracle](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/benchmarks/engine_comparison.py)

| Operator      | 11.14 4T P50 | Main 4T P50 | 4T delta | 4T verdict              | 1T delta | 1T verdict              |
|---------------|--------------|-------------|----------|-------------------------|----------|-------------------------|
| projection    | 49.38        | 53.29       | +7.92%   | inconclusive            | -2.58%   | no-confirmed-regression |
| filter        | 59.53        | 65.58       | +10.17%  | inconclusive            | -1.51%   | no-confirmed-regression |
| group_by      | 294.64       | 294.71      | +0.02%   | no-confirmed-regression | -0.53%   | no-confirmed-regression |
| join          | 693.57       | 632.43      | -8.81%   | no-confirmed-regression | +1.11%   | no-confirmed-regression |
| interval_join | n/a          | n/a         | n/a      | capped at 1M            | n/a      | capped at 1M            |
| sma20         | 481.06       | 584.71      | +21.55%  | regression              | +37.08%  | regression              |
| dual_sma      | 545.00       | 610.78      | +12.07%  | regression              | +14.35%  | regression              |
| average       | 378.58       | 1,208.96    | +219.34% | regression              | +250.89% | regression              |
| argmax64      | 450.26       | 533.18      | +18.42%  | regression              | +17.11%  | regression              |
| argmax256     | 477.75       | 566.43      | +18.56%  | regression              | +20.18%  | regression              |
| unique64      | 906.19       | 964.55      | +6.44%   | inconclusive            | +7.53%   | regression              |
| cs_mean       | 140.62       | 159.95      | +13.75%  | inconclusive            | +0.54%   | no-confirmed-regression |
| window_sum    | 956.09       | 941.58      | -1.52%   | no-confirmed-regression | -1.46%   | no-confirmed-regression |
| asof_join     | 885.02       | 2,553.77    | +188.56% | regression              | +189.87% | regression              |

10M default 下，4T 最大相对回退是按键连续平均 average（+219.34%），ASOF +188.56%。其余确认项是 SMA20、argmax256、argmax64、dual SMA。单线程还确认 unique64 回退。逐轮确切置信区间和每个算子的 P95 在 CSV 中。 [全量行与区间](all-scheduled-cases.csv)

10M 实测还包括每批 1024 行的 projection、静态 Join、ASOF，4T 和 1T 各三项。4T 静态 Join 为 -5.68%，判为 improved，ASOF 为 -7.00%，判为 no-confirmed-regression。Interval Join 的 catalog 将主输入限制在 100 万行，未执行 1000 万行。100 万行默认 4T、1T 分别改善 29.6% 和 36.6%；其余尺寸与模式中仍有两个真实 Interval Join 回退。定时 checkpoint/recovery 变体仅在 10 万与 100 万行档运行。Warm 的 1000 万表示预填历史，随后计时 64 行追加，覆盖 SMA20、dual SMA。唯一 warm 回退是 100 万历史上追加 6.4 万行的 SMA20（+23.10%）。 [覆盖表](stream-operator-coverage.csv) · [精确 catalog](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/scripts/benchmark_suite/catalog.py)

## 全尺度

managed streaming 引擎在目录中的 14 种算子均有数据，包含 280 项独立的 1T/4T、默认/小批及 checkpoint 配置；另有 26 项公开增量 warm。49 个确认回退跨 9 类算子。下表列出每类确认回退中池化 P50 相对差距最大的单个用例，同时给出各自两轮配对中位数下界。 [全部 49 项与上界、下界](confirmed-regressions.csv)

| Operator      | Count | Largest relative case                                                                       | P50 delta | R1 lower | R2 lower |
|---------------|-------|---------------------------------------------------------------------------------------------|-----------|----------|----------|
| asof_join     | 10    | engines/1000000/calc-flow-stream-1t/asof_join                                               | +349.79%  | +298.00% | +338.45% |
| average       | 8     | engines/10000000/calc-flow-stream-1t/average                                                | +250.89%  | +243.29% | +233.53% |
| interval_join | 2     | engines/100000/calc-flow-stream/interval_join/batch-1024/checkpoint-100ms-duration-recovery | +40.94%   | +32.64%  | +38.47%  |
| sma20         | 6     | engines/10000000/calc-flow-stream-1t/sma20                                                  | +37.08%   | +31.16%  | +27.87%  |
| join          | 4     | engines/100000/calc-flow-stream/join/batch-1000000/checkpoint-100ms-duration-recovery       | +32.09%   | +23.02%  | +21.82%  |
| argmax256     | 7     | engines/1000000/calc-flow-stream/argmax256                                                  | +28.48%   | +19.11%  | +18.74%  |
| argmax64      | 6     | engines/1000000/calc-flow-stream/argmax64                                                   | +26.69%   | +16.88%  | +15.87%  |
| unique64      | 4     | engines/10000/calc-flow-stream-1t/unique64                                                  | +19.06%   | +16.36%  | +8.72%   |
| dual_sma      | 2     | engines/10000000/calc-flow-stream-1t/dual_sma                                               | +14.35%   | +12.88%  | +11.86%  |

## Rust、SQL 和对照引擎

Rust 原生分片基线是 main 第一父 `a35ad02a0b065a91505b94304abb6c67e3ac5644`。128 项 Rust 数据包括 112 项 suite-blocks 性能参考和 16 项 allocation 计数。69 项专门面向 ASOF、等值 Join、SQL 增量聚合、窗口和 Union 等流算子，另有独立的 rolling 内核与流基础设施项目。原始 Criterion 对照包含固定夹具和多个统计口径，部分项目未报统一输入行数。原生分片没有覆盖全算子的“同一线程配置、原生 1000 万行、对 11.14”的配对结论。原生与 Python 公开数据需分别解读。 [Rust 分片](https://github.com/wegamekinglc/calc-flow/actions/runs/45496822000/artifacts/14980977810)

194 个 calc-flow 有限 SQL 数据行、354 个外部参考行（DataFusion、Polars、TA-Lib、Finance-Python）均单独标记。外部引擎记录是同场参考速度，没有 calc-flow 的 11.14→main 变化。312 个 Python 独立 pytest benchmark 和 1 个生命周期验证各自沿用专门口径。仓库并未为每一个受支持的特殊形态提供 10M 用例，例如通用基于持续时间的 rolling 并未得到 10M 发布配对，目录中的 window_sum 是按事件时间分组的 10 秒固定窗口（tumbling）聚合，不能外推到任意滑动时间窗口。

## 复算与字段

存在发布对照的引擎和 warm 使用两轮，每轮 10 个交替 AB/BA 配对；名义 95% 的精确顺序统计区间在 10 对样本下覆盖率为 97.85%。combined P50 百分比与配对中位数是两个统计量。只有两轮下界都超过 +5% 才记为确认回退。正式原始数据中的错误、缺失和未配对样本须保留其状态。此次 49 条总报告错误逐条对应上述 49 个确认回退，原始 10M 分片的 106 条数据与全量汇总逐条相等。

CSV 的 dimension_rows 记录原报告的行维度，primary_fixture_rows 只填写明确的主输入行数，right_fixture_rows 只在可从公开 streaming 夹具确认右输入规模时填写。history_rows 与 timed_append_rows 单列。unknown 数字留空。suite-blocks 的时钟数值属于 informational-only，allocation 计数属于 metric-only。历史已配对表包含 4T/1T、P50/P95、每轮区间、正确性、样本计数及分片链接。 [原始 10M 分片结果](original-10m-shard-results.json.gz) · [精确构建和原始文件摘要](sources.json)

以下命令在上述 main checkout 中使用本证据目录中已保存的原始报告、分片数据和摘要离线复算表格。 [复算脚本](make_audit.py)

```bash
PYTHONPATH=. python EVIDENCE_DIRECTORY/make_audit.py EVIDENCE_DIRECTORY OUTPUT_DIRECTORY
```
