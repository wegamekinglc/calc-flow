# Python 安装包的千万行 ASOF 水位分解

两侧各 1,000 万行、64 个键。使用单个物理核，对照 11.14 发布版 7012b5b4 和合并后的 de20f821。分别测量普通构建与仅加水位计时的构建。每组预定两轮，每轮十对正式观测，每轮另有一次双臂预热。

每次验证全部投影输出行的 sequence、value、物理顺序和作业结束后的状态。分段记录 finalize 调用的墙钟时间，包含其中的异步等待及发送背压；入口更新进度、记录重放以及发布版在进入回调前的淘汰等待均在该时钟之外。

原始实验在独立 GitHub 托管主机执行。[运行](https://github.com/wegamekinglc/calc-flow/actions/runs/45627251427) · [冻结的协议与源码](https://github.com/wegamekinglc/calc-flow/tree/18be04ec374b9318b6efd29afecedb26b2e6d1de/benchmarks/evidence/20261125-hosted-asof-watermark)

## 无插桩

从就绪输入到最终输出的 p50 为 1,166.645 → 838.007 ms，p95 为 1,195.265 → 881.571 ms。两轮配对判定为 `improved`。

## 开启分段计时

从就绪输入到最终输出的 p50 为 1,215.120 → 855.301 ms，p95 为 1,300.228 → 1,025.822 ms。两轮配对判定为 `improved`。

| 回调阶段           | 发布 p50 (ms) | 候选 p50 (ms) | 同对平均差 (ms) |
|--------------------|---------------|---------------|-----------------|
| 回调总计           | 696.581       | 334.952       | -356.285        |
| 匹配和就绪判断     | 587.577       | 242.032       | -344.506        |
| Arrow 输出         | 5.962         | 23.404        | +17.591         |
| 其中，输出预算     | 0.068         | 11.678        | +11.759         |
| 其中，列定位与收集 | 5.791         | 11.569        | +5.760          |
| 其中，组装及边预算 | 0.052         | 0.068         | +0.015          |
| 其中，其他输出     | 0.040         | 0.097         | +0.058          |
| 左侧提交           | 0.250         | 14.336        | +15.563         |
| 右侧淘汰           | 104.095       | 40.769        | -62.558         |
| 发送               | 0.106         | 0.149         | +0.045          |
| 剩余回调开销       | 0.079         | 13.499        | +17.580         |

分段是每次作业中水位及 EOF 回调的累计值。独立公开时钟在提交 EOF 前结束。子阶段的配对平均差可以相加，独立分位数不能相加。

在同主机同核先后运行的两组中，插桩组相对普通组的 p50 变化为发布版 +4.16%、候选 +2.06%。

此机的每个正式作业有 23 次回调，其中 10 次产生输出，2 次记录双侧结束。两版在所有分段都没有获准的 gather-service 请求。累计回调中，匹配与就绪判断是最大的改善项；Arrow 输出预算是输出内最大的剩余差距。组装的差距接近零。未细分的残余开销与整个 Arrow 的增量接近，应独立追踪。

当前实现先核验完整输入段用于预算，随后在输出路径重新核验同一组定位行。此重复遍历是可检验的优化点，其耗时尚未从预算与列收集时钟中独立隔离。[预算路径](https://github.com/wegamekinglc/calc-flow/blob/de20f8216c8dcb1bfce3e3811c3884a3080157df/crates/calc-flow/src/operator/asof/columnar/materialize/charge.rs#L39) · [列输出路径](https://github.com/wegamekinglc/calc-flow/blob/de20f8216c8dcb1bfce3e3811c3884a3080157df/crates/calc-flow/src/operator/asof/columnar/materialize.rs#L97)

本次结论限定为该托管单核、单工作线程与这一默认负载。此前完成的本机四线程 #747 对照使用另一个候选；相同合并树的本机四线程对照因换页中止。[四线程证据](https://github.com/wegamekinglc/calc-flow/tree/babbbc10c869fdc3a682a348e78c67be406e5f14/benchmarks/evidence/20261125-asof-watermark-phases/merged-main-10m)

## 离线复算

在固定协议提交 18be04ec 的仓库及其锁定基准依赖环境中，令 CALC_FLOW_REPO 为仓库绝对路径，在本报告目录执行以下命令。复算会校验原始 ZIP 的 SHA-256、自动解包并重新生成 analysis.json。

```sh
PYTHONPATH="$CALC_FLOW_REPO/benchmarks/evidence/20261125-hosted-asof-watermark:$CALC_FLOW_REPO" python replay.py .
```

[逐对阶段统计](analysis.json) · [原始附件](original.zip) · [复算程序](replay.py)。原始附件 SHA-256 为 `230e2f5c7745549f60f1e591f33c707ff375d099aaca85cbab12cef01541c641`。
