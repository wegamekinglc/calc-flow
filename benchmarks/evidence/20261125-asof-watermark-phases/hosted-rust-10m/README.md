# Rust 原生作业的千万行 ASOF 水位分解

两侧各 1,000 万行、64 个键。使用单个物理核，对照 11.14 发布版 7012b5b4 和合并后的 de20f821。分别测量普通构建与仅加水位计时的构建。每组预定两轮，每轮十对正式观测，每轮另有一次双臂预热。

每次验证全部投影输出行的 sequence、value、物理顺序和作业结束后的状态。分段记录 finalize 调用的墙钟时间，包含其中的异步等待及发送背压；入口更新进度、记录重放以及发布版在进入回调前的淘汰等待均在该时钟之外。

原始实验在独立 GitHub 托管主机执行。[运行](https://github.com/wegamekinglc/calc-flow/actions/runs/45627258042) · [冻结的协议与源码](https://github.com/wegamekinglc/calc-flow/tree/18be04ec374b9318b6efd29afecedb26b2e6d1de/benchmarks/evidence/20261125-hosted-asof-watermark)

## 无插桩

该组未完成。已接受的预热及正式观测保存在复算文件中。资源不满足门限或未形成完整样本的组不计算分位数。

`ExceptionGroup('unhandled errors in a TaskGroup', [ValueError("sparse changelog resource gate: {'cpu': 2, 'cpu_busy': 98.0, 'siblings': [3], 'sibling_busy': 6.0, 'host_busy': 50.625, 'available_bytes': 15130476544, 'major_faults': 0, 'swap_pages': 0, 'swap_in_pages': 0, 'swap_out_pages': 0, 'suitable': False}")])`

接受 2 次预热、9 对完整正式观测。

## 开启分段计时

该组未完成。已接受的预热及正式观测保存在复算文件中。资源不满足门限或未形成完整样本的组不计算分位数。

`ExceptionGroup('unhandled errors in a TaskGroup', [ValueError("sparse changelog resource gate: {'cpu': 2, 'cpu_busy': 100.0, 'siblings': [3], 'sibling_busy': 4.0, 'host_busy': 50.175, 'available_bytes': 15244431360, 'major_faults': 0, 'swap_pages': 0, 'swap_in_pages': 0, 'swap_out_pages': 0, 'suitable': False}")])`

接受 2 次预热、5 对完整正式观测及下一对中的一个观测。

## 离线复算

在固定协议提交 18be04ec 的仓库及其锁定基准依赖环境中，令 CALC_FLOW_REPO 为仓库绝对路径，在本报告目录执行以下命令。复算会校验原始 ZIP 的 SHA-256、自动解包并重新生成 analysis.json。

```sh
PYTHONPATH="$CALC_FLOW_REPO/benchmarks/evidence/20261125-hosted-asof-watermark:$CALC_FLOW_REPO" python replay.py .
```

[逐对阶段统计](analysis.json) · [原始附件](original.zip) · [复算程序](replay.py)。原始附件 SHA-256 为 `1b6cc0048e2efcf8dffd06576992821a93788f1861c6b363459fe33552652f26`。
