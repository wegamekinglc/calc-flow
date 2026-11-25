# ASOF 的 Rust 四线程整作业对照

11.14 发布版 `7012b5b4` 对照 #744 最终提交 `df01447e`。每侧 1,000 万行，分十批，有 64 个字符串键。两端使用四个 Tokio worker 和相同的四个物理核 2、4、16、20。原生 Rust 测量从投递首批已就绪数据前计至收齐输出并封装为 Batch。随后关闭源并验证作业终态。完整校验覆盖两侧接纳、输出各行与终态。

p50 为 711.068 → 1036.494 ms，p95 为 742.494 → 1083.438 ms。两轮配对判定为 `regression`。

此前 Python 安装包对照基于计费重构前的 `ade0e864`。其发布版到候选的 10M p50 为 632.36 → 969.47 ms。两个入口独立报告。[安装包对照](https://github.com/wegamekinglc/calc-flow/tree/f7915aee6d159f8a9f0d122a14a4fb757ca670d4/benchmarks/join-asof/20261125-asof-dense-charge-public-10m)

[复算记录](four-replay.json) · [原始数据与校验](managed-asof-10m-four-workers.tar.gz)，SHA-256 `24af30662d7194972d33097a19477bb1da7d709b2f54d663ca7628affb86694d`。
