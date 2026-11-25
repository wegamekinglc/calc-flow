# 已安装 ASOF，单个 Tokio 工作线程与进程可见 CPU

使用 45623973262 官方全量运行的同一对 wheel，11.14 为 7012b5b4，候选为 de20f821。左右各 100 万行，64 个键，默认托管 ASOF，单个 Tokio 工作者。两组在同一台托管虚拟机上依次执行，先允许四个逻辑 CPU，再绑定一个物理核，顺序固定。每组内部交替对照发布版与候选，两轮各十对正式观测。计时从就绪输入到结果 Arrow 表，物理序列、投影及 EOF 终态在计时结束后校验。[登记协议](https://github.com/wegamekinglc/calc-flow/blob/7f3548e6b1b54da66cdb41f9978b89454d15f230/benchmarks/evidence/20261125-hosted-asof-watermark/wheel-affinity-protocol.md)

| 进程亲和性     | 发布 p50   | 候选 p50   | 发布 p95   | 候选 p95   | 两轮结论       |
|----------------|------------|------------|------------|------------|----------------|
| CPU 0、1、2、3 | —          | —          | —          | —          | 资源门限中止   |
| 仅 CPU 2       | 139.457 ms | 126.123 ms | 149.522 ms | 135.693 ms | 未确认回退     |

四 CPU 组通过预检，随后的资源观测记录到一个选中 CPU 的 SMT 同胞忙碌度达到 100%，超过维护的运行时门限。尚未收到通过校验的预热或正式观测，无法估计两个掩码之间的差异。该守卫将同胞的忙碌计入判断，记录不能区分测试自身和其他任务造成的负载。单核组共有 44 次正确性校验，两轮配对变化中位数为 −8.09% 和 −14.07%，区间分别为 [−11.67%, −4.82%] 和 [−39.98%, −6.12%]。沿用维护的判据。

官方完整分片中，同一单 Tokio 线程用例可见四个逻辑 CPU，发布与候选 p50 为 95.823 ms 和 151.142 ms，确认回退。这仍是该调度配置的有效结果。运行时会查询操作系统报告的可用并行度来配置 morsel 工人容量，具体耗时贡献尚待隔离。两次运行使用不同虚拟机。[官方分片](https://github.com/wegamekinglc/calc-flow/actions/runs/45623973262/artifacts/15035372449) · [运行时配置](https://github.com/wegamekinglc/calc-flow/blob/de20f8216c8dcb1bfce3e3811c3884a3080157df/crates/calc-flow/src/runtime/streaming/context.rs#L195)

[本次运行](https://github.com/wegamekinglc/calc-flow/actions/runs/45643368973) · [原始压缩包](original.zip) · [复算脚本](replay-affinity.py) · [逐组结果](summary.json)
