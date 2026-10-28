# DAL-417 当前版本补测与归因约定

本轮启动时固定 `main@b9a10d4fa4b109b810d2f5839951912b92e597f8`，
calc-flow 版本 2026.10.27；Arroyo 继续使用 v0.15.0 / `4ef15f6`，
二进制 SHA-256 为 `558b28b4c2aa3153500308684fd121dea2f040b5599d9f3342830834c24a319f`。
历史 `dab98a5` 和 #494 的文件、数据及结论保持原样，不把跨日期差异归因为版本优化。
当前源码/lock 已变化，不能将历史 native 冒充当前构建；新 release 构建时间单列。

原测量使用 TableExpr.stream，内部构造 Program(engine="streaming")，
是持续 StreamingRunner 加表达式/原生窗口节点，没有 SQL 节点。
本轮两引擎使用相同表达式图：engine="streaming" 的持久 native job，
与 engine="sql" 的有限 collect_async。将有限计算按每个输入 frame 调用是应用适配，
不是将 SQL engine 变成原生持续流。另记录预编译 batch plan 的对照与实际 DataFusion 计划。

固定五个 workload，不做全矩阵或全仓扫描：

1. `live-512`：原 32,768 行投影/过滤，512 行帧，CF streaming / CF sql 每帧 collect / Arroyo。
2. `live-8192`：同一有序行集和 JSON 字节，合并成 8,192 行帧；相同三组，隔离调用频率影响。
3. `arrow-full`：同一行集一次 Arrow 表，CF streaming / CF sql collect / CF sql 预编译计划。
   这是明确的有限输入计算对照，计时不包含网络和 JSON，不称 Arroyo 批处理结果。
4. `arrow-512`：同一 Arrow 行集切成 64 批，仍是上述三组；隔离每批编译/执行和调度成本。
5. `window-live-512`：原四秒 tumbling SUM/COUNT 跨两个帧，CF streaming / Arroyo。
   SQL engine 对同一原生窗口声明的拒绝必须保留；有限全表 SQL 可以对齐最终值，
   但不等价于 watermark 提前输出，不参与原生窗口吞吐比。

每 workload 固定 5 次每组的新进程观测。三组交替 ABC/CBA，二组 AB/BA；
同 workload 对应的输入哈希、顺序、CPU affinity 和 correctness 必须一致。
有效数据与 #494 相同：16 key，x=(id*17)%101-50，512 行推进事件时间两秒；
live 均追加 512 行过滤 marker，窗口由显式进度关闭，保持输入开放直至结果校验。
控制/观察进程 affinity 0-1；所有引擎子进程 affinity 4-7、Tokio 4 threads，Arroyo -p 1。
同机内存可用资源，不设单独硬上限；保留主机负载与采样 RSS，不能称 CPU 独占。

正式样本不启用 cProfile/callback tracing。另固定 8 次诊断：live-512/8192 的两种 CF
路径、arrow-full 的预编译 SQL、arrow-512 的预编译 SQL 和 streaming、窗口 streaming。
诊断时间与正式时间分开。使用显式 JSON 解码、Arrow 转换、JSON 输出、线程派发阶段计时，
Python cProfile、现有 native callback profiling、DataFusion 的计划/阶段 metrics，
以及 finite plan reuse 对照。没有 perf 工具时记录实际缺口，不伪造 Rust 内核采样。
Native/异步重叠阶段不得简单相加；累计 CPU/回调时间不能当端到端独占墙钟。

总测量预算 600 秒，包括 fixture、预检、正确性、启动、正式样本、诊断、语义 probe 和清理；
不包括构建、编写脚本和阅读源码。不得为有利结论追加重复、规模或重新挑选场景。
构建复用同机原 Cargo cache，默认 file feature、release/thin-LTO、locked dependencies；
每个 worker 核对实际载入 native hash，保留编译源 SHA 与完整命令/日志。

live 端到端仍从首 send 到父进程读到末条结果；Arrow 场景从源放行/首有限调用到所有 Arrow
输出获得，不含 fixture、启动、oracle、JSON 写盘和退出。Arrow stream 在计时前建立 job，
但首批惰性初始化仍在计时内；另记录首批与后续批时刻。两边 checkpoint 与旧报告边界一致。
所有结果精确核对整数 oracle，记录吞吐、P50/P95/P99、范围、RSS、输入/输出计数。
只做自定义跨引擎描述性比较，不冒用仓库 revision regression gate。

方案依据实测阶段占比和分层对照给出 Amdahl 上限；区分已验证瓶颈、重叠/残差及待验证假设。
每项列位置、证据、拟改动、收益区间/上限、工作量、兼容风险和验收指标，不实施产品优化。
