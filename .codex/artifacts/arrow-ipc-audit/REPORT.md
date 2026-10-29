# DAL-417：Arrow/IPC 重测、benchmark 计时审计与优先级更新

> 接续说明：本文保留首阶段数字与当时的构建状态。真实 Arroyo Arrow/Batch 运行、窗口
> 对齐探针及新五对比较已在[补测报告](ARROYO-RESULTS.md)交付，不与本文样本混合统计。

本轮取得了 **CF 两种引擎的 Arrow 核心 API/完整 IPC 往返结果，以及维护中 benchmark
集合的逐组计时审计**。没有取得 Arroyo 同口径的可运行 release/窗口语义证据，因此没有
新的 Arroyo/CF 纯计算倍数排名。#494/#510 的旧数字完整保留，只适用于原 JSON 集成出口，
不再用其中的 JSON 编码热点指导核心产品优化。

核心结论：小批 SQL 的声明准备/重复 DataFusion 规划、持续路径的批调度仍值得优先处理。
维护中的主矩阵未发现把逐行 JSON 出口计入核心 seconds 的证据；它通常计完整 API，
包含正常的规划、包装、同步或生命周期成本。不能把所有 benchmark 都称为裸算子计算。
完整逐组矩阵及具体修正建议见 [计时审计](BENCHMARK-AUDIT.md)。

## 固定版本、资源与实验契约

启动时 main 为 `b5ae227e48ac9bb8a81583a23485e2c6064cee52`，CF 2026.10.27。
它相对上轮构建源码 `b9a10d4` 的差异仅为 #510 研究文件；产品源码、Cargo.lock、
toolchain 和构建配置相同。本轮复用兼容 release wheel，没有冒充重新构建。
wheel SHA-256 为 `c7353872880b2c511aab3d290aff181795809e3818007159379d0f9a02f07753`，
实际 import/native seal 为 `a0708fcf5fa84a1cf2c5347a66b73680d7426a3313746a1ef99519d5b2c7bfb8`。
两种 CF 引擎、所有 worker 核对同一 seal；原构建 9m46s，本轮 CF 新构建用时为零。

Arroyo 继续固定 v0.15.0 / `4ef15f6336a596651f10cf27c1ee29c29bf81577`；原 release
binary SHA 为 `558b28b4c2aa3153500308684fd121dea2f040b5599d9f3342830834c24a319f`。
原生接口 probe 使用该 binary；最小 IPC 适配仅修改隔离的 upstream source/sink，不替换
查询算法，不修改 calc-flow 产品。

同一台 WSL2/i9-13900HX、32 logical CPU、约 31GiB RAM，controller 固定 CPU 0–1，
worker 固定 4–7，Tokio 4 threads，Arrow/BLAS helper 单线程，DF 默认 partitions=1。
不保证 CPU 独占、锁频或无 swap。CPython 3.13.9、PyArrow 24.0.0、NumPy 2.5.4，
依赖 freeze、主机/负载/内存及兼容构建核对保存在 raw provenance。
正式测量开始前已结束本任务的所有构建；不把编译争用计入正式性能。

预记四个场景、每 arm 五次：整表/64×512 转换各有 streaming、SQL collect_async、
SQL 预编译 native plan 三路径；整表/64×512 原生窗口各有 streaming，一共 40 次。
Arroyo 在可运行且语义对齐时才有预留观测，不以替代算法填充单元。

首轮 40 次完成后发现 worker 为取得输出行数，在 core 前、IPC 往返开始后生成完整
oracle；核心时钟未含它，但往返违反预记的准备边界。已保留首轮全部 Arrow 数据和八份
精确源文件，标记排除，仅按原五次/arm 做一次修正采集。最终统计只用修正后的 40 次，
不混合首轮。详见 [计时修正记录](MEASUREMENT-CORRECTION.md)。
首轮采集 15.65s、修正采集 24.25s；预检第二次 2.91s、格式 probe 0.66s。
首次失败预检有 60s 上限，其他定向 probe/离线核对均为秒级，合计远低于原 600s
测量预算。没有因整表噪声增大而追加样本。构建时间另列，编写报告/读源码不计入采样预算。

## 数据、两引擎及结果语义

32,768 行、16 key，`x=(id*17)%101-50`，每 512 行事件时间增加两秒，值与历史输入一致。
本轮直接用 NumPy/Arrow 生成 fixture，不经过 JSON。整表和小批是同一行序列，
输入 IPC 中的 record batch 边界不同；每个场景所有 arm 的输入字节/hash 相同。
转换仍为一个融合表达式图：`id >= 0 AND k < 8`，输出 `id, x*2+1`。
`Program.engine` 选择执行生命周期，图中不是 SQL 节点。

窗口为四秒 tumbling SUM/COUNT、16 key，追加 512 行过滤 marker，发送每批最大 ts−1µs
的显式 watermark，结果在 native source EOF 之前齐备。它是有限输入喂给持续 job，
不是长时间稳态吞吐。整表模式的 watermark 轨迹比 64 小批更粗，最终值相同，
不能把两者当成相同在线输出节奏或直接推广为实时延迟收益。

SQL engine 对同一原生窗口声明的拒绝及有限全表 GROUP BY 最终值对齐证据，复用 #510
已验证 probe：产品、lock、native seal 均未变化。有限 GROUP BY 没有原 watermark
finality，逐帧查询也没有原生跨批窗口状态，因此本轮不伪造 SQL 窗口排名。
首次预检还暴露 COUNT 的 uint64 与 oracle int64 元数据不同；现在在 timer 外做安全
整型 cast 后逐值比较，实际 COUNT=64，未改写结果，未忽略溢出或算术差异。

数据链为 `Arrow IPC input → PyArrow/Batch → 真实 CF 图 → 完整 Arrow Table → Arrow IPC output`。
所有最终 496,640 条输出逐值检查；原始 IPC 含真实 schema、全部列和值。
JSON 只保存命令、配置、图、时钟和摘要，不是表的数据出口。

## 计时边界

- fixture 生成、输入 IPC 编码：父进程开始前准备并单列。
- 编译/启动：Runtime、声明、graph 导出、native plan 编译或 continuous job/source-ready
  在发送输入前完成。SQL collect 的每次 lowering/compile 是 convenience API 工作，仍在
  其 core 时钟内。预编译 native 图不等于跨调用缓存 DataFusion 物理计划。
- 输入 IPC：worker 读完 stream EOS、解码所有 record batches；该 wall span 可能从父进程
  发送前开始等待，包含等待/pipe/IPC 解码，不能称纯 decoder CPU。
- 输入绑定：创建 Arrow 表视图、窗口 watermark 等测试源事件；单独计时。窗口小批约
  3ms 的绑定含 65 个控制事件准备，不计作窗口内核，也不藏进 IPC 编码时间。
- core：从预备输入放行/首有限调用，到全部 Arrow 输出及 concat 完成；包含实际源/通道/
  算子/状态/PyO3/正常 Arrow 包装，SQL 正常规划也在内。结果完全可消费才停钟。
- 输出 IPC：真实 encode/write/EOS/flush 独立 span；父进程同时读取、解码并取得完整表。
  round-trip 从首发送到父进程完整 Arrow 结果可消费；保存原始文件、全量校验、退出/取消在外。
  接收器记录原 IPC bytes 有测试观察成本，RT 不是零开销内核计时。
- 流的首批 lazy DF 初始化可能计入 core，EOF/关闭不计；checkpoint disabled，不比较恢复。
  RSS 是约 10ms 采样的进程树全程峰值，包含启动/退出，短峰值可能漏采，不是 allocation。

core、IPC span 和两进程的调度可能重叠。不能把独立中位数相加，或把 RT−core 叫作
纯 IPC 传输成本；原始 ns 时钟允许逐次核对先后。五次 P95 用 nearest-rank，等于该组
max，是 invocation 完成时长而不是逐行延迟。未把短 burst/有限输入吞吐称为长期流式上限。

## 最终结果：五次中位数与完整范围

| 场景           | 路径      | core ms（min–max）      | IPC 往返 ms（min–max）  | core 百万行/s | RSS MiB（min–max）         |
|----------------|-----------|-------------------------|-------------------------|---------------|----------------------------|
| transform-full | streaming | 3.472（2.534–12.996）   | 6.607（5.104–16.099）   | 9.438         | 193.480（193.234–198.930） |
| transform-full | sql       | 6.076（4.023–9.885）    | 8.616（5.954–12.850）   | 5.393         | 193.461（193.383–201.320） |
| transform-full | sql_plan  | 3.412（2.562–29.259）   | 8.198（4.902–69.883）   | 9.604         | 197.035（193.379–201.188） |
| transform-512  | streaming | 19.842（18.838–29.265） | 22.528（21.445–35.188） | 1.651         | 201.363（201.059–203.426） |
| transform-512  | sql       | 74.594（68.900–86.289） | 78.484（72.105–89.340） | 0.439         | 201.328（200.848–201.688） |
| transform-512  | sql_plan  | 53.018（45.599–65.168） | 55.919（48.438–70.042） | 0.618         | 201.281（200.789–203.203） |
| window-full    | streaming | 2.609（2.414–2.764）    | 5.176（4.601–7.201）    | 12.561        | 200.996（200.855–201.453） |
| window-512     | streaming | 25.706（22.443–27.901） | 33.372（28.091–33.984） | 1.275         | 201.488（201.188–201.676） |

整表 SQL 预编译/collect 的成对 core 吞吐比中位数 **1.366×**，范围 **0.338–1.882×**；
往返比值为 **0.993×**，范围 **0.184–1.496×**。整表有明显噪声/长尾，不能可靠排序。
64 小批对应 core 为 **1.300×（1.121–1.892）**，往返为 **1.289×（1.084–1.844）**。
这些是本轮描述性对照，样本五次，不满足仓库两轮十对的回归判定要求，也不是产品优化收益。

| 场景           | 路径      | input IPC wall ms | input binding ms | output IPC wall ms |
|----------------|-----------|-------------------|------------------|--------------------|
| transform-full | streaming | 3.748             | 0.035            | 0.534              |
| transform-full | sql       | 1.946             | 0.027            | 0.517              |
| transform-full | sql_plan  | 2.137             | 0.027            | 0.459              |
| transform-512  | streaming | 2.583             | 0.131            | 0.639              |
| transform-512  | sql       | 3.042             | 0.124            | 0.603              |
| transform-512  | sql_plan  | 1.758             | 0.126            | 1.888              |
| window-full    | streaming | 2.185             | 0.565            | 0.121              |
| window-512     | streaming | 2.705             | 3.100            | 0.268              |

输入 span 可能含父进程发送前等待，输出 span 含 pipe backpressure；它们是 I/O wall 观测。
核心算法、结果规范化和 parent observer 不一样的路径，必须按这些边界解释，不能从
当前表推导 JSON→IPC 的因果加速：版本/形状/源与计时契约和 #510 旧 JSON 路径不同。

## Arroyo：实际接口与构建尝试及缺口

固定源码 `arroyo-rpc/src/formats.rs:360–408` 没有 Arrow/IPC 格式；真实 release CLI
分别尝试 `arrow`、`arrow_ipc`、`ipc`，三次均 exit=1、明确 unknown format。
内部 `network_manager.rs` 确实用 Arrow IPC，但带任务/路由/信号和 schema 上下文，不是
已暴露的用户 source/sink API。不能把它当可直接接入的标准 Arrow stream 接口。
[官方 connector 文档](https://doc.arroyo.dev/connectors/) 与该格式边界方向一致；版本能力
结论以固定源码和实际 CLI 拒绝为准，不以最新文档推断所有未来版本。

必要最小适配是在隔离 upstream 的 single_file source/sink 添加 `.arrow` 分支：用
StreamReader 读取/按 schema cast 后送入 SourceCollector，由真实引擎执行图；sink
保留 RecordBatch，完整输出后才编码 IPC。只有格式/观测边界变动，没有替代计算算法。
补丁、原始源码 ref、编译命令和日志都交付。

实际尝试依次为：

1. 原始 `cargo check --locked -p arroyo-connectors`：sasl2-sys 老式 C 函数原型与本机默认
   C 标准不兼容。初段 log 活动约 284s；未保存完整 process 起止。
2. `CFLAGS=-std=gnu17` 的适配检查在上一 run 因模型容量被终止，日志可见约 36s 活动，
   没有完整退出证据；接续 65.98s 后遇到 CMake 4 移除旧最低版本兼容。
3. 仅给旧 librdkafka 配置 `CMAKE_POLICY_VERSION_MINIMUM=3.5`，配合 gnu17，
   **cargo check 42.45s 成功**。这是类型检查，不是运行/正确性或性能通过。
4. 最后一次 `cargo build --release --locked -p arroyo --bin arroyo` 设置 440s 硬上限，
   440.12s、exit=124，尚未生成可运行 release。停止进一步构建；不通过减少优化级别或
   用 raw DataFusion 代替来交差。初段/中断的完整构建墙钟不全，不能虚报精确总和或
   已精确证明 900s 聚合构建预算；日志、已知时长和缺失项一并保留。

因此，Arroyo 本轮没有 Arrow 实际输出或新速度数据。适配也尚未验证窗口的开放源
watermark 轨迹；当前 file source 的 Final/EOF 不能冒充原窗口输出时机。缺口需要一个
可运行的封存 release adapter 和语义/完整结果核验之后才能补排名。本报告不据此声称
Arroyo 内部不能处理 Arrow，也不声称已完成对双方纯计算性能的比较。

## 新口径下的归因及分阶段优先级

**P0：先修正性能指标的解释和剩余测试适配。** 位置为 `benchmarks/support.py`、
`symbolic_support.py`、`scripts/benchmark_suite/report.py` 及 acceptance fixture export。
主矩阵没有逐行 JSON 结果出口；warmup DF metrics、reference/candidate 外层计时、RSS snapshot、
codec/存储/lifecycle 等必须分列。工作量约 0.5–2 人日，风险是 workload identity 和历史
报告兼容；收益是更准确的验收，不承诺产品速度。任何边界改变新增 contract，不重写旧基线。

**P1：有限小批使用公开编译复用接口，随后才研究 immutable DF 模板。** 位置为
`python/calc_flow/compute.py:149–180,283`、`pipeline/batch.rs:406–422`、`datafusion.rs`。
本轮小批 core 74.59ms→53.02ms，成对中位数 1.300×，支持 convenience 准备成本可见；
整表长尾不支持稳定收益结论。首先完善公开 `compile_batch`/Runtime plan 用法和绑定示例，
约 0.5–1 人日、低到中风险，验证 fresh-run、取消、不可变输入和 registry/UDF snapshot。
应用不可依赖基准为方便绑定而使用的私有 `_lowered_collect`。

重复物理规划是后续产品优化候选，不能把 native 图已编译误认为计划跨调用复用。
本轮最后一次查询 metrics 和调用链仍支持该机制；#510 已有同 native 的 Arrow 小批
诊断记录 64 次规划，保留为分层证据，不把旧累计耗时直接从新 core 中扣除。
模板复用约 5–8 人日、风险高：并发、volatile/UDF snapshot、schema/config、内存预算与
取消必须隔离。收益只能作用于重复准备部分，当前不能给无条件数字上限；旧诊断的
Amdahl 模型只对旧边界成立。候选验收为 eligible 小批 planning count/费用至少下降 50%，
再依仓库两轮成对 gate 判定收益，不以缓存可变输入/状态换速度。

**P2：小批异步调度/桥接优先于重写 Arrow 数据接口。** 位置为 PyO3
`runtime.rs` await bridge、continuous source/sink、`stream.py` 结果队列。
小批转换 core 19.84ms，native 各算子累计 processing 中位数约 3.15ms；窗口小批 core
25.71ms、各算子累计约 4.95ms。二者有重叠和 await，余项不是独占 GIL 或复制时间。
可做 native 已备 Batch source、通知/回调批量化的受限实验，保持 watermark/barrier/EOF
顺序、背压及 owning references。约 3–5 人日原型，风险高；10–30% 的 core 下降只是
待验证目标，现无可信独占份额上限。#510 裸 bridge 探针约 0.68ms/64 次不足以支持大规模
C Data Interface 重写。验收同时覆盖 Arrow 输出完整性、取消/重入/慢 sink 和有界资源。

**P3：窗口内核按 batch 形状重新评估，先带符号 profile。** 本轮 whole-window operator
累计 processing 中位数 0.844ms/core 2.609ms，小批为 2.121ms/core 25.706ms。
相对占比随边界变化，不能沿用 JSON E2E 的约 2.5% 上限。若假设该累计阶段都在关键
路径且可完全消除，粗略 Amdahl 模型上界分别约 1.48×/1.09×；不是独占计时或承诺。
先约 1 人日采集 native symbols/热点，再按证据决定 key/hash/聚合的 2–5 人日原型；
状态布局、checkpoint lineage/fingerprint、late handling 和恢复兼容风险高。当前 16 key
不能外推大状态/偏斜场景；验收只选实际热点和直接语义边界，不启动全仓优化扫描。

JSON 编码优化从核心产品优先队列移除。IPC codec、记录器和往返调度是独立测试/集成
成本，不能把优化观察器速度记作内核收益。当前没有 native flamegraph、独立 GIL/OS
调度干预，也没有长期稳态/大状态/恢复/事务输出证据。

## 交付与验证

交付目录 `.codex/artifacts/arrow-ipc-audit/`：报告、完整逐组矩阵、预记/修正记录、
可复现采集/传输/worker、静态 inventory 和真实引擎边界 probe、离线审计、Arroyo 补丁。
`data/` 的原始包包含最终及排除轮的全部 IPC 结果、原始时间/资源、精确脚本 hash、
图/DF metrics/native seal、构建与接口失败证据，不包含 native binary/wheel。
实际命令及离线重算见 [README](README.md)。本轮通过的检查及 PR/CI 状态在交付评论中
据实报告；不改变现有复杂度/性能门禁或历史证据，也没有实施产品优化。
