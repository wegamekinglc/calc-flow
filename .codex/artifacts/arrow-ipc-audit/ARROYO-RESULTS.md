# DAL-417：真实 Arroyo Arrow/Batch 邻近对照补测

本次接续 #511 `c4dc6b58`，已取得真实 Arroyo 的 Arrow 输入/输出转换结果。
两个转换场景各五对 AB/BA，共 20 次新观测、327,680 条输出，列名、全部值和原顺序均通过。
整表和 512 行小批的 **Arroyo/CF Batch 核心吞吐比**成对中位数分别为 **1.363×**
和 **8.821×**。这是短 burst、已备原生 Batch 到完整原生 Batch 的描述性比较，
包含实际调度/绑定回调，不能称裸算术内核、长期稳态上限或可直接兑现的优化收益。

窗口小批的单次语义探针也已通过：512 行结果、列/值/顺序、65 个实际返回/广播的
watermark 完全一致，两个 job 都在源仍开放、EOF 前交付完整 IPC 结果。
窗口没有预记五对性能样本，**不参加性能排名**；整表窗口未扩测。

原 CF 两引擎的 40 次观测及 [benchmark 计时审计](BENCHMARK-AUDIT.md) 保持有效，
未整套重跑。[原阶段报告](REPORT.md) 中的构建超时/能力缺口是当时的记录，由本文补足，
旧数字及 #494/#510 不回写。本次没有 calc-flow 产品、维护中 benchmark 或门禁变动。

## 版本、构建与资源

CF 产品继续固定启动时 main `b5ae227e48ac9bb8a81583a23485e2c6064cee52`，2026.10.27；
源码/lock/toolchain 与兼容 wheel 的产品基线一致。复用 sealed native
`a0708fcf5fa84a1cf2c5347a66b73680d7426a3313746a1ef99519d5b2c7bfb8`，新 CF 构建为零。
Arroyo 固定 v0.15.0 / `4ef15f6336a596651f10cf27c1ee29c29bf81577`，release + debuginfo；
完整 IPC/观察补丁 `arroyo-ipc-observed.patch` SHA-256：
`7ed240bd3907b79ccfae007e2a3234f8fcfcfe30adf2a46d85413f57e6d80b15`。
实际运行 binary seal `40588f3d036faf05520103a149a1b5e46680ace8e4e98b68c85082a1cdf68cb0`，
1,374,457,408 bytes，不随研究数据提交。CF DataFusion 55、Arroyo 的固定 fork 48.0.1，
是各自已锁定的依赖，并非相同 DataFusion 版本的裸算子对照。

同一 WSL2/i9-13900HX 主机、32 logical CPU、约 31GiB RAM；worker CPU 4–7，controller
0–1，Tokio 4 threads，BLAS/Arrow helper 单线程，Arroyo embedded scheduler、`-p 1`。
Python 3.13.9、PyArrow 24.0.0、NumPy 2.5.4。没有保证 CPU 独占、锁频或隔绝宿主机负载。
所有正式观测前已停止本任务构建及私有构建数据库；原主机/依赖证据继续保留，新快照标明采于结束后。

必要增量构建预记累计硬预算 **1,800s**，六条同步命令累计 **717.184s**；包括失败及
官方 webui 构建，完整 UTC 起止、命令、退出码/日志在原始包 `build/attempt-*`：

| 尝试 | 秒      | exit | 实际结果                                                  |
|------|---------|------|-----------------------------------------------------------|
| 1    | 195.472 | 101  | API build script 的在线 SQL codegen 需要可访问 PostgreSQL |
| 2    | 24.495  | 101  | 隔离数据库就绪；缺少真实 webui/dist                       |
| 3    | 167.675 | 0    | 官方锁文件、pnpm 9.7.1 的完整 frontend build              |
| 4    | 80.734  | 0    | 首个真实 IPC release 可运行                               |
| 5    | 123.667 | 0    | 加入源开放门控、分段时钟及结果观察                        |
| 6    | 125.143 | 0    | 对结果后的空 batch 加明确 guard，最终适配构建成功         |

继续使用已通过 check 的 gnu17/CMake compatibility 配置，没有改锁文件、降低 release
优化级别或伪造 UI/SQL codegen。PostgreSQL 18 使用任务私有目录、loopback 47836、
独立 dal417/arroyo_build 和固定源码 25 条迁移；未写现存数据库，现已 stop exit=0。
717.184s 是本次受限增量命令墙钟累计，不是冷构建总时间，也不包含旧阶段 440s 超时。
源码调查、私有数据库准备和人工分析不冒充编译时间。原阶段的计时证据缺失仍保留。

正式采集 **62.434s**。共享 600s 测量账本累计 **235.220s**，其中包含给前阶段保守预留
的 120s，以及本轮所有预检/失败和窗口 probe；不是声称前阶段精确耗时恰为 120s。
固定五对，不因离散度补样。相邻 invocation 之间间隔 0.266–9.712ms，完整 AB/BA 顺序及
每对时间在 `results/pairs.json`。旧 CF 样本没有混入这些配对。

## 实际调用链与计时契约

相同 32,768 行、16 key，`x=(id*17)%101-50`，条件 `id>=0 AND k<8`，输出 `id,y=x*2+1`。
每场景两端使用完全相同的 Arrow IPC bytes/hash 和 record batch 边界：1×32,768 或 64×512。
每次结果 16,384 行，按原顺序严格比较，不排序掩盖转换的输出差异。

CF 为 `Program.engine='streaming'` 的融合表达式图，`compile_stream → StreamingRunner →
已备 Data(Batch) → Python SourceBinding → native expression/DataFusion → Python SinkBinding`。
不是 SQL 节点，也不是有限 SQL collect。Arroyo 为真实 `arroyo run -p 1` SQL 编译图；
输入由修改后的 single_file SourceCollector 交给真实运行时。实际执行的 program protobuf
从各 job 的 SQLite 只读导出，随包保存 `executed-program.pb/.txt`：转换图有 ConnectorSource、
ArrowValue → ExpressionWatermark → ArrowValue 链、ConnectorSink。没有直接调用一个替代 DF 算法。

固定源码证据：Arroyo `crates/arroyo-operator/src/context.rs:302` 原样转交 RecordBatch；
`crates/arroyo-worker/src/arrow/mod.rs:49,316` 从真实 physical plan 构造 executor，并在每批
reset/execute；CF `crates/calc-flow-python/src/continuous.rs:568,646` 每批调用 source/sink 的
Python async bridge；`crates/calc-flow/src/operator/expression.rs:271` 调用实际 DataFusion。
补丁只增加隔离 single_file Arrow 接入/观察和可选 watermark trace，不替换算子。

原生用户 format probe 的 arrow/arrow_ipc/ipc 拒绝仍成立；这里验证的是**基准适配后的真实
引擎能力**，不宣称 v0.15.0 发布包已提供公共 Arrow source/sink。DDL 中 format=json 仅是
connector 声明占位；`.arrow` 分支避开逐行 deserializer/serializer 的数据调用，真实行
从 Arrow StreamReader 到 SourceCollector，再到 Arrow StreamWriter；图/时钟 JSON 仅为元数据。

各阶段明确分列：

- 准备：fixture/输入编码在父进程、计时前；CF compile/job ready、Arroyo SQL/物理图准备在 core 前。
- 输入 IPC/绑定：CF 在 ready 后从 pipe 解码，再 Batch.from_pyarrow/创建 Cursor；Arroyo 在
  source-ready 前读 file IPC/cast，补内部 `_timestamp`。Arrow 表不经 JSON。
- **core**：从放行已备原生 Data/RecordBatch，到所有 native Batch 完整进入持有的 sink 集合；
  两端均包括真实源/算子/通道/回调，CF 首批 lazy DF 初始化可能仍在内。
- 输出绑定：CF Batch.to_pyarrow + concat；Arroyo project 去掉内部 `_timestamp`，均在 core 后。
  输出整数列完全匹配；原生 non-nullable schema 与 CF nullable metadata 差异不改成值差异。
- 输出 IPC：StreamWriter EOS/close、file rename 或 pipe flush 后，父进程 read_all 全部结果。
  **observer 直到完整 Arrow 表可消费才结束**，随后全量校验/保存/结束源/退出，不提前在 enqueue 停钟。
- 本地文件输出未 fsync，不测持久性；输入/输出都已完整消费后才允许源 Final/EOF。
  checkpoint 未纳入，Arroyo 1h interval 的这段短观测没有实际 checkpoint，CF checkpointing=False。

CF observer 包含输入 pipe/解码；Arroyo observer 在已解码 input 的 gate 开始。Arroyo 的启动
包括 embedded API/SQL compiler/worker/database，CF 为 Python import/声明/job。
因此 core 是有用的 native Batch 应用执行边界，**不是所有准备位置都一致的内核竞赛**；
observer、IPC/绑定和启动只作为各路径观测，不给跨引擎速度排名。阶段存在并行/调度重叠，
不能相加中位数或把 observer−core 称为纯传输 CPU。
外部适配还使用控制文件 gate/完成标记：start 等待在 core 前，source 的 fed 标记写入及
等待 finish 的文件检查可能与下游执行重叠，未分离其独占 CPU/调度成本。它们不传表数据，
但也是不能将 core 叫作严格无 I/O 裸计算的原因；正式观测未开启 watermark trace 写入。

## 新成对结果

五次中位数（全部 min–max）；吞吐分母为输入行数，延迟为整次完整结果的 core wall time。
P95 若使用 nearest-rank 等于这五次最大值，不提供逐行延迟或窄置信区间。

| 场景           | 路径   | core ms（min–max）      | 百万输入行/s（min–max） | 进程树峰值 RSS MiB（min–max） |
|----------------|--------|-------------------------|-------------------------|-------------------------------|
| transform-full | CF     | 1.718（1.627–1.748）    | 19.069（18.750–20.141） | 201.992（199.930–202.340）    |
| transform-full | Arroyo | 1.282（1.173–1.401）    | 25.563（23.395–27.926） | 125.383（121.266–127.992）    |
| transform-512  | CF     | 10.866（10.080–10.981） | 3.016（2.984–3.251）    | 200.551（199.340–201.578）    |
| transform-512  | Arroyo | 1.215（1.123–1.305）    | 26.961（25.112–29.191） | 122.918（121.336–126.297）    |

每对 `CF core ms / Arroyo core ms` 即 Arroyo/CF 输入吞吐比：
整表 **1.363×（1.162–1.489）**，小批 **8.821×（8.406–9.680）**。
仅本轮主机/负载/观察契约下的五对结果，不满足仓库两轮成对 CI 回归分类门禁，
不作 regression/improvement 判定；与旧 JSON 以及旧 Arrow API core 的计时边界不同。
CF 新 core 不含输出 to_pyarrow/concat，不能把旧 19.84ms 到新 10.87ms 称作产品优化。

下面只列阶段中位数（ms）；每项完整五次范围在 `phase-summary.json`：

| 场景           | 路径   | 启动→ready | input IPC wall | input binding | output binding | output IPC wall | observer wall |
|----------------|--------|------------|----------------|---------------|----------------|-----------------|---------------|
| transform-full | CF     | 297.790    | 1.753          | 0.338         | 0.172          | 0.349           | 4.309         |
| transform-full | Arroyo | 733.526    | 0.401          | 0.118         | 0.004          | 0.274           | 4.671         |
| transform-512  | CF     | 252.878    | 1.647          | 1.142         | 0.476          | 0.486           | 14.768        |
| transform-512  | Arroyo | 737.831    | 0.469          | 0.221         | 0.015          | 0.458           | 3.650         |

CF 声明/compile/job-ready 子阶段中位数整表 5.113ms、小批 4.928ms，已含在 launch→ready，
不是可再相加的成本。Arroyo 当前未细分 compiler 与 embedded 服务启动，不把差值当 planning。
RSS 为约 10ms 进程树采样，全生命周期包含启动/结束；Python/PyArrow 与 embedded 服务常驻
内容不同，短峰可能漏采；不是裸算子 allocation，也未从这些数字证明状态内存复杂度。

Arroyo 成功交付后由 harness 显式 SIGINT 停常驻 CLI，exit=128 是上游
`crates/arroyo-server-common/src/shutdown.rs:235` 的 Signal 分支；CF 自然完成 exit=0。
两端每次都有完整消费/关闭及终止证据，不把已停止的 CLI 当运行失败，也未遗留任务进程。

## 窗口已验证范围与局限

采用原 window-512 同一 32,768 行 + 512 行过滤 marker，四秒 tumbling SUM/COUNT、16 key。
CF 每批实际 SourceProvidedWatermarks 与 Arroyo `WATERMARK FOR ts AS ts-1 microsecond`
广播 trace 各 65 项逐整数微秒相同；原始 trace/时钟/IPC 均保留在 `window-probe-v2`。
每批 ts 常量，所以 Arroyo 的 batch minimum 与 CF 的 batch maximum 相同。
源不会因文件读完就 Final：两端直到父进程完整读取 Arrow 结果后才释放 finish gate，
marker 推进最后窗口，两个 512 行输出所有值/顺序及四秒 window_end 宽度通过。
timestamp ns/us、时区与 COUNT uint64/int64 的安全规范化只在校验外层做，原 IPC 未改写。

这证明该受限轨迹可等价，不证明整表混合时间、乱序/空闲/迟到、更大状态、恢复/事务语义。
whole-window 的单批 min/max watermark 可能不同，未伪造对齐。只有一次语义 probe，
不以其时钟做窗口吞吐排名。前次 `window.start` SQL 别名解析失败的原日志保留，后改为
上游 TUMBLE 子查询 `w.start/w.end` 后真实运行成功，不能将 SQL 书写失败算功能缺失。

## 更新归因与分阶段方案

**已验证：** 移除 row JSON 后仍有小批差距；控制输入行数不变，仅分成 64 批，CF core
1.718→10.866ms，Arroyo 1.282→1.215ms。CF 本轮状态证明 64 次真实 source/operator/sink
交付、无错误/blocked send；operator processing 累计中位数 **2.902ms**（整表 1.324ms），
sink write await 累计中位数 **7.966ms**（整表 0.121ms）。计时位置为
`runtime/streaming/sink_task.rs:582` 和 `metrics.rs:370`，await 包含调度等待，多个阶段重叠。
这把下一步调查收敛到每批执行/回调，**不等于 7.966ms 全是 GIL、复制或可独占消除时间**。
Arroyo 实际图及 StatelessPhysicalExecutor 的跨批已建 plan 复用是机制证据，尚未单独干预归因。

**P0，计时契约/报告修正，0.5–2 人日。** 沿已有审计的 support、symbolic_support、report
及 CI 指标位置，标明 API、startup、planning、消费、codec 和 lifecycle 边界；此项为测试/报告
维护，不是产品提速，计算收益上限记 0。风险是 workload identity/历史报告可比性；验收为
新 contract/provenance 与最小 probe 对齐、旧基线保持、CI 不误把未消费结果记为计算完成。

**P1，持续小批回调/调度定向剖析，约 1 人日；证据成立后原型 3–5 人日。** 热点位置为
`continuous.rs:568,646`、`runtime.rs`/`runtime/dispatch.rs` 及 source/sink tasks。先对 source.next、
sink.write、native 算子和 OS/GIL wait 做符号/队列分段，再比较已备 native source/sink 与
Python callback 层，禁止换掉算子算法。拟改动只在证据允许后研究回调批量通知/原生 binding。
10–30% core 下降是候选目标，当前没有可信独占份额上限，8.821× 不是可达收益预算；仅消除
外层 IPC 不会改善已排除 IPC 的 core。兼容风险高：backpressure、取消、owner 引用、慢 sink、
watermark/barrier/EOF FIFO。验收须输出逐值/顺序、恢复兼容及有界资源同时通过，并按现有
两轮成对性能 gate 评估，不能用移走 Python sink 功能换取名义提速。

**P2，重复规划/模板，先 0.5–1 人日分段定位，候选实施 5–8 人日。** 位置为 CF
`operator/expression.rs:271`、`datafusion.rs`、有限 API 的 `python/calc_flow/compute.py`。
Arroyo prebuilt physical plan 与 CF 小批 operator 2.902ms/源码路径及 #510 的同 native 诊断
支持研究复用，但新观测没有分离 parse/logical/physical/execute，不能把全部差距归为规划。
先复用已公开 compiled plan API 的适用场景，再研究不可变、受预算约束的 DF template；目标
eligible planning cost/count 下降至少 50%，总收益受其独占 critical-path 份额约束，尚未量出
该份额上限。高风险边界是 concurrent run、volatile/UDF snapshot、schema、取消与内存预算；
验收要保持 caller Batch 不变和 fresh execution state，不缓存可变输入/状态。

**P3，窗口状态/内核，约 1 人日符号 profile 后再决定 2–5 人日原型。** 小窗口语义已补齐，
但本轮没有新五对性能/大状态/偏斜证据；继续用原 CF 分层资料作为历史线索，不能从转换
8.821× 外推窗口。若 profile 指向 hash/key/state 才设计实现，收益估计和上限目前未定，
未知不是零成本；风险为 state layout、lineage/fingerprint、checkpoint、late handling。
验收需真实等价 watermark/EOF 前消费及恢复数据完整性，保留现有 complexity/performance gate。

JSON 优化继续从核心产品队列移除。真实 CPU flamegraph、独占 GIL/调度比例、Arroyo 冷启动
compiler 拆分、窗口成对排名均未交付，本报告不以替代指标补造这些结论。

## 交付与复核

[复现说明](ARROYO-REPRODUCE.md) 给出实际命令、固定版本、构建条件、离线复核和缺失范围。
`data/arroyo-extension/` 独立封存原始 IPC、完整 clocks/RSS、失败/预检、计划、构建及预算日志。
旧 `data/DAL-417-arrow-raw.tar.gz` 的 SHA-256 仍为
`13fa0f9e7a64e45c926daf2f2b272547ae6abe2d21451e899fb5f6cfd6d63995`。
新增产物在同一 PR #511；不合并，由协调器安排聚焦审查及必要后续步骤。
