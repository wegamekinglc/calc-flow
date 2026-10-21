# Arroyo 与 calc-flow streaming 对比

日期：2026-10-22。范围：DAL-417，源码/官方资料调研与本机真实流式集成测量。
本变更只包含调研、基准脚本及测量证据，未修改产品行为。
阅读与离线复核入口见 [README.md](README.md)，完整原始数据保存在 `data/`。

## 结论与适用场景

Arroyo 更适合把 SQL 查询作为持续服务部署：已有控制面、作业调度、分布式执行、
增量更新语义及较广的消息/湖仓连接器。calc-flow 更适合在 Python 应用中直接处理
Arrow 表和有状态特征，尤其是按实体的滚动计算、横截面运算和有界 backward ASOF。
两者都是真实的有状态流系统，不能把 calc-flow 当作只有跑批能力，也不能把两者的
普通 `GROUP BY` 当作同一种增量算法。

本次性能对比的对象是各自一条实际可用的 **WebSocket JSON → 计算 → stdout JSON**
集成路径。calc-flow 包含 Python/PyArrow 适配，Arroyo 使用原生连接器；这是应用选型
证据，不是公平剥离所有适配成本的纯算子排名。短时、单机、16 key、小状态的结果不能
推广为分布式、长期运行、Kafka/事务输出或故障恢复的性能结论。

在这条集成路径上，Arroyo 的 burst 吞吐较高：投影/过滤成对比值中位数 2.26×，
窗口 4.50×。paced 时两边均约 8.6–8.8 万行/秒，受生产者节奏限制，不能据此宣布
引擎性能相等。calc-flow 的采样总 RSS 约 205 MiB，Arroyo 约 217–226 MiB；
这些差异只适用于本次包、适配层和控制面组合。

## 固定版本与环境

| 项目        | 固定对象                                                                                 | 来源与边界                                                                                                  |
|-------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| calc-flow   | `2026.10.21`，`dab98a559314bd46d866ed7a6489abcdd3810afe`                                 | 任务专用 checkout；Rust 1.94.0，`maturin build --release --locked`，默认 file feature，安装 wheel 后运行    |
| Arroyo      | `v0.15.0`，`4ef15f6336a596651f10cf27c1ee29c29bf81577`                                    | 官方 `arroyo-linux-x86_64.tar.gz`；本机核实 `arroyo 0.15.0`；对应 tag 源码只读 checkout                     |
| 内部依赖    | calc-flow DataFusion 55.0.0 / Arrow 59.2.0；Arroyo DataFusion 48.0.1 / Arrow 55.2.0 系列 | Arroyo 还锁定上游 fork，详见双方 Cargo.lock；依赖不同，不是同依赖版本回归测试                               |
| 主机        | WSL2 Linux 5.15.167.4，i9-13900HX，32 逻辑 CPU，约 31 GiB RAM，64 GiB swap               | 同一主机顺序运行；没有 CPU 独占、锁频或禁用 swap，其他任务会造成竞争                                        |
| 资源        | 引擎子进程 affinity `4-7`，Tokio 4 worker threads，Arroyo `-p 1`                         | 生产者/观察者 affinity `0-1`；各引擎使用同样 4 个逻辑 CPU，未设置单独内存硬上限；不等同于已优化的并行度配置 |
| Python 适配 | CPython 3.13.9，PyArrow 24.0.0，websockets 15.0.1                                        | 完整已解析版本见 `requirements-resolved.txt`，native/wheel/binary SHA-256 见 provenance                     |

两边均为新进程、新状态目录，正式计时前已导入、声明并建立源连接；文件页缓存可能已热。
首批惰性规划/执行初始化仍可能在计时内；源连接不等同于完整 runner 就绪或 warm 稳态。
Arroyo 用上游发行二进制，无法从下载件独立重建其编译器/编译 flag 来源，记录二进制
SHA-256 与官方发布归属；不把“tag 对应”写成可重复构建证明。

## 测量方法

预先约定见 `PLAN.md`。四个 case 是：无状态投影/过滤、四秒 tumbling SUM/COUNT，
各用 burst 与每 frame 等待 5ms 的 paced 输入。每 case 五组相邻交替 AB/BA，合计
40 次正式进程运行。没有删掉异常慢样本，也没有用最小值替代重复分布。

输入为 32,768 行，字段 `id:int64, k:int64, ts:timestamp, x:int64`，16 个均匀 key；
`x=(id*17)%101-50`，每 512 行一个 NDJSON WebSocket binary frame。每帧事件时间
推进两秒。窗口覆盖两个输入 frame，必须跨批保留状态。末尾追加 512 行会被过滤的
marker，将事件时间推进到全部窗口之后。有效输入/线上输入分别为 32,768 / 33,280 行。
输入完整 SHA-256 在各样本中相同。无状态预期 16,384 行；窗口预期 32×16=512 行，
各窗口各 key 的 count=64。

计时从首 frame `send()` 前到父进程读到最后一条输出 JSON 行。包含生产者 pacing、
loopback WebSocket、JSON 解码/编码、Arrow 转换、native 计算、任务/通道、watermark、
pipe 和观察成本；fixture 构造、进程启动/编译、完整 oracle 比较和退出清理在计时外。
输出与独立 Python 整数 oracle 全量排序后精确比较，键/行数/sum/count 全部覆盖；
JSON 数值按数学值比较，未要求不同引擎的 Arrow nullability/整数位宽元数据相同。

所有预期输出都必须在源连接仍开放时出现，随后开放 100ms 检查额外输出，才停止进程。
窗口由 watermark 关闭，不依赖 EOF。calc-flow 用公开 `SourceProvidedWatermarks`，
每帧数据后发 `ts-1us`；Arroyo 用 DDL watermark 表达式，源码生成器按 batch 最小值
及事件时间间隔处理。两秒的帧间事件时间推进超过其一秒 watermark 周期。

吞吐 = 有效输入行数 / 输出完整可见时间。它包含 rate limit、源和 sink 成本，
是本负载完成吞吐，不是最大可持续吞吐。延迟逐输出行计算：无状态从其输入 frame
send 起点；窗口从最后贡献 frame send 起点，包含等待 watermark 的时间。原始数据
另记从最早可关闭该窗口的 progress frame 到输出的延迟，不能与前者混用。
表中 P50/P95 是各次运行逐行分位数的五次中位数，不是五次拼接后重新计算的分位数。

RSS 每约 10ms 采样引擎子进程树，从启动到停止前（包括正确性观察段），不含父观察器。
这是含解释器/控制面的总 RSS 采样峰值，可能漏掉短峰值；共享页相加可能重复计数，
也不是算子状态字节或长期稳定内存。保留完整采样时间序列。

**检查点边界不同**：calc-flow 明确 `checkpointing=False`；Arroyo interval=1h，
测量中没有主动 checkpoint，仍保留其恢复记账。停止时可能产生 final checkpoint，
不计入吞吐/延迟。两边都经不可 replay 的 WebSocket 和普通 stdout，不验证 exactly-once。
本次不提供 checkpoint/恢复耗时排名，也不声称通过仓库 lifecycle 的至少 20 轮门禁。

## 实测数据

正式采集用时 **76.22 秒**，完成 40/40 次；全量正确性、输出数量、EOF 前可见、
输入哈希、AB/BA 顺序、每个 calc-flow worker 实际加载的 native 哈希及统计边界审计
均通过。无状态共核对 327,680 条输出，窗口共核对 10,240 条输出。以下为五次中位数，
括号为五次 min–max；吞吐单位千行/秒，延迟单位 ms，RSS 单位 MiB。

| 场景            | 引擎      | 吞吐（范围）               | 逐行延迟 P50 / P95 | 采样峰值 RSS（范围） |
|-----------------|-----------|----------------------------|--------------------|----------------------|
| 投影/过滤 burst | calc-flow | 281.7（265.3–288.8）       | 63.68 / 109.46     | 204.8（204.5–205.0） |
| 投影/过滤 burst | Arroyo    | 631.9（483.7–700.0）       | 26.60 / 48.98      | 217.2（216.8–219.0） |
| 投影/过滤 paced | calc-flow | 87.2（83.0–88.1）          | 3.00 / 4.99        | 204.6（204.5–204.9） |
| 投影/过滤 paced | Arroyo    | 87.6（77.9–88.0）          | 1.09 / 1.64        | 219.0（214.8–221.3） |
| 四秒窗口 burst  | calc-flow | 358.9（294.5–382.8）       | 55.73 / 89.58      | 204.6（204.5–204.8） |
| 四秒窗口 burst  | Arroyo    | 1,573.6（1,372.8–2,184.4） | 13.01 / 19.17      | 222.3（221.1–223.9） |
| 四秒窗口 paced  | calc-flow | 86.2（84.5–87.2）          | 14.43 / 15.20      | 204.6（204.3–204.9） |
| 四秒窗口 paced  | Arroyo    | 86.5（84.5–89.2）          | 12.98 / 13.70      | 226.4（224.4–227.7） |

burst 的相邻成对吞吐比 Arroyo/calc-flow，投影/过滤为 1.77–2.44×（中位数 2.26×），
窗口为 3.59–7.20×（中位数 4.50×）。窗口的重复散布较大，Arroyo 五次吞吐自身
max/min 约 1.59；保留全部样本，不将最好的 218 万行/秒称为稳定能力。
投影/过滤 burst 的每次 P95 范围为 calc-flow 108.28–118.77ms、Arroyo 43.97–64.62ms；
窗口 burst 对应 84.17–108.37ms、14.13–22.83ms。其他 P50/P95/P99 和每条输出时间
可在原始数据直接复算。

paced 的相邻成对吞吐比中位数为投影/过滤 0.998×、窗口 1.023×；这只是同一发送节奏下
的完成速度。`sleep(5ms)` 加调度和发送开销，实际源速率低于理想 102,400 行/秒，
各次发送时间均保留。窗口 paced 的额外进度等待也包含在 14.43/12.98ms 内。

启动到源连接的五次中位数，calc-flow 各场景约 0.32–0.35 秒，Arroyo 0.75–0.78 秒；
这不是完整作业就绪/首次结果或恢复时间。采集前/后的 loadavg、可用内存和 RSS 序列
随数据保留；固定 affinity 不提供 CPU 独占或主机噪声的抵消证明。

收尾在验证后主动 SIGINT：Python worker 的退出码 `-2` 和 `KeyboardInterrupt` 是
预定停止结果，Arroyo 为 0；它们不表示计时中的结果失败，也不构成恢复/资源清理门禁。
构建及预检与正式数据分开：初始 harness 使用了 ColumnExpr 不支持的 `%`，在预检中
失败后改为双方一致的 `k < 8` 过滤，再通过双方预检；该失败和旧 harness 均保留。
Arroyo 定向 planner probe 第一次相对 state-dir 失败，改用绝对路径才取得预期拒绝，
两份日志均保留。旧一秒窗口、不同输入规模及构建并发期间的诊断数据不混入表格。

## 功能矩阵

“实测”只表示本次运行过对应路径；“文档/源码”表示已定位实现/契约，未作完整功能验收。
未验证不等于不支持。

| 能力                 | Arroyo v0.15.0                                                                     | calc-flow 基线                                                                                  | 证据与选型意义                                                                                               |
|----------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 无状态持续转换       | SQL projection/filter，原生 WebSocket → stdout                                     | Python 声明 → native StreamingRunner；Arrow 异步输出                                            | 本次实测两边值一致、EOF 前可见                                                                               |
| 事件时间窗口         | TUMBLE、HOP、SESSION                                                               | native tumbling/hopping；count/sum/min/max/avg；geometry 无 session                             | tumbling SUM/COUNT 本次实测；其他为文档/源码 [A2]、[C2]；Arroyo 的窗口类型更广                               |
| 普通 GROUP BY        | 增量 updating 表，撤回/插入等 changelog；默认 flush 1s，TTL 默认 1 天              | SQL 每批在保留输入上重算累计完整快照；没有默认小型行/字节 cap，可设预算                         | [A3]、[C1]；输出语义及长期成本不同，不宜直接按同名 SQL 比吞吐                                                |
| 按行滚动/特征        | SQL OVER 需要已有事件时间窗口，不能直接等同于无界逐行滚动                          | native `ts` 提供跨批 rolling、delta、EWMA；`cs` 提供横截面运算                                  | [A4]、[C3]；本次 Arroyo 未分窗 OVER ROWS 被 planner 拒绝，错误日志随附件；calc-flow 特征列表为源码/文档证据  |
| Join                 | 同窗口 INNER/LEFT/RIGHT/FULL、updating join、Redis lookup                          | 有界 event-time stream join、有限 backward ASOF；不是通用 changelog SQL join                    | [A5]、[C4]；两边 join 本次未跑性能，不能宣称互相完全替代                                                     |
| Watermark/迟到       | 事件时间表达式和窗口 finality；tumbling 实现跳过已过 watermark 的 bin              | source-provided、bounded disorder、idle 等政策；rolling/cross-section 可 route late side output | [A6]、[C1]；本次只验证有序输入，没测乱序、迟到和 idle；不能推断所有算子策略相同                              |
| 背压/资源            | source batch/linger、worker queue、流图任务；并行度与调度配置                      | 有界 edge/输出队列、显式状态预算和拥有者生命周期；逻辑预算不是 RSS cap                          | [A6]、[C1]；两边本次未作慢 sink、过载或长期资源稳定性测试                                                    |
| Checkpoint/恢复/交付 | 周期 barrier checkpoint、本地/对象存储；exactly-once 依赖 source/sink 能力         | aligned epoch、managed v3 manifest、lineage/fingerprint/layout 校验；能力按完整 route 判断      | [A1]、[A7][C1]；仅文档/源码，未作断电/重启实测。临时 iterable stream 不承诺 durable restart                  |
| 部署与扩展           | 单 binary minicluster，process/node/Kubernetes scheduler，SQLite/Postgres 控制配置 | 以嵌入式 Python/Rust runtime 为核心；Studio 位于独立仓库                                        | [A1]、[A8][C5]；Arroyo 现有服务运维能力更完整；calc-flow 本基线不提供等价的分布式 worker 调度服务            |
| 连接器               | Kafka、Kinesis、NATS、MQTT、RabbitMQ、Redis、HTTP/SSE/WS、文件及 Iceberg/Delta 等  | feature-gated file、Kafka、PostgreSQL CDC、MySQL、ClickHouse、HTTP、WebSocket                   | [A9]、[C6]；方向/格式/事务支持不同。本次只用 Arroyo native WS；calc-flow native WS 不在此次 wheel feature 中 |
| UDF/配置             | 支持 Rust/Python UDF，以及 Rust async UDF/UDAF 工作流                              | 可信注册实现，配置只存 `UdfReference(provider,name,version)`；类型化 Python 向量回调            | [A10]、[C7]；执行/部署/信任边界不同，不是对外部任意代码的等价沙箱保证                                        |
| 表/数组与接口        | 服务化 streaming SQL；Arrow/DataFusion 为内部基础                                  | Python immutable expression/Program + Arrow 表；NumPy/JAX 外部 array provider                   | [A8]、[C5]；面向 Python 特征/数组组合时 calc-flow 更自然，数组性能本次未测                                   |

## 优劣与尚缺证据

若任务是 Kafka/消息流到湖仓、SQL changelog 物化视图、多 worker 调度与图形化作业管理，
Arroyo 的现成功能组合更贴近需求。代价是需要管理控制面、状态存储和连接器契约，
开源 controller 高可用也不能默认视为已具备：官方架构文档明确列出单 controller 限制。
[Arroyo 架构](https://doc.arroyo.dev/architecture/)

若已有 Python/Arrow 应用，需要实体时间序列特征、横截面计算、有界 ASOF、不可变声明
和受控的注册 UDF，calc-flow 的库接口能减少服务边界。代价是应用方承担进程/重启和
持久 source/sink 的集成；普通 streaming SQL 累计聚合保留输入并重算，长期流需要关注
状态/时间增长并选择原生算子。其 event window 尚无 session geometry，不能直接替代
Arroyo 的 session/updating SQL 服务。[calc-flow 基线说明](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/introduction.md)

缺少的证据包括：相同原生 source/sink 的纯 runtime 分解、长期 steady state、较大 key
基数/偏斜、浮点/null/溢出/乱序边界、背压、Kafka、分布式扩展，以及两边检查点/恢复与
transactional sink 的成对实验。这些没有被小样本或厂商吞吐替代；只有真实目标场景需要
它们时才值得追加测量。本报告不是版本 regression/no-change/improvement 门禁结论。

## 来源与源码定位

官方文档可能随站点变化，版本结论以本次固定 commit 的源码为锚；线上文档只补充语义。
全文均为概括，没有复用厂商性能数字。

- [A1：发布与 pipeline cluster](https://github.com/ArroyoSystems/arroyo/releases/tag/v0.15.0)，[运行模式](https://doc.arroyo.dev/deployment/pipeline-clusters/)。固定源码 `crates/arroyo/src/run.rs`。
- [A2：Streaming Windows](https://doc.arroyo.dev/sql/windows/)。固定源码 `crates/arroyo-worker/src/arrow/{tumbling,sliding,session}_aggregating_window.rs`。
- [A3：Updating Tables](https://doc.arroyo.dev/sql/updating-tables/)。固定源码 `crates/arroyo-rpc/default.toml`、`crates/arroyo-worker/src/arrow/incremental_aggregator.rs`。
- [A4：Window Functions](https://doc.arroyo.dev/sql/window-functions/)。固定源码 `crates/arroyo-planner/src/plan/window_fn.rs:53`，实际 planner 拒绝见 `probe-over-absolute.log`。
- [A5：Joins](https://doc.arroyo.dev/sql/joins/)。按官方文档区分三种 join，未据未发现 ASOF 推断其绝对不存在。
- [A6：固定版本 watermark 源码](https://github.com/ArroyoSystems/arroyo/blob/4ef15f6336a596651f10cf27c1ee29c29bf81577/crates/arroyo-worker/src/arrow/watermark_generator.rs)，`arroyo-planner/src/extension/watermark_node.rs`、`arroyo-connectors/src/websocket/operator.rs` 与 `arroyo-rpc/default.toml`。
- [A7：File System checkpoint 交付边界](https://doc.arroyo.dev/connectors/filesystem/)。本次未用 file sink，不将此承诺移到 WebSocket/stdout。
- [A8：Architecture](https://doc.arroyo.dev/architecture/)。当前文档含历史 codegen 描述，因此执行细节以 pinned source 为准。
- [A9：固定版本 connector 注册清单](https://github.com/ArroyoSystems/arroyo/blob/4ef15f6336a596651f10cf27c1ee29c29bf81577/crates/arroyo-connectors/src/lib.rs)。文档上的 Postgres/MySQL 图标不作为本版本原生 CDC source 已验证证据。
- [A10：UDF Overview](https://doc.arroyo.dev/udfs/)，[Async UDF](https://doc.arroyo.dev/udfs/rust/async-udfs/)。本次使用不含 Python feature 的官方 binary，Python UDF 未实测。
- [C1：基线 streaming guide](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/streaming-guide.md)，`python/calc_flow/stream.py`、`python/calc_flow/runtime.py`。
- [C2：基线 window 实现](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/crates/calc-flow/src/operator/window.rs)，`python/calc_flow/symbolic/windows.py`。
- [C3：表达式工作流](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/symbolic-workflows.md)。
- [C4：ASOF 契约](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/asof-join-guide.md)。
- [C5：Introduction](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/introduction.md)。
- [C6：Connectors](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/connectors/README.md)，[WebSocket 边界](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/connectors/websocket.md)。
- [C7：Python API/UDF](https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/python-api.md)。

## 重现与交付

将 `measure.py`、`requirements.txt`、`reproduce.sh` 放入固定基线 checkout 的
`.codex/artifacts/arroyo-comparison/`，从 repository root 执行
`bash .codex/artifacts/arroyo-comparison/reproduce.sh`。脚本拒绝不同 HEAD、已有输出目录
和 tracked diff，构建产物全部留在 `target/`；不生成源码目录下的 `_native*.so`。
PR 的 HEAD 包含研究产物提交，应按 [README.md](README.md) 准备独立的固定基线 worktree，
再将脚本复制过去运行；离线复核数据不需要 worktree 或 native 构建。
源码获取可使用 `multica repo checkout git@github.com:wegamekinglc/calc-flow.git --ref dab98a559314bd46d866ed7a6489abcdd3810afe`。
外部读者无 Multica 环境时可自行 clone 并 checkout 同一 SHA。

脚本的重现结果保存在 `target/arroyo-comparison-reproduction/`。原始数据中的绝对路径
只作为当时 provenance，读者不需要拥有这些机器路径。正式数据保留每次 command、
query、原始 stdout、完整 expected/arrivals/send_times、RSS 时间序列、sample 与 summary。
下载 binary/重新构建 wheel 的哈希应与 provenance 核对；重新构建不同 hash 需要重新记账，
不能冒称同一个 sealed binary 的重复测量。

正式采样时的精确脚本另存为 raw-data 根目录的 `measure-measured.py`，其 SHA-256 与
`results/identity.json` 完全一致。交付的 `measure.py` 随后仅把 oracle 的列表推导式
按 Ruff 折为一行；AST 完全相同，双哈希与核对结果见 `format-provenance.json`。
离线审计优先检查原始测量脚本，保留证据对应关系。

本次实际执行的主要命令（均从 calc-flow repository root）：

```bash
CARGO_BUILD_JOBS=8 target/arroyo-comparison/venv/bin/maturin build \
  --release --locked \
  --interpreter target/arroyo-comparison/venv/bin/python \
  --out target/arroyo-comparison/wheels \
  > target/arroyo-comparison/build.log 2>&1

taskset -c 0-1 target/arroyo-comparison/venv/bin/python \
  .codex/artifacts/arroyo-comparison/measure.py \
  --arroyo target/arroyo-comparison/bin/arroyo-linux-x86_64/arroyo \
  --output target/arroyo-comparison/results \
  --rows 32768 --pairs 5 --cpus 4-7 --budget 600 \
  > target/arroyo-comparison/acquisition.log 2>&1

target/arroyo-comparison/venv/bin/python \
  .codex/artifacts/arroyo-comparison/audit.py target/arroyo-comparison \
  > target/arroyo-comparison/audit-reproduction.json
```

构建成功（11m19s），正式采集 40/40 成功（76.22s），离线数据审计成功。
`ruff check`、`ruff format --check` 对两个 Python 产物及 `bash -n reproduce.sh` 通过；
测量阶段未运行无关的全仓测试、CI 或性能门禁；PR 的 CI 结果单独报告。
`audit.py` 是这份固定样本集的离线复核脚本，
与 `measure.py` 放在同一目录即可检查解包后的 raw-data 根目录，无需启动任何引擎。

[A1]: https://doc.arroyo.dev/deployment/pipeline-clusters/
[A2]: https://doc.arroyo.dev/sql/windows/
[A3]: https://doc.arroyo.dev/sql/updating-tables/
[A4]: https://doc.arroyo.dev/sql/window-functions/
[A5]: https://doc.arroyo.dev/sql/joins/
[A6]: https://github.com/ArroyoSystems/arroyo/blob/4ef15f6336a596651f10cf27c1ee29c29bf81577/crates/arroyo-worker/src/arrow/watermark_generator.rs
[A7]: https://doc.arroyo.dev/connectors/filesystem/
[A8]: https://doc.arroyo.dev/architecture/
[A9]: https://github.com/ArroyoSystems/arroyo/blob/4ef15f6336a596651f10cf27c1ee29c29bf81577/crates/arroyo-connectors/src/lib.rs
[A10]: https://doc.arroyo.dev/udfs/
[C1]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/streaming-guide.md
[C2]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/crates/calc-flow/src/operator/window.rs
[C3]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/symbolic-workflows.md
[C4]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/asof-join-guide.md
[C5]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/introduction.md
[C6]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/connectors/README.md
[C7]: https://github.com/wegamekinglc/calc-flow/blob/dab98a559314bd46d866ed7a6489abcdd3810afe/docs/python-api.md
