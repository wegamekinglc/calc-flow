# Arroyo / RisingWave 独立调研与 Calc-Flow 流式演进建议

> - 调研日期：2026-08-02
> - 文档性质：时间点工程调研记录，不是 Calc-Flow 当前 API 或运行时规范
> - 调研快照：Arroyo `f6afb832c00d25a349522ba2678d55a6866a9fab`
>   （`0.16.0-dev`）；RisingWave `v3.0.2`
>   （`391c3a16ef26d0cd86d1236c9b7c122a9a27fb1e`）；Calc-Flow 当前工作区
> - 对应计划：
>   [`Calc-Flow 3.0 Continuous Streaming Development Plan`](../superpowers/plans/2026-08-02-continuous-streaming-v3.md)

## 1. 结论先行

Calc-Flow 当前仍是“静态 DAG 上逐次执行不可变 micro-batch”的单机嵌入式引擎，
不是持续运行的流处理系统。现有 `RuntimeEnvelope::Control`、watermark/epoch
占位标记和控制路由，只证明内部已经为未来语义留出了边界；它们不携带事件时间，
不是公开 API，也没有形成 source 驱动、背压、窗口、barrier checkpoint 或
exactly-once 协议。

Arroyo 与 Calc-Flow 的技术亲缘度最高：二者都使用 Rust、Arrow 和 DataFusion，
Arroyo 展示了如何把列式 batch、控制信号、异步算子、checkpoint 和 connector
组合为持续 dataflow。它最值得借鉴的是运行时协议和工程边界，而不是其对
Arrow/DataFusion/sqlparser 上游 fork 的具体依赖方式。

RisingWave 的核心价值不在于“另一个 Rust 流引擎”，而在于其流式数据库模型：
物化视图、epoch、Hummock 云原生状态存储、计算与持久状态解耦。它适合启发
Calc-Flow 的状态版本和恢复设计，但完整复制 Hummock、PG serving 层或分布式
actor 调度会偏离 Calc-Flow 的嵌入式定位。

最终建议可以概括为：

> 采用 Arroyo 风格的运行时协议，吸收 RisingWave 的状态版本思想，保持
> Calc-Flow 原生的嵌入式 API；保留现有 batch executor，新增 continuous
> executor；先交付 final-only 窗口，再建设 exactly-once，最后扩展 connector。

推荐目标不是“缩小版 Arroyo”或“缩小版 RisingWave”，而是：

> 单机、嵌入式、Arrow-native、Python 友好的有状态流式计算运行时。

## 2. 调研方法与证据边界

本报告按以下顺序独立核验：

1. 阅读 Calc-Flow 当前 Rust 核心、runner、checkpoint、I/O 和 runtime envelope
   实现，确认已实现与未实现边界。
2. 在固定提交上阅读 Arroyo 的消息类型、执行循环、背压、checkpoint、状态和
   connector 注册代码，再用官方文档补充部署与产品定位。
3. 在 RisingWave `v3.0.2` 固定提交上阅读 `StreamChunk`、barrier、watermark、
   Hummock 与 sink delivery 相关代码，并与官方架构、交付语义和扩缩容文档交叉
   核验。
4. 对厂商博客中的性能、成本和恢复数字只作为产品方陈述，不将其作为独立基准
   结论。

因此，本报告区分三类内容：源码可以直接证明的实现事实、官方文档描述的产品
契约，以及基于二者作出的 Calc-Flow 架构建议。

## 3. Calc-Flow 当前基线

### 3.1 已实现能力

- Rust 2024 单机核心，当前版本 `2.0.0`，表计算统一使用 DataFusion 54。
- `Batch` 是公开、不可变的数据 envelope，可承载 Arrow table 或显式注册的
  external provider payload。
- `PipelineBuilder::compile()` 负责编译期端口、类型、schema、单写入者、环、UDF、
  拓扑和 fingerprint 校验。
- `ExecutionPlan::execute()` 为一次执行创建 run-scoped DataFusion session，并按
  编译后的确定性拓扑顺序执行节点。
- `MicroBatchRunner` 从一个可重放 `Source` 拉取一个 batch；`StreamingRunner`
  接受调用方逐次 `step(batch)`。
- runner 在所有 sink 写成功后提交 checkpoint，因此当前交付契约是
  at-least-once。
- `FileCheckpointStore` 和 `FileProjectStore` 使用受大小限制的原子 JSON 文档。

对应实现入口：

- [`pipeline.rs`](../../crates/calc-flow/src/pipeline.rs)
- [`runtime/micro_batch.rs`](../../crates/calc-flow/src/runtime/micro_batch.rs)
- [`runtime/streaming.rs`](../../crates/calc-flow/src/runtime/streaming.rs)
- [`checkpoint.rs`](../../crates/calc-flow/src/checkpoint.rs)
- [`io.rs`](../../crates/calc-flow/src/io.rs)

### 3.2 尚未实现的流语义

当前执行仍由一次调用处理一组完整输入。节点在一个执行路径中按拓扑序依次运行，
不存在长生命周期算子 task、per-edge channel、队列容量、背压传播或多 source
调度。

crate-private `RuntimeEnvelope` 虽然有 `Data` 与 `Control` 两种变体，但目前的
watermark/epoch marker 只携带 opaque occurrence：

- watermark 没有事件时间值、单调性、idle 或迟到数据含义；
- epoch 没有 checkpoint ID、对齐、快照边界或提交协议；
- 控制入口不是 runner 或公开 API；
- 可达 multi-input 节点时控制路由 fail-closed；
- 现有控制路径没有持续 per-edge 队列。

这一边界已经在 [`runtime-envelope.md`](../runtime-envelope.md) 中明确记录。换言之，
当前拥有的是可继续演进的内部 envelope 和路由骨架，而不是“部分可用的流式
watermark/checkpoint”。

### 3.3 当前状态模型的容量边界

当前 checkpoint 是一个有大小上限的 JSON 文档，包含 pipeline fingerprint、source
cursor、sequence 和节点 snapshot。它适合轻量 operator state，但不适合作为大型
keyed window state 的长期承载形式。若先实现窗口、再继续把全部窗口状态塞入 JSON，
状态增长会直接转化为 checkpoint 延迟、文件大小和恢复成本。

## 4. Arroyo 独立调研结果

### 4.1 定位与项目状态

Arroyo 是 Rust 实现的分布式流处理引擎，以 SQL 编译持续 dataflow 为主要产品
形态。调研源码快照为
[`f6afb832`](https://github.com/ArroyoSystems/arroyo/tree/f6afb832c00d25a349522ba2678d55a6866a9fab)，
crate 版本为 `0.16.0-dev`；调研时最新稳定版为 `v0.15.0`。该快照就是调研当日
默认分支的 HEAD（提交时间 2026-08-01），因此下文源码结论描述的是当时的最新
开发状态，而不是某个已发布版本。仓库同时提供 Apache-2.0 与 MIT 许可证文件。

Cloudflare 于 2025-04-10 宣布收购 Arroyo 团队，并将其技术用于 Cloudflare
Pipelines。开源仓库此后仍有提交，但产品与社区演进需要同时观察开源项目和
Cloudflare 平台方向，不能仅凭“仍有提交”推导长期治理承诺。这一判断有具体
证据：`v0.15.0` 发布于 2025-12-01，到调研日已约八个月没有新的稳定版，而默认
分支仍在持续提交。开发活跃度与发布节奏在该项目上并不同步。

### 4.2 控制面与数据面

Arroyo 的主要分层是：

```text
API / Web UI
      │
      ▼
Controller：作业状态机、调度、checkpoint 协调
      │
      ▼
Scheduler / Node：Process、Kubernetes、Embedded 等部署形态
      │
      ▼
Worker / Subtask：持续执行算子
      │
      ├── Arrow IPC 数据面
      └── 对象存储中的 checkpoint / state
```

逻辑计划使用有向图表达节点和边，worker 再按并行度展开为 subtask。`Forward`
链可以融合，shuffle 边负责重分区。控制面主要通过 gRPC 组织，worker 数据面使用
Arrow IPC over TCP/TLS，而不是把 RecordBatch 当作普通 gRPC payload 传输。

### 4.3 消息、执行循环与背压

链路消息是 `ArrowMessage`：

```rust
pub enum ArrowMessage {
    Data(RecordBatch),
    Signal(SignalMessage),
}
```

signal 包括 barrier、watermark、stop 和 end-of-data。数据与控制信号走同一条
有序通道，这使“barrier 之前的数据”和“watermark 之前的数据”具有清晰的
per-channel 顺序边界。

算子循环通过 `tokio::select!` 多路复用输入、控制消息、tick 和算子 future。
先到达 barrier 的输入会被暂时阻塞，直到同一 checkpoint 的其他输入 barrier
到齐，然后快照并继续推进。这是多输入算子 barrier alignment 的关键机制。

Arroyo 的 `batch_bounded(size)` 有一个容易误读的细节：真正阻塞发送端的是逻辑
row/message 计数；queued bytes 会被统计为指标，但不是第二个强制容量配额。因此，
将它描述为“消息数 + 字节数双配额背压”并不准确。相关实现可直接查看
[`context.rs` 固定提交代码](https://github.com/ArroyoSystems/arroyo/blob/f6afb832c00d25a349522ba2678d55a6866a9fab/crates/arroyo-operator/src/context.rs#L88-L157)。

这对 Calc-Flow 的启示不是照搬该 limiter，而是同时强制逻辑行数和估算字节数
上限，避免少量超大 Arrow batch 绕过仅按消息或行数设置的容量。

### 4.4 时间、窗口与迟到数据

Arroyo 的 watermark 是 `EventTime(SystemTime)` 或 `Idle`。watermark 可由 source
提供，也可由表达式 watermark 算子定期生成。多输入算子需要根据各输入进度决定
可以安全推进的最小事件时间。

窗口覆盖 tumbling、sliding、session 等形态，聚合计算复用 DataFusion 物理表达式
和聚合能力。调研快照中的 late-data 处理相对保守：窗口路径会过滤已经落后于
watermark 的记录。需要注明证据强度：该快照与官方文档中都没有出现 Flink 式
allowed lateness + side output 的产品契约，但本次核验未能从一手源逐条确认其
全部 late-data 分支行为，因此这里只作“未见等价产品契约”的弱结论，不作“确定
不存在”的强结论。

### 4.5 状态、checkpoint 与 exactly-once

Arroyo 的算子状态主要在 worker 内存中，checkpoint 以增量 Parquet 写入对象存储，
并定期 compact。controller 向 source 注入带 epoch 的 barrier；barrier 沿数据流
传播，多输入节点对齐后建立一致快照。

端到端 exactly-once 不是 checkpoint 单独提供的。它需要同时满足：

1. source offset/cursor 与算子状态处于同一 checkpoint 边界；
2. sink 支持事务或幂等提交；
3. checkpoint 完成后再提交 pre-commit 输出；
4. 恢复时能够依据 checkpoint 状态补提交或回滚。

Arroyo 为文件、Delta、Iceberg、Kafka 等适用 sink 提供两阶段提交流程；stdout、
webhook 等非事务 sink 仍只能提供较弱的交付保证。

### 4.6 Connector 与部署

源码注册表在该快照构造了 21 个 connector entry，但“21 个 connector”不应直接
等同于 21 个双向、同等成熟的用户能力：有些仅 source、有些仅 sink，有些是
测试、preview 或特定存储变体。官方 connector 文档应作为用户可用性和配置契约的
最终入口。

Arroyo 支持 process、Kubernetes、embedded 等调度方式。开源 controller 的高
可用性不是其当前强项；手动 rescale 通常需要 final checkpoint、停止和按新并行度
恢复，不能等同于 RisingWave 的存算分离弹性模型。

### 4.7 对 Calc-Flow 最有价值的部分

- 数据与控制消息共享有序 channel 的协议模型；
- source、普通算子、sink 的不同生命周期；
- 多输入 barrier 对齐和 checkpoint completion 回调；
- Arrow/DataFusion 作为列式执行底座；
- connector 与 format 分离、注册表驱动构造；
- 增量状态文件与周期 compaction；
- transactional sink 的独立能力接口。

不建议复制的部分包括：对 DataFusion、Arrow、sqlparser 的深度 fork patch，以及在
Calc-Flow 尚未形成单机持续运行时之前引入 controller、worker、slot 和跨节点
shuffle。

## 5. RisingWave 独立调研结果

### 5.1 定位与组件

RisingWave 是流式数据库，而不是通用嵌入式 pipeline 库。其核心用户抽象是
持续维护的物化视图，结果可通过 PostgreSQL wire protocol 查询。调研固定版本为
[`v3.0.2`](https://github.com/risingwavelabs/risingwave/releases/tag/v3.0.2)。

主要节点角色为：

- Serving Node：SQL 接入、优化和批查询；
- Streaming Node：执行 actor graph 并持续维护状态；
- Meta Node：catalog、调度、barrier、checkpoint 和恢复协调；
- Compactor Node：Hummock SST compaction。

持久状态位于 S3、GCS、Azure Blob 或兼容对象存储中，compute 本地内存和磁盘主要
用于 cache。

### 5.2 数据消息是列式 changelog

RisingWave 的 `StreamChunk` 不是“内部行式 chunk”。它由列式 `DataChunk`、每行
操作类型和 visibility 组成，操作类型包括 insert、delete、update-delete 和
update-insert。固定版本源码见
[`stream_chunk.rs`](https://github.com/risingwavelabs/risingwave/blob/391c3a16ef26d0cd86d1236c9b7c122a9a27fb1e/src/common/src/array/stream_chunk.rs#L38-L109)。

这项区别非常重要：RisingWave 能持续维护会被更新或撤回的关系结果，依赖的是
changelog 语义，不只是“不断到达的 RecordBatch”。Calc-Flow 当前 `Batch` 没有
insert/delete/update/retract 标记，因此第一版窗口应优先选择关闭后只输出一次的
final-only 模式。

### 5.3 Barrier、epoch 与 Hummock

stream message 包括 chunk、barrier 和 watermark。Meta Node 默认按秒级周期注入
barrier，barrier 划分 epoch，并协调算子进度、状态提交和图变更。

Hummock 是面向对象存储的 LSM-tree 状态后端。executor 先写本地 shared buffer，
再将不可变 SST 上传到对象存储；Hummock version 记录各 epoch 可见的 SST 集合，
旧版本由 compaction 和 low watermark 回收。key 与 epoch 共同支持 MVCC 快照读取。

把 RisingWave checkpoint 描述成“只提交一个 epoch 元数据，几乎免费”过于简化。
barrier checkpoint 仍可能触发 dirty state flush、SST 构建和对象存储上传；其优势是
状态本来就采用共享、版本化存储，不需要每次把完整本地 RocksDB 状态重新打包成
独立快照。

### 5.4 恢复与扩缩容

RisingWave 恢复时可以重新挂载共享状态，不要求先把所有状态完整下载到 compute
节点。这使恢复和扩缩容避免了按完整状态量搬迁本地数据库文件。不过，“恢复时间
与状态大小完全无关”仍然过强：actor 重建、元数据规模、cache 重新预热、对象存储
吞吐和工作集大小都会影响恢复后的延迟与稳定时间。

vnode 将逻辑分片所有权与物理 SST 文件解耦。adaptive parallelism 与 vnode
重分配可以在不重写全部状态文件的前提下调整 actor 并行度，但新节点仍会经历
cache miss 和远端读取成本。

### 5.5 Exactly-once 是逐 sink 能力

RisingWave 内部状态一致性由 barrier、epoch 和 Hummock version 协调。对外 sink
是否 exactly-once 则取决于 connector。官方 delivery 文档给出的分档很明确：
Iceberg sink 在 `is_exactly_once = true`（默认值）且启用 sink decoupling 时提供
exactly-once，一旦关闭 sink decoupling，exactly-once 会被自动禁用；Kafka sink
的官方表述是“非事务写入，通过重试提供 at-least-once”；其余 sink 除非另有说明
一律为 at-least-once。因此不能把“RisingWave 内部 epoch 一致性”直接推广为“所有
外部 sink 都端到端 exactly-once”，也不能把 Iceberg 的 exactly-once 当成无条件
能力——它依赖一个可以被用户关掉的配置组合。

### 5.6 SQL、窗口、CDC 与 serving

RisingWave 的优势来自能力组合，而不是某一个单独算子：

- PostgreSQL 协议和 `CREATE MATERIALIZED VIEW`；
- TUMBLE、HOP、session 等窗口；
- watermark 和 `EMIT ON WINDOW CLOSE`；
- insert/delete/update changelog；
- PostgreSQL、MySQL、SQL Server、MongoDB 等原生 CDC；
- 多类消息系统、数据仓库、搜索和湖仓 sink；
- 可直接查询已物化结果的 serving 层。

temporal join 仍应区分处理时间和事件时间能力。调研版本提供 process-time temporal
join，并支持 ASOF、window/interval 等其他 join 形态，但不能据此宣称具备任意
event-time temporal table 语义。

### 5.7 对 Calc-Flow 最有价值的部分

- epoch 既是处理进度，也是状态版本可见性边界；
- 状态数据与 checkpoint manifest 分离；
- 增量 SST/segment 与后台 compaction；
- 物化结果可查询带来的易用性；
- final-only window close 作为无需 retract 的首个稳定语义；
- source、状态、sink 三方共同决定 delivery guarantee。

不建议复制的部分包括：Hummock 完整分布式 LSM、PG wire serving、Meta/Compactor
集群角色和全套 CDC 生态。这些能力的成本远超 Calc-Flow 当前产品边界。

## 6. 对原始参考报告的关键修正

| 原始表述或倾向                             | 独立核验后的结论                                                                               |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Arroyo 队列按消息数和字节数双配额阻塞      | 行数/消息计数实际执行阻塞；bytes 在该实现中主要用于指标，不是第二个硬配额。                    |
| RisingWave 内部使用行式 chunk              | `StreamChunk` 基于列式 `DataChunk`，另带操作数组和 visibility。                                |
| RisingWave checkpoint 只是提交 epoch       | dirty state 仍需 flush、生成 SST 并上传；版本提交避免的是重复的全量独立快照。                  |
| RisingWave 端到端一律 exactly-once         | 内部状态一致，不代表所有 sink 一致；外部 delivery guarantee 是 connector-specific。            |
| RisingWave 恢复时间与状态大小无关          | 不必预下载全部状态，但元数据、工作集、cache 和对象存储仍会影响恢复与预热。                     |
| Arroyo 有 21 个等价 connector              | 这是源码 registry entry 数；source/sink 方向、用途和成熟度并不相同。                           |
| Calc-Flow 已有 watermark/epoch 语义骨架    | 只有 opaque marker 与私有控制路由骨架，没有事件时间、barrier 或 runner 契约。                  |
| 直接把 `execute_nodes` task 化即可持续执行 | 现有 `Operator::process` 是整组输入语义；需要独立 stream operator 生命周期和执行器。           |
| M1 到 M5 可在约 16 至 22 周完整生产化      | 可做原型；按本报告 §9 分解，覆盖 Rust、Python、schema、Studio 与故障测试更可能需 38 至 59 周。 |

## 7. 三者架构对比

| 维度               | Calc-Flow 当前                         | Arroyo                                  | RisingWave                                      |
| ------------------ | -------------------------------------- | --------------------------------------- | ----------------------------------------------- |
| 产品定位           | 单机嵌入式 batch/micro-batch 引擎      | 分布式流式 pipeline 引擎                | 分布式流式数据库                                |
| 用户主抽象         | Builder DAG 与一次执行                 | SQL/逻辑 dataflow 与持续 job            | 物化视图与可查询关系                            |
| 数据载体           | Arrow `Batch` 或 external payload      | Arrow `RecordBatch`                     | 列式 `StreamChunk` + changelog op                |
| 执行生命周期       | 每次调用按拓扑顺序执行                 | 长生命周期 subtask                     | 长生命周期 actor                               |
| 节点间队列         | 无                                     | 有，异步 push 与逻辑容量背压            | 有，actor message channel                       |
| 时间语义           | 无公开事件时间语义                     | event time watermark + idle             | watermark + epoch                               |
| 窗口               | 无                                     | tumbling/sliding/session 等             | TUMBLE/HOP/session 等                           |
| 更新与撤回         | 无 changelog                           | 依查询/算子能力                         | insert/delete/update changelog                  |
| 状态后端           | 节点 JSON snapshot                     | 内存状态 + 对象存储增量 Parquet         | Hummock LSM on object storage                   |
| checkpoint         | sink 成功后提交整图 JSON               | 对齐 barrier + epoch checkpoint         | barrier + epoch + Hummock version               |
| 外部交付保证       | at-least-once                          | sink-specific，可达 exactly-once        | sink-specific，可达 exactly-once                |
| 并行与部署         | 单进程，节点顺序执行                   | 多 worker/subtask，支持 Kubernetes      | 多节点 actor/vnode，存算分离                    |
| connector          | 用户实现 trait，无内置生态             | registry 驱动，内置多类 connector       | 丰富 connector 与原生 CDC                       |
| serving            | 返回 terminal batch / Studio 预览      | 主要 push 到 sink                       | PG 协议直接查询物化结果                         |
| 与 Calc-Flow 关系  | 本体                                   | 运行时设计最接近                        | 状态、epoch 与 materialization 思想最有参考价值 |

## 8. Calc-Flow 最终架构建议

### 8.1 保留 `BatchExecutionPlan`，新增 `StreamExecutionPlan`

不建议直接把 `ExecutionPlan::execute_nodes()` 改造成一组永久 task。这会同时改变现有
batch API 的延迟、错误、取消、rollback、指标和确定性契约，并把流式生命周期硬塞入
`Operator::process(BTreeMap<...>)` 的整组输入模型。

建议保留：

```text
BatchExecutionPlan
  └── execute(inputs) -> RunResult
      现有有界、确定性、一次性 batch 执行路径
```

新增：

```text
StreamExecutionPlan
  ├── source tasks
  ├── per-edge bounded channels
  ├── long-lived stream operator tasks
  ├── watermark / barrier coordinator
  ├── state backend
  └── sink tasks
```

两种执行计划共享图定义、端口、schema、UDF/provider snapshot 和可复用的纯计算
内核，但不共享互相冲突的生命周期。

### 8.2 引入专用 `StreamOperator` 生命周期

持续算子至少需要表达以下事件：

```rust
enum StreamMessage {
    Data(Batch),
    Watermark(EventTime),
    Barrier(Epoch),
    Idle,
    EndOfInput,
}
```

实际公开 API 应在设计评审后确定，但内部能力需要覆盖：

- 按 ingress 接收数据，而不是等待所有输入组成完整 map；
- 对 join、union 和 window 保存跨消息状态；
- 接收 watermark、barrier、idle 和 end-of-input；
- 在一个输入 barrier 先到时阻塞该 ingress，同时继续消费未到 barrier 的输入；
- 发射零个、一个或多个数据/控制消息；
- 在 checkpoint 完成后接收 commit 通知；
- 支持 cancellation、close、snapshot、restore 和错误收敛。

可复用 `Operator` 的纯 batch 计算内核，但不应假设所有现有 operator 自动具备正确
的流式多输入行为。

### 8.3 Watermark 使用强类型 UTC 时间

不建议公开裸 `i64` 微秒时间戳。应使用明确的 `EventTime` newtype，固定 UTC、精度、
序列化和越界规则。每条 ingress 上 watermark 必须单调不降，多输入输出 watermark
通常取所有非 idle 输入的最小值。

精度转换必须一并固定：内部单位确定后，Arrow 的 second/millisecond/microsecond/
nanosecond 输入都要做 checked conversion，且截断方向统一向下取整。事件时间向下
取整保证行不会被推进到更晚的窗口，watermark 向下取整保证进度估计保持保守；两者
方向不一致会同时产生错分窗口和过早关闭窗口两类错误。

迟到判定的边界必须按窗口而不是按行定义。窗口算子中，一行迟到当且仅当它所属
窗口的 `window_end <= 当前输入 watermark`，即该窗口已经关闭。若错误地采用
`event_time <= watermark` 作为判据，则只要 watermark delay 小于 window size，
大量本应进入尚未关闭窗口的正常数据都会被丢弃：例如 1 小时 tumbling 窗口、
watermark 为 10:30 时，事件时间 10:15 的行属于 `[10:00, 11:00)`，该窗口尚未关闭，
必须接受而不是丢弃。

第一阶段迟到数据只支持：

- `Drop`，并暴露按 operator/window/source 统计的指标；
- 可观测的 late row count 和最大迟到量。

暂缓 `Allow`、side output 和更新已关闭窗口，因为这些能力需要明确的 changelog、
retract 或 upsert 输出契约。

### 8.4 第一版窗口采用 final-only

第一版建议只实现：

- tumbling window；
- hopping window；
- watermark 越过 `window_end` 后输出一次最终结果；
- 迟到数据丢弃并记录指标；
- window state 可 checkpoint 和恢复。

session window、early trigger、processing-time trigger、allowed lateness 和
emit-on-update 后置。原因是 session 合并和早期更新都可能修改已发结果，而当前
Calc-Flow `Batch` 没有 RisingWave 式 changelog/retract 语义。

### 8.5 先建立状态后端，再承载大型窗口

建议把 checkpoint 拆成控制面 manifest 和数据面 state segment：

```text
checkpoint manifest (bounded JSON)
  ├── pipeline fingerprint
  ├── epoch
  ├── source cursors
  ├── watermarks / idle state
  ├── operator state references
  └── sink pre-commit metadata

state backend
  ├── immutable Arrow IPC / Parquet segments
  ├── epoch/version metadata
  ├── incremental dirty-key writes
  └── periodic compaction and retention
```

首版只需本地文件后端即可，不需要对象存储和 Hummock。关键是让大量 keyed state
不再膨胀 bounded JSON checkpoint，同时维持原子 manifest 提交。

这里有一个容易被忽略的一致性要求：state segment 的发布和 checkpoint manifest 的
发布是两个不同的原子提交点，二者必须只有一个是“最近完成 epoch”的唯一真相。
建议规定 checkpoint manifest 是唯一真相，并固定提交顺序为：先落盘并校验 state
segment，再原子发布 checkpoint manifest。恢复只读取 checkpoint manifest，任何未被
保留 manifest 引用的 segment 一律按垃圾回收，不参与恢复判定。若不固定这条顺序和
唯一真相，崩溃窗口会同时产生两类不可判定状态：manifest 引用了不存在的 segment，
以及 segment 已提交但没有任何 manifest 指向它。

### 8.6 Exactly-once 必须显式组合能力

只有满足以下条件的 pipeline 才能声明端到端 exactly-once：

- source 可重放，并能在 barrier 边界持久化 cursor/offset；
- operator 是确定性的，状态在 epoch 边界一致快照；
- 多输入 barrier 能对齐，或未来有经过验证的 unaligned checkpoint 协议；
- sink 提供事务或基于 epoch 的幂等提交；
- 恢复逻辑能够处理未完成 pre-commit；
- connector 明确声明并由编译器验证 delivery capability。

建议保留现有 `Sink` 作为 at-least-once 简单接口，另增事务 sink 能力，而不是让每个
sink 被迫伪装成支持 2PC。

barrier 注入还有一条容易被忽略的 source 契约：source task 必须能在等待外部数据的
同时响应 barrier 请求，否则一个长期无数据的 source 会让每次 checkpoint 都超时。
这要求 source 的“取下一项”操作要么明确声明可取消安全，要么由 runtime 用预取槽位
把外部 I/O 与控制响应解耦。二者必须择一写进规范，不能留给实现临时决定。

默认背压策略应是 `Block`。`DropOldest` 只能用于用户显式选择的 lossy source，且
编译时必须判定它与 exactly-once 不兼容。

### 8.7 Connector registry 应为 plan-scoped snapshot

可以借鉴 Arroyo 的类型擦除 registry，但不建议使用运行期可变的全局单例。更符合
Calc-Flow 当前 `ProviderRegistry` / `UdfRegistrySnapshot` 设计的方案是：

1. 构建期向 registry 注册 trusted connector factory；
2. 编译时解析数据-only connector spec；
3. 编译结果捕获不可变 registry snapshot；
4. runner 启动时创建 source/sink 实例；
5. checkpoint 仅保存配置引用、cursor 和事务元数据，不序列化 executable object。

format 应与 transport 正交，例如 Kafka + JSON、Kafka + Avro、file + Parquet，而不是
为每个组合创建一个独立 connector 类型。

### 8.8 Project 文档使用显式 v3 迁移

当前 project v2 是严格、data-only 的稳定契约。connector、watermark、window 和
runtime mode 会引入新的持久语义，不应在 v2 schema 中静默扩展或复用含义模糊的
现有字段。建议设计 project v3；v2 文档返回 `UnsupportedVersion`，与计划文档
决策一致。

## 9. 推荐开发路线

### M0：语义规格与故障模型，2 至 3 周

交付物：

- continuous execution、message ordering 和 cancellation 规范；
- event time、watermark、idle、late-data 规范；
- epoch、barrier alignment、checkpoint completion 规范；
- source/sink capability matrix；
- final-only window 与 project v3 决策记录。

验收重点：用可执行状态机测试或模型测试描述双输入 barrier、idle input、取消和故障
恢复，不在语义未定时先固化 public API。

### M1：独立持续执行器，5 至 8 周

交付物：

- `ContinuousExecutionPlan` 与 `StreamOperator` 内部生命周期；
- source-driven 多输入调度；
- 真正按 logical rows 和 estimated bytes 双上限的 per-edge channel；
- `Block` 背压和显式 lossy 策略；
- end-of-input、排空、取消和优雅停止；
- 基础队列、吞吐、阻塞时间和内存指标。

验收重点：双 source 持续运行、慢 sink 反压至 source、超大单 batch 受 bytes 限制、
EOF 排空、取消无 task 泄漏、现有 batch executor 行为不变。

### M2：事件时间，3 至 5 周

交付物：

- 强类型 `EventTime` 和 watermark message；
- source-provided 与 generated watermark；
- per-ingress 单调校验、multi-input minimum、idle/reactivate；
- late-row drop 和指标；
- watermark state checkpoint/restore。

验收重点：乱序数据、多输入快慢分区、idle 后重新活跃、恢复后 watermark 不回退。

### M3：状态后端与 final-only 窗口，6 至 10 周

交付物：

- keyed state API；
- 本地增量 Arrow IPC/Parquet state backend；
- manifest、retention 和 compaction；
- tumbling/hopping aggregate；
- watermark close 后一次性输出。

验收重点：与离线 group-by 结果一致；大量窗口状态不进入 JSON；恢复后不会因为
重放而改变最终窗口结果；compaction 前后结果一致。需要注意，“同一窗口只对外
关闭一次”在 at-least-once 下只能在事务/幂等 sink 边界成立；算子边界在故障重放
时必然可能重新发出已关闭窗口，因此这条断言应该写在 sink 层，而不是算子层。

### M4：Epoch checkpoint 与事务 sink，8 至 12 周

交付物：

- source barrier injection；
- multi-input alignment、timeout 和失败收敛；
- operator/source/sink 同一 epoch 快照；
- transactional/idempotent sink capability；
- pre-commit、commit、abort/recover；
- delivery guarantee 编译期诊断。

验收重点：在 barrier 前后、snapshot 中、pre-commit 后、manifest 提交后和 commit
回调中分别注入故障；文件 sink 恢复后无重复、无丢失、无不可见临时文件泄漏。

### M5：Connector 与 project v3，16 至 24 周

建议顺序：

1. bounded file/Parquet source 与 transactional Parquet sink；
2. Kafka source/sink 与 offset/transaction 协调；
3. PostgreSQL snapshot/incremental/`pgoutput` CDC source，以及 append/upsert/
   ledger-transaction sink；CDC 以 append-only change-event envelope 输出；
4. ClickHouse snapshot/incremental polling source 与 batch sink；通用交付契约为
   at-least-once，insert token 只声明 retry-deduplicated；
5. HTTP polling source；
6. WebSocket 等不可重放 source，明确 weaker delivery；
7. 可选 NATS 和 lookup connector。

同时更新 Rust、PyO3、Python functional API、project v3、Studio、OpenAPI 和生成的
TypeScript 类型。

验收重点：从 data-only project 声明端到端 pipeline；非法 capability 组合在编译期
失败；secret 不进入 project、fingerprint、日志或 checkpoint；schema migration 可
重放且确定。

### 9.1 排期判断

原始 3 至 6 周级别的各里程碑估算适合做窄原型，不足以覆盖 Calc-Flow 当前所有
公共表面及可靠性要求。加入 PostgreSQL CDC/transactional sink 与 ClickHouse
source/sink 后，按一名熟悉 Rust async、Arrow/DataFusion、数据库协议、PyO3 和
Studio 的工程师估算，完整生产化更合理的总量级约为 11 至 16 个月；并行投入可
缩短日历时间，但状态、checkpoint 和故障测试仍存在强顺序依赖。

上述量级与实现计划的分解一致：计划把同一范围拆成 M0 至 M7，合计约 47 至 70
engineer-weeks，正好落在 11 至 16 个月区间内。两份文档的排期口径应始终保持
一致；本报告 §9 的里程碑划分与计划的里程碑编号并不一一对应，引用时需注明依据
的是哪一份分解。

## 10. 自研还是采用现成系统

| 需求                                                                   | 推荐选择                                           |
| ---------------------------------------------------------------------- | -------------------------------------------------- |
| 嵌入 Python/本地进程，复用 Calc-Flow DAG、数组 provider 和 Studio      | 演进 Calc-Flow continuous runtime                  |
| 分布式 Arrow/DataFusion pipeline、SQL-first、外部 sink 为主            | 优先评估 Arroyo                                    |
| 多租户 SQL、物化视图、可查询 serving、大状态、原生 CDC、集群 HA        | 优先评估 RisingWave                                |
| 近期只需 Kafka/CDC 到可查询表                                           | 不应等待 Calc-Flow 自研 connector 与状态系统       |
| 量化本地数据流、Python UDF/JAX 边界、轻量部署                           | Calc-Flow 有明确差异化价值                         |

Calc-Flow 的建设理由应来自嵌入式、Python/Arrow/array 混合工作负载和可视化
编排，而不是追赶分布式流数据库的功能清单。

## 11. 主要风险与决策门槛

- **语义风险**：watermark、barrier 和 cancellation 若先写代码后补规范，很容易形成
  无法兼容的公开行为。
- **内存风险**：只按消息数量限制队列无法约束超大 Arrow batch；必须有 bytes 预算。
- **状态风险**：窗口先于状态后端会把 JSON checkpoint 变成技术债。
- **一致性风险**：把普通 `Sink` 宣传为 exactly-once 会产生错误安全感，必须按
  capability 编译和展示。
- **changelog 风险**：没有 retract/upsert 契约时不应承诺 early update、session
  merge 后修正或完整流式 SQL。
- **生态风险**：Kafka、Avro、CDC 和云存储 connector 的维护成本往往高于核心
  算子实现。
- **依赖风险**：应复用 DataFusion 公共 extension 能力，避免走向 Arroyo 式深度
  fork，除非有独立、长期维护预算。
- **定位风险**：一旦引入分布式调度、PG serving 或 Hummock 等价物，产品就不再是
  当前的轻量嵌入式 Calc-Flow，需要重新做产品和组织决策。

## 12. 参考来源

### Arroyo

- [GitHub 仓库固定快照](https://github.com/ArroyoSystems/arroyo/tree/f6afb832c00d25a349522ba2678d55a6866a9fab)
- [官方架构文档](https://doc.arroyo.dev/architecture/)
- [核心概念](https://doc.arroyo.dev/concepts/)
- [Connector 文档](https://doc.arroyo.dev/connectors/)
- [v0.15.0 发布说明](https://doc.arroyo.dev/releases/v0.15.0/)
- [ArrowMessage 与 SignalMessage 源码](https://github.com/ArroyoSystems/arroyo/blob/f6afb832c00d25a349522ba2678d55a6866a9fab/crates/arroyo-types/src/lib.rs#L175-L193)
- [背压 channel 源码](https://github.com/ArroyoSystems/arroyo/blob/f6afb832c00d25a349522ba2678d55a6866a9fab/crates/arroyo-operator/src/context.rs#L88-L157)
- [Cloudflare 收购公告](https://blog.cloudflare.com/cloudflare-acquires-arroyo-pipelines-streaming-ingestion-beta/)

### RisingWave

- [v3.0.2 release](https://github.com/risingwavelabs/risingwave/releases/tag/v3.0.2)
- [官方架构文档](https://docs.risingwave.com/get-started/architecture)
- [Hummock state store 设计](https://risingwavelabs.github.io/risingwave/design/state-store-overview.html)
- [Delivery guarantee 概述](https://docs.risingwave.com/delivery/overview)
- [Kubernetes 扩缩容](https://docs.risingwave.com/deploy/k8s-cluster-scaling)
- [时间窗口](https://docs.risingwave.com/processing/sql/time-windows)
- [Watermark](https://docs.risingwave.com/processing/watermarks)
- [CDC ingestion](https://docs.risingwave.com/ingestion/cdc-with-risingwave)
- [Join 能力](https://docs.risingwave.com/processing/sql/joins)
- [StreamChunk 固定版本源码](https://github.com/risingwavelabs/risingwave/blob/391c3a16ef26d0cd86d1236c9b7c122a9a27fb1e/src/common/src/array/stream_chunk.rs#L38-L109)

### 数据库 Connector 设计依据

- [PostgreSQL Logical Decoding](https://www.postgresql.org/docs/current/logicaldecoding.html)
- [PostgreSQL Streaming Replication Protocol](https://www.postgresql.org/docs/current/protocol-replication.html)
- [PostgreSQL `pg_replication_slots`](https://www.postgresql.org/docs/current/view-pg-replication-slots.html)
- [ClickHouse 26.1：insert deduplication token](https://clickhouse.com/blog/clickhouse-release-26-01)
- [ClickHouse ReplacingMergeTree 与 `FINAL`](https://clickhouse.com/resources/engineering/clickhouse-optimize-table-final)

### Calc-Flow

- [`docs/introduction.md`](../introduction.md)
- [`docs/runtime-envelope.md`](../runtime-envelope.md)
- [`crates/calc-flow/src/pipeline.rs`](../../crates/calc-flow/src/pipeline.rs)
- [`crates/calc-flow/src/runtime/streaming.rs`](../../crates/calc-flow/src/runtime/streaming.rs)
- [`crates/calc-flow/src/runtime/micro_batch.rs`](../../crates/calc-flow/src/runtime/micro_batch.rs)
- [`crates/calc-flow/src/runtime/envelope.rs`](../../crates/calc-flow/src/runtime/envelope.rs)
- [`crates/calc-flow/src/pipeline/control.rs`](../../crates/calc-flow/src/pipeline/control.rs)
- [`crates/calc-flow/src/checkpoint.rs`](../../crates/calc-flow/src/checkpoint.rs)
- [`crates/calc-flow/src/io.rs`](../../crates/calc-flow/src/io.rs)
