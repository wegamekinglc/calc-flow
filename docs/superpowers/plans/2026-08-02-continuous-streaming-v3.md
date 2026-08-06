# Calc-Flow 3.0 持续流计算详细开发计划

> **状态：** M0 与 M1 已完成；M2 的 crate-private runtime internals 已在
> code-approved head `7deda2a0` 完成。当前 public v2 runner 保持不变；public A6
> 必须等 M4/M5 完整状态与 checkpoint 语义完成后另行评审、原子集成。
>
> **依据：**
> [`Arroyo / RisingWave 独立调研与 Calc-Flow 流式演进建议`](../../research/2026-08-02-arroyo-risingwave-streaming-research.md)
>
> **兼容性决策：** 本计划面向 Calc-Flow `3.0.0` 的主动破坏性升级。不保留
> project v2、checkpoint v2、现有 runner API、Python 签名或 Studio `/api/v2`
> 的兼容层，也不提供自动迁移 shim。

**总目标：** 将 Calc-Flow 建设为单进程、嵌入式、Arrow-native、Python 友好的
有状态流计算引擎，同时保留一个职责清晰的有限批执行器。

**核心架构：** 把当前含义混合的执行模型拆成 `BatchExecutionPlan` 和
`StreamExecutionPlan`。流计划由 source task 主动驱动，节点间通过同时限制 envelope
数量、行数与估算字节数的有界 channel 连接；data、watermark、barrier、idle 和
end-of-input 在同一条 edge 上保序。coordinator 统一管理 epoch、barrier 对齐、状态
manifest 和事务 sink 提交。

---

## 1. 3.0 交付边界

### 1.1 必须交付

- 有限输入的一次性 `BatchExecutionPlan`。
- source 驱动、可长期运行的 `StreamExecutionPlan` 和 `StreamingRunner`。
- 多 source 并发、独立分支、fan-out 和同 schema 多输入 union。
- 每条 edge 上 data/control 统一 FIFO。
- 用两个配置字段独立限制 envelope 数量、rows 和 estimated bytes 的异步 channel。
- 从 sink 向 operator、source 逐级传播的背压。
- event time、watermark、idle input 和 late-row 指标。
- final-only tumbling/hopping aggregate window。
- 本地增量状态 segment、bounded checkpoint manifest 和后台 compaction。
- epoch barrier 对齐 checkpoint。
- 普通 at-least-once sink 和经能力验证的 transactional sink。
- 文件/Parquet、Kafka、PostgreSQL、ClickHouse、HTTP polling、WebSocket
  connector。
- 严格、data-only 的 project v3。
- Rust、PyO3/Python、Studio 三个完整操作面：启动、状态、checkpoint、优雅停止、
  取消和恢复。

### 1.2 3.0 明确不做

- 分布式 worker、shuffle、slot、rescale 或 controller HA。
- PostgreSQL wire protocol 或 RisingWave 式 serving database。
- Hummock 等价物、通用分布式 LSM 或强制对象存储。
- session window、early trigger、allowed lateness、side output。
- engine-level changelog/retract/upsert 流；PostgreSQL CDC 只输出 append-only
  change-event envelope，不自动物化数据库表状态。
- multi-input 增量 SQL join 和 event-time temporal join。
- 自动 schema evolution、ClickHouse CDC 或 PostgreSQL 之外的数据库原生 CDC。
- DataFusion、Arrow、sqlparser 的深度 fork。
- 对 project/checkpoint/API v2 的兼容和自动迁移。

### 1.3 允许并要求发生的破坏性变更

- 用 `BatchOperator` 与 `StreamOperator` 替换现有 `Operator`。
- 用 `BatchExecutionPlan` 与 `StreamExecutionPlan` 替换现有
  `ExecutionPlan`。
- 用 `compile_batch()` 与 `compile_stream()` 替换 `compile()`。
- 删除 `MicroBatchRunner`。它当前提供的“可重放 source + 逐 batch checkpoint +
  at-least-once”能力由新的 `StreamExecutionPlan` 承接：使用有界 source，配合
  `EndOfInput` 与 epoch checkpoint 即可覆盖同一场景。这是公共能力的替换而不是
  净删除，`CHANGELOG.md` 与迁移指南必须显式写出这条替换路径。
- 重新定义 `StreamingRunner`：它成为 source-driven continuous runner；删除
  当前调用者逐批 `step(batch)` 的模型。
- 用流原生 source/sink trait 和 binding 替换 `Source`、`SourceItem`、`Sink`、
  `SinkRouter`。
- checkpoint 格式直接升级为 v3；v2 文档返回 `UnsupportedVersion`。
- project 格式直接升级为 v3；v2 文档返回 `UnsupportedVersion`。
- Python API 直接换成 v3，不保留弃用别名。
- Studio 直接切换到 `/api/v3`，删除 `/api/v2`。
- Rust crate、Python core、Studio backend、frontend 最终统一升级到 `3.0.0`。

`tests/fixtures/v1/` 仍按仓库要求保持不变。旧 v2 schema 可以作为历史证据移动到
历史文档目录，但任何 3.0 runtime 路径都不得读取它。

## 2. 目标架构

```text
Project v3 / Rust builder / Python builder
                    │
                    ▼
              共享图编译器
                │       │
                ▼       ▼
       BatchExecution  StreamExecution
             Plan          Plan
              │             │
              │      StreamingRunner
              │             │
              │     ┌───────┴────────┐
              │     ▼                ▼
              │ source tasks     coordinator
              │     │          epoch / barrier
              │     ▼                │
              │ bounded edge channels
              │     │                │
              │     ▼                ▼
              │ operator tasks ─ checkpoint ack
              │     │                │
              │     ▼                ▼
              │  sink tasks ─── transactional commit
              │                      │
              ▼                      ▼
          RunResult          local state backend
                              + checkpoint v3
```

### 2.1 统一流消息

新的内部消息直接表达真实语义，不再沿用 opaque UUID occurrence：

```rust
enum StreamMessage {
    Data(Batch),
    Watermark(EventTime),
    Barrier(Epoch),
    Idle,
    EndOfInput,
}
```

必须满足以下不变量：

- 同一 edge 上 data 与 control 统一 FIFO。
- 同一 ingress 上 watermark 单调不降。
- source task 必须在此前数据全部入队后再把 barrier 放入 edge；coordinator 不能
  直接向下游队列插 barrier，以免越过数据。
- `Idle` 只把 ingress 排除出 watermark minimum，不排除 checkpoint 参与资格。
- idle ingress 收到新 data 时先恢复 active，再判断数据是否 late。
- `EndOfInput` 对单个 ingress 永久终止；全部 ingress 结束后 operator flush 一次，
  每个 output 只转发一个结束标记。
- runtime 是 control 转发的唯一所有者。operator 可以在 watermark 到来时输出已关闭
  窗口，但不能自行伪造、压制或重排 barrier/epoch。
- source task 必须能在等待外部数据的同时响应 barrier 请求。这条不变量决定了
  source trait 的形状，必须在 M0 二选一并写进规范：要么显式声明“取下一项”的
  future 是取消安全的（丢弃 future 不丢数据、不前进 cursor），要么由 runtime 用
  预取槽位把外部 I/O 与控制响应拆到两个协作单元。若两者都不做，一个长期无数据的
  source 会让每次 checkpoint 都走到超时失败。

### 2.2 批算子与流算子分离

以下是 M0 评审的目标形态，不是预先冻结的最终签名：

```rust
#[async_trait]
pub trait BatchOperator: Send + Sync {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>>;

    fn snapshot(&self) -> Result<Value>;
    fn restore(&mut self, state: &Value) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
}

#[async_trait]
pub trait StreamOperator: Send + Sync {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    async fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorCheckpoint>;
    async fn restore(&mut self, checkpoint: &OperatorCheckpoint) -> Result<()>;
}
```

上面的草图只画出两者的差异部分，省略了编译器今天就依赖的成员。M0 冻结签名时，
`name()`、`input_ports()`、`output_ports()`、`configuration()` 和
`udf_references()` 必须在两个 trait 上都保留或提升到共享的 `OperatorMetadata`
supertrait，否则 `compile_batch()` 与 `compile_stream()` 无法完成端口、schema 和
UDF 校验。`reset()` 目前只出现在 batch 侧，流算子在重启、`reset()` 与从
checkpoint 恢复三种入口下的状态语义也必须一并冻结。

首版支持矩阵：

| 算子                         | 3.0 流式行为                                                      |
| ---------------------------- | ----------------------------------------------------------------- |
| Expression                   | 单输入；一个输入 batch 产生零或一个输出 batch                     |
| 单 alias SQL                 | 单输入；逐 batch 执行                                             |
| 多 alias SQL                 | 编译失败；尚未定义增量 join 语义                                  |
| External provider            | factory 必须显式构造 `StreamOperator`                             |
| Union                        | 新增多输入算子；转发同 kind、同 schema batch                      |
| Window aggregate             | 新增有状态 tumbling/hopping 算子                                  |
| Session window               | 3.0 不支持                                                        |

每个 table stream-operator task 拥有一个 lazy、operator-scoped
`DataFusionRuntime`，避免所有节点争用一个 job-wide query lock，也避免临时表 alias 跨
节点污染。纯 array task 不初始化 DataFusion。

### 2.3 Runner 生命周期

目标 Rust 使用方式：

```rust
let plan = builder.compile_stream(&udfs)?;
let runner = StreamingRunner::new(
    plan,
    source_bindings,
    sink_bindings,
    state_backend,
    checkpoint_store,
    StreamRuntimeConfig::default(),
)?;
let job = runner.start().await?;

job.status();
job.trigger_checkpoint().await?;
job.shutdown().await?;
job.wait().await?;
```

`StreamingJob` 是资源所有者，不是一次 `step()` 的返回值。M0 必须明确：谁拥有 task、
drop 的行为、`shutdown` 与 `cancel` 的区别、`wait` 是否幂等、手动 checkpoint 与周期
checkpoint 如何串行化。

### 2.4 Delivery guarantee 不是全局宣传语

```text
不可重放或主动丢数据的 source
    => best effort / 局部 at-most-once

可重放 source + 普通 sink
    => at-least-once

可重放 source + 确定性 operator
    + aligned epoch checkpoint + transactional/idempotent sink
    => exactly-once
```

用户请求 `delivery = "exactly_once"` 时，只要 source、operator、backpressure policy、
state backend 或 sink 任一能力不满足，编译器必须在任何外部副作用前失败。

## 3. 建议模块布局

```text
crates/calc-flow/src/
  batch.rs
  operator/
    mod.rs
    batch.rs
    stream.rs
    expression.rs
    sql.rs
    union.rs
    window.rs
  pipeline/
    mod.rs
    compile.rs
    batch.rs
    stream.rs
  runtime/
    mod.rs
    batch.rs
    streaming/
      mod.rs
      channel.rs
      context.rs
      coordinator.rs
      job.rs
      message.rs
      metrics.rs
      operator_task.rs
      progress.rs
      sink_task.rs
      source_task.rs
      supervisor.rs
      transaction.rs
  time/
    mod.rs
    event_time.rs
    watermark.rs
  state/
    mod.rs
    backend.rs
    local.rs
    manifest.rs
    segment.rs
  checkpoint/
    mod.rs
    model.rs
    store.rs
  connector/
    mod.rs
    capability.rs
    format.rs
    registry.rs

crates/calc-flow-connectors/
  src/
    lib.rs
    csv.rs
    clickhouse.rs
    database_types.rs
    file.rs
    http.rs
    json.rs
    kafka.rs
    parquet.rs
    postgresql.rs
    websocket.rs
```

核心 trait、能力验证、编译器、runtime、state 和 checkpoint 协议留在
`calc-flow`。`rdkafka`、PostgreSQL、ClickHouse、HTTP、WebSocket 等较重客户端依赖
进入独立 `calc-flow-connectors` crate，避免污染核心 crate。

新增第三个 workspace crate 会牵动打包、CI 与覆盖率，必须在 M0 一次性决定，不能留到
M6 才发现：

- **依赖边**：`calc-flow-python` 是否依赖 `calc-flow-connectors`。若依赖，rdkafka、
  TLS 与压缩库会进入 wheel 的原生模块，需要确认 manylinux/macOS/Windows 的构建与
  SBOM；若不依赖，则 M6.7 中“Python/Studio 运行 PostgreSQL CDC 到 ClickHouse”的
  验收门不可达，必须改写。二者只能择一。
- **feature gate**：每个 connector 一个 cargo feature，默认全部关闭。否则任何一次
  普通构建都会拉起 rdkafka 与数据库客户端。
- **`--all-features` 影响**：`AGENTS.md` 的 clippy、`cargo llvm-cov` 和 `cargo doc`
  都使用 `--all-features`，启用全部 connector feature 后 CI 需要系统级依赖，必须
  同步更新命令与开发环境说明。
- **覆盖率底线**：workspace 有 90% 行覆盖门，而 connector 的主要路径靠容器测试，
  且计划明确要求容器测试不混入普通单测。必须在 M0 决定是把
  `calc-flow-connectors` 排除在 workspace 覆盖门之外并给它单独的门槛，还是把容器
  测试纳入覆盖率采集。不做决定就会在 M6 撞上无法通过的门。
- **版本与发布**：release invariant 要求各包版本同步。`calc-flow-connectors` 的
  版本策略、是否随核心 crate 发布、以及 crate 打包内容都要一并写入 M7.4。

## 4. 里程碑、依赖与排期

```text
M0 语义、API、故障模型
        │
        ▼
M1 v3 类型和编译器重构
        │
        ▼
M2 持续 task runtime
        │
        ▼
M3 event time / watermark
        │
        ▼
M4 state backend / window
        │
        ▼
M5 epoch checkpoint / exactly-once
        │
        ▼
M6 connector / project v3 / Python / Studio
        │
        ▼
M7 hardening / 3.0 release
```

| 里程碑 | 核心产出                                      | 单工程师估算 | 前置依赖 |
| ------ | --------------------------------------------- | ------------ | -------- |
| M0     | 通过评审的语义、API、故障模型                 | 2 至 3 周    | 无       |
| M1     | v3 trait、plan、compiler、message、channel    | 4 至 6 周    | M0       |
| M2     | crate-private source-driven runtime internals | 5 至 7 周    | M1       |
| M3     | event time、watermark、idle、late metrics     | 3 至 5 周    | M2       |
| M4     | 增量本地状态与 final-only window              | 6 至 9 周    | M3       |
| M5     | epoch checkpoint 与 transactional sink        | 8 至 11 周   | M4       |
| M6     | connector、project v3、Python、Studio         | 16 至 24 周  | M5       |
| M7     | soak、性能、安全、打包与发布                  | 3 至 5 周    | M6       |

顺序总量约 47 至 70 engineer-weeks，与调研报告 §9.1 给出的“单工程师 11 至 16 个月”
量级一致。两名熟练工程师可并行数据库 connector、其他 connector、Python、Studio、
测试和文档，但 state/checkpoint 位于强依赖关键路径，更可信的日历时间约 34 至 46 周。
调研报告 §9 的里程碑划分与本表的 M0 至 M7 并非一一对应，引用排期时必须注明依据的是
哪一份分解。

### 4.1 可合并性策略

每个合入分支都必须可编译、可测试，不能因为是破坏性升级就把 `main` 长期留在红色：

- M1.1 与 M1.2 是一个原子 PR：trait 和 plan 公共名称必须一起切换。
- M1.3、M1.4 可在其后分别合入。
- M2.1 至 M2.5 以 crate-private 模块完成。public runner 不在 M2.4 公开；它作为
  post-M5 A6，在 M4/M5 状态与 checkpoint 语义完整后另行评审并原子集成。
- M5.1 至 M5.4 使用一个 milestone integration branch 接受 stacked PR，完整协议通过
  后一次合入 `main`，避免暴露半套 checkpoint 公共 API。
- M6.7 至 M6.9 同样使用 stacked PR：Rust project v3、Python、Studio、生成文件全部
  green 后再完成公共 v2 删除。
- 临时并存只能是开发分支内部脚手架，不是 3.0 产品兼容承诺。
- 不建立贯穿 M1 到 M7 的超长 feature branch。

## 5. 每个开发任务的统一执行规则

所有行为变更必须按以下顺序执行：

1. 增加最小、聚焦的失败测试。
2. 运行该测试并记录符合预期的 RED 原因。
3. 只实现本任务要求的行为。
4. 运行聚焦测试至 GREEN。
5. 在不改变已证明行为的前提下重构。
6. 运行受影响的 Rust、Python、backend、frontend、schema 或 packaging 检查。
7. 运行 `git diff --check`，核对所有生成文件是否按预期变化。
8. 由 reviewer 检查 exact final diff；公共 API 变更必须先有 API note 和 critique。

全局约束：

- 保持 `unsafe_code = "forbid"`。
- 文件、压缩、connector、compaction 不能阻塞 Tokio executor thread。
- manifest、fingerprint、错误聚合、metrics snapshot 和生成文档使用确定性
  `BTreeMap` 排序。
- 不序列化函数、connector object 或其他 executable object。
- secret value 不得进入 project、fingerprint、日志、指标、state segment 或
  checkpoint manifest。
- 正确性测试不得使用时间性能断言。
- 不允许 detached Tokio task。success、failure 和显式 cancel 三条路径都必须 join。
  `Drop` 里无法 await，也不允许为了 join 而阻塞 Tokio worker thread，因此 drop 的
  契约是“取消并释放所有权”，join 由 `wait()` 承担；M0 必须明确 drop 之后未 join 的
  task 何时被回收，以及测试如何在不依赖 `Drop` 内 join 的前提下断言无泄漏。
- 所有 edge queue 在入队前同时验证 rows 与 estimated bytes。
- 单 batch 超过 edge byte limit 时直接报错，不允许“一条超限消息”例外。为此
  `compile_stream()` 必须校验 source 侧配置的最大 batch bytes/rows 不超过其下游
  edge 容量，把这类错误提前到 source 打开之前；同时注意现有
  `io.rs` 的 `BatchingSource` 对首个超限 item 是照常发出而不是报错，M1.3 重构时
  必须显式统一这两处策略，并在 `CHANGELOG.md` 记录行为变化。
- `tests/fixtures/v1/` 保持不变。

## 6. M0：实现前冻结语义

### Task M0.1：编写规范性流语义规格

**文件：**

- 新建：`.codex/artifacts/specs/continuous-streaming-runtime.md`
- 读取：`docs/runtime-envelope.md`
- 读取：`docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`

**步骤：**

- [ ] 定义 data、watermark、barrier、idle、end 在 edge 上的 FIFO。
- [ ] 定义 source sequence/cursor 的严格递增、重复、回退和恢复规则。
- [ ] 定义 fan-out 顺序，明确 sibling branch 之间没有全局顺序保证。
- [ ] 定义 union 跨 ingress 的选择顺序：每 ingress FIFO，但 ready ingress 间不承诺
  业务顺序。
- [ ] 定义 watermark 单调性、idle、reactivate、end 和最终 flush。
- [ ] 冻结 `EventTime` 内部精度，规定 Arrow second/millisecond/microsecond/
  nanosecond 的 checked conversion，并统一截断方向为向下取整：事件时间向下取整
  保证行不会被推进到更晚的窗口，watermark 向下取整保证进度估计保持保守。
- [ ] late boundary 按窗口而不是按行定义：一行迟到当且仅当它所属窗口的
  `window_end <= 当前输入 watermark`，即该窗口已经关闭。**不得**采用
  `event_time <= output_watermark`：只要 watermark delay 小于 window size，该判据
  就会丢弃大量本应进入尚未关闭窗口的正常数据（1 小时 tumbling、watermark 10:30、
  事件时间 10:15 的行属于尚未关闭的 `[10:00, 11:00)`，必须接受）。非窗口算子若
  需要独立的 late 判据，必须单独记录并说明与窗口判据的关系。
- [ ] 定义 source 在等待外部数据时如何响应 barrier 请求：明确“取下一项”是否取消
  安全，或改由 runtime 预取槽位解耦；这项选择将直接决定 `StreamSource` 的签名。
- [ ] 定义 state segment 发布与 checkpoint manifest 发布的先后顺序，并指定
  checkpoint manifest 为“最近完成 epoch”的唯一真相。
- [ ] 定义 final-only tumbling/hopping 的触发与输出顺序。
- [ ] 定义 barrier alignment、timeout、cancel 和 source cursor 边界。
- [ ] 定义 job terminal state 和并发多错误的确定性选择规则。
- [ ] 定义 at-least-once/exactly-once 的完整能力矩阵。
- [ ] 将本计划所有 non-goal 写入规范。

**验收门：** 每条语义都有至少一个可执行 acceptance test。watermark、EOF 或
checkpoint 顺序仍有歧义时禁止进入 M1。

### Task M0.2：设计并对抗性评审公共 API

**文件：**

- 新建：`.codex/artifacts/api-notes/continuous-streaming-runtime.md`
- 新建：`.codex/artifacts/critiques/continuous-streaming-runtime.md`
- 读取：`crates/calc-flow/src/lib.rs`
- 读取：`python/calc_flow/runtime.py`
- 读取：`web-ui/backend/src/calc_flow_studio/models.py`

**步骤：**

- [ ] 冻结 Rust 类型名、所有权、async 方法和错误 variant。
- [ ] 验证 `StreamCollector` 的 object safety、生命周期和 error propagation。
- [ ] 冻结 `StreamingRunner` 的 start/wait/shutdown/cancel/checkpoint/status。
- [ ] 冻结 Python async/blocking API；blocking API 在 running event loop 中必须拒绝。
- [ ] 冻结 project v3 顶层结构和 secret reference。
- [ ] 冻结 Studio `/api/v3` job route 与 SSE event model。
- [ ] 对 cancellation safety、reentrancy、task ownership、checkpoint recovery、
  capability spoofing、secret persistence 做对抗性评审。
- [ ] 明确 `StreamSource` 取下一项的取消安全契约，以及 `Drop` 只取消不 join 的
  所有权模型；这两点必须由 critique 判定为无 `Block` 才能进入 M1。
- [ ] 决定 barrier 转发时机：operator 完成同步快照后立即转发，还是等待 coordinator
  ack。若选择后者，必须记录 checkpoint 延迟为 O(图深度 × 往返) 的代价与上限。
- [ ] 所有 `Block` 结论清零后才进入 M1。

**验收门：** spec、API note、critique 使用同一个
`continuous-streaming-runtime` slug，且没有互相矛盾的定义。

### Task M0.3：记录正确性与性能基线

**文件：**

- 读取：`crates/calc-flow/benches/core.rs`
- 读取：`crates/calc-flow/benches/allocation_regression.rs`
- 新建：`docs/superpowers/handoffs/2026-08-02-continuous-streaming-v3-baseline.md`
- 仅生成：`target/cargo/criterion/`

**步骤：**

- [ ] 在精确起始 commit 运行 Rust、Python、Studio backend、frontend 和 schema
  检查。
- [ ] 保存 expression、SQL、external passthrough、DataFusion runtime creation、
  checkpoint persistence 的 Criterion baseline。
- [ ] 记录 `allocation_regression` 当前的冻结分配基线，并明确 M1 的破坏性重构将
  改写该 harness；重写后必须重新建立基线，而不是静默放宽阈值。
- [ ] 记录编译器、CPU、内存、target 路径、point estimate、confidence interval 和
  失败项。
- [ ] 性能回归门设为 5%，但必须由同机配对数据和置信区间共同支持，不能只比较
  单次点估计。

**验收门：** 后续可以把 stream runtime 回归与既有 DataFusion、Python、Studio
成本分开归因。

## 7. M1：重建 v3 核心类型与编译器

### Task M1.1：原子拆分 operator 与 plan

**文件：**

- 替换：`crates/calc-flow/src/operator.rs`
- 新建：`crates/calc-flow/src/operator/{mod,batch,stream}.rs`
- 替换：`crates/calc-flow/src/pipeline.rs`
- 新建：`crates/calc-flow/src/pipeline/{mod,compile,batch,stream}.rs`
- 修改：`crates/calc-flow/src/lib.rs`
- 修改：`crates/calc-flow/tests/{operator,pipeline_compile,pipeline_execute}.rs`
- 修改：`crates/calc-flow/tests/{properties,workspace,v1_fixtures}.rs`
- 修改：`crates/calc-flow/benches/{core,allocation_regression}.rs`
- 新建：`crates/calc-flow/tests/{stream_operator,stream_compile}.rs`

以上后两组是本次破坏性重构必然波及、但容易被漏掉的既有产物：它们直接依赖
`Operator`、`ExecutionPlan`、`ExecutionOptions` 和 `PipelineBuilder::compile()`。
`tests/fixtures/v1/` 的数据文件保持不变，但读取它的 `tests/v1_fixtures.rs`
harness 必须迁移到 `compile_batch()` 路径。

**先写 RED：**

- [ ] batch operator 收到完整、不可变 input map。
- [ ] stream operator 每次收到一个命名 ingress batch。
- [ ] collector 在入队前拒绝未知 output port 或 schema mismatch。
- [ ] stream operator 不能直接 emit barrier 或任意 watermark。
- [ ] `compile_batch()` 接受当前合法 batch graph。
- [ ] `compile_stream()` 接受 unary chain、fan-out、独立分支和 same-schema union。
- [ ] `compile_stream()` 拒绝 multi-input SQL 和不支持 stream 的 provider。
- [ ] graph 插入顺序不同但语义相同，fingerprint 保持确定。
- [ ] batch/stream 语义不同时 fingerprint 必须不同。
- [ ] 只改 channel 容量或 checkpoint interval 时语义 fingerprint 不变，runtime
  config hash 改变。

**实现：**

- [ ] 用 `BatchOperator`、`BatchOperatorContext` 替换旧 trait/context。
- [ ] 引入 `StreamOperator`、`StreamOperatorContext`、`StreamCollector`、
  `OperatorCheckpoint`。
- [ ] 提取共享 node/edge/port/schema/cycle/UDF/topology 校验。
- [ ] 引入 `BatchExecutionPlan` 和 `StreamExecutionPlan`。
- [ ] 替换 builder 的 `compile()`。
- [ ] 为 stream plan 编译稳定 edge ID、source binding slot、sink binding slot。
- [ ] 新增至少两个输入的 `UnionOperator`。
- [ ] stream 模式只允许 unary expression、single-alias SQL 和显式 stream provider。
- [ ] 拆成两个独立摘要：语义 fingerprint 只包含 execution mode、图结构、operator
  配置、UDF catalog 和影响状态布局的 window/state 语义，决定 checkpoint 兼容性；
  runtime config hash 单独记录 channel 容量、checkpoint interval 等可调项，只用于
  可观测性与诊断。调大队列容量不得作废既有 checkpoint。
- [ ] 删除旧 `SignalAwareOperator` 测试桥和 batch plan control-route，同时删除
  `crates/calc-flow/src/pipeline/{signal_allocation_tests,runtime_envelope_tests}.rs`，
  并把其中仍然有效的顺序、分配和 fail-closed 断言迁移到新的 stream 测试中，不得
  静默丢失覆盖。

**聚焦验证：**

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" cargo test -p calc-flow --test operator \
  --test stream_operator --test pipeline_compile --test pipeline_execute \
  --test stream_compile
```

**验收门：** 公共类型不再用一个 `Operator` 或 `ExecutionPlan` 混淆两种生命周期；
所有合法/非法 stream graph 在 source 打开前完成判定。

### Task M1.2：用强类型 stream message 替换 opaque marker

**文件：**

- 删除：`crates/calc-flow/src/runtime/envelope.rs`
- 新建：`crates/calc-flow/src/runtime/streaming/message.rs`
- 新建：`crates/calc-flow/src/time/{mod,event_time}.rs`
- 新建：`crates/calc-flow/tests/stream_message.rs`
- 重写：`docs/runtime-envelope.md`

**先写 RED：**

- [ ] `EventTime` 正确排序 Unix epoch 前后的 UTC 微秒值。
- [ ] Arrow timestamp second/millisecond/microsecond/nanosecond 转换执行精度和溢出
  校验。
- [ ] `Epoch` 遵循 M0 规定的起始值和递增规则。
- [ ] fan-out clone 保持 `Batch` 底层不可变共享。
- [ ] Debug 输出不包含 batch payload 或 secret attribute。

**实现：**

- [ ] 引入 `EventTime`、`Epoch` newtype，禁止 public API 使用裸 `i64`。
- [ ] 固定内部时间精度和 checkpoint 序列化精度，并统一向下取整的截断方向；
  超出内部精度的 Arrow 输入要么按该方向截断，要么在入队前带列路径报错，不允许
  按实现方便逐处选择。
- [ ] 引入 `StreamMessage` 和只用于诊断的 private metadata。
- [ ] 删除 UUID occurrence。
- [ ] control 构造保持 crate-private；source 只能通过验证后的 `SourceEvent` 构造器
  发控制事件。
- [ ] 将 `docs/runtime-envelope.md` 重写为真实 v3 contract。

**验收门：** watermark/epoch 都具有业务值和强类型，不再是占位 UUID。

### Task M1.3：统一 batch 内存计量

**文件：**

- 修改：`crates/calc-flow/src/batch.rs`
- 重构：`crates/calc-flow/src/io.rs`
- 修改：NumPy/JAX external payload 实现
- 新建：`crates/calc-flow/tests/batch_cost.rs`

**先写 RED：**

- [ ] table batch 使用 Arrow slice memory size 计算实际可见 slice 成本。
- [ ] external payload 必须提供 exact 或 conservative byte estimate。
- [ ] rows/bytes 求和溢出返回 typed error。
- [ ] NumPy/JAX view、owned array 和空 array 都有可解释的估算。

**实现：**

- [ ] 新增 `Batch::estimated_bytes()`。
- [ ] v3 的 `ExternalPayload` 强制实现 `estimated_bytes()`，不提供 `Option` 绕过。
- [ ] 将 `io.rs` 现有 Arrow memory measurement 移到可复用位置，并统一超限策略：
  现有 `BatchingSource` 对首个超限 item 是照常发出，v3 的 edge 入队则要求报错，
  两者必须显式收敛到一条规则。
- [ ] 明确共享 payload 的 queue charge 是逻辑占用，不声称等于进程 RSS。

**验收门：** 任意能进入 stream queue 的 `Batch` 都能在入队前给出保守 byte cost。

### Task M1.4：实现 envelope、rows 与 bytes 三重限制 channel

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/channel.rs`
- 新建：`crates/calc-flow/tests/stream_channel.rs`

**先写 RED：**

- [x] `max_rows` 或 `max_bytes` 为零时构造失败。
- [x] 单消息大于 byte limit 时入队前失败。
- [x] rows 已满时 sender 阻塞，receiver 消费后恢复。
- [x] bytes 已满时即使消息数很少也阻塞。
- [x] 即使 row/byte cost 为零，第 `R + 1` 个 envelope 仍会在
  `EdgeBudget::new(R, B)` 上阻塞。
- [x] cancelled send 只释放一次 reservation。
- [x] receiver close 唤醒全部 blocked sender。
- [x] data/control 混合消息仍 FIFO。

**实现：**

- [x] 定义 `EnvelopeCost { messages, rows, bytes }` 并使用 checked arithmetic。
- [x] 明确并在编译期依赖“每条 edge 恰有一个生产者”这一不变量（沿用现有单写入者
  校验，fan-out 的每条出边是独立 channel）。它是下面单一预算设计成立的前提，
  避免多生产者场景下 `Notify` 唤醒到无法推进的等待者而丢失唤醒。
- [x] 用单一 mutex-protected budget + `Notify` 原子检查并预留 envelope slot、rows
  和 bytes，避免多个 semaphore 分别申请造成死锁或饥饿。
- [x] 入队前 reserve，receiver 取出或 drop 时 release。
- [x] fan-out 每条 edge 独立计费，虽然 Arrow buffer 实际共享。
- [x] 暴露 queue depth、charged rows/bytes、blocked sends、blocked duration。

**验收门：** `EdgeBudget` 的 public shape 与调用签名保持不变，但 `(R, B)` 现在分别
表示最多 `R` 个 envelopes、`R` rows 和 `B` bytes。三条谓词独立成立；直接 channel
调用者必须选择 `R >= max(required_row_limit, required_simultaneous_messages)`。该窄化
只覆盖 admission 谓词，不改变 FIFO、oversize error、reservation lifecycle、single-
producer ownership 或其余 S10 行为。

## 8. M2：构建持续运行 task runtime

**当前交付状态：** M2.1-M2.5 的 runtime internals 已完成；whole-job preflight、
source/operator/sink tasks、private runner/job/reaper、status/metrics、stress 与 universal
soak 均有实现和证据。所有新增 runner/control surface 仍为 crate-private。

### Task M2.1：实现 job context 与结构化 task supervisor

**状态：** 已在 crate-private runtime 完成。

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/{mod,context,supervisor}.rs`
- 新建：`crates/calc-flow/src/runtime/streaming/runner.rs`
- 聚焦测试位于对应 private 模块的 inline test 中。

**先写 RED：**

- [x] 第一个 task failure 会 cancel 并 join 所有 sibling。
- [x] 已经同时观测到的多个错误按稳定 task ID 排序返回。
- [x] panic 转成带 task identity 的 `CalcFlowError::TaskPanicked`。
- [x] drop owning handle 会 cancel 并回收 task，不产生 detached task；断言方式不得
  依赖在 `Drop` 内 join，而应通过 supervisor 侧的注册表在 runtime 关闭时校验。
- [x] deadline 和显式 cancel 收敛到唯一 terminal state。

**实现：**

- [x] 引入 immutable `StreamJobContext`：job ID、fingerprint、settings、deadline、
  cancellation token。
- [x] 派生 source/node/sink scoped context，不修改 caller mapping。
- [x] 使用 `JoinSet` 或等价的 owned supervisor。
- [x] task 必须先登记再运行。
- [x] failure 时先 cancel、关闭入口 sender、join 全部 task，再形成稳定错误。
- [x] 定义 running、draining、completed、cancelled、failed、recovery-required。

**验收门：** success/failure/cancel/drop 四条路径都没有后台 task 泄漏。

### Task M2.2：实现 stream source binding 与 source task

**状态：** 已在 crate-private runtime 完成；public A4 source surface 仍与 A6 一起
延后到 post-M5。

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/source_task.rs`
- 新建：`crates/calc-flow/src/runtime/streaming/job.rs` 的 whole-job preflight
- 聚焦测试位于对应 private 模块的 inline test 中。

**先写 RED：**

- [x] 每个 external input 恰有一个 source binding。
- [x] unknown/duplicate binding 在任何 source open 前失败。
- [x] source 先按 recovered cursor open，再开始 poll。
- [x] 每个 source 的 sequence 严格递增。
- [x] `None` 只产生一个有序 `EndOfInput`。
- [x] 下游 edge 满时暂停后续 `next()`。
- [x] source 在 `next()` 长时间未返回时仍能观察 cancel；teardown 丢弃该 future 后
  不再 poll source，并在 pump 退出前调用 close。Barrier injection 延后到 M5。
- [x] source error 通过 supervisor cancel sibling。
- [x] connector I/O 取消后能够 join。

**实现：**

- [x] 用 private `StreamSource -> SourceEvent` seam 驱动 M2，不替换 public v2
  `Source`。
- [x] 用一项预取槽把外部 I/O 拆到独立 pump；`next()` 可以被 teardown 丢弃但该
  source 随后只允许 close，契约固定在 private trait 上。
- [x] `SourceEvent` 支持 data、source watermark 和 idle；`None` 产生 end。
- [x] binding ID 是 `BatchMetadata.source` 的权威来源；用 `with_metadata()` 不可变地
  写入 source/sequence 并保留 attributes。
- [x] 每个 binding 一个 source pump 和一个 source task。
- [x] fan-out 通过 bounded channel 发送到所有 ingress。
- [x] 控制事件必须验证后才进入 edge。
- [x] 分开记录 latest observed cursor 与 durable recovery cursor，为 M5 做准备。

**验收门：** 两个独立 source 可并发持续运行，slow sink 会停止 source polling。

### Task M2.3：实现 operator task 与 runtime-owned control forwarding

**状态：** 已在 crate-private runtime 完成。每个 compiled operator 对应一个
task；per-ingress FIFO、bounded fan-out、unary watermark/idle 的 runtime-owned
转发、multi-ingress control fail-closed、barrier fail-before-output、显式 EOF 与
`on_end` exactly once 已由当前实现固定。

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/operator_task.rs`
- 修改：`crates/calc-flow/src/operator/{stream,expression,sql,union}.rs`
- 新建：`crates/calc-flow/tests/stream_operator_task.rs`

**先写 RED：**

- [x] unary expression/SQL 按 ingress FIFO 处理每个 batch。
- [x] union 保证每 ingress FIFO，但不伪造跨 ingress 全局顺序。
- [x] fan-out 共享 batch payload，同时每条 edge 独立收费。
- [x] output kind/schema 在任何 successor 观察前校验。
- [x] operator error 后不转发半个 control 事件。
- [x] array-only chain 不初始化 DataFusion。
- [x] 每个 table operator 只创建并关闭一个 lazy DataFusion runtime。

**实现：**

- [x] 每个 compiled node 一个 task。
- [x] `select!` 多个 ingress receiver，同时保持各自 FIFO。
- [x] 只有 data 调用 `process_data()`。
- [x] control 转发始终留在 runtime。
- [x] `on_watermark()` 先输出已关闭窗口，再由 runtime 转发 watermark。
- [x] 通过 `StreamCollector` 校验并 enqueue output。
- [x] 所有 terminal path 关闭 operator-scoped DataFusion runtime。

**验收门：** two-source union -> expression 图完整输出全部数据，保持 per-source 顺序，
且 slow downstream 同时反压两条分支。

### Task M2.4：实现 crate-private sink task、runner 与 job handle

**状态：** 已完成 private M2.4。该 task 不公开或替换任何 runner；现有 public v2
`StreamingRunner` 与 `MicroBatchRunner` 保持不变，public source-driven A6 延后到
post-M5 的独立评审与原子集成。

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/{job,runner,sink_task}.rs`
- 保留：`crates/calc-flow/src/runtime/streaming.rs`
- 保留：`crates/calc-flow/src/runtime/micro_batch.rs`
- 聚焦测试位于对应 private 模块的 inline test 中。

**先写 RED：**

- [x] Whole-job preflight 在任何 source/sink lifecycle 前校验 fingerprint、topology、
  delivery、binding 和 capability。
- [x] 一个 output 的多个普通 sink 按稳定配置顺序观察数据，且只承诺 process-local
  ordered delivery。
- [x] graceful shutdown、cancellation、natural EOF 和 task failure 各自产生稳定的
  terminal outcome，并收敛到零 live task/queue/reservation/reaper。
- [x] `wait()` 幂等观察同一 terminal result；drop wait future 不取消 job，drop owning
  private job handle 会取消并由 reaper 回收。
- [x] launch failure 或 cancellation 会按稳定 resource ID 关闭全部已 begin 的资源；
  每个资源独立最多等待五秒，close panic/failure 成为 typed secondary diagnostic，
  不覆盖 primary outcome。

**实现：**

- [x] 使用 private source/sink bindings、`ContinuousRunner`、`ContinuousJob`、driver
  与 reaper，完成 validate -> begin -> publish 三阶段 launch。
- [x] 分开 graceful shutdown 与 cancellation，并用稳定 task/resource ID 聚合错误。
- [x] private status/metrics 只暴露 payload-free 的稳定 ID 与数值；不公开
  cursor、watermark 或 epoch 状态。
- [x] 将 panic 转成 `CalcFlowError::TaskPanicked`；非字符串 panic 使用固定文案，
  字符串按 UTF-8 安全边界限制到 1,024 bytes（含省略号）。
- [x] 保留旧 push `step()`、public `StreamingRunner` 和 `MicroBatchRunner`。

**验收门：** M2 internals 是可验证的有界 source-driven runtime skeleton，但不是
public continuous runner，也不承诺 durable at-least-once、checkpoint recovery 或
event-time 完整性。

### Task M2.5：补齐 metrics、stress 与 soak

**状态：** 已完成。下面的 soak 标准适用于 calc-flow 当前和未来所有 soak；其他
名称、时长、采样默认值或命令均不构成通过证据。

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture
```

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/metrics.rs`
- 新建：crate-private `crates/calc-flow/src/runtime/streaming/soak.rs`
- 修改：`crates/calc-flow/benches/core.rs`

**步骤：**

- [x] 记录 input/output batches、rows、bytes、errors、blocked sends、queue high-water。
- [x] 提供 per-edge slot/row/byte charge 与 source/sink progress gauge。
- [x] 记录 operator processing 和 backpressure duration，禁止 batch ID 高基数 label。
- [x] 用 paused Tokio time 写短 CI stress。
- [x] 增加 opt-in Linux two-source slow-sink soak：精确 1,200 秒 measured workload，
  10 秒 cadence、120 samples、前 30 samples/300 秒 warm-up。
- [x] 增加 channel、unary stream overhead、fan-out Criterion case；benchmark targets
  必须编译，paired comparison 仅在存在匹配 base case 时判定。

**验收门：** code-approved head `7deda2a0` 的
[durable evidence bundle](https://github.com/wegamekinglc/calc-flow/pull/83#issuecomment-5201266650)
记录 120/120 samples、每个 sink 96,124 accepted batches、24,032 zero-cost
envelopes、零 missing/duplicate/leak、三条 saturated/blocked edges 与 -2.609 MiB/hour
RSS slope。raw log SHA-256 为
`bc97c8f736ad41a4f228e07300f1ecd23c9af9fb09dc1be1718823430bd05f35`。

## 9. M3：Event time 与 watermark

### Task M3.1：实现 source watermark policy

**文件：**

- 新建：`crates/calc-flow/src/time/watermark.rs`
- 修改：`crates/calc-flow/src/runtime/streaming/source_task.rs`
- 新建：`crates/calc-flow/tests/watermark_source.rs`

**首版 policy：**

```rust
enum WatermarkPolicy {
    SourceProvided,
    BoundedOutOfOrderness {
        event_time_column: String,
        delay: Duration,
        emit_interval: Duration,
        idle_timeout: Duration,
    },
    Disabled,
}
```

**先写 RED：**

- [ ] missing/non-timestamp event-time column 带 source/column path 报错。
- [ ] 四种 Arrow timestamp unit 做 checked conversion。
- [ ] generated watermark = observed max event time - delay。
- [ ] 后续旧数据不能让 watermark 回退。
- [ ] source-provided 回退 watermark 在入队前失败。
- [ ] 全 null timestamp batch 不产生新 watermark。

**实现：**

- [ ] compile_stream 时校验 policy。
- [ ] 在 Rust/Arrow 中求 timestamp max，不做逐行 Python callback。
- [ ] 用 Tokio time 周期发射，测试使用 paused time。
- [ ] policy state 进入 source checkpoint metadata。
- [ ] 3.0 只支持 table source + timestamp column + fixed delay；任意 SQL watermark
  expression 后置。

**验收门：** source 在 replay 下产生确定、单调的 watermark 序列。

### Task M3.2：实现 multi-input progress、idle 与 end

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/progress.rs`
- 修改：`crates/calc-flow/src/runtime/streaming/operator_task.rs`
- 新建：`crates/calc-flow/tests/watermark_progress.rs`

**先写 RED：**

- [ ] 两个 active input 输出 minimum watermark。
- [ ] fast input 不能越过 slow active input。
- [ ] idle input 排除出 minimum。
- [ ] data 到来时 idle input 先 reactivate。
- [ ] reactivate 后可能产生 late row，但 output watermark 不回退。
- [ ] ended input 永久排除出后续 minimum。
- [ ] 全部 input ended 只触发一次 final flush 和 end forwarding。

**实现：**

- [ ] 每 ingress 记录 `Active`、`Idle`、`Ended` 和 last watermark。
- [ ] 仅在 minimum 严格推进时产生 output watermark。
- [ ] operator 先处理 watermark，再由 runtime 转发。
- [ ] input progress 进入 operator checkpoint。
- [ ] status snapshot 使用稳定 ingress ID 排序。

**验收门：** fast/slow/idle/reactivated/ended 组合在 union 和 window 图上全部正确。

### Task M3.3：实现 late-data drop 与指标

**文件：**

- 修改：`crates/calc-flow/src/operator/window.rs`
- 修改：`crates/calc-flow/src/runtime/streaming/metrics.rs`
- 新建：`crates/calc-flow/tests/late_data.rs`

**步骤：**

- [ ] 3.0 只实现 `LateDataPolicy::Drop`。
- [ ] 严格使用 M0 批准的 late boundary：一行迟到当且仅当其所属窗口
  `window_end <= 当前输入 watermark`。hopping 场景下同一行可能同时命中已关闭和
  未关闭的窗口，此时只丢弃已关闭窗口的那一份，其余窗口正常累加。
- [ ] 记录 late rows、affected batches、maximum lateness。
- [ ] 不记录行 payload。
- [ ] project 中出现 allow/side_output 时编译失败。
- [ ] 恢复后 watermark 和 late metrics 不回退、不重复累计已提交值。

**验收门：** late row 不会重开 final window，且用户能观察丢弃数量。watermark 尚未
越过 `window_end` 的乱序行必须被正常累加，不得计入 late 指标。

## 10. M4：增量状态与 final-only window

### Task M4.1：定义 StateBackend 与 immutable StateHandle

**文件：**

- 新建：`crates/calc-flow/src/state/{mod,backend,manifest,segment}.rs`
- 新建：`crates/calc-flow/src/checkpoint/model.rs`
- 修改：`crates/calc-flow/src/error.rs`
- 新建：`crates/calc-flow/tests/state_backend.rs`

**前移说明：** epoch 与 checkpoint manifest 的数据模型必须在本任务一次定型，而不是
先在 M4 自造一套临时 state manifest、再在 M5.1 推倒重来。M5.1 只负责替换
`CheckpointStore` 的读写路径与旧 runner 的 sequence-only checkpoint，不再重新定义
manifest 结构。

**目标概念：**

```rust
struct StateHandle {
    operator_id: String,
    epoch: Epoch,
    segment_id: String,
    relative_path: String,
    byte_len: u64,
    sha256: String,
}
```

**先写 RED：**

- [ ] handle 拒绝 absolute path、traversal、空 ID、错误 checksum、错误 pipeline/operator。
- [ ] staged segment 在 epoch manifest commit 前不可见。
- [ ] stage failure 不破坏上个 committed epoch。
- [ ] checksum mismatch 在 decode 前失败。
- [ ] unknown/duplicate handle fail closed。
- [ ] compaction 前后逻辑 row 和确定性顺序一致。

**实现：**

- [ ] M0 在 Arrow IPC 与 Parquet 间确定首版 segment format。
- [ ] 固定 state segment 与 checkpoint manifest 的提交顺序：先落盘并校验 segment，
  再原子发布 checkpoint manifest。checkpoint manifest 是“最近完成 epoch”的唯一
  真相，恢复只读取它；任何未被保留 manifest 引用的 segment 一律按垃圾处理，不参与
  恢复判定，也不得让恢复失败。
- [ ] checkpoint JSON 只保存 handle，不保存 keyed row。
- [ ] pipeline/operator 名称先 hash 再用于路径。
- [ ] staging 与 committed state 位于同一受管 filesystem root。
- [ ] 校验目标后才 atomic rename。
- [ ] compression/compaction 放到 Tokio executor 外。
- [ ] manifest 发布前记录 segment byte length 与 checksum。

**验收门：** 超过 10 MiB 的 state 能 checkpoint/restore，但 manifest 本身保持有界。

### Task M4.2：实现 LocalStateBackend、retention 与 compaction

**文件：**

- 新建：`crates/calc-flow/src/state/local.rs`
- 新建：`crates/calc-flow/tests/local_state.rs`
- 修改：`crates/calc-flow/benches/core.rs`

**步骤：**

- [ ] 创建并 canonicalize 唯一 state root。
- [ ] 受管路径边界拒绝 symlink 和意外 file type。
- [ ] 写 epoch staging directory，再原子发布 manifest。
- [ ] 保留可配置数量的 completed epoch。
- [ ] 不删除任何 retained manifest 可达的 state。
- [ ] 已提交但没有任何 retained checkpoint manifest 引用的 segment 视为孤儿，可以
  安全回收，且其存在不得让恢复失败。
- [ ] 只 compact immutable committed segment。
- [ ] locked/delete failure 立即停止，不扩大 cleanup target。
- [ ] 模拟 segment/manifest rename 前后 crash。
- [ ] 分开 benchmark incremental write、full restore、compaction。

**验收门：** crash 后只选择最近完整 committed manifest，从不读取 partial epoch。

### Task M4.3：冻结 window 与 aggregate spec

**文件：**

- 新建：`crates/calc-flow/src/operator/window.rs`
- 修改：`crates/calc-flow/src/pipeline/compile.rs`
- 新建：`crates/calc-flow/tests/window_compile.rs`

**3.0 scope：**

- group key 是命名 Arrow column；
- event time 是一个 timestamp column；
- window 只支持 tumbling/hopping；
- aggregate 首批支持 `count`、`sum`、`min`、`max`、`avg`；
- 支持类型矩阵必须显式列举；
- output 包含 `window_start`、`window_end`、group keys、named aggregates；
- output 顺序为 window start、window end、稳定编码 group key。

**先写 RED：**

- [ ] zero/negative size/slide 失败。
- [ ] 3.0 要求 hopping size 可被 slide 整除，除非 M0 明确批准通用规则。
- [ ] output name 与 window/group column 冲突时失败。
- [ ] unsupported aggregate/type 在 compile 时失败。
- [ ] output schema 确定且与插入顺序无关。
- [ ] session/early/allowed-lateness/update config 全部拒绝。

**实现：**

- [ ] 引入严格可序列化的 `WindowSpec`、`AggregateSpec`、
  `WindowAggregateOperator`。
- [ ] 稳定时使用 DataFusion public expression API 做 input projection/filter。
- [ ] Calc-Flow 自己维护 incremental accumulator，不依赖 DataFusion private planner
  node 或 fork。
- [ ] 明确 null、NaN、overflow、decimal、avg 语义。
- [ ] 所有语义进入 fingerprint。

**验收门：** 支持矩阵清晰，所有不支持组合都在 source.open 前失败。

### Task M4.4：实现 tumbling/hopping 执行

**文件：**

- 修改：`crates/calc-flow/src/operator/window.rs`
- 新建：`crates/calc-flow/tests/window_{tumbling,hopping,properties}.rs`

**先写 RED：**

- [ ] Unix epoch 前后时间都正确分配 tumbling window。
- [ ] hopping row 进入精确的多个重叠 window。
- [ ] incremental 结果与 finite batch group-by oracle 一致。
- [ ] watermark close 后每个 window 只输出一次。
- [ ] late row 不改变已关闭结果。
- [ ] empty/all-null aggregate 遵循已批准语义。
- [ ] 任意 batch boundary 恢复后 final output 一致。
- [ ] proptest 随机切分相同 rows，结果仍一致。

**实现：**

- [ ] 稳定编码 `(window_start, window_end, group_key)`。
- [ ] dirty accumulator 与 committed handle 分离。
- [ ] checkpoint 只 flush dirty key。
- [ ] watermark 推进时先 emit closed window，再 forward watermark。
- [ ] window output 在 checkpoint 协议下 durable 后再清理/tombstone state。
- [ ] hopping update 避免不必要克隆 Arrow payload。

**验收门：** batch partition、checkpoint recovery、compaction 都不改变 final result。
注意“同一窗口只关闭一次”只能在事务/幂等 sink 边界成立：at-least-once 下故障重放
必然可能让算子重新发出已关闭窗口，因此该断言写在 sink 层，算子层只断言重放后最终
结果不变。

## 11. M5：Epoch checkpoint 与 exactly-once

### Task M5.1：用 manifest v3 替换 checkpoint v2

**文件：**

- 替换：`crates/calc-flow/src/checkpoint.rs`
- 新建：`crates/calc-flow/src/checkpoint/{mod,model,store}.rs`
- 替换：`crates/calc-flow/tests/checkpoint.rs`
- 修改：`crates/calc-flow/src/lib.rs`

**v3 manifest 必含：**

- `format_version = 3`；
- pipeline name 与 stream fingerprint；
- epoch 与 created time；
- per-source cursor、sequence、watermark policy state、end state；
- per-operator input progress、inline metadata、state handles；
- per-sink delivery capability 和 pre-commit metadata；
- 恢复完成提交所需的 manifest status；
- state metadata 的确定性 checksum。

**先写 RED：**

- [ ] v2 文档按 expected 3 拒绝。
- [ ] unknown field、missing nullable field、duplicate key、过深、过大全部 fail closed。
- [ ] source/operator/sink ID 必须与 plan 精确相等。
- [ ] handle 必须属于同 pipeline/epoch。
- [ ] fingerprint mismatch 在任何 restore side effect 前失败。
- [ ] canonical serialization 确定。

**实现：**

- [ ] 用 `CheckpointManifest` 替换 `Checkpoint`，沿用 M4.1 已定型的 manifest 模型，
  不重新定义结构。
- [ ] 重写 `CheckpointStore` 为 manifest operation。
- [ ] 保留 bounded atomic JSON store，但只存 metadata/handle。
- [ ] 删除旧 runner 的 sequence-only checkpoint 和 compensation path。
- [ ] 错误保留准确 format/path/source。

**验收门：** v3 checkpoint JSON 不含 window key/value state。

### Task M5.2：在精确 cursor 边界注入 source barrier

**文件：**

- 新建：`crates/calc-flow/src/runtime/streaming/coordinator.rs`
- 修改：`crates/calc-flow/src/runtime/streaming/source_task.rs`
- 新建：`crates/calc-flow/tests/source_barrier.rs`

**先写 RED：**

- [ ] barrier 位于其 snapshot cursor 所覆盖数据之后。
- [ ] post-barrier data 不能出现在 barrier 前。
- [ ] idle source 仍注入 barrier。
- [ ] ended source 报告 final cursor 且不 reopen。
- [ ] 并发 checkpoint request 串行成严格递增 epoch。
- [ ] timeout 按 M0 规则失败，不静默跳过 epoch。

**实现：**

- [ ] coordinator 向每个 source task 发 barrier request。
- [ ] source task 停止 poll，记录已完全入队 cursor，通过全部 output edge 发 barrier，
  ack cursor，然后 resume。
- [ ] coordinator 不直接写 graph edge。
- [ ] 3.0 同时只允许一个 in-flight checkpoint。
- [ ] checkpoint interval/timeout 来自已校验 runtime config。

**验收门：** 所有 source cursor 与 edge barrier 共同描述一个可重放 prefix。

### Task M5.3：实现 multi-input barrier alignment 与 operator snapshot

**文件：**

- 修改：`crates/calc-flow/src/runtime/streaming/{operator_task,coordinator}.rs`
- 新建：`crates/calc-flow/tests/barrier_alignment.rs`

**先写 RED：**

- [ ] ingress A 先到 barrier 后停止消费 A，B 继续。
- [ ] 同 epoch 所有 required ingress 到齐后才 snapshot。
- [ ] future/regressed epoch fail closed。
- [ ] barrier 前 data 在 state，barrier 后 data 不在该 snapshot。
- [ ] snapshot 成功后才 forward barrier；转发时机按 M0 的结论执行，并在实现注释中
  写明选的是“同步快照完成即转发”还是“等待 coordinator ack 再转发”，以及对应的
  checkpoint 延迟量级。
- [ ] snapshot failure 不 forward barrier 并 cancel job。
- [ ] idle 不免除 alignment。

**实现：**

- [ ] per-ingress 记录 barrier state。
- [ ] `select!` 中禁用 blocked ingress，不 drain、不 reorder。
- [ ] 未 blocked ingress 继续消费。
- [ ] full alignment 后调用 `checkpoint(epoch)`。
- [ ] stage dirty state 并向 coordinator 发送 `OperatorCheckpoint`。
- [ ] 按 M0 选定的时机 forward barrier，并同时恢复全部 ingress。

**验收门：** union/window 在所有双输入 barrier 到达排列下都通过 before/after state
断言。

### Task M5.4：实现 transactional sink 与 commit protocol

**文件：**

- 修改：`crates/calc-flow/src/connector/capability.rs`
- 修改：`crates/calc-flow/src/runtime/streaming/sink_task.rs`
- 新建：`crates/calc-flow/src/runtime/streaming/transaction.rs`
- 新建：`crates/calc-flow/tests/transactional_sink.rs`

**目标 lifecycle：**

```rust
#[async_trait]
trait TransactionalSink: StreamSink {
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<SinkPreCommit>;
    async fn commit(&mut self, epoch: Epoch, state: &SinkPreCommit) -> Result<()>;
    async fn abort(&mut self, epoch: Epoch, state: Option<&SinkPreCommit>) -> Result<()>;
    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()>;
}
```

**先写 RED：**

- [ ] ordinary sink 不能满足 exactly-once plan。
- [ ] sink 收到 barrier 及此前全部 data 后才 pre-commit。
- [ ] manifest durable 后才开始 external commit。
- [ ] commit retry 幂等。
- [ ] recovery 完成 durable pre-commit，不重写数据。
- [ ] manifest 前失败 abort staged output。
- [ ] manifest 后失败不得 abort 恢复所需 output。

**实现：**

- [ ] 为 source/operator/sink/backpressure 声明显式 capability。
- [ ] 对完整 graph 编译 requested delivery guarantee。
- [ ] 收齐一个 epoch 的 source/operator/sink ack。
- [ ] 原子 commit checkpoint manifest。
- [ ] manifest durable 后 commit transactional sink。
- [ ] manifest 保存幂等恢复所需 pre-commit metadata。
- [ ] 明确 all sink committed 后的 completion 标记或推导规则。

**验收门：** 普通 sink 或 lossy source 永远不会被标为 exactly-once。

### Task M5.5：执行 crash consistency fault matrix

**文件：**

- 新建：`crates/calc-flow/tests/exactly_once_faults.rs`
- 扩展：`crates/calc-flow/tests/support/mod.rs`
- 新建：`docs/superpowers/handoffs/continuous-streaming-v3-fault-matrix.md`

在以下每一点注入 cancel、I/O error、panic、restart：

1. source data enqueue 后；
2. 一个 source barrier 后；
3. multi-input alignment 中；
4. operator state staging 中；
5. sink pre-commit 后；
6. manifest rename 前；
7. manifest rename 后；
8. 多 sink 的第一个 commit 中；
9. 全部 commit 后、completion bookkeeping 前；
10. retention/compaction 中。

每个故障点断言：

- [ ] recovered source cursor；
- [ ] recovered watermark/idle；
- [ ] recovered window state；
- [ ] visible sink rows/files；
- [ ] duplicate/missing count；
- [ ] staged artifact cleanup；
- [ ] latest completed epoch；
- [ ] deterministic terminal error。

**验收门：** transactional file sink 在完整矩阵中零重复、零丢失；ordinary sink
测试明确展示允许重复的位置。

## 12. M6：Connector、Project v3、Python 与 Studio

### Task M6.1：建立 immutable connector registry 与 format layer

**文件：**

- 新建：`crates/calc-flow/src/connector/{mod,capability,format,registry}.rs`
- 新增 workspace crate：`crates/calc-flow-connectors/`
- 修改：根 `Cargo.toml`
- 新建：`crates/calc-flow/tests/connector_registry.rs`

**先写 RED：**

- [ ] duplicate connector identity 原子拒绝。
- [ ] unknown connector/format 在 source construction 前失败。
- [ ] compile 后 registry snapshot 不受后续 registration 影响。
- [ ] project capability 只是 data，不能注入 executable callback。
- [ ] connector config serialization 结构上不能包含 secret value。
- [ ] decoder expansion 受 rows/bytes limit 限制。
- [ ] 数据库 snapshot、polling、CDC、append、upsert 和 transactional write 是
  相互独立的 capability，不能由一个笼统的 `database = true` 代替。

**实现：**

- [ ] connector identity 固定为 `(provider, name, version)`。
- [ ] transport factory 与 `FormatDecoder`/`FormatEncoder` 分离。
- [ ] compile 时捕获 plan-scoped immutable registry snapshot。
- [ ] `SecretResolver` 只接受 secret reference。
- [ ] 网络客户端依赖不进入 core crate；`calc-flow-connectors` 内每个 connector 由
  独立 cargo feature 控制，默认关闭，并同步更新 `--all-features` 相关的 CI 命令、
  覆盖率范围和开发环境前置依赖。
- [ ] 按 M0 结论落实 `calc-flow-python` 与 `calc-flow-connectors` 的依赖边，并让
  Python 侧能够注册可用 connector；若决定不依赖，则同步收窄 M6.7、M6.8 的验收门。
- [ ] capability 包含 delivery、replay、watermark、transaction、lookup、snapshot、
  polling 和 CDC。
- [ ] 公共 `database_types.rs` 只保存经过验证的 Arrow/database 类型映射，不包含
  连接池或具体客户端类型。

**验收门：** project 只用 data 选择 connector/format，factory 解析仍由 trusted
process-local registry 完成。

### Task M6.2：先实现 file/Parquet connector

**文件：**

- 新建：`crates/calc-flow-connectors/src/{file,parquet,csv,json}.rs`
- 新建：`crates/calc-flow-connectors/tests/file_connector.rs`

**Source scope：**

- finite file/directory snapshot；
- 可选轮询新发现 immutable file；
- cursor 为稳定排序 file identity + completed row group；
- CSV、newline JSON、Parquet；
- 显式 schema 与 bounded decode。

**Sink scope：**

- 按 pipeline/output/epoch 写 Parquet；
- staging file 与 target 位于同 filesystem；
- 通过 epoch manifest/directory atomic rename 幂等 commit；
- 不覆盖无关用户文件。

**测试：**

- [ ] deterministic discovery/cursor replay；
- [ ] partial file、schema mismatch、corrupt Parquet、oversized row group；
- [ ] staging file 不会被当作 committed output；
- [ ] retry/recovery 幂等；
- [ ] symlink、traversal、wrong type、locked file fail closed；
- [ ] 使用真实 file sink 跑完整 M5 fault matrix。

**验收门：** 在明确支持的本地 filesystem 假设下，file-to-Parquet exactly-once。

### Task M6.3：实现 Kafka source/sink

**文件：**

- 新建：`crates/calc-flow-connectors/src/kafka.rs`
- 新建：`crates/calc-flow-connectors/tests/kafka_connector.rs`
- 修改：workspace dependency、audit、release 配置

**步骤：**

- [ ] 添加依赖前确定 Linux/macOS/Windows/wheel 的 `rdkafka` build/link 策略。
- [ ] partition 到本地 source task 的映射必须确定。
- [ ] checkpoint 保存 per-partition offset。
- [ ] bounded-channel backpressure 暂停消费。
- [ ] JSON/Avro/Protobuf 只通过注册 format 解码。
- [ ] transactional ID 由 pipeline/sink identity 稳定派生，不含 secret。
- [ ] 显式处理 producer fencing、rebalance、lost partition、timeout、recovery。
- [ ] container integration test 不混入普通 unit test。
- [ ] Kafka fault matrix 通过前不宣传 exactly-once。

**验收门：** restart 后 partition replay 正确，transactional sink 对 committed epoch
无重复记录。

### Task M6.4：实现 PostgreSQL source/sink

**文件：**

- 新建：`crates/calc-flow-connectors/src/{postgresql,database_types}.rs`
- 新建：`crates/calc-flow-connectors/tests/postgresql_connector.rs`
- 新建：`crates/calc-flow-connectors/tests/postgresql_cdc.rs`
- 修改：workspace dependency、audit、release 配置

**Source mode：**

1. `snapshot`：在 read-only repeatable-read transaction 中读取一致快照；
2. `incremental_query`：按严格有序 composite cursor 周期拉取；
3. `logical_cdc`：使用 publication、logical replication slot 和标准 `pgoutput`
   protocol 连续读取事务变更。

**CDC 输出契约：**

- 输出仍是 append-only Arrow event batch，不改变 Calc-Flow 的关系语义；
- 每条 event 明确包含 operation、relation、transaction ID、commit LSN、commit time、
  key、before 和 after；
- `INSERT`、`UPDATE`、`DELETE` 由 operation 字段区分；
- `before` 可用性取决于 PostgreSQL replica identity；缺少 required old row 时 fail
  closed，不伪造旧值；
- 一个 PostgreSQL transaction 可以拆为多个 Arrow batch，但 source barrier 只能在
  transaction commit boundary 注入；
- DDL/schema change 在 3.0 中停止 source 并报告新旧 schema，不自动演进。

**先写 RED：**

- [ ] connection URL、password、TLS key 只能通过 `SecretResolver` 获取，序列化、日志、
  error、metrics 中不可见。
- [ ] snapshot 使用同一数据库快照，分页期间并发写入不会混入 snapshot 结果。
- [ ] incremental query 必须配置唯一、单调的 cursor；非唯一列必须增加 primary-key
  tie-breaker。
- [ ] cursor predicate 使用 bound parameter，table/column identifier 严格验证与引用，
  禁止字符串拼接用户值。
- [ ] logical slot exported snapshot 与 initial table copy 无 gap 衔接。
- [ ] CDC 严格保持 PostgreSQL transaction commit order。
- [ ] replication slot LSN 回退或重复只导致安全 replay，不导致 checkpoint cursor
  前进错误。
- [ ] slot `lost`、publication 缺失、replica identity 不足和 WAL gap 都 fail closed。
- [ ] barrier 到达 transaction 中间时等待 transaction commit 后再注入。
- [ ] source watermark 只从配置的 event-time 字段产生；delete event 没有 after 值时
  不伪造 event time。

**Source 实现：**

- [ ] M0 评审后在 `tokio-postgres`、`sqlx` 或专用 replication client 中选择能同时
  满足普通 query、COPY 和 logical replication protocol 的组合，不提前锁定库版本。
- [ ] 创建 slot 时使用 exported snapshot 完成 gap-free initial snapshot，再从创建
  slot 返回的 LSN 进入 `pgoutput`。
- [ ] checkpoint cursor 使用 commit LSN，而不是 row index 或 32-bit xid。
- [ ] 只有 Calc-Flow checkpoint manifest durable 后才向 PostgreSQL 确认可 flush
  LSN；恢复时核对本地 LSN 与 slot `confirmed_flush_lsn`。
- [ ] 默认不自动删除持久 replication slot；创建、复用、删除策略必须显式配置。
- [ ] 暴露 replication lag、retained WAL、`restart_lsn`、`confirmed_flush_lsn`、slot
  active/lost 状态，防止废弃 slot 无限保留 WAL。
- [ ] CDC metadata 使用固定 Arrow schema；表字段映射为 typed `before`/`after`
  struct，不能退化为无界 JSON 字符串。

**Sink mode：**

- `append`：批量插入目标表；
- `upsert`：要求显式 conflict key，并生成参数化 `INSERT ... ON CONFLICT`；
- `transactional`：目标写入和 Calc-Flow epoch ledger 在同一 PostgreSQL transaction
  中提交。

**Sink 实现与测试：**

- [ ] 为 PostgreSQL/Arrow 建立显式类型矩阵：bool、整数、浮点、decimal、text、bytea、
  date/time/timestamp/timestamptz、UUID、JSON/JSONB；不支持类型在 compile 时失败。
- [ ] `NUMERIC` precision/scale、`TIMESTAMPTZ` UTC、NaN、infinity、array/domain/enum
  的支持边界必须测试并文档化。
- [ ] 使用 COPY 或参数化 batch insert；禁止为 row value 拼 SQL。
- [ ] transactional mode 先把 epoch rows 预提交到 Calc-Flow state segment；manifest
  保存 segment handle、schema hash、row count 和 target identity。
- [ ] manifest durable 后开启数据库 transaction，先尝试插入唯一
  `(pipeline_fingerprint, sink_id, epoch)` ledger，再写 target rows，并在同一 transaction
  commit。
- [ ] ledger 已存在时视为该 epoch 已提交，不能重写 target rows。
- [ ] commit acknowledgement 丢失后恢复必须查询 ledger 并得出确定结果。
- [ ] ledger 与 target table 不在同一 database/transaction、缺少 ledger DDL 权限或
  用户禁用 ledger 时，只能声明 at-least-once。
- [ ] PostgreSQL `PREPARE TRANSACTION` 仅作为后续可选模式，不是 3.0 默认实现；若
  后续启用，必须处理 `max_prepared_transactions`、GID、锁、VACUUM 影响和遗留
  `pg_prepared_xacts`。
- [ ] 使用容器测试 connection loss、statement timeout、serialization failure、
  deadlock、failover/reconnect、commit ack loss、ledger conflict 和完整 M5 fault
  matrix。

**设计依据：** PostgreSQL logical decoding 通过 replication slot 输出有序 change
stream；slot 可能在 crash 后重发近期数据，consumer 必须以 LSN 安全去重；slot 还会
保留 WAL，因此需要 lag/retention 监控。参见
[Logical Decoding](https://www.postgresql.org/docs/current/logicaldecoding.html)、
[Streaming Replication Protocol](https://www.postgresql.org/docs/current/protocol-replication.html)
和
[`pg_replication_slots`](https://www.postgresql.org/docs/current/view-pg-replication-slots.html)。

**验收门：** PostgreSQL snapshot + CDC 无 gap、按事务顺序输出；polling cursor 可
恢复；transactional sink 在 commit-ack 丢失和进程重启后无重复、无丢失。

### Task M6.5：实现 ClickHouse source/sink

**文件：**

- 新建：`crates/calc-flow-connectors/src/clickhouse.rs`
- 新建：`crates/calc-flow-connectors/tests/clickhouse_connector.rs`
- 修改：`crates/calc-flow-connectors/src/database_types.rs`
- 修改：workspace dependency、audit、release 配置

**Source scope：**

- `snapshot`：带启动时 upper bound 的有限一致读取；
- `incremental_query`：按 event-time/sequence + unique tie-breaker 周期拉取；
- 3.0 不提供 ClickHouse CDC，也不把后台 MergeTree part merge 当成 change feed。

**先写 RED：**

- [ ] database/table/column identifier 严格验证，query value 全部参数化。
- [ ] 无 unique tie-breaker 的 polling 配置失败。
- [ ] snapshot 在启动时固定 upper cursor bound，分页期间新 row 不混入本次快照。
- [ ] 相同 cursor value 的多行不会遗漏或无限重复。
- [ ] DateTime/DateTime64 timezone 和 Decimal precision/scale 映射准确。
- [ ] Nullable、LowCardinality、Enum、UUID、IPv4/IPv6、Array 的支持矩阵明确；未知
  类型 compile fail。
- [ ] query/response rows、bytes、timeout 受配置限制。

**Sink scope：**

- 默认 `at_least_once` 批量 insert；
- 每个 Calc-Flow epoch 使用稳定 `insert_deduplication_token`，提供
  `retry_deduplicated` 能力；
- `retry_deduplicated` 不等同于 Calc-Flow 的通用 `exactly_once`；
- 可选 ReplacingMergeTree versioned mode 支持最终去重，但用户查询需要理解
  background merge/`FINAL` 语义。

**Sink 实现与测试：**

- [ ] M0 选择 native/HTTP Rust client，并验证 Linux/macOS/Windows/wheel 的 TLS 和
  compression 构建。
- [ ] 聚合小 batch，限制 block rows/bytes，避免产生大量小 part。
- [ ] checkpoint pre-commit 保存稳定 insert block、token、schema hash、target 和
  row count。
- [ ] retry 必须重用完全相同 token 与 row order，不能重新随机分批。
- [ ] 使用 async insert 时必须等待 server 确认；不允许 fire-and-forget 通过
  checkpoint。
- [ ] 启动时检查 table engine、dedup setting 和 distributed topology，并据此声明
  capability。
- [ ] dedup token 的保留窗口有限时，不能把它升级成无限期 exactly-once 承诺。
- [ ] 一个 insert block、跨 partition insert、Distributed table 和 materialized view
  分别测试 atomicity/dedup 行为。
- [ ] ReplacingMergeTree 只声明 eventual dedup；测试普通查询与 `FINAL` 的可见差异。
- [ ] 容器测试 timeout、unknown commit outcome、retry、replica unavailable、quota、
  schema mismatch 和 M5 at-least-once fault matrix。

**设计依据：** ClickHouse 官方资料说明 MergeTree-family insert 可以使用 deduplication
token 过滤重试，但 table/物化视图分别维护去重状态；ReplacingMergeTree 的行级去重
依赖后台 merge，查询时可能需要 `FINAL`。因此 3.0 只把通用 ClickHouse sink 标为
at-least-once，把 token 能力单独标为 retry-deduplicated。参见
[Automatic deduplication for idempotent inserts](https://clickhouse.com/blog/clickhouse-release-26-01)
和
[ReplacingMergeTree 查询时去重说明](https://clickhouse.com/resources/engineering/clickhouse-optimize-table-final)。

**验收门：** ClickHouse snapshot/polling 可恢复且不漏 composite cursor；sink 对同
epoch retry 不产生额外 insert block；API 与文档不把 eventual/有限窗口去重宣传为
unconditional exactly-once。

### Task M6.6：实现 HTTP polling 与 WebSocket source

**文件：**

- 新建：`crates/calc-flow-connectors/src/{http,websocket}.rs`
- 新建：对应 connector tests

**步骤：**

- [ ] HTTP 支持 response size、timeout、retry、conditional request。
- [ ] ETag/Last-Modified 可作为可选 replay cursor。
- [ ] WebSocket 限制 frame 和 decoded batch 大小。
- [ ] 能暂停读取时默认 `Block`。
- [ ] `DropOldest` 必须显式、可观测，并与 exactly-once 不兼容。
- [ ] TLS verification 默认开启；insecure 模式必须显式且告警。
- [ ] authorization header、含凭据 URL、payload 全部脱敏。

**验收门：** capability 准确区分 replayable HTTP、unreplayable HTTP、lossy
WebSocket。

### Task M6.7：用严格 project v3 替换 v2

**文件：**

- 替换：`crates/calc-flow/src/config.rs`
- 移除 canonical：`schemas/project-v2.schema.json`
- 新建：`schemas/project-v3.schema.json`
- 替换：`crates/calc-flow/tests/config.rs`
- 修改：`crates/calc-flow/src/project_store.rs`
- 修改：examples/docs

**v3 顶层概念：**

- `format_version: 3`；
- `runtime.mode: batch | stream`；
- 对应 mode 的 runtime options；
- graph nodes/edges；
- source binding：connector、format、watermark、secret reference；
- sink binding：connector、format、requested delivery；
- 数据库 binding：snapshot/polling/CDC 或 append/upsert mode、cursor、目标表、
  capability requirement；
- state/checkpoint config；
- strict data-only contract。

**先写 RED：**

- [ ] v2 拒绝。
- [ ] 每层 unknown field 都失败。
- [ ] connector options 是 bounded JSON 且 defensive copy。
- [ ] 结构上只接受 secret reference，不接受 secret value。
- [ ] runtime mode 与 operator capability 必须一致。
- [ ] exactly-once 不兼容时报告精确 path。
- [ ] canonical serialization/fingerprint 确定。
- [ ] generated schema 与 `project-v3.schema.json` 精确一致。

**实现：**

- [ ] 新建 v3 model，不向 v2 填 optional field。
- [ ] inline `DataSourceSpec` 改为 batch fixture 或 stream connector variant。
- [ ] continuous project 一次编译出 graph plan、source/sink factory、state config、
  delivery proof。
- [ ] PostgreSQL slot/publication/ledger 与 ClickHouse table engine/dedup 配置全部进入
  data-only spec；credential 仍只能使用 secret reference。
- [ ] 保持 data-only、max JSON depth/size。
- [ ] v2 schema 若保留，只移入历史文档，不被 runtime import。

**验收门：** 单个 project v3 可以无 executable object/secret 地定义
PostgreSQL CDC -> window -> ClickHouse/Parquet stream。

### Task M6.8：替换 PyO3 与 Python API

**文件：**

- 修改：`crates/calc-flow-python/src/{lib,pipeline,runtime,config,store}.rs`
- 替换：`python/calc_flow/runtime.py`
- 修改：`python/calc_flow/{__init__,pipeline,config,store,capabilities}.py`
- 替换相关：`python/tests/`
- 更新：`python/calc_flow/_native.pyi`

**目标行为：**

- `PipelineBuilder.compile_batch()` / `compile_stream()`；
- async source protocol；
- ordinary/transactional sink protocol；
- `StreamingRunner.start_async()` 返回 owning job；
- `status()`、`checkpoint_async()`、`shutdown_async()`、`cancel()`、
  `wait_async()`；
- blocking convenience 在 active event loop 中拒绝；
- cancel 等待 native/Python task 清理。

**先写 RED：**

- [ ] source/sink input 校验并 defensive copy。
- [ ] async cancel 后无 pending Python task/native lease。
- [ ] job 参与 GC cycle 时正确释放。
- [ ] Python error 保留 source/sink identity 且不泄漏 secret。
- [ ] custom Python transactional sink 未实现完整 protocol 时不能声称能力。
- [ ] stub 与 runtime member 精确一致。

**实现：**

- [ ] 删除 Python `MicroBatchRunner` 与 push `step()` adapter。
- [ ] 围绕 v3 native job handle 重建 PyO3 ownership。
- [ ] 保持明确 GIL boundary，不阻塞 Tokio thread。
- [ ] NumPy/JAX payload 保持 immutable 并能 byte-account。
- [ ] capability schema version 与 project format list 改为 v3 only。
- [ ] 按 M6.1 的依赖边决定，实现 Python 侧可用 connector 的注册与能力枚举；若
  wheel 不携带原生 connector，`capabilities.py` 必须如实反映这一点，而不是宣告
  project v3 中不可达的 connector。

**验收门：** Python 与 Rust 能运行相同 start/status/checkpoint/stop/recover 场景，
重复 cancellation stress 通过。

### Task M6.9：替换 Studio API 并实现持续 job UI

**文件：**

- 修改：`web-ui/backend/src/calc_flow_studio/{app,models,run_manager}.py`
- 替换：`web-ui/backend/tests/`
- 更新：`web-ui/openapi.json`
- 更新：`web-ui/src/api/schema.d.ts`
- 修改：`web-ui/src/api/{client,decoders}.ts`
- 新增/替换：相关 React component/test

**Backend route：**

- `/api/v3/projects`；
- `/api/v3/jobs`；
- `/api/v3/jobs/{id}`；
- `/api/v3/jobs/{id}/events`；
- `/api/v3/jobs/{id}/checkpoint`；
- `/api/v3/jobs/{id}/shutdown`；
- `/api/v3/jobs/{id}/cancel`。

**步骤：**

- [ ] 删除 `/api/v2`。
- [ ] 保持 spawned-worker 的 CPU、memory、output、cancel 限制。
- [ ] 持续作业天然没有运行时长上限，因此原有的 worker timeout 不再是有效边界。
  删除它之前必须给出等价的替代上限：最大并发 job 数、单 job 与全局的常驻内存
  上限、最大 checkpoint/state 磁盘占用，以及必须由用户显式 stop 的生命周期。
  不允许只是把 timeout 置空。
- [ ] 增加适合 long-running job 的 persistent worker ownership。
- [ ] worker death 与 checkpoint recovery/terminal status 一致。
- [ ] `serve()` 保持 loopback-only。
- [ ] SSE 不发送 secret 或 raw payload。
- [ ] UI 展示 queue、watermark、epoch、throughput、backpressure、late row、job state。
- [ ] 清理 EventSource、timer、listener、request、worker。
- [ ] 同一 commit 生成 OpenAPI 与 TypeScript type。

**验收门：** 浏览器能创建 v3 stream project，start、observe、checkpoint、stop，并在
重连后看到 terminal status；无 worker/EventSource 泄漏。

## 13. M7：Hardening 与 3.0 发布

### Task M7.1：性能与内存门禁

- [ ] 同机比较全部 M0 Criterion baseline。
- [ ] 超过 5% 的回归必须经过置信区间与重复测量评审。
- [ ] 按 universal soak 标准运行 two-source backpressure soak：精确 1,200 秒 measured
  workload、10 秒 cadence、120 samples、前 30 samples/300 秒 warm-up。
- [ ] 运行大于旧 10 MiB JSON 上限的高基数 window-state soak。
- [ ] checkpoint duration 按 dirty-key volume 分析，不按 total retained state 混报。
- [ ] recovery 分开测 cold cache 与 warm cache。
- [ ] Python/Studio overhead 与 native runtime 分开归因。
- [ ] external array byte cost 保守且有测试。

### Task M7.2：安全与供应链门禁

- [ ] threat-model secret、path、symlink、decompression bomb、oversized message、
  malicious schema、deep connector option、SQL identifier/query、replication slot、
  WAL retention、database ledger 和 ClickHouse dedup token。
- [ ] 验证 TLS default 和 credential redaction。
- [ ] 运行 `cargo audit`、`cargo deny`、`npm audit --omit=dev` 和 artifact inspector，
  覆盖启用全部 connector feature 的配置。
- [ ] 审查 `rdkafka`、PostgreSQL、ClickHouse、HTTP、WebSocket、Avro、compression 的
  license/platform build。
- [ ] 按 M6.1 的决定核对 workspace 覆盖率门：确认 `calc-flow-connectors` 是被排除
  并单独设门，还是已把容器测试纳入采集，且 `--fail-under-lines 90` 在最终配置下
  真实通过。
- [ ] checkpoint/state cleanup 不遍历 symlink 或宽泛路径。
- [ ] fuzz/property test project、checkpoint、format、state metadata decoder。

### Task M7.3：文档、示例和指导文件

- [ ] 重写 `docs/introduction.md`，明确 batch/stream plan。
- [ ] 重写 Rust/Python API guide。
- [ ] 用实际 v3 contract 更新 runtime-envelope 文档。
- [ ] 按 connector 记录 delivery guarantee。
- [ ] 增加 file、Kafka、PostgreSQL snapshot/CDC/sink、ClickHouse polling/sink、
  watermark、tumbling、hopping、recovery、transactional sink 示例。
- [ ] 增加 v2 -> v3 breaking guide，但不承诺自动迁移。
- [ ] 根据最终源码更新 `AGENTS.md` 和 `CLAUDE.md` 架构摘要。
- [ ] `CHANGELOG.md` 列出所有删除/替换的 public surface。

### Task M7.4：版本与 release 验证

- [ ] workspace crate、Python core、Studio、frontend 同步升级到 `3.0.0`，并明确
  `calc-flow-connectors` 的版本策略与是否随核心 crate 一起发布。
- [ ] PyO3 crate 对 core Rust dependency 保持 exact version。
- [ ] 从最终源码生成 project v3 schema、OpenAPI、TypeScript type。
- [ ] 构建 core wheel、sdist、crate、Studio wheel。
- [ ] 检查每个 artifact，并在 clean environment 安装 smoke。
- [ ] 运行 `AGENTS.md` 全部 Rust、Python、backend、frontend、browser、security、
  release 命令。
- [ ] 确认 source tree 没有 `python/calc_flow/_native*.so`。
- [ ] 在 release commit 再跑一次完整 exactly-once fault matrix。

**发布门：** 无 unresolved high-severity review finding；无 schema drift；fault matrix
全部通过；exact final head 全绿；公共文档只描述该 commit 已实现行为。

## 14. 验证矩阵

| 关注点               | Unit/property test                              | Integration/soak                              |
| -------------------- | ----------------------------------------------- | --------------------------------------------- |
| Edge ordering        | mixed-message FIFO、fan-out                     | two-source slow-sink                          |
| Backpressure         | envelope/rows/bytes reservation、cancel         | 20-minute bounded-memory soak                 |
| Source recovery      | sequence/cursor validation                      | restart at every barrier boundary             |
| Watermark            | monotonic、minimum、idle、reactivate            | out-of-order multi-source                     |
| Window correctness   | randomized batch partition                      | large state checkpoint/recovery               |
| State durability     | checksum、atomic manifest、compaction           | crash around every rename                     |
| Barrier alignment    | all arrival permutations                        | slow/idle/ended source combinations           |
| Exactly-once         | transaction state machine                       | file/Kafka/PostgreSQL fault matrix            |
| Connector safety     | bounds、path、SQL、redaction、capability        | broker/database/HTTP failure integration      |
| Python lifecycle     | await/cancel/GC                                 | repeated native-Python cancellation           |
| Studio lifecycle     | API/React cleanup                               | Playwright start-observe-checkpoint-stop      |
| Packaging            | artifact inspector                              | clean install/smoke                           |

## 15. 各里程碑合入门槛

### M1

- batch/stream trait 与 plan 完全分离。
- 不支持的 stream graph 在副作用前失败。
- typed message 与 envelope/rows/bytes 三重限制 channel 通过测试。
- 不再存在语义含混的 v2 public name。

### M2

- **内部 gate 已完成：** source 持续驱动 private runtime graph。
- slow sink 能反压到 source。
- graceful shutdown drain；cancel join 全部 task。
- two-source union 通过 stress 与 universal 20-minute soak。
- public v2 runner 保持不变；public A6 仍为 post-M5 独立 gate。

### M3

- watermark 强类型、单调、可恢复、multi-input min。
- idle 不阻塞进度且能正确 reactivate。
- late row 被丢弃并可观测。

### M4

- 大量 keyed state 不进入 JSON。
- tumbling/hopping final result 与 batch oracle 相同。
- restore/compaction 不改变结果。

### M5

- source offset、operator state、watermark、sink pre-commit 共享同一 epoch 边界。
- file sink 通过完整 exactly-once fault matrix。
- capability-invalid plan 编译失败。

### M6

- file/Kafka/PostgreSQL/ClickHouse 可从严格 project v3 运行。
- Rust/Python/Studio 暴露一致生命周期和 delivery guarantee。
- schema/OpenAPI/client 与源码一致。

### M7

- performance、soak、security、packaging、docs 在 exact release commit 全绿。
- 所有 package version 为 `3.0.0`。

## 16. 推荐工程协作方式

所有 specialist artifact 使用同一 slug：`continuous-streaming-runtime`。

```text
M0 spec writer
  -> API designer
  -> critic
  -> milestone implementer
  -> focused tester
  -> reviewer
  -> performancer（hot path 变更）
  -> simplifier（每个完整 milestone 后）
  -> doc writer
```

每个 task 或紧耦合 task pair 使用独立 `feature/<description>` branch/PR。PR body 必须
包含：

- controlling spec/API artifact；
- RED test 与观测到的 failure；
- focused/full verification；
- delivery/cancellation/checkpoint 影响；
- hot path 的配对性能证据；
- 明确列出的未包含 follow-up。

## 17. 第一段实际实现切片

M0 批准后，只启动以下 vertical slice：

1. 原子拆分 `BatchOperator`/`StreamOperator`；
2. 原子拆分 `BatchExecutionPlan`/`StreamExecutionPlan`；
3. 引入 typed `StreamMessage`；
4. 引入 envelope/rows/bytes 三重限制 channel；
5. 跑通 in-memory source -> unary expression -> recording sink；
6. 证明 backpressure、FIFO、cancel、end-of-input、无 task leak；
7. 暂不公开 Python/Studio API。

这段切片只验证最核心的执行模型。在 state、time、connector 和跨语言 API 放大错误
成本之前，先证明 runtime contract 正确。
