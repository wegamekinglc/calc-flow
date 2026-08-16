# Calc-Flow 3.0 M6/M7 执行计划（Post-A6 基线）

> **状态：** 已建立 issue 与 PR 映射；M6.0 未启动
> **基线：** `main@858199f6df0161801bb6028f37f3ebbeb1684e3e`（PR #131 之后）
> **控制链：** 本文件负责执行顺序、结构性决策、issue/PR 映射与集成策略；
> 各 task 的完整 RED/实现/验收细节以
> [`2026-08-02-continuous-streaming-v3.md`](2026-08-02-continuous-streaming-v3.md)
> §12、§13 为准；[M6-00] 冻结后以 `m6-connectors-project-v3` slug 的
> spec/api-note/critique 为实现契约。冲突时以 delta package > 本文件 > 主计划
> 历史正文的顺序裁决。
>
> **范围：** Public A6 已交付 Rust/Python 公共持续运行时并删除旧 v2 runner。
> M6 负责 connector、project v3、Python capability 接线与 Studio v3；M7 负责
> hardening 与 `3.0.0` 发布。旧 continuous runner 删除已由 A6 完成，不属于
> 本计划。

## 1. 基线事实（M6 的受保护起点）

以下能力已由 A6 交付并有证据，M6 任何 task 不得重复实现、回退或绕过：

- crate-root 公共 `StreamingRunner` / `StreamingJob` / managed checkpoint
  owner：one-shot `start(self)` ownership、status/checkpoint/shutdown/cancel/
  wait 契约由 `crates/calc-flow/tests/continuous_public.rs` 与
  `python/tests/test_continuous_runtime.py` 固定。
- PyO3/Python 暴露同一 lifecycle：`compile_stream()`、`start_async()`、
  blocking 便捷方法在活动 event loop 中拒绝；无第二运行时。
- checkpoint manifest v3 与 `LocalStateBackend` 是唯一 durable 真相；
  48-case fault catalog 与 exact-head 20 分钟三进程 soak 已通过 public
  facade。
- project format 仍为 v2（`PROJECT_FORMAT_VERSION = 2`）；stream 模式经
  `compile_stream_project()` 表达，尚无 connector、secret reference、
  runtime.mode 字段。
- Studio 仍为 `/api/v2`；A6 只保留其私有 checkpoint 持久化接线。
- 全部包版本仍为 `2.0.0`；`3.0.0` 统一升版属于 [M7-04]。

## 2. 结构性决策（[M6-00] 冻结）

主计划 §3 要求新增 `calc-flow-connectors` workspace crate 前一次性裁决五个
边界。本节给出建议与理由；[M6-00] 的 spec/api-note/critique 必须逐项确认或
推翻这些建议并记录理由，之后才允许 [M6-01] 动工。

### D1 依赖边：`calc-flow-python` <-> `calc-flow-connectors`

**建议：feature-gated 可选依赖。**

- `calc-flow-python` 通过 `connector-file` 等 feature 依赖
  `calc-flow-connectors` 对应 feature；默认只启用 `file`。
- 默认 published wheel 携带 file/CSV/JSON/Parquet connector（无重系统依赖、
  供应链面小）；Kafka/PostgreSQL/ClickHouse/HTTP/WebSocket 为构建期 opt-in，
  由 feature-enabled CI legs 与容器测试证明。
- 主计划 M6.7/M6.8 的“从 project v3 定义并运行 PostgreSQL CDC ->
  ClickHouse/Parquet stream”验收门保持可达：在 feature-enabled 构建 +
  容器环境中验证；默认 wheel 的 `capabilities.py` 只枚举实际携带的
  connector。

**否决的替代：** 完全不依赖（M6.7/M6.8 验收门必须改写，Python 失去全部
原生 connector）；非 gate 的全量依赖（rdkafka/TLS 无条件进入每个 wheel 的
原生模块，manylinux/macOS/Windows 构建与 SBOM 成本不可控）。

### D2 feature gate

**建议：** `calc-flow-connectors` 每个 transport 一个 feature：`file`、
`kafka`、`postgresql`、`clickhouse`、`http`、`websocket`，默认只开 `file`。
纯 Rust 轻量 format codec（CSV、newline JSON）不设 feature、始终编译。
依赖方向恒为 `calc-flow-connectors -> calc-flow` 与
`calc-flow-python -> calc-flow-connectors`；core crate 永不依赖 connectors。

### D3 `--all-features`、CI 与覆盖率门

**建议：**

- `rdkafka` 使用 vendored/cmake 源码构建，保证无系统 librdkafka 时
  `--all-features` 仍可编译；PostgreSQL、ClickHouse、HTTP、WebSocket 客户端
  选纯 Rust + rustls 组合，避免系统级依赖。
- 需要真实服务的集成测试用 `#[ignore]` + 环境变量
  （`CALC_FLOW_CONNECTOR_CONTAINERS=1`）门控，在 ci-linux 的独立 service
  legs 上运行，不混入普通单测，也不计入
  `cargo llvm-cov --workspace --all-features --fail-under-lines 90` 采集。
- 传输客户端 glue 收敛到薄 shim trait 后面，普通单测用 fake 覆盖逻辑；
  workspace 90% 行覆盖门保持不变。若某 connector 的非容器路径确实达不到
  90%，必须在 [M6-00] spec 中按 connector 单独评审并显式设门，不得静默
  放宽。

### D4 版本与发布

**建议：** `calc-flow-connectors` 使用 `version.workspace = true`（当前
`2.0.0`），与 workspace 各包在 [M7-04] 一起升 `3.0.0`。是否随核心 crate
发布到 crates.io 由 [M7-04] 决定；自 [M6-02] 起保持打包卫生（license、
readme、exclude 容器测试与 docker 资产），release workflow 增加带默认
feature 的 connectors 构建 smoke。

### D5 交付顺序

**建议：** file/Parquet 先行（验证 registry/capability/format 设计与
exactly-once fault matrix）；Kafka 与 PostgreSQL 随后并行；ClickHouse 在
PostgreSQL 的 `database_types` 与容器基建就绪后进入；HTTP/WebSocket 最后。
project v3（[M6-07]）在 [M6-01] 的 capability 模型稳定后即可启动，并随每个
connector 落地增量扩展 capability 校验。

## 3. 任务、issue 与分支映射

每个 task 一个 GitHub issue、一个 feature branch、一个 PR。分支命名遵循
`feature/<description>`；connector task（[M6-02] 至 [M6-06]）与 [M6-01]
直接以 additive、feature-gated 变更合入 `main`；产品面 task（[M6-07] 至
[M6-09]）在 milestone integration branch `feature/m6-integration` 上以
stacked PR 开发，[M6-10] 完成原子切换后一次合入 `main`（沿用 M5 与主计划
§4.1 的策略）。

| Issue   | Task  | 交付                       | 前置      | 单轨周 |
| ------- | ----- | -------------------------- | --------- | ------ |
| [M6-00] | M6.0  | freeze M6 delta docs       | —         | 1      |
| [M6-01] | M6.1  | core connector registry    | M6.0      | 2      |
| [M6-02] | M6.2  | file/parquet connector     | M6.1      | 3      |
| [M6-03] | M6.3  | kafka connector            | M6.2      | 3.5    |
| [M6-04] | M6.4  | postgresql connector       | M6.2      | 5      |
| [M6-05] | M6.5  | clickhouse connector       | M6.4      | 2.5    |
| [M6-06] | M6.6  | http + websocket connector | M6.2      | 1.5    |
| [M6-07] | M6.7  | project v3 format          | M6.1      | 2.5    |
| [M6-08] | M6.8  | python v3 + capability     | M6.7      | 1.5    |
| [M6-09] | M6.9  | studio /api/v3 jobs        | M6.8      | 3.5    |
| [M6-10] | M6.10 | m6 integration cutover     | M6-02..09 | 1.5    |
| [M7-01] | M7.1  | perf + memory gates        | M6-10     | 2      |
| [M7-02] | M7.2  | security + supply chain    | M6-10     | 1.5    |
| [M7-03] | M7.3  | docs + migration guide     | M6-10     | 1.5    |
| [M7-04] | M7.4  | 3.0.0 release verify       | M7-01..03 | 1.5    |

周数为单工程师估算的中位值。各 task 单轨中位合计约 27.5 engineer-weeks，
略高于主计划 §4 的 M6 里程碑估算（16 至 24 周）：task 级分解把容器测试、
CI 基建与集成切换显式计价。M7 单轨合计约 6.5 周。

## 4. 各 task 执行要点

### [M6-00] Task M6.0：冻结 M6 delta 规格

**目标：** 以本执行计划为输入，产出同 slug 的
`.codex/artifacts/specs/m6-connectors-project-v3.md`、
`.codex/artifacts/api-notes/m6-connectors-project-v3.md` 与
`.codex/artifacts/critiques/m6-connectors-project-v3.md`；逐项裁决 D1 至
D5；冻结 connector capability 词表、`SecretResolver` 契约、project v3
顶层结构、Studio `/api/v3` route/SSE 模型、connector 错误投影与脱敏清单；
更新主计划 §12 挂接该 delta。

**验收门：** 三份文档同 slug 无矛盾；critique 记录 `BLOCKS REMAINING: 0`；
文档 PR 不声称任何 M6 实现证据。分支 `feature/m6-delta-spec`。

### [M6-01] Task M6.1：核心 connector registry 与 capability 层

**目标：** 在 core crate 建立 immutable connector registry 与 format 层：
`crates/calc-flow/src/connector/{mod,capability,format,registry}.rs`。

**关键交付：**

- connector identity 固定为 `(provider, name, version)`；
- capability 按 delivery、replay、watermark、transaction、lookup、snapshot、
  polling、CDC 独立声明，禁止单一 `database = true` 笼统代替；
- compile 时捕获 plan-scoped immutable registry snapshot；
- `SecretResolver` 只接受 secret reference，connector config 序列化结构上
  不能包含 secret value；
- `FormatDecoder` / `FormatEncoder` 与 transport 正交，decoder expansion 受
  rows/bytes 限制；
- 请求 `delivery = "exactly_once"` 时对完整 reachable
  source/operator/edge/sink capability 编译期验证，失败在任何 source open
  前。

**与主计划的偏差：** 本 task 不创建空的 `calc-flow-connectors` crate（仓库
禁止 placeholder module）；该 crate 随 [M6-02] 的首个真实 connector 建立，
workspace/CI/coverage mechanics 一并在 [M6-02] 落地。

**RED 重点与验收门：** 主计划 Task M6.1 清单。分支
`feature/m6-connector-registry`。

### [M6-02] Task M6.2：file/Parquet connector 与事务 sink

**目标：** 建立 `crates/calc-flow-connectors/` crate 并交付首个 connector。

**关键交付：**

- crate 骨架：workspace 成员、`version.workspace`、feature gates（D2）、
  license/readme/exclude、cargo audit/deny 更新；
- source：finite file/directory snapshot（CSV、newline JSON、Parquet）、
  稳定排序 file identity + completed row group cursor、显式 schema 与
  bounded decode、symlink/traversal/wrong type/locked file fail closed；
- sink：按 pipeline/output/epoch 写 Parquet、staging 与 target 同
  filesystem、epoch manifest/directory 原子 rename 幂等 commit、不覆盖无关
  用户文件；
- CI feature matrix 与容器门控基建（本 task 无需服务容器）；
- 用真实 file sink 复跑完整 M5 fault matrix。

**验收门：** 在明确支持的本地 filesystem 假设下 file-to-Parquet
exactly-once。分支 `feature/m6-file-parquet-connector`。

### [M6-03] Task M6.3：Kafka connector

**关键交付：** `rdkafka` vendored/cmake 构建与三平台 wheel link 验证；
partition 到本地 source task 的确定映射；checkpoint per-partition offset；
bounded-channel backpressure 暂停消费；JSON/Avro 只经注册 format 解码；
transactional ID 由 pipeline/sink identity 稳定派生且不含 secret；
producer fencing、rebalance、lost partition、timeout、recovery 的显式处理；
CI service leg（Kafka/Redpanda）；容器测试不混入普通单测。

**验收门：** restart 后 partition replay 正确；transactional sink 对
committed epoch 无重复记录；fault matrix 通过前不宣传 exactly-once。分支
`feature/m6-kafka-connector`。

### [M6-04] Task M6.4：PostgreSQL connector

**关键交付：** source 三 mode（`snapshot` repeatable-read 一致快照、
`incremental_query` 严格有序 composite cursor、`logical_cdc` publication +
slot + `pgoutput`）；CDC 输出 append-only change-event envelope（operation、
relation、transaction ID、commit LSN、commit time、key、before、after）；
barrier 只在 transaction commit boundary 注入；slot exported snapshot 与
initial copy 无 gap 衔接；checkpoint cursor 用 commit LSN；manifest durable
后才 confirm flush LSN；slot 生命周期与 lag 监控显式配置。sink 三 mode
（`append` 批量插入、`upsert` 参数化 `INSERT ... ON CONFLICT`、
`transactional` epoch ledger 同事务提交）；Arrow/PostgreSQL 显式类型矩阵；
secret 只经 resolver。容器测试覆盖 connection loss、timeout、serialization
failure、deadlock、failover、commit-ack loss、ledger conflict 与 M5 fault
matrix。

**验收门：** 主计划 Task M6.4 验收门。分支
`feature/m6-postgresql-connector`。

### [M6-05] Task M6.5：ClickHouse connector

**关键交付：** `snapshot`（启动固定 upper bound）与 `incremental_query`
（event-time/sequence + unique tie-breaker）source；identifier 严格验证、
value 全参数化；DateTime/DateTime64/Decimal/Nullable/LowCardinality/Enum/
UUID/IPv4/IPv6/Array 支持矩阵，未知类型 compile fail。sink 默认
at-least-once 批量 insert；每 epoch 稳定 `insert_deduplication_token`
（`retry_deduplicated` 不等于 `exactly_once`）；小 batch 聚合与 block
rows/bytes 限制；retry 重用相同 token 与 row order；table engine/dedup
setting 启动检查并据此声明 capability。

**验收门：** 主计划 Task M6.5 验收门。分支
`feature/m6-clickhouse-connector`。

### [M6-06] Task M6.6：HTTP polling 与 WebSocket connector

**关键交付：** response size/timeout/retry/conditional request；ETag/
Last-Modified 可选 replay cursor；frame 与 decoded batch 大小限制；能暂停
读取时默认 `Block`，`DropOldest` 显式且与 exactly-once 互斥；TLS 验证默认
开启，insecure 必须显式且告警；authorization header、含凭据 URL、payload
脱敏。

**验收门：** capability 准确区分 replayable HTTP、unreplayable HTTP、lossy
WebSocket。分支 `feature/m6-http-websocket-connectors`。

### [M6-07] Task M6.7：project v3 严格 data-only 格式

**关键交付：** `format_version: 3`、`runtime.mode: batch | stream`、对应
mode 的 runtime options、graph nodes/edges、source binding（connector、
format、watermark、secret reference）、sink binding（connector、format、
requested delivery）、数据库 binding（snapshot/polling/CDC 或
append/upsert mode、cursor、目标表、capability requirement）、
state/checkpoint config；v2 拒绝并移入历史文档目录；每层 unknown field
fail closed；canonical serialization/fingerprint 确定；generated schema 与
`schemas/project-v3.schema.json` 精确一致。在 `feature/m6-integration` 上
开发。

**验收门：** 单个 project v3 无 executable object/secret 地定义 PostgreSQL
CDC -> window -> ClickHouse/Parquet stream。分支 `feature/m6-project-v3`
（base 为 `feature/m6-integration`）。

### [M6-08] Task M6.8：Python project v3 与 capability 接线

**关键交付（按主计划 PR #131 后重写的 M6.8）：** project v3 document 编译
到现有 A6 native job handle，不新增第二套 owner；connector 注册、format、
secret reference 与 capability 枚举在 Rust/Python 间一致；wheel 只暴露实际
携带或可注册的 connector capability；project v3 validation error 保留稳定
field path 且不泄漏 secret；`_native.pyi` 与 runtime member 精确一致。

**验收门：** Python 与 Rust 从同一 project v3 得到一致的 connector、
capability、delivery guarantee 与既有 A6 lifecycle；重复 cancellation
stress 保持通过。分支 `feature/m6-python-capabilities`（stacked）。

### [M6-09] Task M6.9：Studio `/api/v3` 持续 job API 与 UI

**关键交付：** `/api/v3` projects/jobs/events/checkpoint/shutdown/cancel
routes 与 SSE event model；删除 `/api/v2` 前先立等价资源上限（最大并发
job 数、单 job 与全局常驻内存上限、最大 checkpoint/state 磁盘占用、用户
显式 stop 的生命周期），不允许只把 worker timeout 置空；long-running
persistent worker ownership 与 worker death -> checkpoint recovery/terminal
status 一致性；`serve()` loopback-only；SSE 不发送 secret 或 raw payload；
UI 展示 queue、watermark、epoch、throughput、backpressure、late row、job
state；EventSource/timer/listener/worker 清理；OpenAPI 与 TypeScript type
同 commit 生成。

**验收门：** 浏览器创建 v3 stream project，start、observe、checkpoint、
stop，重连后看到 terminal status，无 worker/EventSource 泄漏。分支
`feature/m6-studio-v3`（stacked）。

### [M6-10] Task M6.10：M6 集成切换与 v2 删除

**关键交付：** 汇总 `feature/m6-integration` 上的 stacked PR；project/REST
v2 删除（旧 continuous runner 删除已由 A6 完成，不在此范围）；
`schemas/project-v3.schema.json`、`web-ui/openapi.json`、
`web-ui/src/api/schema.d.ts` 同 commit 生成；全量 Rust/Python/Studio/
frontend/supply-chain/generated-file/diff gates；exact-head 20 分钟 soak
（按 M5-D12-E1 精神对 connector 路径补充证据）；原子合入 `main`。

**验收门：** 主计划 §15 M6 门槛。分支 `feature/m6-integration`。

### [M7-01] Task M7.1：性能与内存门禁

同机比较全部 M0 Criterion baseline；5% 回归门由配对数据与置信区间共同
支持；universal 1,200 秒 two-source backpressure soak；大于旧 10 MiB JSON
上限的高基数 window-state soak；checkpoint duration 按 dirty-key volume
归因；cold/warm cache recovery 分开测；Python/Studio overhead 与 native
runtime 分开归因；external array byte cost 保守且有测试。

### [M7-02] Task M7.2：安全与供应链门禁

威胁模型覆盖 secret、path、symlink、decompression bomb、oversized
message、malicious schema、deep connector option、SQL identifier/query、
replication slot、WAL retention、database ledger、ClickHouse dedup token；
TLS default 与 credential redaction 验证；`cargo audit`、`cargo deny`、
`npm audit --omit=dev` 与 artifact inspector 覆盖启用全部 connector
feature 的配置；`rdkafka` 等依赖 license/platform build 审查；覆盖率门在
最终配置下真实通过；checkpoint/state cleanup 不遍历 symlink；fuzz/
property test project、checkpoint、format、state metadata decoder。

### [M7-03] Task M7.3：文档、示例与迁移指南

重写 `docs/introduction.md` 与 Rust/Python API guide；按实际 v3 contract
更新 runtime-envelope 文档；按 connector 记录 delivery guarantee；file、
Kafka、PostgreSQL snapshot/CDC/sink、ClickHouse polling/sink、watermark、
tumbling、hopping、recovery、transactional sink 示例；v2 -> v3 breaking
guide（不承诺自动迁移）；`AGENTS.md`/`CLAUDE.md` 架构摘要按最终源码更新；
`CHANGELOG.md` 列出全部删除/替换的 public surface。

### [M7-04] Task M7.4：版本与发布验证

workspace crate、Python core、Studio、frontend、`calc-flow-connectors`
同步升 `3.0.0`；PyO3 对 core 保持 exact version；从最终源码生成 project
v3 schema、OpenAPI、TypeScript type；构建 core wheel、sdist、crate、Studio
wheel 并逐个 `scripts/inspect_wheel.py`；clean environment 安装 smoke；确认
source tree 无 `python/calc_flow/_native*.so`；release commit 重跑完整
exactly-once fault matrix。

## 5. CI 与 feature 矩阵

| CI leg                  | 启用 feature | 服务容器                  | 覆盖率      |
| ----------------------- | ------------ | ------------------------- | ----------- |
| linux unit（既有）      | all          | 无                        | 计入 90% 门 |
| linux connector-file    | all          | 无                        | 计入 90% 门 |
| linux containers        | all          | kafka/postgres/clickhouse | 不计入      |
| windows / macos（既有） | default      | 无                        | 计入 90% 门 |

ci-linux 新增 service legs 时使用 GitHub Actions `services:`（Kafka、
Postgres、ClickHouse），本地开发用 `crates/calc-flow-connectors/` 下的
docker compose 文件与相同环境变量门控。容器 legs 只跑
`CALC_FLOW_CONNECTOR_CONTAINERS=1` 的 `#[ignore]` 测试，失败即 leg 失败，
但不进入覆盖率分母。

## 6. 双轨排期

| 轨道           | 任务序列                                    | 工程周 |
| -------------- | ------------------------------------------- | ------ |
| A（connector） | M6.1 -> M6.2 -> M6.3 / M6.4 -> M6.5 -> M6.6 | 16.5   |
| B（产品面）    | M6.7 -> M6.8 -> M6.9                        | 7.5    |
| 共同           | M6.0；M6.10 集成                            | 2.5    |

关键路径为轨道 A（约 16.5 周）；两轨并行时 M6 日历时间约 15 至 20 周，
M7 约 4 至 6 周；自 [M6-00] 起至 `3.0.0` 约 20 至 26 个日历周。总量口径与
主计划 §4 相比的差异见 §3 的说明。

## 7. 风险与缓解

- **rdkafka 三平台构建**是最大外部风险：vendored/cmake 构建在 [M6-03]
  第一步先行验证 manylinux/macOS/Windows wheel link，失败则回退为
  “Kafka 不进默认 wheel、只提供源码 feature”并收缩验收门，决策记录回
  [M6-00] 的 delta。
- **WSL2 容器测试可行性**：[M6-02] 期间验证本机 docker 可用性；不可用则
  容器测试只在 CI service legs 运行，本地以 fake shim 覆盖逻辑。
- **90% 覆盖率压力**：传输 glue 薄 shim + fake 单测；确实达不到的
  connector 在 [M6-00] spec 中显式单独立门，禁止静默放宽 workspace 门。
- **Studio timeout 等价上限设计**：[M6-09] 动工前必须在 [M6-00] 的
  api-note 中冻结上限词表与默认值，避免“只是把 timeout 置空”。
- **schema/OpenAPI 漂移**：每个 stacked PR 内同步再生成年文件并跑
  `git diff --exit-code` 三件套。
- **范围蔓延**：connector 的功能以主计划 §1.2 的 3.0 non-goals 为准；
  session window、early trigger、changelog、multi-input join 等一律拒绝。
