# Latest-relation Update Contract — API Note

状态：DAL-513 / S1a 局部合同冻结，2026-11-11，供 implementer 直接采用。**本文新增类型、方法和字段均为本片待实现合同；本轮只做文档与源码合同自检，没有产品运行证据。**

基线：`f524493cb80546f35b65db0ecb9d2c86f1a2ea2e`，分支 `agent/dal-orchestrator/1946ef7bb730`，开始时工作树干净。协调器已核实 PR #591 合并 SHA `7ed2773bea2eac3c4256b8c8221f08d98971f8c4` 为本 HEAD 祖先；本轮另行核对本 HEAD 的实际接口，不把前序测试转记为本轮通过。

## Audiences and scope

Rust 用户通过公开 `calc-flow` 的 DataFrame / SQL 声明和 Program，Python 用户通过根包 `calc_flow` 的同类入口，显式选择更新解释，取得逐节点、逐边诊断，并在读取 Source 前获知不支持的组合。两类声明共用一个 Program 选择；batch / stream 继续是独立生命周期维度。

S1a 交付**表达、事实传播和强制准入**。不新增 DataFrame aggregate，不实现普通 SQL 对完整关系的替换消费，不承诺 SUM→SUM 的 latest 结果 6 已可计算。该组合的本片结果是准确拒绝；DAL-514 / S1b 才实现其获准支持域。旧默认的 7 继续合法。

依据：

- DAL-513 当前描述及触发评论 `01a1c9c9-f9fc-7238-beca-7cb4450093fb`；父 DAL-498 的状态生命周期、四单元目标及顺序交付讨论。
- [first-class-relational-frontends](first-class-relational-frontends.md)、[query-history-portability](query-history-portability.md) 及其 [分析证据](../analysis/query-history-portability.md)、[前端分析](../analysis/first-class-relational-frontends.md)。
- 父计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58` 的 S1 / G1、C02/C12/C13、A04/A06/A09、L01–L03、O01；main 复审附件 `01a1c33e-a567-774a-b132-9d1f20b09d2b` 的 G2。附件通过 Multica CLI 下载核对；历史分析不能替代本 HEAD 验证。

按 `AGENTS.md`、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md` 使用 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint 词汇。本仓库实际边界为 Rust core → PyO3 → Python adapters；没有 `dal-public/src/`、C++/Excel binding 或 `.codex/references/component-boundaries.md`，不据此虚构 facade、绑定或路径。现有 Rust/Python example 32 的位置与用途保留。

## Surface today

以下均为基线源码事实，后文新增签名不能写作已有 API。

| Surface                                                                    | Current contract                                                                                                                                      |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/src/dataframe.rs`、`sql.rs`、`program.rs`                  | typed TableExpr、列、row-local 变换、SQL、Program、for_mode、schema/analyze/compile/to_project 已公开；没有更新语义选择。Program fingerprint 排除 mode；native plan 身份包含 mode。 |
| `python/calc_flow/symbolic/sql.py`、`expr.py`、`program.py`                  | sql(query, **tables) 与 TableExpr.sql 共用关系声明；Program 接受 mode 或 legacy engine，提供 owned stream；没有 update_semantics 属性或选择方法。                              |
| `query_analysis.rs:34`、`:193`、`:216`                                       | UpdateKind=append/complete_snapshot/unknown，与 InputInterpretation=append/complete_replacement/unknown 分开；报告 schema_version=1、analysis_revision=1。     |
| `query_analysis.rs:504`、`:566`、`:629`                                      | 投影/按批变换已有快照传播规则；普通消费者一律报告 append。SUM→SUM 的 mismatch 拒绝依赖额外 final_relation 请求；不是一个持久图语义开关。                                                             |
| `operator/sql.rs:1190`、`:1210`                                             | 普通 aggregate 累计接受事件，发完整累计快照；retained 候选在 emit 成功后替换。非 aggregate SQL 保留按批行为，不能仅见到 SELECT 就授予 row-local 证明。                                             |
| `operator/expression.rs:309`、`:390`                                        | 原生表达式处理当前 Batch，检查取消后 emit 结果；这支持空结果传递的实现方向，不能仅据此宣称 fusion、路由和 sink 全路径已验。                                                                            |
| `operator/snapshot_sql/`、`pipeline/stream.rs:1343`                         | snapshot_sql 是现有独立完整替换消费者；保留全 alias 初始化、whole-Batch、空替换、source slicing 准入及输出直达 sink 限制。                                                               |
| `config.rs:124`、`pipeline/compile.rs:508`                                  | PipelineSpec 尚无更新选择字段；strict project-v3 与 native graph configuration/fingerprint 是持久身份入口。                                                             |
| `symbolic/lower/program.py:768`、`symbolic/query_analysis.py`、`_native.pyi` | 准备缓存包含 mode、Program fingerprint，并受注册 scope 隔离；query facts 由单一 Rust 规则集投影；私有 native seam、错误转换和 stub 需随合同同步。                                            |
| 两端 `examples/32_sql_expression_pipeline`、`docs/relational-frontends.md`    | 已有可独立使用的公开 mixed DAG、双 wrapper 和状态所有权示例；继续增量维护，不移动 core 示例。                                                                                           |

Rust 未写完整前缀的文件在 `crates/calc-flow/src/`；Python `symbolic/` 在 `python/calc_flow/`。本片不重新调查 Polars 版本，也不增加全面兼容承诺。

## Proposed surface: one immutable Program selection

冻结名称 **UpdateSemantics**，取值 **legacy_append**、**latest_relation**。选择属于 Program 的声明语义，覆盖所有输出可达边；本片不提供节点级 override，也不将它放进运行时资源配置或 SQL alias 字典。

- `legacy_append` 是默认：保留普通 SQL 的快照追加、其他 SQL 的 per-batch 解释，以及显式 snapshot_sql 已有替换行为。
- `latest_relation`：当生产者发完整快照时，消费边必须保持并按完整替换解释；不能把每一版快照继续追加进消费者历史。真实 append 行仍是 append，不把普通源批次或按批 DISTINCT 自动升级为完整历史快照。
- 这只选择**更新解释**。输入历史、终态截止、行序和跨 mode 等价仍由现有 PortabilityRequest 明示并证明；latest 不等于 global，也不意味着任意 SQL 双端等价。

新增公开签名清单如下；impl 中省略方法体，并非可直接编译的定义。

~~~rust
// crate-root and program module re-export.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UpdateSemantics {
    #[default]
    LegacyAppend,
    LatestRelation,
}

impl Program {
    pub const fn update_semantics(&self) -> UpdateSemantics;
    pub fn with_update_semantics(&self, semantics: UpdateSemantics) -> Self;
}

impl PipelineBuilder {
    pub fn with_update_semantics(self, semantics: UpdateSemantics) -> Self;
    pub const fn update_semantics(&self) -> UpdateSemantics;
}
~~~

~~~python
from typing import Literal

# Export from calc_flow and calc_flow.symbolic.
UpdateSemantics = Literal["legacy_append", "latest_relation"]

class Program:
    @property
    def update_semantics(self) -> UpdateSemantics: ...

    def with_update_semantics(
        self, semantics: UpdateSemantics, /
    ) -> Program: ...
~~~

选择方法只复制声明并设置枚举，不能规划、读行、启动 job 或持有 state。Rust enum 输入无验证失败，返回 Self；Python 精确验证 str / 已知值后返回新对象。原构造签名、SQL 签名、CompileOptions 和 AnalysisOptions 不增加同义开关。默认 Program 的原有声明身份保持；显式切回 legacy 是创建另一 wrapper 的有意选择，不修改原 wrapper 或已经启动的 job。

`with_input`、`output`、`for_mode` 等复制操作保留选择；模式变化仍不改变该 Program 的声明 fingerprint。新对象重新从声明计算身份，不在旧 fingerprint 上反复叠加标记。两个不同选择的 wrapper 不共享准备结果或 state；没有可变 setter。

四个接口单元均使用同一选择方法，构造/执行无需写物理节点或端口。下表是**确定采用的调用形状**，不是四套同义配置。

| Cell             | Declaration and explicit choice                                                                                                                                                                               |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Rust DataFrame   | `let r = input.select(columns)?; let p = Program::new("df", mode, BTreeMap::from([("out".into(), r)]))?.with_update_semantics(UpdateSemantics::LatestRelation);`                                              |
| Rust SQL         | `let r = calc_flow::sql::sql(query, tables)?; let p = Program::new("sql", mode, BTreeMap::from([("out".into(), r)]))?.with_update_semantics(UpdateSemantics::LatestRelation);`；TableExpr.sql 是既有单 alias 便利入口。 |
| Python DataFrame | `p = cf.Program("df", mode=mode, outputs={"out": source.select("x")}).with_update_semantics("latest_relation")`                                                                                               |
| Python SQL       | `p = cf.Program("sql", mode=mode, outputs={"out": cf.sql(query, s=source)}).with_update_semantics("latest_relation")`；TableExpr.sql 同样使用这个 Program。                                                           |

编译、analyze、explain、collect / collect_async、stream / execute、to_project 的现有返回类型与 owned 生命周期保留。TableExpr 的 compute/stream 便利方法没有新的局部开关；需要此声明语义时建立 Program。ArrayExpr、Parameter 及混合 array/table 输出在 legacy 下不变；latest 的输出可达 opaque/provider/array 语义若无证明，按 unknown 拒绝，不静默退回 legacy。

低层 PipelineBuilder / project 编译也必须携带同一选择并强制 gate，以防导出或重载丢掉合同；其原同步 compile 方法保持同步签名，不能隐藏 block_on。所需证明使用同一 Rust 解析/分类规则，未提供足够 schema/端口证据则 unknown；高层 async schema planning 可提供充分元数据。私有实现组织由 implementer 决定，不新增第二套 SQL 关键词分析器。

## Edge facts and admission

复用 QueryNodeFacts、QueryModeFacts、QueryEdgeFacts、QueryEdgeModeFacts 的字段，不把 `complete_snapshot` 改名为 `latest_relation`。前者是产出形状，后者是图的选择；history、update、finality、ordering 各有独立含义。

报告增加 required 字段 `QueryAnalysis.update_semantics: UpdateSemantics`，Python 投影为相同 literal。报告 wire 升 **schema_version=2、analysis_revision=2**：有新增安全字段且规则改变。Python 手工构造 QueryAnalysis 的旧位置字段保持顺序，末尾新增默认 `update_semantics="legacy_append"`；新运行报告总是 v2。旧 v1 报告仅可按 legacy 展示，不能作为 latest 准入证明；未知 schema/enum 拒绝解析或证明，不自动授予 supported。capability 的现有 schema/revision 不因此自增。

逐边规则如下。先形成两种 mode 各自的真实事实，再检查**实际编译 mode** 的 latest 义务；可移植请求另做两 mode 的比较检查。

| Actual edge / shape                                                     | legacy stream facts and behavior                                                    | latest stream admission in S1a                                                                                                               |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| append 源 → 单 alias 普通 aggregate SQL                                     | append → append；节点 accepted_history / complete_snapshot / revisable_snapshot；原累计行为。 | 保留原累计 producer，可作为终端完整快照输出。latest 不要求它把原始 append 源改作 replacement。                                                                            |
| append → 已证明 row-local DF / SQL                                         | append → append；保留当前批域和可证明的 finality。                                               | 支持原计算；不增加 retained 历史或快照。                                                                                                                    |
| 完整快照 → 确定性单输入投影、过滤、with_columns / row-local SELECT 或 CTE                | producer_update=complete_snapshot；原 per-event 计算，快照事实继续传播。                          | consumer_interpretation=complete_replacement；产出仍 complete_snapshot / revisable_snapshot，history=latest_relation。必须证明完整事件及空结果均保留；不得返回 append。 |
| 完整快照 → 普通 aggregate SQL                                                 | producer_update=complete_snapshot，consumer_interpretation=append；可执行，旧事件解释可得7。      | unsupported + update_interpretation_mismatch，定位这条消费边。保留实际 consumer=append，不能为了迎合请求把事实改成 replacement。不产生6。                                    |
| 完整快照 → DISTINCT/ORDER/LIMIT/OVER 等非 row-local 按批 SQL                    | 继续报告其真实按批范围；不因 latest 意图抹掉事实。                                                       | 本片没有 replacement allowlist 证明，unknown + update_semantics_unproved；已有明确不支持约束仍为 unsupported。后续可逐类扩证明，本片不扩大 S3。                                 |
| 完整快照 → union/window/rolling/cross-section/join/ASOF 等 append-only 状态消费者 | 其当前真实消费解释按已有实现报告。                                                                   | 已知 append-only 则 unsupported + update_interpretation_mismatch；不依节点名字断言其可消费 replacement。                                                      |
| 明确 whole relation 输入或累计 aggregate → 既有 snapshot_sql                     | 该节点原本就是 complete_replacement consumer；全 alias 初始化后输出完整最新关系。                         | 只复用其已有 stream 支持域、source 完整性和预算证明；不去掉输出直达 sink、ExactlyOnce 等限制，不新增 Rust snapshot convenience API。                                            |
| snapshot_sql 输出 → 其他 Operator                                           | 原有 mode/边限制拒绝。                                                                      | 仍 unsupported + unsupported_mode；即使目标是 row-local 也不借本片放宽。                                                                                    |
| unknown 生产者/消费者、opaque UDF/provider、混合形状或无法证明边界                         | 普通入口按现有准入，不因新报告改计算。                                                                 | unknown + update_semantics_unproved 或 update_event_boundary_unproved；在 Source 前拒绝。                                                           |
| 完整快照 → 命名终端输出 / sink                                                    | sink 接收的是完整关系事件，不是增量行。                                                              | 允许直接交付同形状事件；StreamOutput 仍是既有 name/table，无新控制注入或 delta/retraction 协议。消费方按报告识别输出。                                                             |

允许的快照中间变换限定为现有 Rust 分类能证明的确定性单输入 row-local 域，排除 aggregate、DISTINCT、ORDER/LIMIT、analytic/window、opaque UDF、多 alias SQL。不是仅凭无 aggregate 即通过。普通 append 输入上的旧按批查询不会仅因选择 latest 被改成 global；其 history=current_batch、finality=per_batch 继续展示。若另要求全历史 final_relation，PortabilityRequest 仍可拒绝。

任一快照经多级投影/过滤不能洗成 append；unknown 不能在下游变成 supported。相同节点有多个消费者时，每条实际边分别报告，任何输出可达的不兼容边使本次 latest 编译失败。不自动剪掉失败分支或改变共享上游 owner。

对于 latest 的可支持快照变换，history 表示当前收到的 latest_relation；普通累计 producer 的 accepted_history 仍是原始 append 接受历史。一个错误的下游累计器的 accepted_history 实际是上游快照事件历史，不能误称原始 H。SQL 仍建立新 row lineage、清除 temporal identity；保留更新事实不恢复 event/entity/sequence 证明，ORDER BY 也不建立该证明。

batch 普通有限 SQL / row-local 计算仍对本次完整输入执行，保持其已有 current_batch / append 事实。latest 不创造跨次 replacement 状态，不因报告中的 stream 反例拒绝本来合法的 batch 编译。stream-only 节点用于 batch 仍按原 mode 限制拒绝。因此 batch SUM→SUM 可编译，而同声明 stream latest 拒绝；若请求跨 mode latest 等价，该请求也应拒绝。

## Empty relation, event boundary and failure atomicity

完整关系事件的单位是**一个 Data Batch**：其中全部 RecordBatch 属于一次替换。latest 的完整快照和所派生的结果不可被优化器、fusion、队列或 binding 切成多次替换；完整事件超过边 row/byte 上限就报原预算错误，不能靠分块“成功”。append 输入仍可按原规则切批。

状态语义固定为：

| Observation                     | Meaning                                                               |
|---------------------------------|-----------------------------------------------------------------------|
| 尚未收到 Data / 迭代器没有事件             | 未初始化或无本次更新；不制造空关系、不重算、不清除上次关系。                                        |
| 收到精确 schema、0 行的完整 Data Batch   | 一次真实空替换，同时完成初始化；若此前关系非空，最新关系现在为空。                                     |
| 输入多个 RecordBatch 的一个 Data Batch | 一个完整替换，不能逐 RecordBatch 触发多次关系变更。                                      |
| 完整快照经 filter 得0行                | 必须发出一个带正确输出 schema 的空 Data Batch；保留 complete_snapshot，不能省略事件或保留旧非空关系。 |
| Idle / Watermark / Barrier      | 沿原控制合同推进，不伪造 Data 或初始化关系；Barrier 可捕获此前已提交状态。                          |
| EOF                             | 保留最后已提交关系；原本从未初始化不能假装已收到空表。全 alias 未初始化时沿既有 snapshot_sql 规则不发结果。      |

标记 complete_snapshot 只说明实际产出事件完整，不保证空历史一定存在事件。空源与一条空 Data 是不同测试；没有空历史终态归约证明时 portability 仍可 unknown。空/全 NULL 的 SUM、COUNT、GROUP BY、HAVING 结果沿现行 SQL 合同，不能统一设为0。batch 仍返回一次执行的表，不能把其空表与 stream 无事件混为一谈。

实现者必须检查原生计算、优化/fusion、路由、Python stream 适配及终端命名输出的事件完整性。必要时在 latest 快照路径禁用不能保持边界的优化；未能证明时用 update_event_boundary_unproved 预拒绝，不标 supported 之后再丢空事件。本片不建立通用碎片重组协议。

原子性按实际可承诺的边界区分：

1. **本片预检拒绝是整次启动前原子拒绝。** 不 open/poll/next Source，不创建输入 bindings、Runner、临时 state root 或 job，不提交状态/checkpoint，不向 sink 发 Data；命中准备缓存也一样。
2. **获准执行的单 Operator 候选**先验证输入 schema、计算和取消、结果完整性及边预算，成功后才提交其候选状态。失败候选不能替代此前已提交的 latest 关系或进入新 checkpoint。S1a 不新增普通聚合 replacement 候选；snapshot_sql 和旧 SQL 的既有候选/预算合同不回退，S1b 新状态必须遵循此原则。
3. 已交付的成功前缀仍可见。fan-out enqueue、下游 sink 写入和取消竞态没有本片新增的全图事务；不能声称任意 sink 失败均撤销所有外部副作用。沿原有 job failure、replay、checkpoint/source cut 和逐输出 delivery 合同处理，不将候选原子性冒充 ExactlyOnce。

多 source 的 snapshot 继续是 latest-available，不保证同一事件时间的原子切面。budget、overflow、实际数据错误或执行期间取消仍可能发生；静态准入不证明这些数据相关条件永不失败。

## Stable preflight codes, paths and error projection

沿 DAL-511 / DAL-512 现有逻辑地址，默认不改变既有 issues 的历史 path：

- SQL 消费边：`P.sql.tables.<alias>`；snapshot 为 `P.snapshot_sql.tables.<alias>`。
- Rust typed DF：`P.project.value` / `P.filter.value`；Python DF 沿现有 primitive/role 地址，不为跨语言表面相似强制相同 path。
- 输入/输出：`inputs.<name>`、`outputs.<name>`，非 identifier 名用 JSON string 下标，如 `outputs["risk.total"]`；同声明/版本内稳定，不承诺改图或重命名后地址不变。
- 低层 node：`graph.nodes["id"]`；边：`graph.edges["source"]["out"]["target"]["in"]`。project 重载使用物理端点地址；不伪造已丢失的原 symbolic path。
- 配置：高层 `Program.update_semantics`；project `graph.update_semantics`。事实子路径可后接 `.batch` / `.stream` 和已有 aspect。
- 共享节点只有一个 canonical path，但消费边各保留一次；沿前序确定性遍历与 UTF-8 path 排序，diagnostics 按 (path, code, status) 排序去重。不得只返回 producer path 而遗漏 offending consumer。

| Code                                                       | Status and fixed meaning             | Required path / example                                                                             |
|------------------------------------------------------------|--------------------------------------|-----------------------------------------------------------------------------------------------------|
| invalid_literal                                            | 配置值不在两个枚举值中；不是 unknown 查询。           | `Program.update_semantics` 或 `graph.update_semantics`；Python 非 str 用 TypeError，未知 str 用 ValueError。 |
| update_interpretation_mismatch                             | unsupported；已知完整快照要求替换消费，但目标实际上只追加。  | 精确消费 edge path，沿既有 code 含义；例如 `outputs.result.sql.tables.input`。                                    |
| update_semantics_unproved（新增）                              | unknown；生产/消费更新解释缺充分证明，本片不能准入。       | 精确消费 edge path；终端产出未知则节点的 `.stream.update`。                                                         |
| update_event_boundary_unproved（新增）                         | unknown；完整事件、空替换或不切片性无法证明。           | 失去证明的 producer `.stream.update`，或具体消费 edge；说明欠缺哪项。                                                  |
| unsupported_mode                                           | unsupported；已有生命周期或 snapshot 输出路线限制。 | 沿既有 node/edge 的 mode.support 路径；不被 latest 覆盖。                                                       |
| schema_mismatch / unresolved_type / ordering_unproved 等既有码 | 保留现有错误分层、意义与 offending field。        | 保留具体列、alias、identity、SQL 节点或边；未知 schema 不用空 schema 冒充。                                              |

新增 Rust 错误采用 `CalcFlowError::QueryUpdateSemantics { mode: CompileMode, report: Box<QueryAnalysis> }`，返回原 Result。复用 Python CompileError 并附不可变 `query_analysis`，无需新异常类。错误首行固定 `{path}: {code}: {message}`，从**本次 mode 的更新义务失败**中按上述顺序选择首项，其余保留报告；不能选另一 mode 的无关失败作为首因。message 是人类解释，不能用于 switch。

示例：
`outputs.result.sql.tables.input: update_interpretation_mismatch: latest_relation requires complete replacement at input; producer emits complete_snapshot but consumer appends`。

该错误不同于已有 QueryPortability：即使 `report.request=None`、`report.portability=None`，显式 latest 的本 mode 准入仍须失败。不要伪造 PortabilityRequest 来复用只能检查 portability 的旧 check。若两者均请求，更新准入和比较证明都必须通过；优先报告配置/声明错误，其次本 mode 更新拒绝，再报告额外比较义务，不吞掉原编译器拒绝。

analyze/explain 继续返回报告/文本，可呈现 unsupported/unknown，不因最新选择直接抛执行准入错误。Python AnalysisResult.issues 投影当前诊断 mode 的失败，query 中保留双 mode 事实。collect_schema 只规划精确 schema，to_project 可导出语义明确但当前不可执行的声明；它们不构成运行许可。

强制 gate 发生在所有 Program compile / execute / collect / collect_async / stream 路径的准备缓存读取之前，使用同一不可变注册 snapshot；Python 严格 batch 路径也必须先于遍历和包装输入。async 错误在 await 或 async with 启动时交付，阻塞 planning 留在已有工作线程。low-level builder、project compile/validate、runner 的最终计划准入同样不能丢弃选择；不要求用户再传 portability 才安全。schema/原 compiler、mode、delivery、Source 完整性及恢复身份检查继续全部生效。

## Serialization, cache and checkpoint identity

选择是执行语义，不能套用 DAL-511 的“纯诊断不入身份”规则。冻结持久字段为 **`graph.update_semantics`**；私有 data-only DeclarationGraph / PyO3 分析输入也携带同名选择，缺省仅表示 legacy，不能丢失 latest 后再重新分析。持久字段形状如下：

~~~json
{
  "graph": {
    "name": "latest",
    "nodes": [],
    "edges": [],
    "datafusion": {},
    "update_semantics": "latest_relation"
  }
}
~~~

这是字段位置片段，不是可执行的完整 ProjectSpec。Project format 仍为3；schema 与严格 serde 同步扩字段，未知值/字段仍拒绝。

1. `PipelineSpec` 新增 `pub update_semantics: UpdateSemantics`，serde default=legacy_append，**legacy 值必须 skip_serializing_if**。旧 JSON 缺字段、显式 legacy 经规范化、默认 Program 导出和 native configuration 仍归同一 legacy 语义；旧有省略字段的文档重载/再导出、节点 ID、canonical bytes 与 native fingerprints 不变。不能全局补写一个 default 字段导致旧图变 hash。
2. **Rust 源码兼容例外明确披露：** PipelineSpec 当前是 public struct，新增字段会使下游完整 struct literal 需要补 `update_semantics: UpdateSemantics::LegacyAppend`（或 Default）。这是本片选定的最小源码迁移；不能称其完全 source-compatible。旧 Program、SQL、PipelineBuilder 方法调用签名不变；仓内 literals、公开 docs 和外部 consumer fixture 同步迁移。不得把字段藏进 description/datafusion 配置来规避语义边界，也不因此重构所有配置类型。
3. Program 内部默认值 legacy；该值按**原编码原样**计算声明 fingerprint。latest 使用独立、确定性的版本化语义域参与 fingerprint；从同一声明切换选择必须改变身份，for_mode 保持选择/声明身份。两语言内部声明编码各沿现有体系，不新增跨语言相同 hash 承诺。
4. Native graph fingerprint 只在非 legacy 时加入 `update_semantics="latest_relation"`。低层 builder、typed lowering、项目重载均走同一规则，batch 与 stream 原 execution_mode 仍入身份。不要仅更改 Python cache key 而让 native fingerprint 相同。
5. 准备缓存继续受 mode、声明、schema、注册 session/revision、late settings 隔离；新选择通过新 fingerprint 或显式 key 参与。检查/证明缓存另包含完整选择、report schema_version=2、analysis_revision=2 及 portability 请求。每次读取执行缓存前重做或复用同一 snapshot 下的有效 gate；不能凭旧 supported 报告绕过。
6. 报告 schema/revision、explain 文本和 PortabilityRequest 仍是诊断，不写入 native 执行身份或 state payload。不能让分析修订使原 legacy checkpoint 失配；真正的 update 选择必须使 legacy/latest 原生身份不同。
7. S1a 没有新增 state payload/layout，不升级普通 SQL state_version 或 compact3 / retained4 / ordered5，也不转换旧快照。已有 mode、engine、operator、state/layout、UDF、source/connector、epoch/source cut、checkpoint runtime-config 等校验保持。新 latest 图恢复 legacy checkpoint 必须在 source 拉取前按原身份校验失败；反向同样失败，不能在 restore 时“猜”解释。
8. 未改选择的 legacy 图继续按旧布局恢复，SUM→SUM 仍可得7；不能只测两者 fingerprint 不同而不验实际拒绝。后续 S1b 如果引入 replacement 状态，应另设其 state/layout 身份及初始化标记，不复用旧累计 payload。
9. 新 binary 接受旧省略字段的 project；旧 strict binary 不识别 latest 字段时明确拒绝，这是正确的降级边界。不提供剥字段后继续执行的新旧解释转换。
10. batch snapshot 对内置 SQL 仍为 NULL，execute 的临时计算状态每次隔离；项目保存、计划缓存和 stream checkpoint 均不变成 batch 查询中途续算。

## Typical use and rejection

以下是待实现签名的示例种子，语法自检不等于运行验证。Python 例演示 SQL 累计快照经 DF filter 的安全终端输出；第二批使累计和变负，因此应产生一个空关系更新。

~~~python
import calc_flow as cf
import pyarrow as pa

source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
total = source.sql("SELECT SUM(x) AS total FROM input")
visible = total.filter(total["total"] >= cf.lit(0))
program = cf.Program(
    "latest-filter", mode="stream", outputs={"result": visible}
).with_update_semantics("latest_relation")

async def observe():
    seen = []
    data = [pa.table({"x": [1]}), pa.table({"x": [-2]})]
    async with program.stream({"events": data}) as events:
        async for event in events:
            seen.append(event.table.to_pydict())
    assert seen == [{"total": [1]}, {"total": []}]

# Declare the unsupported consumer without collecting intermediate rows.
chained = total.sql("SELECT SUM(total) AS value FROM input")
rejected = cf.Program(
    "latest-chain", mode="stream", outputs={"result": chained}
).with_update_semantics("latest_relation")
# analyze/explain shows the edge; compile_stream/stream rejects before reading data.
~~~

Rust 同义关键链如下；schema construction、udfs snapshot 和 stream requirements 沿既有公共 API，不引入新 runner。

~~~rust
let total = source.sql("SELECT SUM(x) AS total FROM input")?;
let visible = total.filter(total.column("total")?.ge(lit(0_i64)?))?;
let program = Program::new(
    "latest-filter", CompileMode::Stream,
    BTreeMap::from([("result".into(), visible)]),
)?.with_update_semantics(UpdateSemantics::LatestRelation);
let report = program.analyze(&udfs, &AnalysisOptions::default()).await?;
let plan = program.compile_stream(
    &udfs, &requirements, &CompileOptions::default(),
).await?;
~~~

SUM→SUM 用 `[1]`、`[2,3]` 的独立手表：上游快照为1、6，旧下游追加终值7；latest stream 本片拒绝这条下游边，不执行到7后才警告。batch 给完整 H=[1,2,3] 可得6；stream 同 job 单 SUM 为1、6，同 batch plan 两次独立执行为1、5。不能把这些不同输入历史的执行混作等价证据。

## Why this shape and rejected alternatives

Program 统一选择保留现有 SQL 的任意 alias 命名，避免增加 `sql(..., update_semantics=..., **tables)` 抢占合法 alias；也避免给每个 select/filter/with_columns 添加同义参数。四单元使用相同词汇与 gate，未来 aggregate 不必另发一套运行协议。本片粒度足以满足整图 latest 安全准入；需要同一 Program 内混合两套解释时，未来另定局部作用域，不能隐式 override。

不把 latest 当 PortabilityRequest 别名：请求只检查跨 mode 比较，新选择必须持久化并决定消费准入。不默认切换 snapshot_sql、不增加全历史 retained、不重写 SQL 或 reset stream 来修成6。也不把 UpdateKind.complete_snapshot、HistoryScope.latest_relation 和声明选择合并为一个枚举。

本片选择一个显式图字段及已披露的 Rust struct literal 迁移，优于把行为藏入描述、opaque metadata 或资源 limits；保留省略 legacy 的 canonical identity，避免全体旧图、缓存、checkpoint 被一次默认字段插入改义或失效。

## Acceptance evidence for implementation

下列是后续实现者必须取得的定向证据，**本轮均未执行产品测试**。复用现有测试模块，不默认增加 spec、critic、独立 tester、性能阶段或全量矩阵。

| Evidence                    | Minimum observable result                                                                                                                                                        |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 四单元 / A04、A06、A09           | 外部 Rust consumer 分别构造 typed DF、SQL 及 mixed DAG，根包 Python 同义构造；选择可发现、只读、复制保留；Rust re-export、Python exports/type annotations/native seam/error stub/docs 同步。DF aggregate 未实现须如实写明。 |
| C02 / 旧默认                   | 对 [1]/[2,3] 保留无选择和显式 legacy 的旧 SUM链7；terminal SUM 1/6；不以只取下游最后快照冒充6。                                                                                                             |
| C02 / latest mandatory gate | 不传 portability 的 latest SUM链在启动前拒绝；直接 SQL→SQL 及 SQL→DF投影/过滤→SQL 均定位实际消费边；普通 analyze 返回准确双 mode facts。                                                                            |
| 传播及空关系 / O01                | 上述 SUM→filter 的1→空更新到达命名 stream 输出；至少一条 SQL row-local 中间链同样保留 complete_snapshot；一个 Batch 多 RecordBatch 仍一个事件；无事件/一条空 Data 不混淆。不能仅断言 metadata。                                    |
| 未知与明确拒绝 / C13               | opaque SQL/UDF 或无边界证明得到 unknown 新code；已知 append consumer 得 unsupported mismatch；schema/ordering/volatile 和 mode 的旧拒绝继续准确。                                                        |
| source 前零副作用 / A09          | 自定义 Source 或 iterator 的 open/poll/next、bindings factory、job/sink 调用计数为0；普通/strict、同步/async、stream入口、低层 builder、导出重载均不能绕过。检查不读取行；缓存暖后重试仍在 gate 拒绝。                                |
| 路径稳定                        | 同一声明/注册 snapshot 的重复报告相同；含特殊输出名及共享 producer 的每条消费边可定位。不能用优化后物理 node ID 替代 typed symbolic 地址。                                                                                     |
| 序列化与身份                      | 旧 project fixture 规范化/再导出和 legacy Program/native fingerprints 字节级不变；latest 字段往返不丢、身份不同；两 mode 保持原分离。实际 restore 验 legacy↔latest 错配在 source前拒绝，未改 legacy 恢复继续成功。                   |
| C12 受影响回归                   | 复用 snapshot_sql 初始化/空替换/whole-Batch/EOF/恢复及预算候选证据；输出仍直达 sink、原 source slicing 与 ExactlyOnce 拒绝不放宽。只在改动影响这些路径时跑最小对应子集。                                                            |
| C13 原子性                     | 预检失败无 source/state/output；若触及空事件/路由，验超边预算先拒绝、schema保留、不拆完整事件。若触及已有状态路径，验失败候选未替换已提交关系、checkpoint不含失败候选；不把外部 sink失败宣称全图回滚。                                                         |
| L01–L03                     | 同 stream job跨批累计；同 batch plan 暖缓存无reset两次1/5；新job fresh，显式恢复另验同逻辑历史。便利调用每次新plan不能代替同plan隔离测试。                                                                                    |
| 检查及来源                       | native/Python 修改后使用本 checkout 匹配扩展，记录源码SHA、解释器/extension来源；仅受影响编译、类型/格式、schema往返及示例检查。全量/coverage交CI，不转抄旧测试为本SHA通过。                                                              |

建议落点：`crates/calc-flow/tests/query_history_portability.rs`、`relational_program.rs` / `relational_frontend.rs`、`pipeline/batch/template_tests.rs`、SQL / snapshot_sql 邻近测试；Python `test_query_history_portability.py`、`test_relational_frontend.py`、`test_symbolic_compile_cache.py`、`test_program_engine.py`、`test_stream_results.py` 及既有 snapshot 模块。实现按触及路径选择用例，不要求全跑这些模块。

## Open questions and delivery boundary

**当前无须阻塞实现的未决公共合同。** Program 选择层级、默认、边义务、空关系、错误及身份已经确定。PipelineSpec 的 Rust struct literal 迁移是本文明确选择并披露的兼容成本，须随实现落实，不留作“默认完全兼容”的假设。不存在的 dal-public / Excel / component-boundaries 路径不构成本仓库 blocker。

实现中若已有优化路径无法保持完整/空事件，依本文收紧 latest allowlist、禁用该优化或准确预拒绝；这不要求先扩建运行协议。只有无法在上述明确成功域完成完整事件传递、无法保留旧 legacy canonical identity，或实际受限接口阻止持久化/恢复校验时，才记录具体源码与失败证据作为 blocker；不能把 S1b 尚未实现当本片阻塞。

留给 DAL-514：typed DF aggregate 的具体函数/数值/nullability 签名，普通 SQL 完整 replacement 候选、预算和新state/layout及恢复。S2/S3 的有限 union/window/join/ASOF、global DISTINCT/top-k/OVER 依原切片；snapshot_sql batch适配、一般 outer/semi/anti、external强制隔离和 batch中途恢复不扩入本片。

本轮只新增此 API note 并附线程完成评论；文档结构、签名/引用/路径、Python示例语法、空白和写入范围自检后交协调器安排 implementer → 独立 reviewer。未改产品、schema、测试、agent配置，未提交、推送、建PR或扩派。DAL-513 仍 in_progress，不能把设计交付当成本 issue 已合并完成。
