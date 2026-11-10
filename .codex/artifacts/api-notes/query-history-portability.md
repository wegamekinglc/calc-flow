# Query History and Portability — API Note

状态：DAL-511 / S0a-2 的局部公共合同决策，供 implementer 采用；本文不表示产品已实现。2026-11-11，静态核对 HEAD `2114e374441fa5099902f511bf23914c286d0c87`，分支 `agent/dal-orchestrator/94f7a11bdce2`。只新增本文，不修改产品、提交或创建 PR，不运行产品测试。

## Audiences and scope

Rust 用户从现有 `PipelineBuilder` 获取声明图诊断、显式检查后编译，随后继续使用 `BatchExecutionPlan` 或 `StreamingRunner`。Python 用户从现有 `Program` 获取不可变事实，并在编译或执行时显式提出严格请求。DataFrame/SQL 是声明接口类别，batch/stream 是生命周期；本片不交付 DAL-512 的 Rust typed relation 前端，也不迁移 `engine`。

本决策承接父 DAL-498 的 API note 附件 `01a1bc01-9dbe-7c02-ac79-02093c8a178e`、计划 v2 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`、main 复审 `01a1c33e-a567-774a-b132-9d1f20b09d2b`。三件均已通过 CLI 下载读取；它们未在本 HEAD 入库。既有 `symbolic-relational-dag`、窗口、join、ASOF 的身份、排序和迟到合同继续约束诊断。

仓库没有 `dal-public/src/`、C++/Excel binding、methodology 目录及 `.codex/references/component-boundaries.md`。实际边界是 `crates/calc-flow` → PyO3 `crates/calc-flow-python` → `python/calc_flow`；用 `docs/introduction.md` 的 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint 词汇。既有 Rust/Python 示例位置和用途保留。

## Surface today and evidence

以下为本 HEAD 已有事实，不是新签名：

| Surface / evidence                                                                 | Current contract                                                                                                                                                                                |
|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `python/calc_flow/symbolic/analyzer.py:225`                                        | `AnalysisIssue(path, code, message)`；`AnalysisResult(mode, program_fingerprint, capability_session_id, capability_revision, issues)`，均不可变。没有公开 query history/update/finality facts。 |
| `symbolic/program.py:516`、`:551`、`:565`                                          | `Program.analyze/explain(runtime=None, /, *, mode=None)`；`compile_batch`、`compile_stream`。分析的 mode override 仅诊断，不改变绑定 engine。                                                   |
| `symbolic/analyzer.py:885`、`:3024`、`:3081`                                       | 遍历声明并取 capability snapshot；SQL schema 委托 native。SQL 结果建立新 lineage，清除 temporal ordering；stream 多 alias 普通 SQL 拒绝。                                                       |
| `crates/calc-flow/src/lib.rs:64`、`:117`                                           | 导出 `PipelineBuilder`、两种 plan、`SqlOperator`、`Result`、`CalcFlowError`，尚无 typed Rust Program/TableExpr 分析入口。                                                                       |
| `pipeline/batch.rs:237`、`pipeline/stream.rs:792`                                  | 现有 compile 同步且消费 builder；共享图校验，保留 stream-only / volatility / snapshot 边限制。                                                                                                  |
| `expression.rs:60`、`:77`、`:102`                                                  | `ValidatedQuery` 持有解析结果；`has_stream_aggregate` 检查 grouping / 非窗口 aggregate；simple projection 判据比“没有 aggregate”严格。                                                          |
| `operator/sql.rs:486`、`:1190`、`:1207`、`:1353`                                   | schema-only planning；普通 aggregate 走累计历史，非 aggregate fallback 使用当前 batch；compact/retained 是执行策略。特殊 native SQL window 路径另有规划证明，不能仅见 `OVER` 就推断。           |
| `pipeline/batch.rs:383`、`:504`；`runtime/streaming/operator_task.rs`              | 内置 batch 每次独立 execute 创建运行上下文/session；stream 同一 job 复用算子状态。                                                                                                              |
| `python/calc_flow/pipeline.py:218`、`:239`；`symbolic/lower/program.py:765`        | schema/准备缓存按运行时注册 scope 隔离，准备 key 含 mode；stream 只缓存声明并创建独立拥有的 plan。                                                                                              |
| `python/calc_flow/stream.py:112`、`:249`；`compute.py:183`                         | stream 编译先于 source bindings 和 `runner.start_async`；batch collect 目前先包装输入再编译，严格路径需提前 gate。                                                                              |
| `examples/29_sql_stream_switch.py`、Rust 同名例；两端 `32_sql_expression_pipeline` | 已有同声明双生命周期及 mixed DAG 示例，可保留并补诊断；无需移动示例或新建 facade 程序。                                                                                                         |

表内 `symbolic/` 前缀为 `python/calc_flow/symbolic/`，Rust `pipeline/`、`operator/` 等前缀为 `crates/calc-flow/src/`。本轮只读源码和现有断言，不将历史附件的运行结果当作本 SHA 测试结果。

## Decisions: three states and two kinds of support

冻结 `SupportStatus = supported | unsupported | unknown`：

- `supported`：针对确切声明形状、schema、注册 snapshot 及请求，已有规则证明该项成立；结论附明示比较前提。
- `unsupported`：已有反证或实现限制，例如 stream-only 节点用于 batch、按批 DISTINCT 请求全历史去重、快照被下游累计追加。
- `unknown`：缺少足够证据，例如 opaque/复杂 SQL、外部算子未声明历史或 row order 不能证明。它不表示已知不能计算。

每个 mode 的 `support` 表示已知生命周期准入；整图 `portability` 表示本次请求的跨 batch/stream 关系比较结论，两者独立。普通 SUM→SUM 可在两 mode 准入，旧 stream 结果 7 合法，但其 final-relation 请求为 `unsupported`。SQL 能 infer schema、编译、checkpoint，或 capability 有 `stateful=True`，均不能单独得到 portability=`supported`。

没有请求时 `portability=None`（JSON `null`），不得默认为 supported。严格入口只接受 supported；unknown 与 unsupported 均在拉取 Source 前拒绝，但在报告和错误中保留不同 status。规则合取时，相关依赖有明确反证取 unsupported，否则有未知取 unknown，全通过才 supported；只对输出可达路径及请求相关义务求合，不让无关未证明事实遮盖已可证明结论。

## Proposed surface: data types

下列名字及字段是本片决定。Rust 从 crate root 导出；数据类型用 owned 值、`Clone + Debug + Serialize`，请求额外 `Deserialize`、严格字段校验，枚举 JSON 为 snake_case。报告只要求 Serialize，不提供“导入报告即获准执行”的入口。Rust extensible 报告 struct/enum 标 `#[non_exhaustive]`；请求结构提供构造函数，避免要求下游以 struct literal 绑定未来字段。

Python 从 `calc_flow` 及 `calc_flow.symbolic` 导出同名数据类型，采用仓库 `_compat.dataclass(frozen=True, slots=True)`。集合为 tuple；入参序列先复制。报告中的 enum 是受注解约束的字符串 literal。Python 高层类型无需全部变成 PyO3 class。

```rust
pub enum SupportStatus { Supported, Unsupported, Unknown }
pub enum HistoryScope {
    CurrentBatch, AcceptedHistory, EventWindow, LatestRelation, Unknown,
}
pub enum UpdateKind { Append, CompleteSnapshot, Unknown }
pub enum InputInterpretation { Append, CompleteReplacement, Unknown }
pub enum QueryFinality {
    PerBatch, PerRowFinal, GroupFinal, RevisableSnapshot, Unknown,
}
pub enum Comparison { PerBatch, FinalRelation }
pub enum ComparisonCutoff { BatchBoundary, AllSourcesEof, ClosedWindow }
pub enum ComparisonOrdering { Unordered, Declared }

pub struct PortabilityRequest { /* private fields, immutable getters */ }
impl PortabilityRequest {
    pub fn new(
        comparison: Comparison,
        history: HistoryScope,
        cutoff: ComparisonCutoff,
    ) -> Result<Self>;
    pub fn with_ordering(self, ordering: ComparisonOrdering) -> Self;
    pub fn comparison(&self) -> Comparison;
    pub fn history(&self) -> HistoryScope;
    pub fn cutoff(&self) -> ComparisonCutoff;
    pub fn ordering(&self) -> ComparisonOrdering;
}

#[derive(Default)]
pub struct AnalysisOptions {
    pub portability: Option<PortabilityRequest>,
}

pub struct QueryAnalysis {
    pub schema_version: u32,
    pub analysis_revision: u64,
    pub request: Option<PortabilityRequest>,
    pub portability: Option<SupportStatus>,
    pub nodes: Vec<QueryNodeFacts>,
    pub edges: Vec<QueryEdgeFacts>,
    pub diagnostics: Vec<QueryDiagnostic>,
}
pub struct QueryNodeFacts {
    pub path: String,
    pub schema: Option<arrow::datatypes::SchemaRef>,
    pub batch: QueryModeFacts,
    pub stream: QueryModeFacts,
}
pub struct QueryModeFacts {
    pub support: SupportStatus,
    pub history: HistoryScope,
    pub update: UpdateKind,
    pub finality: QueryFinality,
    pub ordering: QueryOrderingFacts,
    pub reason: String,
}
pub struct QueryEdgeFacts {
    pub path: String,
    pub source_path: String,
    pub target_path: String,
    pub batch: QueryEdgeModeFacts,
    pub stream: QueryEdgeModeFacts,
}
pub struct QueryEdgeModeFacts {
    pub support: SupportStatus,
    pub producer_update: UpdateKind,
    pub consumer_interpretation: InputInterpretation,
    pub reason: String,
}
pub struct QueryOrderingFacts {
    pub kind: QueryRowOrder,
    pub keys: Vec<QuerySortKey>,
    pub ties: QueryTieOrder,
    pub temporal_identity: Option<QueryTemporalIdentity>,
}
pub enum QueryRowOrder { Unspecified, Ordered, Unknown }
pub enum QueryTieOrder { Unspecified, Total, Unknown }
pub struct QuerySortKey {
    pub field: String,
    pub direction: SortDirection,
    pub nulls: NullPlacement,
}
pub struct QueryTemporalIdentity {
    pub entity_by: Vec<String>,
    pub event_time: String,
    pub sequence_by: Vec<String>,
}
pub struct QueryDiagnostic {
    pub path: String,
    pub code: String,
    pub status: SupportStatus,
    pub message: String,
}
```

`arrow` 表示 core 已使用的 Arrow re-export/dependency 类型路径，不新增版本。`SortDirection`、`NullPlacement` 复用 crate 现有类型。Rust schema 以现有 `ArrowFieldSpec` 数据形状序列化，Python 投影为 `tuple[Field, ...] | None`；unknown schema 用 None，已知零列用空 tuple，不能混淆。保留 names、列序、dtype、nullable，不把 DataFusion Debug 文本当 schema。

Python 除下列默认外，报告字段全部 required；向量映为 tuple，嵌套类型同名：

```python
@dataclass(frozen=True, slots=True, init=False)
class PortabilityRequest:
    comparison: Literal["per_batch", "final_relation"]
    history: Literal[
        "current_batch", "accepted_history", "event_window", "latest_relation"
    ]
    cutoff: Literal["batch_boundary", "all_sources_eof", "closed_window"]
    ordering: Literal["unordered", "declared"]

    def __init__(self, *, comparison, history, cutoff,
                 ordering="unordered") -> None: ...

@dataclass(frozen=True, slots=True, init=False)
class AnalysisOptions:
    portability: PortabilityRequest | None

    def __init__(self, *, portability=None) -> None: ...

# Existing positional fields retain their order; append only this defaulted field.
# AnalysisResult(..., issues, query: QueryAnalysis | None = None)
```

`_compat.dataclass` 的 Python 3.9 分支不支持 stdlib `kw_only`，因此以自定义关键字 `__init__`（参数类型同字段注解）实现上述合同，构造中验证后设置 owned 冻结字段，不提高 Python 下限。所有新运行产生的 AnalysisResult.query 均非 None；None 仅允许既有代码按旧五字段手动构造报告。原 `issues` 不因新增普通事实而增加失败；显式请求产生的失败投影为既有 `AnalysisIssue(path, code, message)`，完整 status 留在 query.diagnostics。

`reason`、`message` 是可改进的人类解释，不是 switch key；`code` 才是机器判断依据。无结论字段必须明确 Unknown，不用空字符串或“默认 append”填充未知。事件窗口是声明的窗口范围，`GroupFinal` 指真实 group/window 闭合后的追加，不表示每个输入 batch 都闭合。

history 描述节点实际收到的输入作用域；下游 accepted_history 可能累计的是上游快照事件，不能等同于原始 Source 的 H。必须沿 source_path/target_path 和 update/consumer_interpretation 判定整图请求，保留这种历史来源差异。

ordering.kind=ordered 时 keys 必须非空且可解析为输出字段；表达式排序若无法映射到这些字段则 unknown，不能省略排序项。unspecified 的 keys 为空；ties=total 需要完整唯一排序元组证明，单有键名不够。temporal_identity=None 表示没有可用的 temporal proof，不能反推“数据实际无序”。模式不支持时仍可展示已知历史事实，其余字段保留 unknown。

### Request interpretation and validity

请求是比较和准入合同，不改图、不增加 retained state、不把普通 SQL 切换到 snapshot SQL。它比较同声明的 batch 和 stream，受两 mode 各自原准入限制约束。

| Input                                                 | Validity and meaning                                                                                     |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `per_batch / current_batch / batch_boundary`          | 每个 stream 输入 batch 与独立 batch execute 对比，包含事件边界；不比较整个 job 的累计结果。              |
| `final_relation / accepted_history / all_sources_eof` | batch 显式接收同一完整接受历史 H；stream 到所有相关 Source 的真实 EOF，按产出种类归约。                  |
| `final_relation / event_window / closed_window`       | 同一声明窗口、接受集合、late 策略及真实闭合；若无 batch adapter，unsupported。                           |
| `final_relation / latest_relation / all_sources_eof`  | 请求完整关系的 latest 解释；本片仅识别既有能力，凡需改变旧快照追加边或 snapshot_sql 的 mode 限制均拒绝。 |
| 其他组合、`history=unknown`、bool 冒充数值或未知 enum | `invalid_literal` / host 类型错误；配置无效，不是 unknown 查询。                                         |

`ordering="unordered"` 按多重集比较，保留重复数、schema/NULL/数值合同；它不允许忽略 LIMIT/OVER 对所选行的影响。`declared` 要求结果行序可证明，ties 的相对次序只有完整 tie key 才保证；不能用 SQL ORDER BY 补 temporal identity。

比较的共同前提写进用户文档及 explain：输入是同一接受集合/合法次序，batch 由调用者提供本次完整 H（或逐批 bi）；late drop 比较接受集合，error 比较成功前缀。诊断不读取行、不验证两次实际输入相等，不保证数据相关预算、overflow、取消或 ExactlyOnce。EOF/closed_window 是有效比较截止条件，不强迫 Source 有界，不伪造 watermark，也不保证无界 job 最终达到它。输入为空、没有输出事件与一条空完整关系必须区分；没有可证明的空历史归约规则时 strict 结果 unknown。

## Proposed surface: analysis, compile and execution

Rust 复用现有 builder，不发明 Rust Program：

```rust
impl PipelineBuilder {
    pub async fn analyze(
        &self,
        udfs: &UdfRegistrySnapshot,
        options: &AnalysisOptions,
    ) -> Result<QueryAnalysis>;

    pub async fn compile_batch_checked(
        self,
        udfs: &UdfRegistrySnapshot,
        request: &PortabilityRequest,
    ) -> Result<BatchExecutionPlan>;

    pub async fn compile_stream_checked(
        self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
        request: &PortabilityRequest,
    ) -> Result<StreamExecutionPlan>;
}
```

新方法 async，因 schema / SQL logical planning 已有 async native 路径；不能在 Rust 内隐藏 `block_on`。原同步 `compile_batch/compile_stream` 签名与行为不变。检查后仍调用原 compiler，原 compiler 的全部错误和 mode/checkpoint/delivery 校验仍有效；分析 supported 不覆盖最终编译。report 返回已知语法/schema/mode 失败作为 diagnostics；无效 options、无法建立分析基础的内部错误以 Result error 返回，不能以 unknown 吞掉引擎内部失败。

Python 冻结这些增量（其余现有参数/返回保持）：

```python
def analyze(self, runtime=None, /, *, mode=None,
            options: AnalysisOptions | None = None) -> AnalysisResult: ...
def explain(self, runtime=None, /, *, mode=None,
            options: AnalysisOptions | None = None) -> str: ...
def compile_batch(self, runtime=None, /, *,
                  portability: PortabilityRequest | None = None) -> BatchExecutionPlan: ...
def compile_stream(self, runtime=None, /, *,
                   allowed_lateness_micros=0, late_policy="error",
                   portability: PortabilityRequest | None = None) -> StreamExecutionPlan: ...
```

同一可选 keyword `portability=None` 追加到 `Program.collect`、`collect_async`、`stream`、`execute` 的现有 keyword 参数末尾，类型仍为 `PortabilityRequest | None`。execute 按既有 engine 分派并透传，返回类型不变。异步 collect 和 stream 在其编译工作线程中 gate，异常在 await/`async with` 启动时交付；不在 asyncio 线程执行阻塞 planning。同步 API 延续现有阻塞限制。required 参数仍在前，不加 positional 参数。

本片明确的严格入口是 Rust `*_checked` 和 Python Program 上的 `portability=`。不扩展 `cf.compute`、TableExpr 快捷方法、低层 project import/export 或 runner 配置；需严格行为的快捷调用者先建 Program。analyze/explain 带 request 仍返回报告/文本，不抛 portability 拒绝；先 analyze 再无 request execute 不构成检查凭据。避免添加模糊的 `strict=True`。

Rust 严格拒绝新增 `CalcFlowError::QueryPortability { report: Box<QueryAnalysis> }`，仍返回 `calc_flow::Result<T>`。Python 对应既有 CompileError（含 native 转换），携不可变报告 `query_analysis: QueryAnalysis`，无需新异常类；旧异常类分层、原始错误保持。错误首行 `{path}: {code}: {message}`，选择排序后的首条请求 blocker；其他失败在 report。已知不支持与证据不足不共用一句“unsupported”。

PyO3 复用 Runtime 的注册 snapshot/schema-only planning 机制，提供内部 data-only 图/SQL 分析 seam，核心解析和事实规则只在 Rust 一处维护；Python 做 symbolic 地址映射和不可变报告投影，不实现第二套 SQL 关键字分析器。binding 具体私有方法名不属于公共合同，但所有新增 native 方法须同步 `_native.pyi`；不能只把 native JSON dict 暴露给 Python 用户。JSON 只携 schema/声明/请求和事实，不接受数据、Source、Sink、Runner、callable、import path 或 checkpoint state。

## Stable paths and codes

原 AnalysisIssue 的 path/code 和项目 ValidationIssue 路径保持；不修改既有 fingerprint 编码、SQL query、物理 node ID 或项目图以制造诊断地址。新增事实地址遵循所属声明的逻辑路径：

| Context                              | New fact address                                                                                                                                                                        |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python root                          | `outputs.<name>`、`inputs.<name>`、`static_inputs.<name>`，沿现有 analyzer。                                                                                                            |
| Python SQL producer / consuming edge | SQL 节点自身位于 P；入边固定为 `P.sql.tables.<alias>`，snapshot 为 `P.snapshot_sql.tables.<alias>`。source_path 指向上游节点，target_path=P。                                           |
| Python other arguments               | 沿现有 `P.<primitive>.<role>`；没有既有命名 role 时固定 `P.<primitive>.args[<index>]`，从 0 起；字段表达式保留 `P.<field_name>`。                                                       |
| Rust builder node                    | `graph.nodes[<JSON string node_id>]`，例如 `graph.nodes["sum2"]`；不使用节点数组下标或 topology 次序。                                                                                  |
| Rust builder edge                    | `graph.edges[<JSON source id>][<JSON source port>][<JSON target id>][<JSON target port>]`，例如 `graph.edges["sum1"]["output"]["sum2"]["input"]`。                                      |
| Fact aspect                          | 节点/边 path 后可加 `.batch` / `.stream`，再加 `.history`、`.update`、`.finality`、`.ordering`、`.support` / `.consumer_interpretation`；请求配置路径是 `options.portability.<field>`。 |

新增 Python 地址中，名称是标识符时用 `.name`，否则用 JSON string 下标，如 `outputs["risk.total"]`、`.tables["a.b"]`；控制字符按 JSON 转义。既有 issues 的历史 path 文本不追溯改写，适配器保留两者来源，不能声称旧歧义路径变成新唯一 ID。新事实不额外输出未经转义的物理 SQL/query 文本。

每个唯一逻辑节点只列一次：Python 从逻辑输出名按 UTF-8 字节序遍历、按上述 role 路径选最小到达路径；已声明输入始终使用 inputs/static_inputs 根。共享节点的每条实际消费边各列一次；edge.source_path 指向该唯一节点。Rust 使用 builder node IDs 和四元组。增加新输出若提供更小地址可改变共享节点 canonical path；此范围明确只保证“同一声明和版本”稳定，不保证重命名、改图、跨前端不同构造或不同 lowerer 的路径相同。Python lower 到 native 时维护地址映射，不能把 `cf_sql_<digest>`、优化后节点序号当 Python 公共 path。

facts 按 path 的 UTF-8 字节序排列；diagnostics 按 `(path, code, status)` 排列，重复同键合并，message 不参与排序。相同声明、注册 scope、options 和 analysis_revision 得相同事实；报告不含随机 job ID、时间、内存地址。SQL span 不作为 v1 必填字段：目前不能可靠映射时定位 SQL 节点/边，message 明说未定位子表达式，不伪造 span。

| Code                                                                  | Status / fixed meaning                                                                           |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `unsupported_mode`（沿用）                                            | unsupported：现有生命周期/图边限制，不含“可能以后实现”。                                         |
| `ordering_required`（沿用）                                           | unsupported：当前声明缺少算子规定的 temporal identity；ORDER BY 不补该证明。                     |
| `capability_mismatch`（沿用）                                         | unknown：注册 snapshot 未证明所需 provider/UDF 合同；已知缺失注册则沿既有编译错误。              |
| `history_scope_mismatch`（新增）                                      | unsupported：已知实际 current_batch 等作用域不满足请求。                                         |
| `update_interpretation_mismatch`（新增）                              | unsupported：完整快照沿 append 消费导致请求关系解释不成立，必须定位边。                          |
| `sql_semantics_unproven`（新增）                                      | unknown：schema/解析成功，但 SQL 形状不足以证明请求的历史/等价合同。                             |
| `query_semantics_unproven`（新增）                                    | unknown：非 SQL 子图或组合缺少所需证明。                                                         |
| `finality_unproven`（新增）                                           | unknown：没有请求截止处的合法归约/闭合证明，包括未解决的空历史产出。                             |
| `row_order_unproven`（新增）                                          | unknown：请求 declared 顺序但缺少所需 row/tie order 证明。                                       |
| `invalid_literal` / `schema_mismatch` / `unsupported_type` 等（沿用） | 保留既有含义及错误路径；新 facts 的对应 mode 标 unsupported。不是 portability 的新替代错误体系。 |

新增 code 不改已有 code 含义，须加入现有 frozen symbolic code 清单及公共文档。同一明确 code 不因未知变成拒绝而复用；将来扩充证明可以让某个声明不再产生该 code，但不能把 code 改解释。新错误不得包含行值、payload、凭据、callback repr、完整 DataFusion plan dump；用 alias/field 和 expected/actual 约束定位。

## Facts and required classifications

本表是最低实现域及 handoff oracle；“supported”仅针对表列确切形状和前提，不外推到所有 SQL。

| Case                                                                                | Required observed facts                                                                                                                                                                    | Requested comparison                                                                                                                                           |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C01 deterministic row-local projection/filter/cast/NULL，含 simple single-alias SQL | 两 mode supported；current_batch / append / per_row_final；SQL 不继承 temporal identity，filter/投影仅保留可证明的键。                                                                     | per_batch supported；对可分配子集，accepted_history final relation 按 append final 行归约也 supported。不可用“无 aggregate”代替 row-local 证明。               |
| C04 单个普通 SQL `SUM(x)` 或已证明 aggregate                                        | batch current_batch / complete_snapshot / per_batch；stream accepted_history / complete_snapshot / revisable_snapshot，compact 或 retained 不改变此事实。                                  | per_batch unsupported；完整 H 与 stream 末完整快照可 supported，但必须证明该具体查询及空历史归约。未证明的 HAVING/empty-output 路径 unknown。                  |
| C02 `SUM(x)` → 普通 `SUM(total)`                                                    | 两端各可准入；stream 入边 producer_update=complete_snapshot、consumer_interpretation=append。row-local 中间节点不能把快照误标为普通新增关系。                                              | 旧执行 `[1] / [2,3]` 末值7保持；final-relation / latest 请求在该边 unsupported + update_interpretation_mismatch。不能只取下游末快照就称6；本片不实现替换消费。 |
| C03 非 aggregate `DISTINCT`、普通 ORDER/LIMIT/OFFSET、一般 OVER                     | 只有复用实际调度/规划证明时才标 current_batch；单独 ORDER BY 记录键及 tie 限制，不创造全局历史。                                                                                           | 简单已证明 per-batch 形状可 supported；global DISTINCT/top-k 反例为 unsupported。复杂嵌套、CTE 或 SQL window 若有专门路径但无法静态证明则 unknown。            |
| Aggregate + ORDER/LIMIT/HAVING，ordered LAST_VALUE                                  | 先看原 stream aggregate 分类，再看完整 logical plan；可能是累计快照上的排序/截断，不一律 current_batch。                                                                                   | 只有整形状、NULL、ties、空输出和终态归约有证明才 supported，否则 sql_semantics_unproven；不能按关键词统一拒绝/批准。                                           |
| Event window / union / bounded join / ASOF / snapshot_sql                           | 保留已知 stream 事实；event window 为 event_window、闭合后 group_final；snapshot 为 latest_relation / complete_snapshot / revisable_snapshot；当前 batch adapter 缺失按 unsupported_mode。 | 本片不通过重写另一 SQL、切换 engine 或清空 stream 状态授予双端证明。snapshot 多 alias 是 latest-available，不承诺原子时间切面。                                |
| C13 SQL/字段/type 错误、volatile/clock、identity 缺失                               | 复用解析、schema、原 mode/ordering 约束，报具体输入和路径；没有分析数据行或启动 job。                                                                                                      | 严格编译仍受原编译器拒绝；运行时 overflow/预算/取消保留原执行边界，分析不谎称提前证明。                                                                        |

非 aggregate 的 per-batch 是 SQL 执行范围；update 必须按整图传播区分“本批新增关系”和“处理上游完整快照所得关系”。例如 aggregate→filter 可以保留 complete_snapshot 性质，但若当前 filter 空输出抑制事件，不能证明空替换。聚合器消费 snapshot、LIMIT 依赖分批等非分配组合需检查边，不能只给每节点贴局部支持标签后全图 AND。

局部 current_batch 与请求 accepted_history 不同，不应机械判为 mismatch：确定性 row-local 可分配，纯 ORDER BY 在 unordered 比较下也可能保持完整多重集；但 DISTINCT、LIMIT、OFFSET、OVER 对行选择/值的影响不能忽略。只有证明整图分配/归约成立才能接受，已知反例拒绝，尚无证明标 unknown。

初始可证明 allowlist 要小且明确：已验证的确定性 row-local 子集、简单普通聚合及显然不兼容的边。复杂 SQL 不为提高 supported 数量而猜测；已有 C01/C04 简单用例必须真正归类，不能全量填 unknown 作为交付。对 retained/compact 和 native SQL window，静态事实不承诺在第一批前选择动态物理策略，也不运行 probe 数据来取得证明。

## Revisions and compatibility

| Revision / identity                                                     | Rule                                                                                                                                                                                               |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `QueryAnalysis.schema_version = 1`                                      | 新独立线协议，不修改 capability schema 当前版本5。字段形状、enum 词义或 required 字段变动升 schema_version；首版宁可升版，不依赖读者忽略未知安全字段。未知 schema/enum 不得得到支持或准入。        |
| `QueryAnalysis.analysis_revision = 1`                                   | 静态证明规则集版本；支持域、组合传播、分类或影响 gate 的修复均单调递增。纯措辞、排版不增。同 schema 下 status 可因新证据变化，消费者不能以旧 supported 跨 revision 作凭据。                        |
| Existing `AnalysisResult.capability_revision` + `capability_session_id` | 保持运行时注册 snapshot 的现有身份；不是 schema version，也不因一次 analyze 自增。新 query 字段不冒充 capability 注册。Rust 绑定到调用传入的不可变 UDF/provider 事实，不能虚构 Python session ID。 |
| Program fingerprint / native fingerprint / mode / cache / checkpoint    | 新 facts/request/revisions 不写入 Program canonical bytes、project-v3、原 native fingerprint 或 checkpoint state。旧 SQL layout 3/4/5、state_version=2 及原恢复身份保持。                          |

analysis_revision 的改变只失效诊断/证明缓存；原执行准备缓存、mode key 和已有 capability 注册失效规则不变。若缓存检查结果，其 key 必含完整声明/schema、注册 scope、schema_version、analysis_revision、请求四字段和影响规则的 compile settings。不得仅以 fingerprint 或成功编译记录作为证据。

严格 gate 每次位于原编译缓存读取之前，在同一 registration lock/snapshot 下完成并编译；命中旧普通 compile 缓存也不能跳过。可以复用不变 snapshot 下的有效诊断缓存。请求不修改算法或 graph，因此不为它改变旧执行 fingerprint；未来真正引入 replacement 语义的图必须在 DAL-513/514 按原计划增加其语义/恢复身份，不能借本片的“诊断元数据不入身份”规避。

## Integration into the existing lifecycle

1. 捕获声明、精确 schema、现有不可变 capability/UDF snapshot 和 options；只用解析/逻辑规划，复用 ValidatedQuery、schema inference 和原 compiler 校验。允许准备只读 plan 元数据，不初始化运行时状态、执行 SQL、读数据行、open/poll Source、创建 Source bindings、启动 job 或调用外部 provider 工厂求证语义。分析的这两项计数必须为0：读取行数、启动 job 数。
2. 分别形成 batch/stream 准入事实；不因某 mode 不支持就调用该 mode 真正编译以试错或放弃所有事实。解析/schema 已知失败保留 diagnostics；未能 lower 的 symbolic 子图保留逻辑路径并 unknown，禁止消失。Python 的既有 declaration-time engine 拒绝保持；可用 `analyze(mode=...)` 观察的反向模式仍只是诊断。
3. 请求检查按事实及消费边组成比较义务；失败报告在编译/执行入口转成上述 CompileError/QueryPortability；成功后再走原 compile 流程。Rust checked 方法在消费 builder 前借用分析，随后调用原 compile；不增加第二套执行内核。
4. Python `compile_*` 在 preparation cache 命中前 gate。`collect/collect_async` 的严格分支在 `compute._prepare_collect` 的 `_collect_batches` 前 gate，避免先遍历/包装 Arrow 行；正常分支保持原顺序。`execute` 同规则透传。
5. `Program.stream(..., portability=...)` 把请求保存于不可变 stream request；`StreamResults._start` 的编译阶段（`stream.py:249` 起）完成 gate，早于 `_bindings`、临时 state root、Runner 和 `start_async`。调用者可先传 source 对象引用，但不得为分析调用其 iterator/open/poll。低层使用先 checked compile 再绑定 Source/Sink。
6. 最终 plan 仍由原 batch execute / stream runner 管理。stream 同 job 跨批保留状态，batch 同 plan 不同 execute 无跨次计算历史；不加 reset，不增加 batch 中途 checkpoint 续算，不放宽 mode、delivery、预算或恢复校验。

## Error cases

| Input / request                          | Required error example                                                                                                                                                                                         |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 旧快照追加边要求关系解释                 | `outputs.total.sql.tables.input: update_interpretation_mismatch: final_relation requires latest complete relations at this input; producer emits complete_snapshot and consumer appends`；status=unsupported。 |
| 单批 DISTINCT 要求 accepted_history 终态 | `outputs.unique.stream.history: history_scope_mismatch: requested accepted_history; this DISTINCT reads current_batch`；status=unsupported。                                                                   |
| 复杂 SQL 缺少证明                        | `outputs.result.stream: sql_semantics_unproven: no rule proves final_relation for this SQL shape; diagnostic location is the SQL node`；status=unknown。                                                       |
| Rust stream-only 节点要求 batch          | `graph.nodes["join"].batch.support: unsupported_mode: stream_join has no batch lifecycle adapter`；status=unsupported。                                                                                        |
| 请求非法组合                             | `options.portability.cutoff: invalid_literal: per_batch comparison requires current_batch history and batch_boundary cutoff`。                                                                                 |
| SQL 后接 temporal 算子但缺 identity      | 沿既有 operand path + `ordering_required`，指出缺少 event_time / sequence 证明；不因 ORDER BY 存在而接受。                                                                                                     |

同一查询可以同时有 mode 和历史失败；完整报告保留所有可确定事实，不以一个通用 compile message 覆盖它们。运行时失败仍走现有 Execution/Checkpoint/Cancelled 分类；analysis 未运行的约束不能被报告为已验证通过。

## Typical example

以下为实现后的 Python 用法（本轮只做设计，未执行）；同一声明可在两个独立 Program wrapper 使用，engine 仍沿旧合同。例子只比较确定性 row-local，无跨批快照或隐式输入推断：

```python
import pyarrow as pa
import calc_flow as cf

events = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
positive = events.filter(events["x"] > cf.lit(0)).select("x")
program = cf.Program("positive", engine="sql", outputs={"result": positive})
request = cf.PortabilityRequest(
    comparison="per_batch",
    history="current_batch",
    cutoff="batch_boundary",
)
analysis = program.analyze(options=cf.AnalysisOptions(portability=request))
assert analysis.query is not None
assert analysis.query.portability == "supported"
# Rechecks under the current snapshot; the analysis object is not a permit.
plan = program.compile_batch(portability=request)
result = program.collect(
    {"events": pa.table({"x": [-1, 2, 3]})}, portability=request
)
assert result["result"].to_pydict() == {"x": [2, 3]}
```

Rust 对现有 `declaration() -> Result<PipelineBuilder>` 使用：

```rust
let request = PortabilityRequest::new(
    Comparison::PerBatch,
    HistoryScope::CurrentBatch,
    ComparisonCutoff::BatchBoundary,
)?;
let builder = declaration()?;
let report = builder.analyze(&udfs, &AnalysisOptions::default()).await?;
let plan = builder.compile_batch_checked(&udfs, &request).await?;
// Existing plan.execute(inputs, ExecutionOptions::default()).await? follows.
```

Rust snippet 是现有例程内的增量，不虚构已发布的 TableExpr/Program，也不要求移动 core 示例。发布时在既有相关例子补齐实际 imports 和正确输入/输出绑定；Python 严格 stream 示例须在 async with 启动失败时证明 Source 未被拉取。

## Rationale and rejected alternatives

保留 AnalysisResult 五个旧字段并追加 query，比替换其 issues 或另建互不相关的 Program 分析接口兼容；独立 schema_version / analysis_revision 避免把 proof 规则、runtime registration 和 native state 混作一个 revision。显式请求是数据配置，可以透过 Rust/PyO3/JSON，不接受 callback 或运行对象。

拒绝以 `stateful`、operator 名称、关键词、成功编译或相同 fingerprint 证明等价；拒绝给未知默认 supported；拒绝用全图“可 stream”掩盖 snapshot→append 的边语义；拒绝修改旧 SQL defaults、把所有 LIMIT 判为 per-batch、在分析中执行样本数据。另加 mutable SQLContext、Rust 新前端或统一 run 返回 Any 均超出本片。

仅增加分析而要求用户自行在 execute 前处理所有错误，容易让缓存/便利 stream 绕过请求；因此 Program 的编译和执行共享同一 gate。checked 编译不接收旧报告，避免 scope/revision 失效后仍凭缓存报告执行。未知采用明确可见的保守拒绝，不将其伪装为永久能力缺失。

## Handoff, checks and open questions

本片公共选择已定，无需阻塞 implementer 的额外公共 API 选择。缺失 component-boundaries 文件及父附件未入库均已用实际边界/附件核对解决，不是当前阻塞。内部模块划分、私有 PyO3 seam 名称和首版额外 SQL proof allowlist 可由 implementer 根据本合同选择；额外形状不能证明则标 unknown，无需为此扩派。

待后续切片决定：DAL-513/514 的 replacement 消费/空更新/恢复身份，DAL-512 的 Rust typed 前端与 mode 迁移，S3 的完整历史 DISTINCT/top-k/OVER 支持域。一般 outer/semi/anti、snapshot_sql 有限适配、external 强制 batch 隔离与 batch 中途恢复保持原范围；本片不替这些选择背书。

实现验收需覆盖 C01–C04/C13 的报告和 strict 行为、读取行数/job 数为0、缓存命中仍 gate、未知和明确拒绝可区分、特殊名称/共享子图 path 稳定；用独立小表保留旧7、stream1/6、batch同 plan1/5、per-batch DISTINCT/top-k/OVER 及原 mode 拒绝。复用既有效证据，产品检查由 implementer/reviewer 按实际改动选择，本 API note 不要求新独立 tester。

本文自检范围：指定文件独占修改、Markdown 结构/表格/空白/末行、引用存在、Python 示例语法、签名参数顺序和状态/revision/gate 一致性。未运行产品测试、native 编译、性能或 CI；不声称运行行为已验证。交 orchestrator 推进 implementer → reviewer；DAL-511 继续 in_progress。
