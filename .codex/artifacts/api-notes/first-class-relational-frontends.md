# First-class Relational Frontends — API Note

状态：DAL-512 / S0b 的局部公共接口冻结，2026-11-11；供 implementer 采用。**下文新增签名是本片实现合同，不是基线已有能力，也不是运行验证结果。**

精确基线：`775ea9738265597de492b848f8e0763962c7041b`；分支 `feature/dal-512-relational-frontends`，开始时工作树干净。前序 PR #588 的交接已确认合并；本轮直接核对该 SHA 的源码，不将历史测试转记成本 SHA 的 GREEN。

## Audiences and scope

- Rust 用户仅通过公开 `calc-flow` 构造 typed DataFrame / SQL relation、取 schema 和诊断、命名输出、双生命周期执行并导出项目；不依赖 Python、仓内 example helper、手写 Project JSON、ports 或 node IDs。
- Python 用户继续从 `calc_flow` 根包使用 `TableExpr` / `ColumnExpr` / `sql` / `Program`，获得明确的 mode 兼容入口、schema 方法、类型注解及已有 async / owned stream 行为。

DataFrame / SQL 是声明类别，batch / stream 是独立生命周期。S0b 的共同成功域为确定性 row-local 列运算、投影、过滤、单 alias SELECT/CTE 及其 mixed DAG；不新增 DataFrame aggregate、union、window、join、ASOF 或完整历史 DISTINCT/top-k/OVER。既有更宽 SQL 仍按既有准入和诊断使用，不被本片删减或自动标成可移植。

依据为父 DAL-498 的 API note 附件 `01a1bc01-9dbe-7c02-ac79-02093c8a178e`、计划 v2 `01a1bc0b-3060-7c9d-ad57-b23a24daca58` 和 main 复审 `01a1c33e-a567-774a-b132-9d1f20b09d2b`；本轮均通过 CLI 下载并读取相关章节。沿用已合并的 [query-history-portability API note](query-history-portability.md) 与 [分析证据](../analysis/query-history-portability.md)，历史附件的候选由本文仅在 S0b 范围内具体化。

具体仓库合同是 `AGENTS.md`、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md` 的 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint 词汇。实际组件依赖为 Rust core → PyO3 → Python adapters；本仓库没有 dal-cpp、dal-public、Excel binding 或 component-boundaries 文件，不据其虚构设计约束。

## Surface today and evidence

以下是本 SHA 已有能力；后文标为新增的名称不能写进当前能力清单。

| 位置                                                                                                 | 已有合同                                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/src/lib.rs`                                                                      | 导出 Batch、PipelineBuilder、两种 plan、StreamingRunner、SqlOperator，以及 DAL-511 的 QueryAnalysis / AnalysisOptions / PortabilityRequest 等；尚无公开 Rust TableExpr / ColumnExpr / Program。 |
| `pipeline/analysis.rs`、`query_analysis.rs`                                                         | builder 的 async analyze / checked compile、版本化三态报告、严格请求及 CalcFlowError::QueryPortability 已实现。                                                                                 |
| `pipeline/mod.rs`、`pipeline/compile.rs:427`                                                        | 低层外部 binding 名按端口唯一性使用裸 port 或 node.port；RunResult 保留 outputs、timings、metrics、metadata、context。                                                                              |
| `pipeline/batch.rs:383`、`continuous.rs:953`                                                        | BatchExecutionPlan.execute async；StreamingRunner 消费 stream plan，start 返回拥有生命周期的 StreamingJob。                                                                                |
| `python/calc_flow/symbolic/expr.py`、`symbolic/sql.py`                                              | 根表、列、lit、select / filter / with_columns、sql 返回同一 TableExpr 声明；table.sql 固定 alias 为 input。                                                                                    |
| `symbolic/program.py:384`                                                                          | engine 是必填 keyword-only，值 sql / streaming；Program 不可变，with_input / output 返回副本，analyze / explain 支持诊断 mode override。没有构造 mode、mode 属性或 for_mode。                             |
| `symbolic/program.py:470`、`:652`、`compute.py:397`                                                  | execute 按 engine 返回 dict[str, pa.Table] 或 StreamResults[StreamOutput]；collect_async 是同步捕获后返回 Awaitable 的普通 def。                                                              |
| `stream.py:206`、`_stream_inputs.py`                                                                | StreamResults 只进入一次，async with / async for，bounded queue；启动、取消和关闭等待原生 owner 清理，table form 与 Program form 分别产出 pa.Table / StreamOutput。                                       |
| `crates/calc-flow-python/src/config.rs:756`、`python/calc_flow/_native.pyi:333`                     | 已有私有 data-only 分析与 SQL schema-only planning seam；不是公开 Rust typed frontend。                                                                                                   |
| `examples/32_sql_expression_pipeline.py`、`crates/calc-flow/examples/32_sql_expression_pipeline.rs` | Python 已有 typed mixed pipeline；Rust 同主题例仍使用 support helper、SQL/公式字符串及手工连边，不能替代外部 typed consumer。                                                                             |
| `python/tests/test_program_engine.py`、`test_symbolic_compile_cache.py`                             | 既有断言覆盖 immutable engine、原拒绝/返回值、声明 fingerprint 与 mode cache / stream owner 隔离。只读这些断言，不称本轮已运行。                                                                                |

表中无完整前缀的 Rust 路径在 `crates/calc-flow/src/`，Python `symbolic/` 等路径在 `python/calc_flow/`。

## Proposed surface: modules and types

以下为**新增合同**。Rust 公共模块固定为 `calc_flow::dataframe`、`calc_flow::sql`、`calc_flow::program`；内部 IR、lowering 和 schema planner 不成为公共 ABI。

```rust
// lib.rs: new public modules/re-exports; definitions are specified below.
pub mod dataframe;
pub mod sql;
pub mod program;
pub use datafusion::arrow;
pub use dataframe::{ColumnExpr, InputOptions, LiteralValue, NamedExpr, TableExpr, lit, table_input};
pub use program::{CompileMode, CompileOptions, Program};
// sql is a module: use calc_flow::sql::sql; do not add a second root sql function.
// Existing QueryAnalysis / AnalysisOptions / PortabilityRequest and plan exports stay.
```

Arrow re-export 只提供与本 crate 依赖匹配的 Arrow 数据类型和 Batch 构造材料，使外部 consumer 不必通过 DataFusion 的私有路径找 SchemaRef / RecordBatch。不公开 DataFusion Expr / LogicalPlan，不新增 Arrow 版本；不承诺跨 Arrow major 的类型兼容。

TableExpr / ColumnExpr / NamedExpr / Program 是 `Clone + Debug + Send + Sync`、私有字段的不可变声明句柄。clone 只共享不可变声明/schema，不持有 rows、注册表活引用、plan、job、source/sink 或 checkpoint。声明不接受 callable、Python source、import path 或 executable object；项目导出才是持久化合同，不新增 serde 导入 Rust 句柄的入口。

### Rust DataFrame declarations

```rust
#[derive(Clone, Debug, Default)]
pub struct InputOptions {
    pub entity_by: Vec<String>,
    pub event_time: Option<String>,
    pub sequence_by: Vec<String>,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum LiteralValue {
    Null,
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    Utf8(String),
}
// From<bool>, From<i64>, From<u64>, From<f64>, From<String>, From<&str> for LiteralValue.

pub fn table_input(
    name: &str,
    schema: arrow::datatypes::SchemaRef,
    options: InputOptions,
) -> calc_flow::Result<TableExpr>;
pub fn lit(value: impl Into<LiteralValue>) -> calc_flow::Result<ColumnExpr>;

impl TableExpr {
    pub fn column(&self, name: &str) -> calc_flow::Result<ColumnExpr>;
    pub fn select(&self, columns: Vec<NamedExpr>) -> calc_flow::Result<Self>;
    pub fn with_columns(&self, columns: Vec<NamedExpr>) -> calc_flow::Result<Self>;
    pub fn filter(&self, predicate: ColumnExpr) -> calc_flow::Result<Self>;
    pub fn sql(&self, query: &str) -> calc_flow::Result<Self>;
    pub async fn collect_schema(
        &self, udfs: &UdfRegistrySnapshot, mode: CompileMode,
    ) -> calc_flow::Result<arrow::datatypes::SchemaRef>;
}
impl ColumnExpr {
    pub fn alias(&self, name: &str) -> calc_flow::Result<NamedExpr>;
    pub fn cast(&self, data_type: arrow::datatypes::DataType) -> calc_flow::Result<Self>;
    pub fn eq(&self, rhs: Self) -> Self;
    pub fn ne(&self, rhs: Self) -> Self;
    pub fn lt(&self, rhs: Self) -> Self;
    pub fn le(&self, rhs: Self) -> Self;
    pub fn gt(&self, rhs: Self) -> Self;
    pub fn ge(&self, rhs: Self) -> Self;
    pub fn is_null(&self) -> Self;
    pub fn is_not_null(&self) -> Self;
}
// Implement std::ops::{Add, Sub, Mul, Div, BitAnd, BitOr} for owned ColumnExpr,
// each taking ColumnExpr and returning ColumnExpr; Neg and Not are unary.
// No PartialEq/PartialOrd returning host bool for expression comparisons.
```

这些是签名清单，非可直接编译的 impl 定义；作用域中的 UdfRegistrySnapshot / CompileMode 使用上列 crate exports。少量纯操作直接构造 typed IR；名称/字面量验证失败用 Result，schema-dependent 类型检查留到 planning，不在 `+` 中 panic 或用 Result 实现算术 trait。`.eq()` 等产生声明而非比较句柄；调用者显式 `lit(1_i64)?`，首版不增加隐式所有 scalar / column 组合重载。

确定的细则：

- input name、field reference、output alias 非空；input schema 字段唯一，names / 顺序 / dtype / nullable 精确，options 按现有 entity / event_time / sequence 规则验证。没有 options 时传 Default，不以零值伪造 temporal identity。
- column(name) 绑定当前 relation 身份，立即检查名称；对 SQL 输出不触发阻塞 planning，未知列在 analyze / collect_schema / compile 定位。来自无关 row lineage 的列用于 select/filter 报 schema_mismatch，不按相同长度或字段名偷偷对齐。
- select 必须至少一列，按 Vec 顺序输出，重复结果名拒绝。直接引用本表 x 并 alias 为 x 对应 Python select("x")；派生表达式不得借 select 覆盖同名现有列，沿用当前 Python 约束，需使用新名称。with_columns 保留原列序并追加新列，重复/替换现有名拒绝；同一调用中的新增列不是其他同批新增表达式的隐式输入。
- literal 类型明确为 bool / int64 / uint64 / float64 / string / null；f64 非有限值在构造时拒绝，不把 NaN 当 NULL。其他宽度/日期字面量用显式 cast；Null 用显式 cast 确定需要的输出类型，不推测 decimal 或 extension 类型。Python lit 的既有签名/推导不变。
- 算术/比较/布尔使用当前 Python row-local 规则与 native ExpressionOperator 的同一类型、nullable 和数值合同；保留必要的显式 cast，不因 DataFusion 能更宽 coercion 就给 Rust 另一个隐式规则。Rust `/` 对应 Python true division，只对 floating operands 成立，整数 operands 与当前 Python 一样以 unsupported_type 拒绝并要求显式 cast，不能自动转换或照搬 SQL 整数除法。BitAnd / BitOr / Not 对应三值布尔；没有新增 Polars 缺失值策略。

### Rust SQL relations

```rust
// calc_flow::sql module
pub fn sql(
    query: &str,
    tables: std::collections::BTreeMap<String, TableExpr>,
) -> calc_flow::Result<TableExpr>;
```

非空 query，至少一个显式 alias；alias 使用现有 portable SQL identifier 范围 `[A-Za-z_][A-Za-z0-9_]*`，按确定顺序复制。别名归一化后的冲突必须准确拒绝，不合并输入。Rust 可在构造时识别该冲突；Python 普通 SQL 继续保留现有构造/规划验证边界，不能把新提早拒绝描述为旧行为。TableExpr.sql(query) 精确等价于 `sql(query, {"input": self.clone()})`。

只读单条 SELECT/CTE，沿用 native parser 的 DDL/DML/utility、多语句、recursive CTE、table-generator 和 backend-selector 限制。普通 append stream SQL 只接受一个 alias；batch 多 alias 仍可用，stream wrapper 结构预检拒绝已知多 alias，不在开始后才打开 Source 试错。SQL 输出建立新 row lineage，清除 temporal identity；ORDER BY 不能恢复 event/entity/sequence 证明。

### Rust Program, schema, analysis and compilation

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompileMode { Batch, Stream }

#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    pub portability: Option<PortabilityRequest>,
}

impl Program {
    pub fn new(
        name: &str,
        mode: CompileMode,
        outputs: std::collections::BTreeMap<String, TableExpr>,
    ) -> calc_flow::Result<Self>;
    pub fn name(&self) -> &str;
    pub fn mode(&self) -> CompileMode;
    pub fn fingerprint(&self) -> &str;
    pub fn inputs(&self) -> &[TableExpr];
    pub fn outputs(&self) -> &std::collections::BTreeMap<String, TableExpr>;
    pub fn with_input(&self, input: TableExpr) -> calc_flow::Result<Self>;
    pub fn output(&self, name: &str, value: TableExpr) -> calc_flow::Result<Self>;
    pub fn for_mode(&self, mode: CompileMode) -> calc_flow::Result<Self>;
    pub async fn collect_schema(
        &self, udfs: &UdfRegistrySnapshot,
    ) -> calc_flow::Result<std::collections::BTreeMap<String, arrow::datatypes::SchemaRef>>;
    pub async fn analyze(
        &self, udfs: &UdfRegistrySnapshot, options: &AnalysisOptions,
    ) -> calc_flow::Result<QueryAnalysis>;
    pub async fn explain(
        &self, udfs: &UdfRegistrySnapshot, options: &AnalysisOptions,
    ) -> calc_flow::Result<String>;
    pub async fn compile_batch(
        &self, udfs: &UdfRegistrySnapshot, options: &CompileOptions,
    ) -> calc_flow::Result<BatchExecutionPlan>;
    pub async fn compile_stream(
        &self, udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements, options: &CompileOptions,
    ) -> calc_flow::Result<StreamExecutionPlan>;
    pub async fn to_project(
        &self, udfs: &UdfRegistrySnapshot,
    ) -> calc_flow::Result<ProjectSpec>;
}
```

选择 CompileMode 与现有 Python / capability 词汇一致，不另建同义 ExecutionMode。Rust 没有 legacy engine，不新增 engine 字段。所有配置 required 在前；defaults 通过 Default 类型，最大方法也只有四个 self 外参数。

new 推导所有输出可达的命名输入，按名字确定排序；空 outputs 允许逐步构造，但取 schema / 编译 / 导出要求至少一个输出。with_input 返回新 wrapper，保留已有输入并明确输入列表；output 对自动推导输入的 wrapper 继续推导，对显式列表则不偷偷补输入；缺失/多余/冲突声明按现有 Program 规则拒绝。自动发现时同一声明去重为一个输入；显式输入列表（含 with_input）重复名字即 duplicate_name，即使声明相同也不静默去重。同名不同 schema/options/声明不成为两个 Source。重复 output 拒绝，不覆盖。

Program.analyze **直接复用 QueryAnalysis**，其中有两种 mode facts、schemas、edges、diagnostics；不再造 Rust AnalysisResult。Python 继续用已有 AnalysisResult.query 投影这份核心报告。explain 为确定性文本，说明绑定 mode、声明 fingerprint、请求前提和事实，不作为机器解析格式。schema/analysis 读取行数、Source/Sink open 数和 job 数均为 0；已知空 schema 与未知 schema 分开，collect_schema 无法确定时报告 unresolved_type，不能以空 schema 假成功。

compile 只接声明、不可变 UDF snapshot、配置/requirements；不接数据、Source/Sink/Runner。schema planning / export / compile 均 async Result，不在 Rust 偷用 block_on。future 调用期间借用 self/snapshot，完成后的报告/schema/project/plan 拥有全部必要数据，不借用 Program 或注册表；独立 plan 可在 wrapper 丢弃后使用。编译不得修改声明或把 plan 缓存在 wrapper 内。

CompileOptions.portability=None 精确保持默认准入。Some(request) 重用已合并规则，在同一注册 snapshot 下重新分析并 check，再调用原 compiler；unknown / unsupported 均拒绝，schema/类型/生命周期及原 compiler 的校验仍全部生效。analyze/explain 带请求返回报告，不等于执行许可。报告不作为 compile 参数。

**命名绑定合同：** 经新 Rust Program 编译产生的现有 plan 类型，以 inputs 的逻辑名接受 Batch / SourceBinding，以 outputs 的逻辑名返回 RunResult.outputs / 接受 sinks / StreamRequirements.delivery。用户只写 events/result，不查 node IDs。lowerer 在构造该 plan 时维护逻辑名到真实 PortEndpoint 的映射，并一致用于输入验证、runtime source/sink routing、delivery proof 与输出收集；不得仅在已构建 plan 上改一张 public map。一个输入、一个共享 relation 的 fan-out 只拥有一个原生 owner，多个命名输出可指向同一上游。

这是新 Rust Program 编译入口的命名便利；**原 PipelineBuilder / project compiler 的 binding 命名算法完全保留**。编译结果仍为原 plan，不引入另一套 BatchProgram / StreamProgram / runner。物理节点/端口由 lowerer 生成，低层 node timings / metrics 的原含义不变。

## Proposed surface: Python additions and compatibility

Python 不新增同义 DataFrame 容器、NamedExpr 容器或 SQLContext；DataFrame 类别继续由 lazy TableExpr 表示。table_input、lit、select / filter / with_columns、sql、TableExpr.sql 与 compute 的既有签名、返回和语义保持。新增 CompileMode 根包和 symbolic re-export，实际类型仍复用 `symbolic.types.CompileMode`。

Program 构造公开类型通过两个 overload 表达互斥必填项；下面是**新增接口签名清单**，其余既有参数照旧，特别不删除 ArrayExpr / Parameter 输入输出能力：

```python
from typing import Literal, overload

class Program:
    @overload
    def __init__(self, name: str, /, *, mode: CompileMode,
                 inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]] | None = None,
                 outputs: Mapping[str, TableExpr | ArrayExpr]
                 | Sequence[tuple[str, TableExpr | ArrayExpr]] = ()) -> None: ...
    @overload
    def __init__(self, name: str, /, *, engine: Literal["sql", "streaming"],
                 inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]] | None = None,
                 outputs: Mapping[str, TableExpr | ArrayExpr]
                 | Sequence[tuple[str, TableExpr | ArrayExpr]] = ()) -> None: ...
    @property
    def mode(self) -> CompileMode: ...
    @property
    def engine(self) -> Literal["sql", "streaming"]: ...
    def for_mode(self, mode: CompileMode, /) -> Program: ...
    def collect_schema(self, runtime: Runtime | None = None, /, *,
                       mode: CompileMode | None = None) -> dict[str, pa.Schema]: ...

class TableExpr:
    def collect_schema(self, *, runtime: Runtime | None = None,
                       mode: CompileMode = "batch") -> pa.Schema: ...
```

运行时用一个私有 omitted sentinel 区分“没有传参数”和“显式传 None”；sentinel 不是公共类型。inspect.signature 需要显示两项均可省略以支持 overload，但两项不能同时省略。`engine.default is inspect.Parameter.empty` 的旧反射断言因新增别名必然改变，更新为互斥/缺失行为测试；这是一项明确的**签名反射兼容变化**，不是旧调用行为破坏。保留 name positional-only、选择项 keyword-only。若生成 stubs，只写上述 overload，不给“同时接受两项”的合法 overload。

| 构造选择                              | 绑定 mode / engine 或错误                                                                               |
|-----------------------------------|----------------------------------------------------------------------------------------------------|
| 只给 engine="sql"                   | batch / sql；纯 DataFrame 与 SQL 图都按 batch 运行。                                                        |
| 只给 engine="streaming"             | stream / streaming；原输出事件和准入不变。                                                                     |
| 只给 mode="batch"                   | batch / sql；与 legacy sql 的输出、属性、声明和 native 身份一致。                                                   |
| 只给 mode="stream"                  | stream / streaming；与 legacy streaming 一致。                                                          |
| engine、mode 都省略                   | TypeError，`Program.mode: invalid_literal: provide exactly one of mode or engine`；没有默认 batch 或自动选择。 |
| 两项同时给出，即便映射一致                     | ValueError，同路径/code，`mode and legacy engine cannot both be supplied`；不设优先级。                        |
| 只给 engine=None / mode=None，或非 str | TypeError，分别指出 Program.engine / Program.mode 与允许的字符串。None 不是缺省生命周期。                                |
| 字符串不在对应枚举内                        | ValueError，指出该字段、invalid_literal 和允许值；不接受 mode="streaming" / engine="batch"。                       |

同时传两项时先报告互斥冲突，再检查单值类型；只有一项时按该项检查。构造成功后的 .mode 和 .engine 都只读，with_input / output 保留选择及输入推导状态，不新增 set_mode / set_engine。本片只文档标示 legacy engine，**不发弃用 warning，也不指定移除版本**。

for_mode 总是返回新 wrapper，包括目标与当前 mode 相同的情况。保留 name、输入推导标志、精确显式 inputs、output 名/顺序、immutable DAG 和声明 fingerprint；执行与 output() 相同的目标结构准入预检，故 SQL 多 alias → stream、window 等 stream-only → batch 立即拒绝。更深的 schema / registration / portability 仍在相应分析或编译阶段检查。转换失败不改变原 wrapper，不启动 job；它不承诺所有图都可跨模式使用。

Program.collect_schema 的 mode=None 使用绑定 mode；显式 mode 与 analyze/explain 一样仅选择诊断视角，不改变 wrapper。TableExpr 的 schema 方法默认 batch，stream-only 图可显式 mode="stream"。结果是全体**表**输出的 logical-name → pa.Schema；ArrayExpr 输出明确以现有 unsupported_type 约束拒绝，不凭空转换数组 schema。schema 方法调用原 schema-only planner，无数据执行。新增 collect_schema 使用现有 blocking convenience 的 event-loop 检查；既有 analyze/explain 等方法的同步合同保持原样。在运行中的 event loop 需要 schema/planning 时由调用方放到 worker，不能在方法内部嵌套运行 loop。本片不新增 collect_schema_async。

### Python existing execution signatures retained

以下签名/返回在本基线已经存在，增加 mode 别名后继续成立；不是新增 execute_async 或统一 Any：

```python
# Existing methods; shown as a signature inventory.
def collect(self, inputs, /, *, runtime=None, options=None,
            portability: PortabilityRequest | None = None) -> dict[str, pa.Table]: ...
def collect_async(self, inputs, /, *, runtime=None, options=None,
                  portability: PortabilityRequest | None = None
                  ) -> Awaitable[dict[str, pa.Table]]: ...
def stream(self, inputs, /, *, runtime=None, config=None, watermarks=None,
           portability: PortabilityRequest | None = None
           ) -> StreamResults[StreamOutput]: ...
def execute(self, inputs, /, *, runtime=None, options=None,
            portability: PortabilityRequest | None = None,
            **streaming: Unpack[_StreamingExecuteOptions]
            ) -> dict[str, pa.Table] | StreamResults[StreamOutput]: ...
```

输入类型继续分别为已有 TableData / NativeArray 的 mapping、或 StreamInput 等 mapping；这里省略的入参注解不是改成无类型。TableExpr.collect/collect_async 返回单表，TableExpr.stream 返回 StreamResults[pa.Table]；Program 返回命名结果。execute 为 batch 返回立即收集的 dict，为 stream 返回待进入的 owning context，不能把 stream context 自动 await 成 dict。batch 的 config/watermarks 与 stream 的 options 仍按原规则拒绝；collect/collect_async/compile_batch 对 stream wrapper、stream/compile_stream 对 batch wrapper 仍保留原异常类别与 legacy engine 提示。to_project(mode=...) 仍禁止与绑定 mode 不同；跨端导出先 for_mode，再 to_project。

### Async capture, cancellation and ownership

- collect_async 保留普通 def → Awaitable 的调用时捕获行为，不能改写成 async def 把捕获推迟到 await。立刻复制 inputs mapping、固定已有 Batch/Arrow 引用；Arrow buffers 只共享只读，不 deep-copy 整表。strict 路径在 worker 先 gate，后包装/访问 sequence 或 native 数据，沿用 DAL-511 已实现的延迟验证边界；非 strict 仍保留原 entry capture/验证。
- 每次便利 collect/collect_async 创建新 batch plan；此外必须保留对**同一已编译 plan**暖缓存多次 execute 的内置状态隔离，不用便利方法的新 plan 掩盖泄漏。一次取消/错误不把输入、累加器或未提交候选留给下一次 execute。
- stream(inputs) 在调用时复制 mapping、水位配置及现有可复制输入容器，保留 iterator/SourceBinding 的所有权引用；不提前迭代、open 或启动 job。strict 路径不能为了捕获而读取用户 Sequence，启动阶段先 gate，再创建 source bindings。Arrow buffers 和 caller 提供的外部源按现有只读/单 owner 规则使用。
- async with 成功进入后拥有一个 native job，async for 受 bounded backpressure 约束，只能在 context 内消费且不可并发 __anext__。正常 EOF、异常、取消、提前退出及进入失败都 await 原有 cleanup；close 幂等，context 不可再次进入。不要共享同一 SourceBinding/iterator 到两个活动 job 来声称 wrapper 状态独立。
- stream 同 job 的计算状态跨批保留；普通新 job 从初始状态开始。临时 checkpoints / 普通 iterable 不承诺 durable restart、replay 或 ExactlyOnce；持久恢复仍用显式 StreamingRunner 与合法 source/sink、managed state。Rust 的 compiled stream plan 被 runner 消费，job.wait/shutdown/cancel 为生命周期所有权入口，不新增第二套结果 iterator。

Python `calc_flow` 和 `calc_flow.symbolic` 同步导出 CompileMode 及既有关系/分析类型；高层 typed Python 声明不必重复注册为 PyO3 class。若为共享 schema/IR/diagnostics 增加私有 native seam，`_native.pyi` 必须同步 exact 参数/返回与异常投影；不能只返回未经封装的 JSON dict，也不能让 Rust typed API 反向调用 Python。保留 Python 3.9 和现有 _compat 写法。

## Schema, diagnostics and data-only export

成功域的 Arrow 字段类型沿现有 TABLE_FIELD_TYPES：bool、各 8/16/32/64 位有/无符号整数、float32/64、string/large_string、date32/64、time32[s]、time64[us]、timestamp[ms] / [us]，以及 timestamp[us, UTC]。Rust 映射到匹配 Arrow DataType；不默认增加 Decimal、nested、dictionary 或 extension 的 portable 保证。更宽 native SQL 的存在不等于 Python symbolic/typed 公共域已支持。

schema 在零行时也规划，返回字段名、列序、dtype、nullable；不看首行猜类型，不随数据改变。声明和输入 schema / Batch.metadata 不被修改，保留既有内部 schema normalization / metadata 行为，不新增“每个计算结果保留所有 Arrow metadata”的承诺。NULL predicate 只保留 true，NULL 与 NaN/Inf 区分；重复行保留，未指定排序按多重集验证。NULL-only、零行、有重复行与 numeric coercion 都用小型独立预期，不用 Polars 结果覆盖 native 合同。

沿用 DAL-511 的 QueryAnalysis 字段、schema_version、analysis_revision、三态、code、稳定排序和严格 gate。新增 frontend 适配不能修改其 wire schema 或把 supported 准入等同于 portability。普通 schema/类型错误继续投影为原路径/code；没有请求时 portability=None。opaque SQL proof 不足仍 unknown，而非假装 unsupported 或 supported。

**地址扩展：** 新 Rust typed Program 使用与 Python 相同的逻辑根 `inputs.<name>` / `outputs.<name>` 和规则化 role 地址；SQL 入边为 `P.sql.tables.<alias>`，字段表达式用原 role 路径。共享节点选择按 UTF-8 输出名遍历的最小到达路径，已声明输入固定 inputs 根，每条实际消费边独立列出。特殊名称使用 JSON string 下标。通过 lowerer 映射核心事实到这些地址，不暴露 cf_sql_<digest> 或优化序号。既有 Rust PipelineBuilder 的 `graph.nodes["id"]` / 四元组 edge 地址完全保留，不重写旧报告。

同声明、注册 scope、请求和 analysis_revision 的事实/诊断确定排序；不承诺重命名、改图、不同语言构造或不同 lowerer 后路径/指纹相同。SQL 无可靠 span 时定位 SQL 节点/alias 并说明无法定位子表达式，不伪造行列号。错误不含数据行、payload、凭据、callback repr 或完整 plan dump。

| 违反输入/约束                                  | 冻结的定位、code 和错误内容                                                                                       |
|------------------------------------------|--------------------------------------------------------------------------------------------------------|
| schema 中 x 重复                            | inputs.events.schema 的 offending field/index；duplicate_name，指出 x 必须唯一。                                 |
| 未知列 missing                              | 所在表达式逻辑路径；schema_mismatch，`column 'missing' is not present in relation schema`，不得只报 generated node id。 |
| filter 是 int64                           | 所在 filter predicate role；unsupported_type，`filter predicate requires bool, found int64`。               |
| 跨无关 lineage 的列相加/投影                      | offending operand 的 lineage role；schema_mismatch，指出必须来自同一 row lineage。                                 |
| SQL 引用不存在的列/非法语句                         | SQL 节点/alias；schema_mismatch / invalid_literal，保留 native 原因；无 span 明示只能定位节点。                           |
| stream SQL 多 alias / stream-only → batch | 对应 output/SQL 节点；unsupported_mode，包含实际 mode、声明形状及单 alias / stream-only 约束；Source 拉取前拒绝。                |
| SQL → temporal 缺身份                       | temporal role；ordering_required，说明 SQL output 没有 temporal identity，ORDER BY 不足以证明。                     |
| 严格请求遇快照追加边                               | 原 edge path；update_interpretation_mismatch，说明 complete_snapshot 被 append 消费；原旧解释仍合法。                   |
| 严格请求缺 SQL/排序证明                           | sql_semantics_unproven / row_order_unproven，status=unknown，与明确反证区分。                                    |
| collect_schema 无法解析输出                    | 对应 output/schema；unresolved_type，不返回空 schema 冒充成功。                                                     |

构造 host 值：Rust CalcFlowError::InvalidArgument 保留 field / message(code + constraint)；Python TypeError / ValueError 沿原分层。planning/compile：Rust 使用既有 Compile 等 typed variant，在可定位错误的 message 中保留 `{path}: {code}: {message}`；analyze 的结构化定位仍在 QueryDiagnostic。strict 拒绝仍为 CalcFlowError::QueryPortability，Python CompileError.query_analysis 为 immutable QueryAnalysis；不新增另一个 diagnostic/error 容器。执行期数据错误、预算、取消、checkpoint 继续原异常/variant 与 source 因果链，不为了统一路径吞掉原错误。

Rust to_project 返回现有 ProjectSpec，Python to_project 返回现有 ProjectDocument；都为 strict project-v3，mode 对应 wrapper，包含声明与精确 schema 占位，**不携实际输入 rows、Python aliases/callables、计划缓存或运行 state**。逻辑名到物理端点的 adapter 不写入新 schema 字段；导出后重载按既有物理 binding 名和显式 runner 使用，不声称仍按逻辑别名接受输入。调用者从重载 plan 的公开 external_inputs/outputs 获得实际名，无需手写节点 ID；文档说明此边界。导出不是 checkpoint，也不支持 batch 查询中途续算。

## Identity and compatibility decisions

1. engine 拼写与 mode 拼写规范化后走**同一现有 Python lowering**；Program v1 声明 fingerprint 的序列化字节不改，name/input/output 身份及其原排序不改。for_mode 保留同声明 fingerprint。新 Rust frontend 给相同 Rust 声明稳定的声明身份，排除 mode；不要求与 Python 或等价 SQL/DF 构造 hash 相同。
2. 编译准备 cache key 继续包括 mode、schema、runtime registration scope / revision、UDF 等原身份；相同声明 hash 不是共享 plan 的许可。可以复用只读准备信息，不能缓存 rows、source bindings、state owner 或 stream plan。strict 即使命中 cache 也重新 gate。
3. native fingerprint、Project.runtime.mode、delivery/config/schema/operator/layout 身份和 source cut / epoch 校验全部保留。别名迁移不能重新编码既有 Python project/node IDs/SQL 文本；新 Rust lowering 也不能去掉 native mode 身份。
4. legacy SQL aggregate 默认累计快照与 downstream append 解释保留：SUM→SUM 的旧终值 7 合法。显式 latest 请求只在已证明域成功，否则按 DAL-511 边诊断拒绝，不在本片“自动修成 6”。stream 两批 SUM=1/6，batch 同 plan 独立 execute=1/5；显式完整历史才是6。
5. for_mode / Clone 不复制 plan、runtime registration、source/sink、checkpoint、query accumulator 或临时目录；目标模式重新编译/启动并独立拥有状态。普通新 stream job fresh，只有显式合法恢复续接同逻辑历史；batch 不承接 stream checkpoint。External 强制纯函数隔离仍不在本片承诺内。

## Minimal mixed-DAG examples

以下是**实现后的验收种子，当前基线不可运行**；不是本轮产品测试。保留现有 32 示例的语言、位置和“SQL 前后复用表达式”用途；implementer 用公开 Rust API 替换其 helper 构图，并在 Python 同主题示例补 mode/schema/双 wrapper 使用。不移动已有例子，不建立不存在的 dal-public facade 程序。

### Python: one declaration, named fan-out and owned events

```python
from __future__ import annotations
import asyncio
import pyarrow as pa
import calc_flow as cf

async def mixed() -> None:
    data = pa.table({"x": pa.array([1, None, 3], type=pa.int64())})
    source = cf.table_input("events", schema=data.schema)
    scaled = source.with_columns(y=source["x"] + cf.lit(1))
    first = cf.sql("SELECT x, y FROM s WHERE y > 1", s=scaled)
    selected = first.select("x", z=first["y"] * cf.lit(2))
    result = selected.sql("SELECT z FROM input")  # DF -> SQL -> DF -> SQL
    batch = cf.Program("mixed", mode="batch", outputs={"scaled": scaled, "result": result})
    stream = batch.for_mode("stream")
    assert batch.fingerprint == stream.fingerprint
    assert stream.engine == "streaming" and batch.engine == "sql"
    schemas = await asyncio.to_thread(batch.collect_schema)
    assert schemas["result"].names == ["z"]
    request = cf.PortabilityRequest(comparison="per_batch", history="current_batch",
                                    cutoff="batch_boundary")
    report = await asyncio.to_thread(
        batch.analyze, options=cf.AnalysisOptions(portability=request)
    )
    assert report.query is not None
    assert report.query.portability == "supported"
    tables = await batch.collect_async({"events": data}, portability=request)
    assert tables["result"].to_pydict() == {"z": [4, 8]}
    async with stream.stream({"events": [data]}, portability=request) as events:
        async for event in events:
            assert event.name in {"scaled", "result"}
    exported = await asyncio.to_thread(batch.to_project)
    assert exported is not None  # data-only project, not an execution snapshot
```

上例把同步 schema / analyze / export 放到 worker；collect_async 在调用时捕获输入，stream context 在进入时才启动。

独立 SQL→DF→SQL 方向使用同一 source，首步为 `first = cf.sql("SELECT x FROM s", s=source)`，接 `mid = first.select(z=first["x"] + cf.lit(1))`、`last = mid.sql("SELECT z FROM input")`；分别建立 batch / for_mode("stream")，预期 z 为 `[2, None, 4]`。纯 DataFrame 是 source.with_columns(...).filter(...).select(...)；纯 SQL 是 source.sql(...)。这些方向要实际验收，不能只在混合例中计数推断。

### Rust: same public types through existing plans

```rust
use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::sql::sql;
use calc_flow::{
    AnalysisOptions, Batch, CompileMode, CompileOptions, ExecutionOptions, InputOptions, Program,
    Result, SinkBinding, SourceBinding, StreamRequirements, StreamingRunner, UdfRegistry, lit,
    table_input,
};
use std::{collections::BTreeMap, sync::Arc};

async fn mixed(
    inputs: BTreeMap<String, Batch>,
    sources: BTreeMap<String, SourceBinding>,
    sinks: BTreeMap<String, Vec<SinkBinding>>,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let source = table_input("events", schema, InputOptions::default())?;
    let scaled = source.with_columns(vec![(source.column("x")? + lit(1_i64)?).alias("y")?])?;
    let first = sql(
        "SELECT x, y FROM s WHERE y > 1",
        BTreeMap::from([("s".into(), scaled.clone())]),
    )?;
    let selected = first.select(vec![
        first.column("x")?.alias("x")?,
        (first.column("y")? * lit(2_i64)?).alias("z")?,
    ])?;
    let result = selected.sql("SELECT z FROM input")?;
    let batch = Program::new(
        "mixed",
        CompileMode::Batch,
        BTreeMap::from([("scaled".into(), scaled), ("result".into(), result)]),
    )?;
    let stream = batch.for_mode(CompileMode::Stream)?;
    assert_eq!(batch.fingerprint(), stream.fingerprint());
    let udfs = UdfRegistry::new().snapshot();
    let schemas = batch.collect_schema(&udfs).await?;
    assert_eq!(schemas["result"].field(0).name(), "z");
    let _report = batch.analyze(&udfs, &AnalysisOptions::default()).await?;
    let options = CompileOptions::default();
    let finite = batch.compile_batch(&udfs, &options).await?;
    let _result = finite.execute(inputs, ExecutionOptions::default()).await?;
    let plan = stream
        .compile_stream(&udfs, &StreamRequirements::default(), &options)
        .await?;
    let job = StreamingRunner::without_checkpoints(plan, sources, sinks)?
        .start()
        .await?;
    let _outcome = job.wait().await;
    let _document = batch.to_project(&udfs).await?;
    Ok(())
}
```

调用者输入为同例 x=[1,NULL,3]，有限命名结果 result.z=[4,8]；stream 按名称聚合 per-row-final 输出后同多重集。不得只看 `_outcome` 存在就判 stream 成功，落地用现有 JobOutcome 成功断言并核对 sink 内容。调用者按逻辑名提供 events 输入/源与 scaled/result sinks；实际 L02 另在同一已编译 plan 上换输入与空输入、多次 execute、不 reset，并验证不残留旧结果。Rust 反向链是 `source.sql("SELECT x FROM input")?` → typed select(z) → table.sql，仍无需手写 DF 算术 SQL。

## Acceptance mapping and handoff

| 验收                        | 本片落实点 / implementer 需要给出的证据                                                                                                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A01 / A02                 | 外部 consumer 仅使用公开 crate API，分别构造纯 DF、纯 SQL；schema/analyze/explain、命名绑定、两生命周期执行。不能用仓内 support 或 SQL 实现 DF 算术。                                                |
| A03                       | 根包/symbolic exports、真实类型注解/必要 native stub、collect_async 的 entry capture、owned context 的取消/关闭与返回类型。                                                          |
| A04                       | 两语言各跑 DF→SQL→DF→SQL 和 SQL→DF→SQL，两个命名输出共享子图、单一原生 ingress/共享 owner，for_mode 声明身份相同、compiled mode 身份不同。                                                       |
| A05 row-local             | 零行、全 NULL、重复行、typed literal/cast、NULL predicate、schema/nullable、跨 lineage 及未知列的 path/code；SQL 后不继承 temporal identity。COUNT/SUM 新 DF 构造及其 widening 完整验收留 S1。 |
| A07                       | engine / mode 四种合法入口等价；缺失/同时/显式 None/非法值/反向方法拒绝；immutable copies、for_mode、签名反射变化及旧 Python fingerprint/project/checkpoint 无漂移。                               |
| A08                       | 两语言 data-only 项目往返；导出后物理 binding 边界清楚，既有 project schema 无新增字段。owning-language 文档与公开示例可独立理解。                                                                 |
| A09 / C01 / C13           | 未支持形状准确准入/诊断；strict supported-only、unknown 区分、Source 前 gate；同步/async 异常与取消仍归原层次。                                                                            |
| C02 / C04 / C14 / L01–L03 | 复用既有 SUM链7、stream1/6、同batch plan1/5、新job fresh 与 mode/恢复身份回归；只有触及相关路径才补必要定向验证，不能借本片宣告 S1/S2 已交付。                                                            |

实现按本仓库行为变更先有定向 RED，再做必要 core/绑定编译、Python exports/types、定向用例、既有 32 示例和文档检查；使用本 checkout 对应 native，不借其他 checkout 的已安装包冒充通过。全量/覆盖率与跨平台留 CI；无新增性能阶段、无默认独立 tester/doc-writer。由协调器继续 implementer → dal-reviewer 独立主审 → 有证据的修复 → 唯一 PR。

## Why this shape and rejected alternatives

保留既有 TableExpr 名称与 native plan/runner，新增 Rust typed relation 和命名 Program 足以覆盖用户实际调用链；另建 DataFrame 物化类、mutable/global SQLContext、第二套 typed runner 或统一 Any 返回值都增加歧义。typed 算术进入同一个 core 表达语义/权威内核；内部 renderer 可以生成 native ExpressionOperator 所需格式，但用户不能被要求编写 SQL/JSON 来完成 DataFrame 查询。

共同 IR 指共享声明语义、native schema/SQL planning、QueryAnalysis 规则及权威执行内核；允许保留已有 Python immutable Node 适配，不先重写全部 Python lowering 或 project 格式。不允许各语言各写 SQL 关键字分析器或计算实现。不同构造首批不必同 hash，**同声明**的 fan-out 和生命周期 wrapper 必须可复用且不共享运行 state。

拒绝同时 engine/mode 时“mode 优先”、自动默认 batch、自动换生命周期、for_mode 改原对象、删除 mode/cache/checkpoint 身份、普通 SQL 自动改 latest、为模拟一致而 batch 跨次累积或 stream 批间 reset。schema planning 不通过跑样本替代。Polars 仅沿父 note 的有出处接口参考与差异界限；本轮不重做版本调查或新增全面兼容承诺。

## Open questions and self-check boundary

**本片无等待用户或其他角色回答的接口阻塞；以上 S0b 选择已经定下，可直接实现。** 内部 IR 文件布局、private PyO3 seam 名称、schema planning 缓存细节由 implementer 选择，必须满足不可变/共享核心/既有身份合同；不需要因此再开 spec 或默认 critic。

真实后续决策：legacy engine 的 warning/移除版本尚未安排，本片明确不触发；DAL-513/514 的 replacement 消费、空更新和恢复身份，后续有限关系与有界全局算法的 API，仍按各自切片冻结。更广 Arrow/SQL 类型、一般 outer/semi/anti、snapshot_sql finite、external 强制隔离、batch 查询中途恢复不借本片扩大。

本轮只写本文。自检覆盖：引用/路径、现有与新增标识、互斥/default 表、参数顺序、逻辑/物理 binding、schema/诊断与 owned lifecycle 一致性、Markdown 结构/表格/空白、示例 Python 语法及 Rust 示例格式。未运行产品测试、native 编译、性能或 CI；不声称新签名可在基线调用。DAL-512 保持 in_progress，附 note 交协调器继续。
