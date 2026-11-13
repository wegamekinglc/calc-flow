# Bounded Global Ordering / Top-k — API Note

状态：DAL-520 / S3-2 / G3，2026-11-13，**局部修订 v2：SortWindow 聚合固定区间参数，可交 implementer**。本文 Rust/Python 的 `SortSpec.closed_window` 签名已替换为 v2，取代初版附件 `01a1d2ee-57dd-756f-94b8-2b3f9469d7f7` 的对应签名；新增 SortWindow，具体迁移见文末修订。本轮只交付设计，产品仍待应用此修订；其余冻结行为和支持域不变，不把文档自检当作产品通过证据。

本次局部核对基线为 PR #606 HEAD `ccda5564809b36e222f4b084ec39cdc20d1b3116`，仍复用原分支；初始工作树干净。触发线程 `01a1d35b-5671-7c90-9ab6-652487e368dd` 要求只解决已复现 PLR0913；下面原 G3 调研基线/源码证据保留历史含义，未重新审查全部实现或 CI。

核对基线：`b29b7a9a164a79c85cbf92f8bf1683a1704c0f3f`，分支 `agent/dal-orchestrator/c9c0751ee030`，HEAD 与本地 origin/main 一致，开始时工作树干净。交接及父讨论确认前序 PR #605 已合并；本轮未重新查询远端 CI。依据为 DAL-520 当前描述、触发线程 `01a1d2e0-0fff-7a98-892e-577022fe014d`、父 DAL-498 的四单元/生命周期讨论，以及经 Multica CLI 下载读取的开发计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`（S3、A05/A06/A08/A09、C03/C13、L01–L03）。

承接 [bounded DISTINCT（含迟到策略增补）](bounded-global-distinct.md)、[finite event window](finite-event-window.md)、[query history](query-history-portability.md) 与 [first-class frontends](first-class-relational-frontends.md)。历史 notes 的旧基线和候选签名不取代本 SHA 的事实。角色合同为 `.codex/agents/cf-api-designer.toml`；实际边界为 Rust core → PyO3 → Python adapters。本仓库没有 `dal-public`、C++/Excel binding、methodology 或 `.codex/references/component-boundaries.md`；按 `docs/introduction.md` 的词汇设计，不补造这些组件。新公开例子另放现有 owning-component 目录，保留旧例子位置及用途。

## Audiences and scope

Rust/Python 用户都能从公开 DataFrame 和 SQL 接口声明有限排序或 top-k，并复用同一关系图的 batch/stream 生命周期。四单元均经过 native schema/AST authority、一个共同关系节点和同一个排序/NULL/截取内核；Python 不实现排序、heap、tie comparator 或历史累积。

本片冻结两个历史域：一个 Bounded Source 的完整接受历史，以及一个明确的固定 UTC 半开区间 `[start,end)` 的接受历史。两者都只发一次 final；有限执行把本次完整输入作为 H。完整排序、offset 后的排序和 top-k 均在此域内，不把普通 SQL 改为全局排序。不加无界滚动 leaderboard、分组 top-k、WITH TIES、增量/retraction/replacement、多个窗口、spill、任意表达式排序或 batch 查询中途恢复。

第一版保留完整域内 H，包括未入选 top-k 的行和重复行。**k 只界定输出，不是 O(k) 状态或时间承诺。** 没有稳定 sequence 时，只保证按声明键有序；带 k/offset 时，相等键中哪些 payload 行被选中也未规定。

## Surface today: source evidence

下表为此 SHA 已有事实；行号仅定位所读代码，不声称运行过测试。

| Source                                                                                           | Existing fact / design implication                                                                                 |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `crates/calc-flow/src/dataframe.rs:35,81,354`、`sql.rs:73`、`program/lower.rs:449`                 | InputOptions 已有 temporal sequence；RelationKind 有 DISTINCT、无 bounded ordering；typed SQL 宏与 DF 可落同一 native operator。 |
| `crates/calc-flow/src/operator/sql/output_order.rs:166,213,268`                                  | 既有 native SQL 输出路径预留排序工作区，用 Arrow lexsort/take 与 skip/fetch；其局部顺序不证明全历史或稳定 ties，saturating arithmetic 也不是新边界规则。    |
| `crates/calc-flow/src/query_analysis/sql.rs:279`、`query_analysis.rs:233,527`                     | 普通 ORDER BY 可报告键，但 ties 未证明；当前报告 schema2/revision9，已有 ordered、total、unspecified 等词汇。                               |
| `crates/calc-flow/src/operator/distinct/spec.rs:18,107,131,153`                                  | 有 BoundedSource/ClosedWindow、显式 StateBudget、copy_visible_v2，以及持久 Error(envelope)/window Drop(metrics_version1)。    |
| `crates/calc-flow/src/operator/distinct/mod.rs:147,175,261,324`                                  | finite 一次处理完整 Batch；stream retained H 后闭合；归一化、final query、emit 后提交分开。可复用模式，不把 DISTINCT 输出当排序输入证明。                  |
| `crates/calc-flow/src/operator/distinct/owned.rs:1,114,137`、`operator/state_budget.rs:9,46`      | 已有节点私有 provenance、逐值 owned normalization 和逻辑预算；外部 backing 不可从可见 size 认证。StateBudget 本身允许 JSON-safe 正数。             |
| `crates/calc-flow/src/operator/distinct/late.rs:18,49,90`、`distinct/checkpoint.rs:27,72,107`     | late drop 预备后提交并持久 counters；checkpoint 有严格身份/control、计费与 cached capture。它们的 layout 是 DISTINCT 自己的命名空间。             |
| `crates/calc-flow/src/continuous.rs:286,360`、`runtime/streaming/job/distinct.rs:14,114`          | SourceBoundedness 已存在；绑定证明及原始时间坐标上的闭合路径在实际启动前检查。排序需扩接此义务，不能假定 DISTINCT 专用逻辑已自动覆盖新节点。                               |
| `crates/calc-flow/src/pipeline/distinct.rs:107`、`query_analysis/distinct.rs:10`                  | DISTINCT 的单源 row-local 路径准入是强制规则；新增排序须接通 native/project 路径的同类规则，不能只加 Python guard。                                 |
| `python/calc_flow/symbolic/expr.py:515`、`symbolic/distinct.py:17,135,206,260`                    | 已有 immutable 配置、声明、native AST/schema seam 与新路径 cache tag；排序需对应公开面，不能只向私有 binding 添加方法。                             |
| `crates/calc-flow-python/src/config.rs:944,962`、`python/calc_flow/_native.pyi:19`                | Rust 是显式方言和 schema authority；新 seam/stub 应投影同一描述，不复制 Python parser。                                                |
| `crates/calc-flow/src/pipeline/batch.rs:449,575`、`examples/bounded_global_distinct.py`、对应 Rust 例 | finite execute 已有运行上下文隔离；已有例子明确 DISTINCT order unspecified，新例子应独立演示排序，不改写旧例子的证明。                                   |

另读取公开 exports、两端 SQL adapter、`docs/introduction.md`、`docs/python-api.md` 的 temporal ordering 约定。没有对源码之外的版本/Polars 协议作新承诺。

## Frozen Rust and Python DataFrame surface

新增 crate-root 导出 `SortKey`、`SortSpec`、`SortHistory`、`SortTies`，v2 追加 `SortWindow`。前四者仍为 owned、不可变使用、严格 serde/JsonSchema 配置；SortWindow 是经过验证的不可变宿主值，构造后投影到原有 SortHistory::ClosedWindow，不增加 wire 包装或字段。私有字段只读，不含 rows、callbacks、Source 或 runtime。复用已有 `SortDirection` 与 `NullPlacement`，但排序只接纳 First/Last，拒绝 Exclude（排序不丢 NULL 行）。不改这些旧枚举的含义。

```rust
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SortHistory {
    BoundedSource,
    ClosedWindow {
        event_time_column: String,
        start_micros: i64,
        end_micros: i64,
    },
}
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SortTies {
    Unspecified,
    Sequence { columns: Vec<String> },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SortWindow { /* private: event_time_column, start, end */ }
impl SortWindow {
    pub fn new(event_time_column: &str, start: EventTime, end: EventTime) -> Result<Self>;
    pub fn event_time_column(&self) -> &str;
    pub fn start(&self) -> EventTime;
    pub fn end(&self) -> EventTime;
}
pub struct SortKey { /* private: column, direction, nulls */ }
impl SortKey {
    pub fn new(column: &str, direction: SortDirection, nulls: NullPlacement) -> Result<Self>;
    pub fn column(&self) -> &str;
    pub fn direction(&self) -> SortDirection;
    pub fn nulls(&self) -> NullPlacement;
}
pub struct SortSpec { /* private fields; no Default */ }
impl SortSpec {
    pub fn bounded_source(state_budget: StateBudget) -> Result<Self>;
    pub fn closed_window(window: SortWindow, state_budget: StateBudget) -> Result<Self>;
    pub fn with_offset(self, offset: u64) -> Result<Self>;
    pub fn with_sequence_by(self, columns: Vec<String>) -> Result<Self>;
    pub fn with_late_policy(self, policy: LatePolicySpec) -> Result<Self>;
    pub fn history(&self) -> &SortHistory;
    pub fn state_budget(&self) -> StateBudget;
    pub fn offset(&self) -> u64;
    pub fn ties(&self) -> &SortTies;
    pub fn late_policy(&self) -> LatePolicySpec;
}
impl TableExpr {
    pub fn sort(&self, by: Vec<SortKey>, spec: SortSpec) -> Result<Self>;
    pub fn top_k(&self, k: u64, by: Vec<SortKey>, spec: SortSpec) -> Result<Self>;
}
```

`sort` 保留全部输入列和重复行，按 by 字典序排序后跳过 offset。`top_k` 在完全相同顺序中跳过 offset，再取至多 k 行；**top 是这个顺序的前 k 行**，不是隐式最大值。要最大值优先须明确 Descending；不另加 largest/reverse/n 同义入口。返回全新 TableExpr，不执行数据或修改原图。排序表达式、输出列选择和改名用前后已有 select/with_columns；by 非空、列名非空且唯一，全部属于本节点输入，不接受跨 lineage ColumnExpr。

Python 从 `calc_flow` 与 `calc_flow.symbolic` 导出 SortKey/SortSpec/SortWindow，沿现有兼容 frozen/slots 数据模型，不提高 Python 3.9 下限。SortWindow 有只读 event_time/start_micros/end_micros 字段及下列显式构造器；复制调用者的 by/sequence_by 序列，不允许 str/bytes 冒充序列。没有公开任意 wire 字段构造器。

```python
class SortWindow:
    event_time: str
    start_micros: int
    end_micros: int

    def __init__(
        self, event_time: str, /, *, start_micros: int, end_micros: int,
    ) -> None: ...

class SortKey:
    def __init__(
        self, column: str, /, *, descending: bool = False,
        nulls: Literal["first", "last"] | None = None,
    ) -> None: ...

class SortSpec:
    @classmethod
    def bounded_source(
        cls, state_budget: StateBudget, /, *, offset: int = 0,
        sequence_by: Sequence[str] = (),
    ) -> SortSpec: ...
    @classmethod
    def closed_window(
        cls, window: SortWindow, /, *, state_budget: StateBudget, offset: int = 0,
        sequence_by: Sequence[str] = (),
        late_policy: Literal["error", "drop"] = "error",
    ) -> SortSpec: ...

class TableExpr:
    def sort(self, by: Sequence[SortKey], /, *, spec: SortSpec) -> TableExpr: ...
    def top_k(
        self, k: int, by: Sequence[SortKey], /, *, spec: SortSpec,
    ) -> TableExpr: ...
```

Python nulls=None 在构造时解析为 ASC→last、DESC→first，匹配当前 native SQL 默认；wire 只保存显式 first/last。Rust 构造器要求明确两值。Python descending 必须是真 bool；k/offset/budget/time 必须是真 int，拒绝 bool/float。空 sequence_by 表示 Unspecified；Rust with_sequence_by 要求非空，避免把“请求稳定但忘传列”规范成 unspecified。重复配置调用返回新 spec，合法最后一次值替换该项；不修改共享副本。

SortWindow::new / SortWindow 构造时验证原有非空时间列、i64 微秒 start<end（负时间合法）。SortSpec.closed_window 只接纳真正 SortWindow，维持 state_budget 必选，offset/sequence_by/late_policy 的默认和约束不变；完整配置仍在原生规划/导入边界验证。SortWindow 只声明一个等待真实闭合的区间，没有几何、聚合、预算、late policy、行或运行状态，不证明未来已经闭合。不得把新的值对象换成任意 dict、匿名 tuple、动态 kwargs 或由调用者提供的 wire 字符串。

## Four units: SQL spelling and shared lowering

Rust/Python SQL 继续使用现有公开入口及关系上的 `.sql(query)`，不新增执行入口：

```rust
pub fn sql(query: &str, tables: BTreeMap<String, TableExpr>) -> Result<TableExpr>;
// TableExpr::sql binds its receiver as input, as today.
```

```python
def sql(query: str, /, **tables: TableExpr) -> TableExpr: ...
# TableExpr.sql(query) binds input, as today.
```

两语言新增相同的 calc-flow 方言宏，由 Rust AST/schema seam 识别。它们不是 DataFusion 内建函数；没有宏的旧 ORDER BY/LIMIT/OFFSET 完全沿旧路径。

```sql
SELECT * FROM CF_ORDER_BOUNDED('a', 1000, 8388608)
ORDER BY x ASC NULLS LAST

SELECT * FROM CF_ORDER_BOUNDED('a', 1000, 8388608, ARRAY['seq'])
ORDER BY score DESC NULLS LAST LIMIT 3 OFFSET 1

SELECT * FROM CF_ORDER_WINDOW('a', 'ts', 0, 10, 1000, 8388608, 'drop', ARRAY['seq'])
ORDER BY x ASC NULLS LAST LIMIT 1
```

BOUNDED 的必选项是 `(alias_string, max_rows, max_bytes)`，可选末项为非空 `ARRAY['sequence_column', ...]`；只允许3或4参数。WINDOW 必选前6项为 `(alias_string, time_column_string, start_micros, end_micros, max_rows, max_bytes)`；可选第7项为 `'error'|'drop'`（默认 error），可选第8项为同一 sequence 数组。若指定 sequence，须提供第7项，不能靠参数 dtype 猜位置。宿主 API 用配置对象，不引入八个宿主位置参数。

固定 SQL 子集：单个顶层 SELECT、单个无修饰 `*`、恰好一个宏 FROM、tables 恰好提供其引用一次的 alias、非空 ORDER BY 直接列引用（可用现有 qualified/quoted identifier 解析）、可选 literal LIMIT k / OFFSET offset。LIMIT 缺省降为 sort，存在时降为 top_k；OFFSET 可以独立使用。ORDER BY 未写方向/NULL 时仍按 ASC、ASC last/DESC first 解析，再规范成明确配置。不能 ORDER BY ordinal、ALL、表达式、collation、WITH FILL 或非本输入列。

不支持 DISTINCT、SELECT 列表达式/别名、WHERE、GROUP/HAVING/QUALIFY、aggregate、OVER、CTE/子查询、JOIN/set operation、FROM AS、FETCH/WITH TIES、UDF/provider/volatile、LIMIT ALL/负数/参数/表达式。需要投影或过滤先声明独立关系再传入宏；SELECT * 是本片和 DF 保留全部列语义的精确对应，不是普通 SQL 的普遍限制。宏名不带 schema、大小写不敏感；alias/time/sequence string 解码后精确匹配。整数只接纳十进制 literal，时间边界允许负号；sequence 数组只接纳非空、不重复的普通字符串 literal，不是 SQL 动态 ARRAY 表达式。

出现任一 CF_ORDER_* 但名字、参数或整形状非法，必须定位违反项并拒绝，不能 fallback per-batch SQL。Rust AST seam 返回同一 data-only 描述 `by/spec/fetch/alias`；schema/analysis/compile/export 和 Python binding/stub 都用它。低层 SqlOperator 不增加另一种宏解释；导出实际 bounded_order 节点。

| Unit               | Declaration seed                                                                            | Common result                                          |
| ------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Rust / DataFrame   | `t.top_k(1, vec![SortKey::new("x", Ascending, Last)?], SortSpec::bounded_source(budget)?)?` | 同一个 bounded_order(input, by, fetch=Some(1), spec) 关系节点 |
| Rust / SQL         | `sql("SELECT * FROM CF_ORDER_BOUNDED('a',1000,8388608) ORDER BY x LIMIT 1", tables)?`       | Rust AST 解析后为该关系节点，不用另一份 batch SQL 实现                  |
| Python / DataFrame | `t.top_k(1, [cf.SortKey("x")], spec=cf.SortSpec.bounded_source(budget))`                    | 绑定到同一 native 配置、schema 和内核                             |
| Python / SQL       | `cf.sql("SELECT * FROM CF_ORDER_BOUNDED('a',1000,8388608) ORDER BY x LIMIT 1", a=t)`        | 相同 native AST 描述和 lowering                             |

native 配置为严格 `kind="bounded_order"`、`by`、`fetch`（null 或整数）和 `spec`。spec 必含 `ingress_policy="copy_visible_v2"`、history、state_budget、offset、ties、late_policy；默认值在新节点中也规范显式写入。四单元相同意思的配置须相同，不要求跨语言/跨 SQL 文本的声明 fingerprint 相同。

复用并按需抽取 `operator/sql/output_order.rs` 的 native Arrow/DataFusion 55 sort/NULL/skip/fetch 路径作为唯一排序内核；这不是允许新增四份 comparator。现有 lexsort/take 需要的合并输入、索引、key arrays、选择与结果分配都进入本节点预留，不能把旧 SQL 的粗估或 pool 当完整 owned peak 证明。新节点不用 saturating_add 掩盖边界；batch 和 stream-final 调同一 finalizer，不让 finite 启动持久 StreamJob。

## Types, ties, schema and composition

| Role                         | First delivery domain                                                                                         | Constraint                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| sort keys / retained columns | Boolean、Int8/16/32/64、UInt8/16/32/64、Utf8/LargeUtf8、Date32/64、timestamp[us] naive/UTC；另支持 timestamp[ms] naive | 可 nullable，保持精确 dtype；ms 的自有复制须显式接通，不能 alias 外部存储 |
| sequence_by                  | 非 NULL 的 signed/unsigned integers、Utf8/LargeUtf8，可为 composite tuple                                           | schema nullable=false，实际整 H 唯一；不要求值按到达顺序递增        |
| window event_time            | timestamp[ms] naive、timestamp[us] naive/UTC                                                                   | 可 NULL；checked EventTime import，原样/纯重命名的来源证明      |
| outside domain               | float32/64、Decimal、binary、nested、dictionary/extension、其他时间类型                                                  | 本片连 retained payload 都准确拒绝；用户可事先投影掉，不缩减普通 SQL 的域  |

选择 scalar owned-normalization 的窄类型域，避免在本片同时决定 NaN/signed-zero、collation、nested copy 和跨语言表示。不是声称 DataFusion 无法排序这些类型。整数/日期/时间按其 native 值；Boolean false<true；字符串按 native UTF-8 二进制字典序，不作 locale/case/Unicode normalization。多键依声明顺序逐级比较，方向和 NULL placement 各自独立；NULLs First/Last 不随 Descending 反转，不删除 NULL，重复 tuple 保留。

**Sequence 的来源和承诺：** 稳定模式只有显式非空 sequence_by。这些是本节点输入里真实、可重放的字段，不是节点自动编号。调用者/Source 须保证在同一逻辑 cut 的 replay 及两 mode 所比较的 H 中，tuple 对应同一输入行；节点在每个完整 Data 的 candidate H 中验证非 NULL 和全 H 唯一（跨 RecordBatch、跨 Data），再提交。可按 native tuple 排序/比较验证，任何索引/scratch 一并预算；不允许未计费的 seen-set。重复/null 的错误只报列/约束及位置，不泄露 tuple 值。有限执行验证本次完整 H；窗口只验证域内成员。schema 可读时即拒绝 nullable 或不支持 sequence 列，数据唯一性不能在 analyze 时假称已验证。

sequence_by 各列必须非空、唯一、确实存在，且与 by 中列不重叠。按声明 tuple **升序**追加为末级 comparator（非 NULL），与主键方向无关。这定义 total order；不承诺 first-arrival/first-seen，也不要求 sequence 是连续整数。输入 `InputOptions.sequence_by` 的 temporal 声明本身只说明原来的 identity，**不自动成为本节点全 H 唯一证明**；用户可显式选择相同列，但仍须通过本片检查。Batch.metadata.sequence、source task 分发次序、RecordBatch offset、DataFusion 分区/take 索引均不是稳定 sequence。

上游已证明 row-local SQL 可用与 window time 类似的局部 AST 映射，把所选 sequence 追溯到原输入的原样列或纯重命名；filter 可筛行。sequence 算术/cast/concat、无法证明的表达式和新生成的序号拒绝稳定模式。不恢复一般 SQL 的 temporal identity，不靠同名/dtype 猜来源。重放稳定性仍受 Source 原有 cursor/cut/replay 合同约束，sequence 不把普通 iterable 变为可恢复 Source。

Unspecified 模式不检查/隐式采用 sequence；主键相等时排列和边界选择未规定。即使当前 helper 稳定，不能宣传跨切批、分区、语言或恢复稳定。Sequence 模式对相同 H 在预算成功下保证同一有序行列及 k/offset 入选集合。DISTINCT 的无序输出既不提供该 sequence，也不提供 first-seen 证明。

输出列名、列序、dtype、nullable 为完整输入的精确 schema（包括 sequence 作为普通数据列），空输出同 schema；不增加 rank、窗口边界或隐藏列。产生新 row lineage，不继承 entity/event_time/temporal ordering。QueryOrderingFacts 的局部有序证明独立于 temporal identity。沿现有 frontend/Port metadata 规范化；不修改 caller Arrow/Batch metadata，也不新增保留任意结果 metadata 的保证。

公共路径为一个 exact-schema input → 已证明 deterministic 单输入 row-local DF/SQL 串 → 一个 bounded_order → row-local DF/SQL 串 → 命名输出。允许共享 fan-out、独立排序节点及无关合法输出；同一声明每个 job/finite execute 只有一个 owner。上游 aggregate/snapshot、union、多源/join/ASOF、rolling/cross-section、window aggregate、DISTINCT 或另一个 bounded_order，及 array/external 都在这条新路径外；强制在 native/project/两语言编译入口拒绝。**本片不放宽 DISTINCT 的既有下游白名单**；DISTINCT→ORDER 两个 finalizer 的 cut/控制传播留后续组合片，不能以相邻节点名宣称可组合。可把另一次执行得到的有限表显式当作新输入排序，其 sequence 须重新满足本合同。

后续 row-local stage 只有在 native 路径证明实际保序且键字段映射可追踪时，才可传播 ordered/total；字段被移除或 SQL 物理顺序无法证明时清除/降为 unknown，并拒绝相应 declared-order portability。不能从“row-local”直接推出跨分区保序。排序节点本身直连命名输出是必交付的完整有序路径；mixed 图仍须有准确 schema/ordering 事实，不能默称整个后处理输出有序。

## k, offset and finite boundaries

`k` 和 `offset` 均为 inclusive `[0, 4294967295]`。两者同时存在时 checked `offset+k<=4294967295`；即使实际数据很小也不截断/饱和这个声明。选此可移植上界，是当前共享 Arrow lexsort 使用 UInt32 row indices 的约束；不是 JSON 或 platform usize 的宽度。SortSpec 的预算另要求 `1<=max_rows<=4294967295`，`1<=max_bytes<=9007199254740991`；底层 StateBudget 接受但大于此 row ceiling 的配置在 SortSpec 构造/严格导入时拒绝。ms/UTC 时间转换、索引数、alignment、offset/string capacity 的 checked 溢出在分配前拒绝。

设域内 H 有 N 行，完全排序结果为 S。sort 返回 `S[min(offset,N):N]`；top_k 返回从 `min(offset,N)` 起的 `min(k,N-min(offset,N))` 行，不用 offset+k 的溢出计算推导。k 可以大于 N 或 max_rows，offset 也可以超过 N；不按声明 k/offset 分配虚构行。k=0、offset>=N、typed empty 和全 NULL 键均合法，闭合时仍产一个精确 schema 的空 final。k=0 **仍接纳/验证/计量完整 H 并等待真实闭合**，不能早发、跳过 sequence/schema 错误或借此运行无界 Source。

有限 process 将本次 Batch 全部 RecordBatch 视为完整 H；window 先筛 membership，再运行同一 finalizer。不模拟水位或 Late Data，也不要求 SourceBoundedness。每次 execute fresh H、candidate、sequence-validation state、closed、output_sequence 和预算 pool；同 plan/Runtime/暖缓存无 reset，成功、失败或取消后再次 execute 都无旧候选。batch snapshot=NULL，非 NULL restore 拒绝；不把 stream checkpoint 当有限查询续算。

## Stream closure, lateness and updates

复用已经实现的 SourceBoundedness，不另发明 bounded=True 或公开 finalize/EOF/watermark 注入 API。bounded_source 必须在实际绑定预检取得 Bounded（固定有限 cut，恢复续同一个 cut）；Unknown/Unbounded 都不能因未来“可能 EOF”获准。ClosedWindow 须有原时间坐标上的可信 source-native/runtime-generated watermark route，或 Bounded EOF route。编译可以结构 supported 并列出绑定义务，但所有启动入口必须在 Source.open/iterator acquisition/next、Sink.open、state root/job 创建前核验冻结 descriptor；不通过试读数据求证。暖缓存、reload、显式 runner 与 Python convenience 同样强制，独立于 portability/latest opt-in。

| Observation / history                | Required behavior                                                         |
| ------------------------------------ | ------------------------------------------------------------------------- |
| bounded_source 的真实正常 FIFO EOF        | 全部先前 Data 已处理后闭合；运行 finalizer，成功 emit 一个完整 table Batch；可以内含多个 RecordBatch |
| window `[start,end)`                 | NULL 时间/区间外行不进 H；有效输入 W>=end 或真实正常 EOF 闭合；W<end 时不因 ts<=W 而提前丢域内行         |
| 空 Data、Idle、Barrier、checkpoint、W<end | 不闭合、不发 provisional、不 reset；Barrier 仍按原协议对齐                                |
| cancellation、sender drop、连接中断或错误     | 失败，不是 EOF；不得补一个“成功 final”                                                 |
| final 成功后的重复 close / 随后正常 EOF        | 不再发 final；早闭合窗口继续原 Source/job 生命周期，不伪造 Source EOF                         |
| 无 Data、k=0 或无入选行                     | 同样在合法闭合处发恰一个空 final，不把尚未初始化等同于空关系                                         |

ClosedWindow 默认持久 `LatePolicySpec::Error { scope: Envelope }`，只另接受 Drop(metrics_version=1)。start/end 是 i64 微秒，start<end、允许负时间；不再加 grace period、不手工二次减 allowed_lateness、不重新开窗。Data 只采样一次已提交有效 W；已水位关闭后属于 `[start,end)` 的行才 late，NULL/范围外不是 late。

Error：任一 late member 令整个 Data 失败，H/accepted_rows/closed/output_sequence/水位/metrics/capture generation 不变；此前 final 不回滚。Drop：丢掉所有 late members，不再保存、输出、撤回或增加 accepted_rows；达到 max_rows 后仍可丢 late rows，不先用 old+late 判超行预算。扫描/metrics 的自有存储仍需 bytes 预留。schema/时间转换/取消/metric overflow 不被 drop 吞掉；EOF 后 Data 仍是非法 runtime 输入，不被改写成 late。

Drop 复用 checkpointed late_rows、affected_batches、max_lateness_micros，初值0/0/None；每个完整 Data 跨全部 RecordBatch 只计一次 affected_batches。每条被丢成员的 lateness 为 checked_u64(i128(W)-i128(ts))，不是 W-end；NULL-time counters 不新增。先准备 delta/totals 和现有 PreparedLateMetrics，再于无 await、不可失败提交段一并安装 counters/metric sink、invalidate capture；恢复 totals 不当作新 delta 重复累加。

bounded_source 只接受默认 Error，Drop/SideOutput 拒绝；window finite 接受 Error/Drop 以复用同一声明，但不产生 late counters。Program 的全局 late_policy/allowed_lateness 仍只配置其原本适用的未标记 rolling/cross-section，**不覆盖排序 spec**。含排序图而没有此类消费者却传非默认全局选项时 capability_mismatch，提示在 SortSpec.closed_window 选本地策略；有独立消费者时只作用于该消费者。默认全局 error 不抹去本地 Drop；同图不同排序策略独立持久化。

stream 输入解释恒 append，输出为 append/group_final，history 为 accepted_history 或 event_window；不是 revisable complete_snapshot。batch facts 为 current_batch/append/per_batch（本次完整 H 的结果）。snapshot 经 row-local 中转仍在 offending edge 以 update_interpretation_mismatch 拒绝；未知 update 不能准入。legacy 也不为新节点接纳 stateful/snapshot 上游。旧快照追加7、已获准 latest6/拒绝、普通每批 SQL 和现有 delivery 语义保持。

## Owned peak budget and failure atomicity

每个节点独立显式 StateBudget。max_rows 计域内累计接受行（重复和最终未入选行也计），并分别限制 candidate history/最终输出行数；不是 k、distinct 数、RecordBatch 数，也不是同时存活副本行数之和。不同角色同时存活的 bytes 则必须相加。窗口只计 membership 内成员；已闭合 drop 不追加。共享声明一个 owner 一次收费，不同节点各自计费。

**强制沿 copy_visible_v2：** 当前 raw ingress 只在同步验证/计数/复制阶段借读，包括它为未知 external parent buffer 保活的临时 envelope；这个未知 backing 在节点 quota 范围外。所有保留的输入列、sequence 值、schema/metadata 都必须从可见值深复制成节点自有、已预留存储；不保存 raw Batch/ArrayRef/SchemaRef/metadata Arc、Python capsule 或捕获它们的闭包。不得在本节点第一次内部 await/emit/query/checkpoint 阶段继续持有 raw 强引用；上游、Source、queue、caller 仍可另持原输入。没有 owner-size 自报、offset=0 或内部来源的 no-copy 旁路。

本预算只约束确定性的**节点自有 participating logical peak**。不承诺整个 graph/job/process/RSS，外部 ingress 父分配没有被测得或计作0；DataFusion pool 未登记的私有开销、allocator bookkeeping 也不因此受限。特别不能复制 DISTINCT 文档后，把其 owned-only 保证改写为“所有输入/进程内存受 max_bytes 限制”。

设 M(t) 为本节点仍活跃的各角色 lease 的 logical charge 和。每次下一阶段分配必须先验证 checked `M(t)+next_charge<=max_bytes`，**先 reserve，再分配/复制**。计数可多遍扫描，不先 collect 行/构建索引/编码大对象再补账。沿已验证 owned buffer 的64字节 alignment/capacity 计量及 schema/metadata 预付规则；自有 Batch 基础 charge 为 `512 + owned_schema_charge + sum(256 + owned_array_charge)`，另加真实 Vec/String/Box/spare capacity。不同角色即使共享自有 buffer 也保守分别收费，不靠指针去重缩短 lease。纯数值 accounting bookkeeping 的既有例外不能扩展到存数据/键/索引的容器。

| Covered role                            | Required simultaneous accounting                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| normalization / committed / candidate H | 所有 retained payload（不限 by）、sequence 校验结构、新旧 history/容器、深复制 schema/metadata，raw 借读时新目标也已预付                     |
| sort / sequence validation workspace    | 合并输入、keys、indices、比较/去重 scratch、native sort/selection/take 的已知分配及 DataFusion reservations；与 H/captures 同时累加   |
| prepared result / compact final         | source result 的完整自有 backing、输出副本、描述符和复制 scratch；top-1 不得经 slice/clone 把整 H 的 lease 过早释放                       |
| output emission                         | 等待 emit 及节点仍持 output 时计费；成功交接并释放节点引用后 solely-downstream 的 owner 才退出本节点预算                                      |
| checkpoint / restore                    | 所有尚活 capture、control/identity/IPC/metadata/segments、解码 candidate 与旧 state、恢复 sequence-validation scratch 同时计费 |
| zero rows / k=0                         | schema/control/metadata/容器与必要 workspace 仍收费，无“空关系零 bytes”的豁免                                                  |

排序 workspace 不是只计最终 indices.len。实现须从受支持 dtype/实际 N/字符串长度/目标分区等推导有限预付上界；Arrow kernel 不进 pool 的新分配须先用 lease 支付，增长前预付旧+新容量。不能用一个已存在的 DataFusion MemoryPool 替代 Arrow/take/结果复制计量，或仅以固定倍率宣称覆盖未知 backing。保守预付可能使最终很小的 top-k 在峰值预算不足时失败，这是公开合同。若具体安全路径无法先算有限需求，应在调用 kernel 前明确拒绝并回报具体路径，不允许运行后才发现越界而仍声称峰值已受限。

Data 事务：验证整个 envelope 的 schema/时间/sequence → checked 行数/计数并预留候选、normalization 和 sequence-validation scratch（旧 H/captures 仍计）→ 完成自有复制、放开 raw → 校验整个 candidate（包括跨批 sequence 唯一）→ 无 await 提交 H/accepted_rows/validation state/capture generation。任一 RecordBatch 失败、预算/取消/duplicate 错误都不部分安装；释放临时 owner/lease。

闭合事务：保留旧 H/watermark/closed/sequence → 预付并排序/截取 → 再预付深复制 compact 输出，不借 H 或临时结果 backing → 验完整 edge row/byte budget、schema 和取消 → await emit 成功 → 无 await、无新的可失败检查地提交 closed/output_sequence=1/触发闭合的水位、清空 retained H 和不再需要的校验结构。失败前不安装 closed/水位/sequence，不伪造空 final；查询/输出 leases 释放，旧 H 可按原 runtime 失败/恢复流程处理。不能在 emit 成功后增加一个取消检查制造“已发出但状态未提交”。

此保证是单 Operator 的状态事务，不是 fan-out/queue/Sink 全图回滚。emit 部分外部效果、失败前缀与 checkpoint replay 仍按原 per-output delivery/source cut 合同处理；不新增 ExactlyOnce。完整 final 超 edge 限额时失败，不拆成几个不完整 final 绕过预算；一个 Batch 内可有多个 RecordBatch 且按其顺序连接。

## Analysis, cache, export and recovery identities

QueryAnalysis 保持 schema_version=2，**analysis_revision 提升为10**：新增支持/准入/组合证明规则，使用已有枚举，不新增报告字段。Rust producer、Python parser、strict/latest gates 与分析缓存同步；旧 r9 可以展示，不成为排序的准入凭据。新配置的 capability/layout/策略 gate 仍独立，不能仅以 revision 数字判断支持。

局部 ordering.kind=ordered，keys 为有效 comparator 顺序（含 sequence 的升序末级键），ties 在 Sequence 时为 total、否则 unspecified，temporal_identity=None。reason 必须公开同一 H、sequence 的数据验证与 replay 前提、闭合/绑定义务、类型/k/offset/owned-only budget 边界。没有 sequence 的完整 sort、offset=0，在 unordered final_relation 比较下保留全部多重集，可获准；declared 比较缺 total 则 row_order_unproven/unknown。**任何存在 fetch 或非零 offset 的 unspecified-ties 声明**，即使请求 unordered final_relation，也因成员选择未证明而 row_order_unproven/unknown；不根据样本无 ties、k=0 或“当前数据够少”在分析中放行。显式 sequence 成功可证明两 mode 同一接受集合/控制截止下的最终有序结果；per_batch 请求不被单次 final 偷换。

下游传播需按实际整图证明；清除键/sequence 或物理重排时不能仅 AND 局部 supported。analyze/explain 仅 schema/AST，不读行、不 open/poll/iterate source、不启动 job。结构错/unsupported type/shape 与 unknown source/proof 区分，均不通过强制准入。compile 准入及 strict 请求在准备缓存读取前执行；真实绑定及恢复 gate 在任何运行资源/Source I/O 之前执行，cache hit 不绕过。数据相关的唯一性、大小和预算只能运行时验证，不在 supported 中谎称已检查。

新增 capability `bounded_order@1`：required table input/output，modes=batch/stream、stateful=true（stream）、requires_datafusion=true、group_final_append_only、checkpointed_state/state_version1，所需 layout 根据本地策略选择。native `StreamOperatorLifecycle::CheckpointedStateful` 的三个 proof flags（microbatch_invariant/deterministic/replay_safe）在 Sequence 时为true，Unspecified时保守为false。后者仍是可捕获 H/control 的内置节点，可用于原有 best-effort/at-least-once 运行及本节所定义的弱 ties 恢复；不能借其 state_version/可编码 checkpoint 推出更强确定性。

`python/calc_flow/capabilities.py:1039` 所示 inventory 的三个 bool 是 kind 级无条件保证，不能表达此处的条件。新 bounded_order inventory 因此三者均为false，requires_watermark=false（存在 EOF 域），state_layouts=(1,2)；这不是“无法运行/无法保存状态”。native/schema analyzer 按实际 Sequence 配置给出局部 total-order proof，只有该证明加运行时值校验才能支持更强请求；不能把同名 kind 的所有节点一律提为true，也不新增第三套 capability 结构。

`pipeline/stream.rs:206` 的 checkpoint capability enum 本身没有 tie-proof 位，`pipeline/stream.rs:1084` 的 lifecycle gate 又只检查 external Stream variant；新 built-in 不能仅注册 state_version1 就让强 delivery 请求通过。**本节点的强制准入还须在任何要求确定性 replay 的路径（含 ExactlyOnce）拒绝 Unspecified**，定位 ties/所要求的保证；不修改其他内置节点的 gate。Sequence 只满足排序自身的必要条件，原 frontend、Source/Sink、per-output delivery 限制仍可拒绝；本片不新增 ExactlyOnce 可用性，也不放宽 external provider 的 lifecycle 校验。

项目仍是严格 data-only project-v3，新增 OperatorSpec/schema variant；导出 by/fetch/完整 spec/精确 schemas，不能以原 SQL 文本在 reload 时改走旧 SQL。逻辑名到物理 bindings 的已有边界不改，reload 用公开 external_inputs/outputs；不导出 rows、sequence 值、Source、plan/cache 或 checkpoint。共享 owner 与无关输出在 round-trip 后保持。

新路径私有 prepare/cache tag 为 `bounded_order_v1/copy_visible_v2/order_accounting1/analysis10`，key 含完整 by（顺序/方向/NULL）、fetch、history/time/cut、offset、ties/sequence 列、本地 late policy、预算和原有 mode/schema/update semantics/registration/engine 身份。for_mode 共享声明，不共享 plan/H/pool/Source 或 state。Boundedness/水位 route 在每次启动采样同一 descriptor，并写入新路径 execution/restore identity；不能复用只对 DISTINCT 冻结的 proof。

冻结新的独立状态命名空间：state_kind=`bounded_order_v1`、state_version=1、accounting_version=1（**排序自己的规则，不是 DISTINCT accounting1 的复活**）、ingress_policy=`copy_visible_v2`；Error 写读 layout1，Drop 写读 layout2。Drop 即使从未迟到，也包含0/0/None metrics 且要求 layout2。capability inventory 真实列 `(1,2)`，按节点策略检查所需布局，不改 DISTINCT 的 kind/accounting2/layout2/3 或旧 SQL layouts3/4/5。

checkpoint 必存：完整配置/schema/engine/Arrow/runtime-config digest、domain/cut 与 source 身份、accepted_rows/retained_rows、规范化自有 H、已提交有效 watermark、closed、output_sequence、segment/content digest、Sequence 校验状态或可预算重建所需字段，以及 Drop metrics。output_sequence 是节点发出 final 的控制计数（0或1），**不是稳定行 sequence**。未闭合 retained_rows=accepted_rows、output_sequence=0；闭合 retained_rows=0、output_sequence=1，accepted_rows 保留并仍受 max_rows 限制。不持久 raw owner、地址、provenance token 或 SQL 临时排序 indices；restore 解码生成新的已预付自有表示，payload 自称 owned 不能跳过计量/唯一性校验。

捕获 cache 随真实数据/水位/metrics/control 提交失效；旧 capture lease 活到最后一个实际 owner 释放，不在 snapshot 发布/新 capture 覆盖变量时提前免账。恢复先小 control/schema/config 检查，再预留有界解码/校验目标，全部成功后一次安装；拒绝未知字段/kind/state/layout/accounting/policy、缺字段、engine/config/schema/cut/Source 能力不匹配、错误行数/sequence/closed/digests、预算不足、Error↔Drop 或任一 by/fetch/offset/ties/预算变更。可由 control/descriptor 确定的失败在大解码目标分配前发生；内容损坏/sequence 重复只能解码后确定时，仍在状态安装/Source.open 前失败，不能谎称读取 control 就已验证内容。

未闭合恢复继续同一 H 到既定闭合点；closed 恢复不重发 final、不重开窗口，后续 late Error/Drop 继续正确。Sink 尚未提交的交付恢复沿原成功前缀协议，不由 closed 位承诺额外去重。没有从 DISTINCT/旧 SQL/别的 mode checkpoint 迁移为排序状态；batch NULL snapshot 与 stream 不互认。新字段/证明/tag 仅用于新路径，旧 graph/project/fingerprint/mode/cache/checkpoint bytes 不重写；原快照链7及显式 latest6/拒绝保留。

## Error contract

P 为 canonical relation path：DF `.sort` / `.top_k`，SQL `.sql.order`；键用 `.by[i].column|direction|nulls`，配置用 `.spec.history|state_budget|offset|ties|late_policy`，k 用 `.k`；真实绑定用 `stream.inputs.<logical_name>.boundedness|watermark_policy`。原生项目保留 graph.nodes/edges 的物理路径，不泄露 generated IDs 当作逻辑用户地址。

| Violation                                            | Required diagnostic / boundary                                                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| k/offset 错类型、负数、越上界或相加溢出                             | TypeError/ValueError 或 Rust InvalidArgument；invalid_literal，点名 k/offset、u32 边界及 sum 约束；读行=0                    |
| 空/重复 by、未知列、NullPlacement::Exclude、unsupported dtype | invalid_literal / duplicate_name / unresolved_type / unsupported_type，unsupported；列/索引及允许域；schema 阶段拒绝，不等首行    |
| nullable/改写/不存在的 sequence、重复或与 by 重叠列                | ordering_required / capability_mismatch / duplicate_name；明确必须原样非 NULL 字段及不重叠；实际重复 tuple 为执行失败，整个 candidate 不提交 |
| SQL 宏名/arity/ARRAY/literal/clause 不合子集               | invalid_literal 或 capability_mismatch/unsupported，定位具体配置/条款；无 legacy fallback                                  |
| Source Unknown / Unbounded                           | source_boundedness_unproved/unknown 或 source_unbounded/unsupported，固定有限 cut 必需；open/next/job=0                 |
| window 无可信 closure route / 时间来源改写                    | finality_unproven/unknown 或 capability_mismatch/unsupported；指出原时间列与 end 义务；open/next/job=0                     |
| snapshot/update unknown、嵌套 finalizer/多源路径            | update_interpretation_mismatch 或既有 update_*_unproved；越界路径 capability_mismatch；native/project/缓存同样拒绝            |
| unspecified ties 请求跨 mode 截取成员或 declared order       | row_order_unproven/unknown；说明 tie/selection 未证明，提示显式 sequence_by，不建议依赖到达顺序                                     |
| runtime schema/时间/sequence/预算/排序/emit/取消失败           | 沿原 typed execution/stream error 与 source cause，保留 node/角色/约束；无候选部分提交，不伪造 final                                 |
| restore 身份/layout/计数/digest/预算错误                     | 既有 CheckpointMismatch/状态错误；点名不兼容项，Source.open=0；不静默清空 H 重跑                                                     |

预算失败须以 `state_budget_exceeded` 提供 node、policy、normalization/history/sequence validation/sort workspace/output/capture/restore 等角色及 required/limit、rows/bytes；不含 payload、sequence tuple、完整 plan dump 或凭据。`input_normalization_unsupported` 只用于具体表示无法安全逐值重建；未知 external 父大小本身不是拒绝条件。不新增平行异常体系；QueryDiagnostic 三态、原 Compile/QueryPortability/QueryUpdateSemantics/stream/checkpoint 分层保持。

## Typical declaration seeds

以下为待实现后的 Python 用法，本文只作语法自检。两种写法共享相同 schema/内核；不同声明不要求 hash 相同。例子只在有限执行检查具体结果，不用它冒充两个 stream Data 边界的验证。

```python
import pyarrow as pa
import calc_flow as cf

schema = pa.schema([
    pa.field("x", pa.int64(), nullable=True),
    pa.field("seq", pa.uint64(), nullable=False),
])
events = cf.table_input("events", schema=schema)
budget = cf.StateBudget(1000, 8 << 20)
spec = cf.SortSpec.bounded_source(budget, sequence_by=["seq"])
out = events.top_k(1, [cf.SortKey("x")], spec=spec)
via_sql = cf.sql(
    "SELECT * FROM CF_ORDER_BOUNDED('a',1000,8388608,ARRAY['seq']) "
    "ORDER BY x ASC NULLS LAST LIMIT 1", a=events,
)
program = cf.Program("smallest", mode="batch", outputs={"out": out})
data = pa.Table.from_pydict({"x": [3, 1, 2], "seq": [10, 11, 12]}, schema=schema)
assert program.collect({"events": data})["out"].to_pydict() == {"x": [1], "seq": [11]}
stream_program = program.for_mode("stream")
# Bind a proven Bounded Source; final appears at real EOF, with one job owner.
```

Rust 的典型声明增量，执行/绑定使用现有 `Program::compile_batch/compile_stream` 和 runner，完整独立例子落 `crates/calc-flow/examples/bounded_global_ordering.rs`：

```rust
use std::{collections::BTreeMap, sync::Arc};
use calc_flow::{CompileMode, InputOptions, Program, Result, SortDirection, NullPlacement,
    SortKey, SortSpec, StateBudget, table_input};
use calc_flow::arrow::datatypes::{DataType, Field, Schema};

fn declaration() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let events = table_input("events", schema, InputOptions::default())?;
    let spec = SortSpec::bounded_source(StateBudget::new(1000, 8 << 20)?)?
        .with_sequence_by(vec!["seq".into()])?;
    let out = events.top_k(1, vec![SortKey::new(
        "x", SortDirection::Ascending, NullPlacement::Last,
    )?], spec)?;
    Program::new("smallest", CompileMode::Batch, BTreeMap::from([("out".into(), out)]))
}
```

Python 新例放 `examples/bounded_global_ordering.py`，不移动/替换 DISTINCT、window、typed core 例子。实现者补相同能力的公开 docs/stubs、实际 imports/执行与绑定说明；本文 Rust/SQL 种子未编译/执行。

## Independent acceptance and handoff

以下是实现的最小独立预期，不用另一个新 frontend/lowerer 自动生成 oracle，不默认展开语言×类型×分区×模式全矩阵。

1. **C03/A05/A06：** 真正两个 stream Data 为 x=[3,1] / [2]，旧 `SELECT x FROM input ORDER BY x LIMIT 1` 仍发 [1] / [2]；新 bounded top_k(1, ASC) 在 EOF 前0事件，后恰1个 final x=[1]；finite 完整 H=[3,1,2] 得[1]。验证完整 sort=[1,2,3]，DESC=[3,2,1]。旧轨迹不能用会合并材料 list 的 convenience Source 替代两个 Data。
2. **ties/NULL：** 精确 schema，行 `(x,seq,label)=(1,20,"b"),(NULL,30,"n"),(1,10,"a"),(2,40,"c")`。ASC NULLS LAST、sequence_by=[seq] 的 label 次序为 a,b,c,n；DESC NULLS FIRST 为 n,c,a,b。同一 H 不同 Data/RecordBatch 切分、checkpoint 中途恢复及独立有限执行均同此序。offset1/k1 选 b；多键按逐级 comparator。无 sequence 时相等键只验有序及允许成员，不断言 a/b；strict unordered top-k 也须拒绝未证明选择。实际重复/null sequence 跨 Data 和 envelope 后一 RecordBatch 失败，旧 H/counters/capture 不推进；不从 metadata.sequence 猜行次序。
3. **边界/空：** k0、k>N、offset=N/>N、纯 offset、N0/typed empty/仅空 RecordBatch、NULL-only 键均按公式且 stream 恰1个 final。k=-1/bool/float/2^32、offset同类、k+offset=2^32、max_rows=2^32 在读取前拒绝；上界2^32-1 可声明，不为测试物化巨大表。by/sequence/类型/Exclude/SQL所有越界均有路径与约束。
4. **window/late：** `[0,10)` 先接纳 (ts,x,seq)=(0,3,10),(9,1,11)；W9 无 final，W10 final x1。后 Data 含 (0,0,12),(9,-1,13),(10,5,14),(NULL,6,15)：Error 整 Data 失败；Drop 不再输出且 accepted_rows仍2，metrics=2/1/Some(10)。跨两 RecordBatch 仍affected_batches1，max_rows2 不挡 late drop。负时间/边界/NULL 时间及 EOF 替代闭合另用小表；普通空批/Barrier/Idle/错误不伪造 close。
5. **预算/ownership/C13：** top1 仍计全 H，2行输入在 max_rows1 失败。复用 DISTINCT 的 foreign slice 反例，验证排序也复制全部保留列/sequence/schema/metadata，raw owner 不进入异步工作；只声明 owned-only，不断言 active Source/caller 的父分配已释放。节点 normalization/candidate、sequence check、sort indices/workspace、compact output、held capture/restore 的代表峰值分别做 B 成功/B-1 失败；预算失败在目标分配/提交前发生，无 hidden heap/no-copy 旁路。截取1行的输出不间接保活整 H；取消/emit失败单节点状态不部分提交，外部 delivery 按原协议。
6. **L01–L03/A06：** 同 job 两批跨批累积、空批不 reset；同一个已编译 finite plan、同 Runtime、暖缓存无 reset 依次完整 H、[9]、typed empty 得对应[1]、[9]、空；在失败/取消后复用仍无旧候选。普通新 job fresh。未闭合 checkpoint→恢复→剩余 Data→EOF 等于未中断 H；closed恢复不重发，Drop恢复后 counters 接续不重复加。改变方向/NULL/k/offset/sequence/budget/late/cut/engine/layout或坏行数/digest在安装/Source open前拒绝。
7. **四单元/A08/A09：** 各自真实公开构造、schema/analyze/两生命周期/导出再加载，不仅比较私有 binding。SQL宏和DF都导出同义 bounded_order 配置；新 capability/layout/r10、prepared cache隔离、严格gate与真实绑定gate一致。DF→row-local SQL→order 及 SQL-order→DF/SQL 后处理、共享fan-out/独立输出不丢节点；只有已证保序输出才通过declared请求。Unknown/Unbounded、错误time/sequence来源、snapshot/嵌套/不支持SQL，在暖缓存/reload/显式Runner/convenience上用读取行/iterator/open/job计数0证明拒绝。

选择 by + 小配置对象，是为了使普通调用短、必需历史/预算显式，并把可选 offset/sequence/late 放最后。选择输入显式 sequence 且验证唯一，是为了不把排队调度、DISTINCT 无序结果或 temporal identity 外推成全 H 稳定行次序。保留 full H 比立即宣称 heap/O(k) 更容易与现有 owned normalization、失败事务和 capture 边界一致；未来优化若改变计量/身份须单独版本化。拒绝无界默认 global SQL、budget overflow 当 EOF、k0早发、只限最终k行、top-k默认为descending、依物理sort稳定性授予ties、先分配后记账及绕过node预检的低层导入。

**真正未决问题：本片无待用户选择或阻塞公共合同的问题。** 实现者仍须验证具体 native sort/take/sequence-validation 分配路径能按所列有限上界预付，及所选后处理路径是否真实保序；这些是实现证据义务，不是放宽预算/排序的授权。若不能满足，带具体 allocation/lifetime/顺序反例回报协调器，准确拒绝该路径，不能换 Python 算法或无声降级。更宽 dtype、可认证零拷贝、O(k)候选/spill、多个finalizer/多源cut和一般analytic均明确留后续，不列为本片未解决需求。

实现范围为四单元公开配置/声明/AST/schema seam、共享native节点/内核、强制路径/绑定准入、owned budget/late事务、capture/restore/inventory、analysis r10/cache/export/schema/stub，以及必要docs/examples和上述定向验证。由协调器接续 implementer → reviewer 独立主审 → 修正并满足门禁后合并；本轮不扩派 spec/critic/tester/性能/独立文档阶段。DAL-520 继续 in_progress，API note 完成不等于本 issue 的 PR 已合并。

本文只做必要源码核对及文档自检：指定文件独占修改、链接/源码位置、围栏/表格/参数顺序、Python种子语法、JSON身份与边界/事务一致性；未运行产品测试、native build、性能或CI，不声称新产品行为已验证。本轮不改产品、提交/推送、建PR或扩派。

## 局部修订 v2：SortWindow 与参数数量门禁（2026-11-13）

**本节与上方已经更新的签名共同取代初版 closed_window 调用形状；只重组宿主参数。** 固定参考为 PR #606 HEAD `ccda5564809b36e222f4b084ec39cdc20d1b3116`。reviewer 按该提交审查初版 note，本工作副本的修订不回写其固定审查输入。

必要核对：`python/calc_flow/symbolic/order.py:72` 的 classmethod 有7个非 cls 参数：event_time、start_micros、end_micros、state_budget、offset、sequence_by、late_policy。`scripts/verify_complexity_gates.py:75` 显式选择包含 PLR0913 的复杂度规则，并与原 baseline 比较；`pyproject.toml:90` 的普通 Ruff select 并未涵盖该规则，所以普通 Ruff 通过不代表此门禁通过。本轮以该 checkout 的 `target/venv/bin/python -m ruff check --select PLR0913 --output-format json python/calc_flow/symbolic/order.py` 定向确认唯一 finding 为 `Too many arguments in function definition (7 > 5)`，与 analysis 的已有复现一致；未重跑全仓复杂度扫描或产品测试。

Rust 当前 `crates/calc-flow/src/operator/order/spec.rs:184` 的四参数构造器与 `sql/order.rs:221` 的调用、Python 窗口测试 `python/tests/test_bounded_global_ordering.py:182`、公开指南 `docs/bounded-global-ordering.md:50` 均仍是初版形状。已核对 Rust `operator/window.rs:117` 的 WindowSpec 是带 geometry/group_by/aggregates 的窗口聚合配置，Python `symbolic/windows.py` 的 RowFrame/DurationFrame/EventTimeBucket 也不是一个任意 `[start,end)` 区间；复用它们会混入不适用的语义。SortHistory 是严格 wire enum，直接拿它作为宿主参数则会允许传 BoundedSource 给 closed_window；不选这种需再拒绝错误 variant 的入口。

因此新增**排序专用、具名、不可变的 SortWindow**，只聚合时间列及两个边界。在根包/所属模块导出，实际持久化仍展平为原 SortHistory::ClosedWindow。Rust 消费一个 owned SortWindow 与预算；Python 的 window 保留唯一必选位置参数，预算继续是必选 keyword。两端精确签名以上方为准，没有并行旧签名、兼容解析分支、deprecated alias、动态参数或隐藏参数。

Python 入口的5个非 cls 参数现在为 window、state_budget、offset、sequence_by、late_policy；SortWindow 构造器有3个非 self 参数。Rust 的 SortWindow::new 为3参数，SortSpec::closed_window 为2参数。可选 offset/sequence/late 的含义、默认值及必选预算均完整保留；不是通过删除配置选项来降低计数。Python 仍采用现有兼容 `_compat.dataclass(frozen=True, slots=True, init=False)` 加显式构造器，不使用 functools、`__signature__`、`*args`/`**kwargs` 或手写 Stub 隐藏真实调用形状。

### 调用及迁移映射

初版 Python `SortSpec.closed_window("ts", start_micros=0, end_micros=10, state_budget=budget, offset=1, sequence_by=["seq"], late_policy="drop")` 只将前三个窗口参数聚合为下面的 SortWindow；其余调用参数原样保留：

```python
import calc_flow as cf

window = cf.SortWindow("ts", start_micros=0, end_micros=10)
spec = cf.SortSpec.closed_window(
    window,
    state_budget=cf.StateBudget(1000, 8 << 20),
    offset=1,
    sequence_by=["seq"],
    late_policy="drop",
)
```

典型默认调用可内联为 `cf.SortSpec.closed_window(cf.SortWindow("ts", start_micros=0, end_micros=10), state_budget=budget)`；没有新增执行/关闭步骤。复用同一 window 构造两个 spec 不共享运行状态。bounded_source、SortKey、sort/top_k 的签名及主示例无需迁移。

初版 Rust `SortSpec::closed_window("ts", start, end, budget)?` 替换为 `SortSpec::closed_window(SortWindow::new("ts", start, end)?, budget)?`，后续不可变配置链保持：

```rust
use calc_flow::{EventTime, LatePolicySpec, Result, SortSpec, SortWindow, StateBudget};

fn selected_window() -> Result<SortSpec> {
    let window = SortWindow::new(
        "ts", EventTime::from_micros(0), EventTime::from_micros(10),
    )?;
    SortSpec::closed_window(window, StateBudget::new(1000, 8 << 20)?)?
        .with_offset(1)?
        .with_sequence_by(vec!["seq".into()])?
        .with_late_policy(LatePolicySpec::Drop { metrics_version: 1 })
}
```

这里的两种 spec 都仍对应原 SQL `CF_ORDER_WINDOW('a', 'ts', 0, 10, 1000, 8388608, 'drop', ARRAY['seq'])`，OFFSET 1 继续在 SQL clause 中表达。Rust SQL parser/严格 spec 反序列化只需先把原三个窗口字段经 SortWindow::new 验证，再调用新构造器；Python 也从值对象读取同三个字段。SQL 方言、参数数目/默认、native 描述 seam 和 PyO3 接收的 spec wire 不变，不把 SortWindow 作为新的 SQL 参数或原生运行对象。

### 验证、身份与实施范围

SortWindow 保存 copied/owned 的时间列名和精确 i64 微秒边界；构造时检查非空、start<end、Python bool/float/越界拒绝，允许负时间。这沿用原窗口值域，不检查数据 schema 或宣称已水位闭合。错误采用原层次：Python 类型错误 TypeError、值域错误 ValueError，Rust InvalidArgument；新值对象参数准确定位 `SortWindow.event_time|start_micros|end_micros`（Rust 对应 `sort_window.event_time_column|start|end`），说明约束。SortSpec.closed_window 传非 SortWindow 定位 window 并给出 `window must be a SortWindow`；原生项目/SQL 的错误继续定位原 `sort.spec.history` / `.sql.order` 路径，不暴露宿主 wrapper。预算、late policy 和 schema/时间来源的验证仍归原各自边界。

相同参数构成的完整 spec 必须生成**相同 canonical wire/config 字节**：原 history.kind/event_time_column/start_micros/end_micros，以及 state_budget/offset/ties/late_policy/ingress_policy 均不变。SortWindow 不是 graph 节点，不进入 declaration bytes、cache key 或 checkpoint；不增加 JSON schema 字段、project/state/accounting/layout 版本、capability 或 analysis revision。不是拿对象的 repr/identity/hash 替代原字段。此要求同时覆盖 SQL 与 DF，保证纯签名重组不导致相同语义换恢复身份。初版接口尚未发布，直接替换调用，无须保留旧调用方式或引入运行期迁移；已有合法项目/checkpoint 继续按原严格合同处理。

由实现者更新 SortWindow 的 Rust/Python 导出/公开类型说明、两个构造器、Rust SQL 与 spec 反序列化调用，以及直接受影响的测试、指南、rustdoc/示例。已定位的调用还包括 `crates/calc-flow/tests/order_closure.rs:64,111` 和 `order_analysis.rs:150`；只做机械迁移，不重新决定闭合/迟到/sequence/类型/预算/排序/输出/恢复规则。Python `_native.pyi` 仅在现有 seam 类型确受影响时同步，不为纯 Python 值对象添加多余 native 方法；普通 package 公共类型须可发现且正确标注。

所需直接验证：新签名（包括实际 `__init__`/classmethod）用现有 Ruff 复杂度规则通过；SortWindow 的 frozen/参数类型/边界和新嵌套调用，Error/Drop 各一个原 fixture；相同完整 spec 的 canonical JSON/native config/导出字节与初版独立预期一致；对应 Rust 新构造/SQL deserialize 调用编译。随后实现者运行原 `scripts/verify_complexity_gates.py` 确认真实门禁消除此增量。无需重跑已匹配提交的全部排序/恢复矩阵；若实现修改超出参数重组，再补直接影响的验证。

本轮仅修改此 note，文档自检包括所有 Python 围栏 AST、替代签名的真实参数计数、以仓库配置和固定 Ruff 对签名片段执行 stdin 复杂度检查、迁移字段/默认对应、相对链接/空白和 diff 范围。**片段检查通过不代表产品门禁已修复**；工作副本产品仍是初版，须由实现者应用并复验。未修改产品/门禁阈值/baseline/suppression，未做全仓扫描、native build、提交/推送或扩派。

本项局部决定已定，无待用户选择或新增未决问题；交协调器与并行固定提交 review 的结果汇总后，安排实现者在同一 PR #606 应用。DAL-520 保持 in_progress。
