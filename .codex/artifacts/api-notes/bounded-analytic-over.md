# Bounded Analytic OVER — API Note

状态：DAL-521 / S3-3 / G3，2026-11-13。**本片最小公共合同已冻结，可交实现；本文新增接口尚未实现。** 精确源码基线为已合并 PR #606 的 `35ff239f7b054b640d120f02925c0113159a8cd3`，复用分支 `agent/dal-orchestrator/4d5e854a24b7`，开始时 HEAD 相同、工作树干净。触发线程为 `01a1d43a-cec1-70ee-9b94-5d463b5fcc1f`。本轮只写本文和完成回复，不提交、推送、建 PR、改产品或跑产品测试；DAL-521 保持 `in_progress`，交协调器接续。

依据为当前 issue、父 DAL-498 的四单元及生命周期讨论、开发计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`（复用先前下载的 v2 副本），以及 [bounded DISTINCT](bounded-global-distinct.md)、[bounded ordering v2](bounded-global-ordering.md)。沿用 [query history](query-history-portability.md)、[first-class frontends](first-class-relational-frontends.md) 的比较与错误分层；旧 note 的候选项不取代本 SHA 的源码事实。角色按 `.codex/agents/cf-api-designer.toml`，词汇按 `docs/introduction.md`，文档按 `.agents/skills/code-style/SKILL.md`。实际边界是 Rust core → PyO3 → Python；仓库没有 `dal-public/src`、C++/Excel、methodology 或 `.codex/references/component-boundaries.md`，不补造这些组件。

## Audiences and adopted minimum

Rust `calc-flow` 与 Python `calc-flow-python` 用户都可用 DataFrame 和 SQL 声明同一有界 analytic 关系，分析 schema/历史/顺序，导出项目，并选择 batch 或 stream 生命周期。Python 不实现排序、分区、frame、历史或求和内核。

本片采用 **ROW_NUMBER、固定 offset=1 / NULL default / RESPECT NULLS 的 LAG，以及整数 SUM 的两个显式 ROWS frame**：`UNBOUNDED PRECEDING ... CURRENT ROW` 和 `N PRECEDING ... CURRENT ROW`。全部表达式共享一个 partition/order；允许 SUM 使用不同的获准 frame。H 为一个 Bounded Source 的完整 cut，或一个明确固定 UTC `[start,end)` 窗口；只在真实闭合时发一次完整 final。stream 保存完整 H，batch 只用本次完整输入。ROWS frame 是在这个 H 内计算函数的行范围，**不是 H 的保留/淘汰规则**；小 N 不承诺 O(N) 状态。

不加入时间 RANGE、GROUPS、FOLLOWING、EXCLUDE、无界在线 OVER、增量修正、replacement、多个时间窗口、不同 partition/order 混合、任意 SQL 窗口函数、spill 或 batch 中途恢复。普通 SQL OVER、旧 SQL 默认、mode/cache/checkpoint 身份及旧快照链7、显式 latest6/拒绝保持。

## Surface today and necessary kernel choice

下列是本基线的静态证据，不是新运行验证。

| Source                                                                                                        | Fact and implication                                                                                                                                                            |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/src/lib.rs:119`、`dataframe.rs:78,415`、`sql.rs:73`                                         | 已有 SortSpec/SortWindow/SortKey、sort/top_k 和显式宏到共享关系的通路；没有本片的 over 关系或公开 analytic 配置。                                                               |
| `crates/calc-flow/src/expression.rs:224,268`、`operator/sql.rs:1417,1446`                                     | OVER 不是 row-local；window SUM 不被视为 non-window aggregate。普通单 alias OVER 经当前批 SQL 路径执行；watermark/EOF hook 不作本片 finalization。不能拿旧 OVER 当跨批 H 实现。 |
| `crates/calc-flow/src/query_analysis/sql.rs:130,182`                                                          | ROW_NUMBER 形状只有旧 Selection/per-batch 证明，其余窗口形状可能 unknown；没有显式有界 analytic 证明。                                                                          |
| `crates/calc-flow/src/operator/order/work.rs:29,39,89`、`order/sequence.rs:61,140`                            | native Arrow concat/lexsort/take、自有结果归一化和整 H sequence 唯一性校验可复用；目前只排序/截取，没有 partition/frame/value evaluator。                                       |
| `crates/calc-flow/src/operator/order/mod.rs:355`、`distinct/owned.rs:135`、`operator/state_budget.rs:9`       | 有受支持标量的 owned normalization、预留和排序 workspace 上界；原排序额度不包含新 analytic 列、frame state 或这些列的输出副本，不能直接照抄额度。                               |
| `crates/calc-flow/src/operator/order/checkpoint.rs:19,29`                                                     | 已有严格 control、config/schema/engine identity、captured H、closed/sequence、水位及策略布局模式；这是可复用模式，不是可以共用 state_kind 的理由。                              |
| `crates/calc-flow/src/pipeline/distinct.rs:1`、`runtime/streaming/job/order.rs:14`                            | 单源 row-local 路径和真实 Source cut/progress 校验已有强制入口；order 的 progress 模块还负责 source proof fingerprint，必须扩到新节点，不能仅靠 Python 或 portability opt-in。  |
| `python/calc_flow/symbolic/order.py:18,90,185`、`_native.pyi:19`、`crates/calc-flow-python/src/config.rs:944` | immutable 宿主值、共同 native AST/schema seam、声明/lowering/cache 是现有模式；两端 SQL 的新方言仍由 Rust 解析。                                                                |
| `crates/calc-flow/src/query_analysis.rs:559`、`python/calc_flow/symbolic/query_analysis.py:668`               | 当前报告为 schema2/revision10；新能力不能继续把 r10 当作 analytic 准入证明。                                                                                                    |
| `Cargo.lock:1368` 与锁定依赖 `datafusion-functions-aggregate-55.0.0/src/sum.rs:316,551,622`                   | 现有整数 SUM accumulator 用 `add_wrapping`/`wrapping_add`；不能隐含承诺新函数遇到越界会失败。已读本机精确锁定版本源码，未采用 55.2 或 latest 推断。                             |

**最小选择：** 抽取复用当前 native Arrow 排序/NULL comparator、owned history、sequence 校验、CPU work/cancellation 和预算设施；为这三个函数增加一个共同的 native partition/ROWS evaluator，四单元和两个生命周期都调用它。不改普通 DataFusion SUM。新 SUM 用足够宽的整数累加并检查每个输出值的 Int64/UInt64 范围；这是新节点的明确语义。DataFusion 55 仍是表/SQL 规划与 Arrow 类型的权威环境，新 SQL AST 只降成此节点，不重写成普通 per-batch SqlOperator。若直接复用 DataFusion window evaluator，仍须独立实现相同溢出检查并证明其所有新分配可预付；本片选择上述较小、可预算的 native evaluator，不留两条执行分支待运行时猜选。

## Frozen DataFrame surface

新增 crate-root 导出 `OverOrder`、`OverWindow`、`OverSpec`、`RowsFrame`、`AnalyticExpr`。全部是 owned、data-only、不可变使用的配置，没有 Source、Batch、callbacks 或 runtime。私有字段、严格 serde/JsonSchema；非法构造/严格导入都拒绝。复用已导出的 SortKey、SortDirection、NullPlacement、StateBudget、LatePolicySpec，不增加同义 comparator 或 late enum。

```rust
pub struct OverOrder { /* private: partition_by, order_by, sequence_by */ }
impl OverOrder {
    pub fn new(
        partition_by: Vec<String>, order_by: Vec<SortKey>, sequence_by: Vec<String>,
    ) -> Result<Self>;
    pub fn partition_by(&self) -> &[String];
    pub fn order_by(&self) -> &[SortKey];
    pub fn sequence_by(&self) -> &[String];
}
pub struct OverWindow { /* private: event_time_column, start, end */ }
impl OverWindow {
    pub fn new(event_time_column: &str, start: EventTime, end: EventTime) -> Result<Self>;
    pub fn event_time_column(&self) -> &str;
    pub fn start(&self) -> EventTime;
    pub fn end(&self) -> EventTime;
}
pub struct OverSpec { /* private: history, order, budget, late_policy, ingress_policy */ }
impl OverSpec {
    pub fn bounded_source(order: OverOrder, state_budget: StateBudget) -> Result<Self>;
    pub fn closed_window(
        window: OverWindow, order: OverOrder, state_budget: StateBudget,
    ) -> Result<Self>;
    pub fn with_late_policy(self, policy: LatePolicySpec) -> Result<Self>;
    pub fn window(&self) -> Option<&OverWindow>;
    pub fn order(&self) -> &OverOrder;
    pub fn state_budget(&self) -> StateBudget;
    pub fn late_policy(&self) -> LatePolicySpec;
}
pub struct RowsFrame { /* private: cumulative or preceding(u32) */ }
impl RowsFrame {
    pub fn cumulative() -> Self;
    pub fn preceding(rows: u64) -> Result<Self>;
}
pub struct AnalyticExpr { /* private: function, operand/frame if any, output name */ }
impl AnalyticExpr {
    pub fn row_number(name: &str) -> Result<Self>;
    pub fn lag(column: &str, name: &str) -> Result<Self>;
    pub fn sum(column: &str, frame: RowsFrame, name: &str) -> Result<Self>;
}
impl TableExpr {
    pub fn over(&self, expressions: Vec<AnalyticExpr>, spec: OverSpec) -> Result<Self>;
}
```

OverOrder 把相关排序声明放在一起，使两个历史构造器都短且必选项明确；没有可执行的 Unspecified-ties 模式或 Default。OverWindow 与既有 SortWindow 是各自特性的宿主配置，复用私有区间验证/扁平 history wire 规则；不重命名/破坏已经合并的 SortWindow，也不让 OVER 接一个包含 offset/fetch/unspecified ties 的 SortSpec。OverWindow 不关闭窗口，不是 graph 节点或 runtime 对象。`window() == None` 仅表示 bounded_source，完整 history 在严格项目配置中明确编码。

Python 从根包和 symbolic 导出同名配置，使用现有 frozen/slots 兼容模型并复制调用者序列。以下为新增公开签名，不是当前可执行代码；与现有 TableExpr/Program/StateBudget 配合，不另建执行入口。

```python
class OverOrder:
    def __init__(
        self, partition_by: Sequence[str], order_by: Sequence[SortKey], /,
        *, sequence_by: Sequence[str],
    ) -> None: ...

class OverWindow:
    def __init__(
        self, event_time: str, /, *, start_micros: int, end_micros: int,
    ) -> None: ...

class OverSpec:
    @classmethod
    def bounded_source(
        cls, order: OverOrder, /, *, state_budget: StateBudget,
    ) -> OverSpec: ...

    @classmethod
    def closed_window(
        cls, window: OverWindow, order: OverOrder, /, *, state_budget: StateBudget,
        late_policy: Literal["error", "drop"] = "error",
    ) -> OverSpec: ...

class RowsFrame:
    @classmethod
    def cumulative(cls) -> RowsFrame: ...

    @classmethod
    def preceding(cls, rows: int, /) -> RowsFrame: ...

class AnalyticExpr:
    @classmethod
    def row_number(cls, name: str, /) -> AnalyticExpr: ...

    @classmethod
    def lag(cls, column: str, /, *, name: str) -> AnalyticExpr: ...

    @classmethod
    def sum(
        cls, column: str, /, *, frame: RowsFrame, name: str,
    ) -> AnalyticExpr: ...

class TableExpr:
    def over(
        self, expressions: Sequence[AnalyticExpr], /, *, spec: OverSpec,
    ) -> TableExpr: ...
```

所有宿主方法必选参数在前、可选项在后；不接受 dict/string wire 代替配置、不接受 str/bytes 冒充序列。Python 配置提供相应只读值，不以对象 identity/repr 生成声明。`RowsFrame.preceding` 接确切 int，范围 `0..=4294967295`；0 是只含当前行。max_rows 必须 `1..=4294967295`（复用 UInt32 sort 索引边界），max_bytes 沿 StateBudget 的正 JSON-safe 上界。时间是 signed i64 微秒且 start<end，允许负值。bool/float 不充当整数；late_policy 必须确切小写 error/drop。Rust builder 消费并返回新配置，Python 配置 frozen，原关系/配置/序列不变。

表达式列表非空，每个输出名非空、唯一且不覆盖任何输入列。输出为全部输入列，后接按声明次序排列的新列；所有表达式只读输入列，不能引用同一 over 中其他表达式的 alias。只取新列或重命名请用后续 select；没有自动覆盖、隐式 cast 或跨 lineage ColumnExpr。

## Partition, order, ties, types and frames

partition_by 可以为空，表示整 H 一个 partition；非空时是有序的直接列名列表。order_by 和 sequence_by 都必须非空。各列表内部无重复；partition/order/sequence 三者列名互不重叠，全部存在于当前输入。这里拒绝冗余重叠是新特性的明确窄域，不改变普通 SQL。SortKey 的 NULL 规则固定为 First/Last，Exclude 拒绝；Python/SQL 省略 NULL placement 时立即规范为 ASC last、DESC first，wire 必写明确值，执行不依赖 DataFusion 默认。

NULL partition key 按相同位置的 NULL 相等分组，复合 tuple 逐项比较；不是丢掉 NULL partition。每个 partition 内依 order_by 比较，再依 **sequence_by tuple 升序**打破 ties。sequence 不受主键方向影响，不自动取 InputOptions.sequence_by、到达次序、Batch.metadata.sequence、RecordBatch offset 或 DataFusion 分区序号。

sequence 字段必须是原输入原样列或经已证明 row-local SQL/DF 纯重命名而来，schema nonnullable，类型为 signed/unsigned integer 或 Utf8/LargeUtf8。调用者/Source 保证同一逻辑 cut 的 replay/两 mode 中同一 tuple 对应同一行；节点在每个 Data 的完整 candidate H 中检查非 NULL、**整 H 全局唯一**，包括不同 partition。选择全局唯一是复用现有 order 校验的最小约束，暂不扩成“只需 partition 内唯一”。值无须连续、递增。唯一性是执行期义务，analyze 只证明字段/配置，不虚报已验数据。候选中任何重复（包括跨批）导致整个 Data 失败，不泄露 tuple 值；窗口外/NULL-time/已 Drop 行不进入检查。

输出采用明确的 canonical 行序：partition_by 各列 ASC NULLS LAST，然后 order_by，然后 sequence_by ASC。partition_by 为空时直接后两项。按该序连接整个 final 的 RecordBatch；不恢复到输入到达序。输出产生新 row lineage，无继承的 temporal identity；QueryOrderingFacts 可报告上述 keys/total。排序 ties 因 sequence 成为总序，ROW_NUMBER/LAG/SUM 的行值及输出顺序在相同 H 下确定。Source replay/同 H 义务仍独立，不因此获得 durable replay 或 ExactlyOnce。

保留列、partition/order 列及 LAG operand 的最小 dtype 域与已合并排序相同：Boolean，Int8/16/32/64，UInt8/16/32/64，Utf8/LargeUtf8，Date32/64，timestamp[us] naive/UTC，timestamp[ms] naive。均可 nullable（sequence 除外），保留精确 dtype。窗口时间只接受 timestamp[ms] naive、timestamp[us] naive/UTC，checked 转换成 EventTime，要求原时间坐标的来源证明。字符串按 native UTF-8 二进制字典序，不作 locale/case/Unicode normalization。float/Decimal/binary/nested/dictionary/extension/其他时间类型在本片连 retained payload 都拒绝，可先投影掉；不宣称底层引擎不支持它们。

设 partition 按总序排为 `r[0..m)`，以下区间均含当前行；H 的事件时间边界不随 ORDER BY 方向或 ROWS 改变。

| Expression                | Exact semantics                                                                                                             | Output field                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| row_number                | 行 i 为 i+1，每个 partition 重置；不读 operand，不使用 frame。                                                              | UInt64，nullable=false。                                           |
| lag(column)               | i=0 为 NULL；否则读取 r[i-1] 的原值，前行 NULL 就返回 NULL。offset 恒1，default 恒 NULL，RESPECT NULLS，不看 SUM 的 frame。 | 与 operand 精确同 dtype，nullable=true。                           |
| sum(column, cumulative)   | 对行区间 [0,i] 的非 NULL 值求和。                                                                                           | signed integer → Int64，unsigned integer → UInt64；nullable=true。 |
| sum(column, preceding(N)) | 对 [max(0,i-N),i] 的非 NULL 值求和；N=0 仅当前行，N 大于 partition 长度时截到 partition 起点。                              | 同上。                                                             |

SUM 只接整数，无隐式 bool/float cast。空/全 NULL frame 为 NULL，含非 NULL 的0为0。用 signed i128 / unsigned u128 或等价精确宽累加：最多 u32 行，64-bit 输入的数学总和在该宽度内有界；**每个要产出的 frame 和都须落在输出 dtype 范围内**，否则 finalization 以 arithmetic_overflow 失败，不 wrap、不改成 NULL、不因为后续抵消就接受已越界的输出。对 bounded frame 不因减去旧值的窄中间步骤制造假 overflow；宽整数/nonnull count 决定结果。不是复用 rolling 浮点 SUM，也不把 DataFusion 当前 wrapping 行为作为默认。

例如 `ROWS 0` 时 LAG 仍可读前行：它不受 SUM frame 限制。ROW_NUMBER/LAG 不接受 frame 参数；SQL 显式给它们任何 frame 一律拒绝，而非悄悄忽略。SUM 必须指定两个获准 frame 之一，缺 frame 拒绝；不默认采用 peer-based RANGE。不同获准 SUM frame 可以同节点，partition/order 不同则必须拒绝本片 SQL；不能偷偷拆出多状态节点或借 alias 串联。

## Four units and the explicit SQL dialect

继续使用现有 Rust `sql(query, BTreeMap<String, TableExpr>)`、Python `cf.sql(query, **tables)` 和关系 `.sql(query)`（绑定名 input）。新增的宏不是 DataFusion 内建函数：

```sql
SELECT *,
  ROW_NUMBER() OVER (PARTITION BY p ORDER BY score ASC NULLS LAST, seq ASC NULLS LAST) AS rn,
  LAG(v, 1) OVER (PARTITION BY p ORDER BY score ASC NULLS LAST, seq ASC NULLS LAST) AS prev,
  SUM(v) OVER (PARTITION BY p ORDER BY score ASC NULLS LAST, seq ASC NULLS LAST
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total,
  SUM(v) OVER (PARTITION BY p ORDER BY score ASC NULLS LAST, seq ASC NULLS LAST
               ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS last_two
FROM CF_OVER_BOUNDED('a', 1000, 8388608, ARRAY['seq'])
```

WINDOW 替换 FROM 为以下任一种，其他子句相同：

```sql
FROM CF_OVER_WINDOW('a', 'ts', 0, 10, 1000, 8388608, ARRAY['seq'])
FROM CF_OVER_WINDOW('a', 'ts', 0, 10, 1000, 8388608, ARRAY['seq'], 'drop')
```

BOUNDED **恰好4项** `(alias, max_rows, max_bytes, sequence_array)`；WINDOW **7或8项** `(alias, time_column, start_micros, end_micros, max_rows, max_bytes, sequence_array[, late_policy])`，末项默认 error。新宏中 sequence 必选，所以排在可选 late policy 前；不要套用 CF_ORDER_WINDOW 的可选参数位置。宿主入口仍是短 sql(query,tables)，不是八个宿主位置参数。

宏名大小写不敏感、不带 schema；alias/time/sequence 是普通单引号 string literal，解码后精确匹配。整数仅十进制 literal；start/end 可负，N/budget 不可负。sequence_array 是非空、无重复的 `ARRAY['column', ...]` literal，不能是动态表达式/NULL/参数。tables 恰好包含所引用一次的 alias。

固定形状是单个顶层 SELECT、第一项恰为无修饰 `*`，后续每项恰为一个带独立非空 alias 的获准 analytic 调用，单个无 AS 的宏 FROM。所有 OVER 的 PARTITION 列列表和完整 ORDER comparator 列表必须规范后相同；允许全部省略 PARTITION（空 partition_by），不允许省略 ORDER。每个 ORDER 的末尾必须恰为宏 sequence_array，逐列 ASC NULLS LAST；前面至少一个 primary SortKey，与 partition/sequence 不重叠。SQL 默认 ASC/NULL placement 在 AST 中立即规范，末尾 sequence 的 DESC/NULLS FIRST 拒绝。序列写在 SQL ORDER 中，不向没有声明这些键的 OVER 偷加 hidden tie breaker。

函数形状：ROW_NUMBER()；LAG(直接列,1)，不接受第三项、其他 offset 或 null-treatment 子句；SUM(直接整数列) 配完整 `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` 或 `ROWS BETWEEN N PRECEDING AND CURRENT ROW`。不接受 SUM DISTINCT/FILTER、frame shorthand、RANGE/GROUPS/FOLLOWING/EXCLUDE、WINDOW 命名、函数 nesting 或 alias 输入。ROW_NUMBER/LAG 的 frame 必须缺省，但其不使用 frame 的含义由本合同决定。

不接受其他 projection、scalar 表达式、WHERE、DISTINCT、GROUP/HAVING/QUALIFY、外层 ORDER/LIMIT/OFFSET/FETCH、CTE/子查询、JOIN/UNION、多 alias、FROM AS、UDF/provider/volatile。投影/过滤/结果截取在节点前后分别声明 row-local 关系；不要为小型 SQL 子集默开一般窗口 planner。检测到任意 CF_OVER_* 后，名字、位置、参数、函数或混合形状有错必须定位并拒绝，**不得 fallback 普通 SQL**。没有宏的原 OVER 保持旧解释，严格 portability 请求仍可按旧事实拒绝。

Rust AST authority 输出 data-only `alias/spec/expressions`，DF 和 SQL 都降成一个共同 `bounded_over` relation；增加与既有 `_order_query/_order_schema` 相同模式的私有 `_over_query(query, aliases)` 和 `_over_schema(spec, expressions, schema)` seam/stub。Python 不另写 parser/schema/求和逻辑。这些私有 seam 不是一等交付本身。

四单元的合法声明种子（t 为已声明精确 schema 的输入，q 为上方完整 SELECT）：

| Unit               | Legal public declaration                                                           | Required rejection example                                                                                                                   |
|--------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Rust / DataFrame   | `t.over(vec![AnalyticExpr::row_number("rn")?], spec)?`，spec 见下方 Rust 种子。    | `OverOrder::new(vec![], by, vec![])?` → over.spec.order.sequence_by / ordering_required，必须非空；不进入 Source。                           |
| Rust / SQL         | `sql(q, BTreeMap::from([("a".into(), t.clone())]))?`。                             | 将 q 的 SUM frame 删除 → sql.over.expressions[i].frame / capability_mismatch，要求显式获准 ROWS；不能按 RANGE 或 per-batch 运行。            |
| Python / DataFrame | `t.over([cf.AnalyticExpr.row_number("rn")], spec=spec)`，spec 见下方 Python 种子。 | `cf.RowsFrame.preceding(True)` → TypeError，over.frame.preceding 要求确切 int；`-1` → ValueError，要求 0..=4294967295。                      |
| Python / SQL       | `cf.sql(q, a=t)`。                                                                 | 把某个 OVER 改为 ORDER BY v,seq（其他仍 score,seq）→ sql.over.expressions[i].order / capability_mismatch，必须共享 partition/order；读行=0。 |

其余最小拒绝例：`LAG(v,2)`、`LAG(v,1,0)`、`LAG(v,1) IGNORE NULLS`、`ROW_NUMBER() ... ROWS ...`、`SUM(v) ... RANGE ...`、`RANK()`、`SUM(v) ... 1 FOLLOWING`、`SELECT DISTINCT * FROM CF_OVER_*`、嵌套宏均拒绝具体函数/条款；SQL 或 native strict import 也不能绕过 DataFrame 的 dtype/alias/schema/sequence 校验。传 nullable sequence 列在 schema 阶段拒绝；actual duplicate 在执行候选阶段拒绝，不能伪称分析期已验值。

## Canonical declaration and schema

新项目 operator kind=`bounded_over`，配置是 `expressions` 和 `spec`，运行配置/state_kind=`bounded_over_v1`；不是 SqlOperator.query 的 hidden mode。下例为规范配置片段（同义的四单元必须相同，不要求不同 SQL 文本/语言的整个声明 fingerprint 相同）：

```json
{
  "expressions": [
    {"kind": "row_number", "name": "rn"},
    {"kind": "lag", "column": "v", "name": "prev"},
    {"kind": "sum", "column": "v", "frame": {"kind": "cumulative"}, "name": "total"},
    {"kind": "sum", "column": "v", "frame": {"kind": "preceding", "rows": 1}, "name": "last_two"}
  ],
  "spec": {
    "ingress_policy": "copy_visible_v2",
    "history": {"kind": "bounded_source"},
    "order": {
      "partition_by": ["p"],
      "order_by": [{"column": "score", "direction": "ascending", "nulls": "last"}],
      "sequence_by": ["seq"]
    },
    "state_budget": {"max_rows": 1000, "max_bytes": 8388608},
    "late_policy": {"kind": "error", "scope": "envelope"}
  }
}
```

window 的 history 是严格 `{"kind":"closed_window","event_time_column":"ts","start_micros":0,"end_micros":10}`；Drop 是 `{"kind":"drop","metrics_version":1}`。新节点默认 Error 也显式写入，不移植 DISTINCT 的历史 omission 规则。RowsFrame 两种 wire 的末端都定义为 CURRENT ROW，没有任意 start/end 或未知字段。LAG wire 没有 offset/default/frame 字段，添加这些字段即 strict import 拒绝，不能解释成尚未支持的能力。

schema 是输入原列名/列序/dtype/nullable 后接新字段（上表 dtype/nullability），零行也相同。不向数据加 partition/window/rank 隐藏列。沿现有 Port/前端 metadata 规范化，不修改 caller metadata，也不新增任意输出 metadata 保留承诺。schema planner 必须在没有 rows 的情况下给出同一结果并验证所有列，包括只保留不计算的 payload。

## H, real closure, late input and lifecycle

复用现有 SourceBoundedness/水位/EOF 协议，不提供新的 public close/finalize/watermark 注入 API。bounded_source 实际绑定必须声明 Bounded 固定有限 cut，恢复续同一 cut；Unknown/Unbounded 在 Source.open/iterator acquisition/next、Sink.open、state-root/job 创建前拒绝。closed_window 则需原事件时间坐标上的可信 source-provided/runtime-generated watermark route，或 Bounded EOF route；不能靠试读、超时、max_rows 达限或看见一条 ts>=end 推断闭合。编译可列绑定义务，真实 runner/convenience/reload/暖缓存入口都必须履行。

| Input/control                             | Effect                                                                                                                             |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| bounded_source 的完整 Data                | 全部行进 H；正常 FIFO EOF 处理完此前 Data 后闭合，按 partition/order/frame 计算 final。                                            |
| window 的 Data、尚未闭合                  | 只有 start<=ts<end 的行进 H；NULL-time/区间外行忽略。即使 ts<=已提交 W，只要 W<end，仍按成员接受，不套 rolling 逐行 late 阈值。    |
| window 的有效 W>=end 或正常 EOF           | 闭合接受历史 H，一次计算并 emit 一个完整 table Batch（可内含多个有序 RecordBatch）。不是把窗口里未来所有时间匹配行都神奇包含进来。 |
| 空 Data、Idle、Barrier、checkpoint、W<end | 不发 provisional、不关闭、不 reset；沿原 barrier/控制 FIFO 和恢复协议。                                                            |
| final 成功后重复 close / 正常 EOF         | 不重复 final，不重新开窗；水位早闭合不提前终止 Source/job。                                                                        |
| 无 Data / H 空                            | 同样等待合法闭合，恰好一个 typed empty final，无 rn=1 或 SUM=NULL 的虚构单行。                                                     |
| cancel、sender drop、断连、错误           | 失败，不代替 EOF，不补“成功 final”。正常 EOF 后继续 Data 是 runtime 协议错误，不改写成 late。                                      |

window 默认本地 Error(scope=Envelope)，另支持 Drop(metrics_version=1)；不支持 SideOutput、其他 scope/version 或 grace period。一个 Data（所有 RecordBatch）只采样一次已提交有效 W。**只有窗口已因水位闭合后到来的域内成员才 late**；NULL-time/范围外行不是 late。不二次减 allowed_lateness，不根据 arrival/ORDER 键重开窗口。闭合及后续判断都用同一原始时间来源，不把 score 的排序换成 watermark 坐标。

Error 遇任一 late member，整个 Data 失败，H/accepted_rows/closed/sequence/水位/metrics/capture generation 不变；先前已成功 final 不回滚。Drop 丢所有 late members、不重算/撤回/追加 final、不保存其 operand/sequence、不增加 accepted_rows；已达到 max_rows 仍能 Drop，不能先做 old+late 行预算检查。schema/checked 时间转换、取消和 metric overflow 不被 Drop 吞掉；扫描/metric 自有目标仍需预留。

Drop 复用持久 late_rows、affected_batches、max_lateness_micros，初始0/0/None。每个完整 Data 丢 n 行，前两者加 n / (n>0?1:0)，第三者合并 checked_u64(i128(W)-i128(ts)) 最大值，非 W-end；跨多个 RecordBatch 只加一次 affected_batches。复用 PreparedLateMetrics，预备 delta/totals/sink 后在无 await、不可失败提交段一起安装；恢复 totals 不是新 delta，不重复累计。无新增 NULL-time counters。

bounded_source 只接受 Error；Drop 构造/导入拒绝。finite window 接受相同 Error/Drop 声明，但完整本次输入仅作窗口筛选，没有模拟水位/late counters 或跨 execute H。要比较 stream final 与 finite，finite 必须提供**相同接受 H**；不能把 stream 水位闭合后被 Error/Drop 的成员再放进 finite H，却要求相等。

Program 全局 late_policy/allowed_lateness 仍只配置原有未标记 rolling/cross-section，不能覆盖/补充 OverSpec。本地 Drop + 默认全局 error 合法且导出 Drop。同图独立已有消费者可用非默认全局选项，只作用于那些消费者；含 OVER 图无此消费者却传全局 drop/非零 allowed_lateness 时 capability_mismatch，提示在 OverSpec.closed_window 选择本地值。不同 OVER 可各自本地策略，同一共享声明只有一个 owner。

batch 每次 execute 新运行状态（直接复用同 plan/Runtime、暖缓存也如此），失败/取消后再执行不见旧 H；普通新 stream job fresh，显式 checkpoint 才延续同逻辑历史。finite 调同一 finalizer，不创建持久 StreamJob、不注入 MAX watermark。内置 batch snapshot 为 NULL、拒绝非 NULL restore；不增加 batch 查询中途恢复。

## Composition, analysis and compatibility

获准路径为一个 exact-schema table input → 已证明 deterministic 单输入 row-local DF/SQL 串 → 一个 bounded_over → row-local DF/SQL 串 → 命名输出。允许共享 fan-out、独立 OVER 和无关合法输出；同声明在每个 job/finite execute 只有一个 state owner。支持 DF→row-local SQL→OVER、SQL-OVER→DF→row-local SQL，不要求物化。

这条新路径不接受上游 aggregate/snapshot/latest replacement、union、join/ASOF、rolling/cross-section、window aggregate、DISTINCT、bounded_order 或另一个 bounded_over；下游也是已证明的 row-local 子集，不能接第二个 finalizer/aggregate/普通 OVER。UDF/provider/array/external 在域外。不能放宽 DISTINCT/order 既有 downstream 白名单来顺手接 OVER。replacement 经投影/过滤中转仍在 offending edge 拒绝；legacy update 模式也不能绕过新节点结构门禁。原项目/低层 native builder、缓存、两语言和便利入口同样强制，不只在 strict/latest 请求时守卫。

上游 window time 和 sequence 的原样/rename 来源必须可追踪；其他 operand/partition/order 可来自已证明 deterministic row-local 计算，随后仍须满足窄 dtype 域。输出不继承 temporal identity。后处理只有在实际保序及 keys 映射都得到证明时才传播 total order；删键/无法证明 SQL 物理保序时清除/降 unknown，declared-order 请求拒绝。OVER 直连命名输出是必交付的完整有序路径。

QueryAnalysis 保持 schema_version=2，**analysis_revision=11**，同步 Rust producer、Python parser/strict/latest gates 和分析缓存；旧 r10 可展示，不作为新 analytic 证明。复用现有枚举：stream history=accepted_history/event_window，update=append，finality=group_final；batch=current_batch/append/per_batch。ordering=ordered/total（partition ASC NULLS LAST + order_by + sequence ASC），temporal_identity=None。reason/explain 写明整个 H、实际闭合义务、sequence 值检查、函数/frame、受检整数溢出、本地 late 策略及 owned-only peak。状态/支持只是事实，不能据函数名证明任意 OVER。

可移植请求只证明相同接受 H/cut 下的 final_relation（bounded_source/all_sources_eof，或 window/closed_window；窗口同完整接受 H 的 EOF 比较亦可），不支持逐批输出轨迹相等。整关系所有列/重复数/函数值均比较；总序成立时可请求 declared order。即使 unordered 比较也不能省略 sequence，因为 ties 会改变附着到行的 analytic 值。H/cut 未证明用 unknown，已知不支持形状用 unsupported，strict 在任何数据访问前拒绝。

新增 capability `bounded_over@1`：required table input/output、batch/stream、requires_datafusion=true、stream stateful、group_final_append_only、checkpointed_stateful、state_version1、state_layouts=(1,2)、requires_watermark=false（有 EOF 域）。获准支持域均有强制 sequence，所以 inventory/native lifecycle 的 microbatch_invariant/deterministic/replay_safe=true；这只是在相同 H、有效 Source/replay/值校验及预算成功的前提下，不是为普通 iterable 承诺恢复或交付。

配置的所有 history/order/functions/frame/name/budget/late/ingress 字段参与声明、native config、prepared/cache 和恢复身份。新的 cache tag 标明 bounded_over_v1/copy_visible_v2/over_accounting1/analysis11；for_mode 只共享声明，不共享 compiled plan/state。project/manifest 格式仍3，不增加全局新 mode；新字段只属于新节点，不重写旧项目/graph/config/checkpoint。原 SQL state_version2/layouts3/4/5、DISTINCT/accounting2/layout2/3、ordering/layout1/2 均保持自己的命名空间。

## Peak budget, transactions and recovery

每节点显式 StateBudget，无默认 unlimited。max_rows 计域内累计接受行，重复、NULL operand、所有 partition 的行都计；分别限制 candidate H 与最终输出行数，不把相同历史的多个活副本行数相加。ROWS N、partition 数或 rn 不替代累计行限制；后处理 filter 不减少本节点 H 额度。H 全保留直到闭合，不因 N=0、只有 row_number 或选少量输出而删历史。

沿 copy_visible_v2：raw ingress（含未知 foreign parent backing）只在同步验证/计数/逐值深复制阶段临时借读，其未知父分配在节点 quota 之外；不能声称测得/计作0或因此要求拒绝全部 PyArrow。保留的所有列、schema/metadata、sequence 和自有容器必须先 reserve 后深复制，不捕获 raw Batch/ArrayRef/SchemaRef/capsule/closure；第一次内部 await/CPU-work submit/emit/checkpoint 前放开 raw 强引用。其他 Source/queue/caller 可以另持原数据，节点不承诺其释放。无可信 provenance 旁路、自报 owner-size 或 offset=0 零拷贝例外。

预算是确定性的**节点自有 participating logical peak**，不承诺 graph/job/process/RSS，也不覆盖 foreign backing、allocator 私有 bookkeeping 或没有登记的引擎私有分配。沿已合并 owned 规则：可观察 buffer 按64字节 alignment/capacity，Batch charge 为 `512 + owned_schema_charge + sum(256 + owned_array_charge)`，另加真实 Vec/String/Box/spare capacity、配置/control。每个活跃角色 lease 相加为 M(t)，下一分配前 checked `M(t)+next_charge<=max_bytes`。共享自有 backing 在不同角色保守分别计费，lease 活到最后实际 owner 释放。

除沿用 history/normalization/sequence/concat/sort/take/capture/restore 覆盖项，必须计入 partition 分界/比较 scratch、宽 accumulator/non-NULL counts、frame 工作状态、每个 analytic 输出数组/validity/descriptor、新输出 schema 和完整 compact final 的副本。固定宽新列可由 N/表达式数/精确 dtype 在分配前计算；LAG 的变长值按实际参与字符串字节预扫并按容量上界预付（不得先收集字符串）。所有 N、N+1、容量乘法/对齐加法 checked；Arrow 不进 pool 的分配也要 lease。现有排序 `workspace_bound` 的两份输入布局+indices 不自动覆盖这些增量，不用任意固定倍率声称覆盖所有新 workspace。

| Concurrent role               | Required accounting/lifetime                                                                                                                                    |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Declaration/old H/candidate H | 新旧容器、完整 retained payload、schema/metadata、复制/sequence 校验 scratch；已有活 capture 同时计费。                                                         |
| Final evaluator               | 合并 H、partition/order keys、排序索引和 scratch、take 源结果、宽 frame state、所有新列和其原生分配。CPU work 持有的 owner/lease 不能在提交 worker 时退出预算。 |
| Prepared output/emit          | 完整自有输出 backing、额外 compact 副本及复制 scratch 与 H/work 同时计；不能通过 slice/clone 让一个小结果暗中留住未计费整 H。                                   |
| Capture/restore               | 所有仍活 captures/segments、control/identity/IPC/metadata、decode candidate/旧 state、sequence 校验及容器同时计费；替换 cache 变量不代表旧 owner 已释放。       |
| Empty H                       | schema、control、metadata、容器和必要 workspace 仍收费；空 final 不豁免。                                                                                       |

实现必须在 kernel 前从窄 dtype/N/字符串长度/列数推导可审计预付上界，沿原 helper 的上界再增加本片真实对象；无法证明可预付的具体路径在分配前准确拒绝并带证据回报，不能跑完再检查 bytes。可保守失败；一列累计 SUM 的最终表很小也不保证整个峰值能放下。验证采用测得本实现逻辑 peak B 的 B/B-1 边界和角色 ledger，不把 B 写成跨实现固定 RSS 数字。

Data 事务：验证完整 envelope/schema/时间/支持表示 → 计算窗口成员及行数、预留候选/normalization/校验 → 全部复制后释放 raw → 验完整 candidate sequence → 无 await、不可失败地安装 H/accepted_rows/validation/capture generation。任何 RecordBatch 失败不部分安装。Drop 的 metric 事务按上节，不建立已丢行的 H。

闭合事务：保留旧 H/closed/output_sequence/水位 → 预付排序/frame/完整输出并计算 → 验 SUM 范围、精确 schema、edge 完整 row/byte budget 和取消 → await emit 成功 → 无 await 且无新可失败检查地提交 closed=true/output_sequence=1/触发闭合 W、释放 H/不再需的 scratch 并 invalidate capture。emit 成功后不再插取消检查制造“已发但未提交”；函数/预算/schema/取消/emit 失败都不安装 closed/sequence/闭合水位，不伪造空 final。此前其他 Data 的已提交 H 不回滚。一个 final Batch 可含多个 RecordBatch，但不能拆成多个部分 final 绕 edge budget。

这只是单 operator 状态原子性，不承诺 fan-out/queue/Sink 全图回滚。部分外部交付、checkpoint 前已发结果的 replay 和成功前缀沿现有 per-output delivery 协议，不增加 ExactlyOnce。

独立恢复命名空间为 state_kind=`bounded_over_v1`、state_version=1、accounting_version=1、ingress_policy=`copy_visible_v2`，Error 读写 layout1、Drop 读写 layout2。Drop 尚无 late 也必须带0/0/None metrics并写layout2。control 含完整 spec/expressions、input/output schema 与 config/engine/runtime-config digest、accepted/retained rows、closed、last effective W、output_sequence(0或1)、H/metadata segment digests、layout/accounting/policy；Drop 另带严格 metrics。未闭合时 retained_rows=accepted_rows，closed 时 retained_rows=0、sequence=1；不保存部分 analytic output 或前一次 final 的 frame accumulator作为继续更新状态。

capture 缓存随真正数据/水位/metric/control 提交失效，captures 生命周期按实际 owners 计费。恢复先校验小 control/identity/descriptor、再预留 decode/sequence 验证存储，全部验证后一次安装，在 Source.open/replay 前完成。控制可知的 mismatch 在大目标分配前失败，坏内容/sequence 重复解码后检查但仍不得先安装/开 Source。拒绝未知字段/kind/state/layout/accounting/engine、错 schema/digest/cut/replay 能力、计数/closed/sequence/水位不一致、超预算、任何 order/frame/function/name/history/budget/policy 变更、Error↔Drop、Drop-layout1/Error-layout2；不静默清空重算。

未闭合恢复继续 H 到同 cut 闭合；closed 恢复不重发、不重开，后续 window late 策略/metrics接续。closed window 可由 EOF 关闭，此时不要求 last_watermark>=end；不能因恢复验证而伪造 W=end。水位早闭合后收到更大有效 W 仍按原协议提交/持久化该 W，供后续 late metrics 使用，不重算 final。Source cut/水位恢复必须协同，不能换 Source 或水位坐标。无旧 SQL/DISTINCT/order 或跨 mode 状态迁移；用户更改语义用新执行，不修改旧 checkpoint 的身份。

## Error contract

DF 路径 P 为逻辑关系 `.over`，SQL 为 `.sql.over`；配置定位 `.spec.history|order.partition_by|order.order_by[i]|order.sequence_by|state_budget|late_policy`，表达式定位 `.expressions[i].name|column|frame`。真实绑定用 `stream.inputs.<logical_name>.boundedness|watermark_policy`；native project 保留 graph.nodes/edges 物理路径，不拿生成 IDs 代替两端逻辑地址。

| Violation                                                    | Required diagnostic and boundary                                                                                                             |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Python 错配置/序列/int/bool、负 frame/越界 budget/时间、空名 | 类型 TypeError、值域 ValueError；Rust InvalidArgument / invalid_literal，指 offending input 和精确约束；构造期读行0。                        |
| 空/重复列表、重复 alias/覆盖输入名、列表列重叠               | invalid_literal/duplicate_name；指出具体列表/表达式索引和名称，不擅自去重。                                                                  |
| 未知列、unsupported retained dtype/SUM 非整数                | unresolved_type/unsupported_type，指出列和允许域；schema 阶段拒绝。                                                                          |
| nullable/改写/空 sequence、无法追溯 time/sequence            | ordering_required / capability_mismatch，说明原样非 NULL、整 H 唯一及来源要求；不能靠同名 dtype 猜证明。                                     |
| 实际 duplicate/NULL sequence                                 | ordering_required，列/位置/候选未提交，不回显 tuple/payload；完整 Data 原子失败。                                                            |
| 不支持函数、frame、SQL 条款/宏 arity/混合 order              | capability_mismatch/invalid_literal，定位函数/参数/frame/order；无 legacy fallback，读取行0。                                                |
| Unknown/Unbounded Source 或 window 无可信 closure route      | source_boundedness_unproved/unknown、source_unbounded/unsupported、finality_unproven/unknown；open/next/job/state-root=0。                   |
| snapshot/unknown update、stateful/多源/嵌套路径              | update_interpretation_mismatch 或现有 update_*_unproved；结构越界 capability_mismatch，指出 offending edge。                                 |
| SUM 输出越界                                                 | arithmetic_overflow，指出表达式/operand/输出 dtype/frame 和约束；不输出 payload/部分 final，闭合状态不提交。                                 |
| owned 峰值或行预算不足                                       | state_budget_exceeded，含 node/policy、normalization/history/sequence/frame/sort/output/capture/restore 等角色、required/limit、rows/bytes。 |
| 运行 schema/时间/worker/emit/取消/restore 失败               | 沿既有 typed execution/stream/checkpoint 层及 cause；不新建平行异常体系，restore mismatch 点名且 Source.open=0。                             |

input_normalization_unsupported 仅用于具体表示不能安全逐值重建，不因 unknown foreign parent 大小本身拒绝。所有诊断不含整 plan dump、payload、sequence tuple 或凭据。

这里的 invalid_literal/ordering_required/arithmetic_overflow 等是沿既有 `dataframe::invalid` 的 InvalidArgument.field/message 诊断 token，不假称当前 `StreamingFailureReason` 已定义所有这些成员。新 SUM 的运行错误沿该既有路径带 `arithmetic_overflow` token、表达式路径及约束；stream 保持现有外层 failure/phase 投影，不新增平行 exception 或 reason enum。预算继续使用既有 StateBudgetExceeded 投影；绑定/检查点沿其原分层。

## Typical declaration seeds

以下仅作未来公开例子的种子，本轮不编译/执行。新例子各自放在 `crates/calc-flow/examples/bounded_analytic_over.rs` 与 `examples/bounded_analytic_over.py`，保留现有 DISTINCT/order/SQL 例子的目录和用途。

```rust
use calc_flow::{
    AnalyticExpr, NullPlacement, OverOrder, OverSpec, Result, RowsFrame,
    SortDirection, SortKey, StateBudget, TableExpr,
};

fn features(events: &TableExpr) -> Result<TableExpr> {
    let order = OverOrder::new(
        vec!["p".into()],
        vec![SortKey::new("score", SortDirection::Ascending, NullPlacement::Last)?],
        vec!["seq".into()],
    )?;
    let spec = OverSpec::bounded_source(order, StateBudget::new(1000, 8 << 20)?)?;
    events.over(vec![
        AnalyticExpr::row_number("rn")?,
        AnalyticExpr::lag("v", "prev")?,
        AnalyticExpr::sum("v", RowsFrame::cumulative(), "total")?,
        AnalyticExpr::sum("v", RowsFrame::preceding(1)?, "last_two")?,
    ], spec)
}
```

```python
import calc_flow as cf


def features(events):
    order = cf.OverOrder(["p"], [cf.SortKey("score")], sequence_by=["seq"])
    spec = cf.OverSpec.closed_window(
        cf.OverWindow("ts", start_micros=0, end_micros=10), order,
        state_budget=cf.StateBudget(1000, 8 << 20), late_policy="drop",
    )
    return events.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.lag("v", name="prev"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.preceding(1), name="last_two"),
        ],
        spec=spec,
    )


# events = cf.table_input(..., schema=exact_schema, event_time="ts")
# out = features(events)
# p = cf.Program("features", mode="stream", outputs={"out": out})
# Bind the original ts watermark route; final appears only at its real close.
# p.for_mode("batch") reuses the declaration, with independent run state.
```

## Independent small-table expectations and delivery checks

以下是手算 oracle，后续实现须断言，不是本轮通过记录。schema 为 `p:Utf8 nullable`、`ts:timestamp[us] nullable`、`score:Int64 nullable`、`seq:UInt64 nonnullable`、`v:Int64 nullable`。采用 window `[0,10)`、partition p、score ASC NULLS LAST、sequence seq；输出表达式为上例 rn/prev/total/last_two。ts 表内数字是微秒坐标，输入保留到达顺序如下：

| Data | p    | ts   | score | seq | v    |
|------|------|------|-------|-----|------|
| D1   | A    | 0    | 2     | 12  | NULL |
| D1   | A    | 1    | 1     | 11  | 2    |
| D1   | B    | 9    | NULL  | 22  | NULL |
| D2   | A    | 9    | 1     | 10  | 3    |
| D2   | A    | 8    | 3     | 13  | 5    |
| D2   | B    | 2    | 0     | 21  | 7    |
| D2   | C    | 5    | 0     | 40  | NULL |
| D2   | NULL | 3    | 1     | 30  | 4    |
| D2   | NULL | 4    | 1     | 31  | NULL |
| D2   | A    | 10   | 0     | 99  | 100  |
| D2   | A    | NULL | 0     | 100 | 100  |

接受9行；ts=0 包含，ts=10 和 NULL 不含。D1、空 Data、W=9、D2 均无结果（D2 中 ts<=W 仍合法）；W=10 才发一次以下完整表。这里有意使用能允许这些乱序到达的真实 watermark route，不能用默认 nondecreasing iterable 验证器伪称通过。

| p    | seq | rn  | prev | total | last_two |
|------|-----|-----|------|-------|----------|
| A    | 10  | 1   | NULL | 3     | 3        |
| A    | 11  | 2   | 3    | 5     | 5        |
| A    | 12  | 3   | 2    | 5     | 2        |
| A    | 13  | 4   | NULL | 10    | 5        |
| B    | 21  | 1   | NULL | 7     | 7        |
| B    | 22  | 2   | 7    | 7     | 7        |
| C    | 40  | 1   | NULL | NULL  | NULL     |
| NULL | 30  | 1   | NULL | 4     | 4        |
| NULL | 31  | 2   | 4    | 4     | 4        |

原输入列全部保留，此表只省略 ts/score/v 以便读 oracle。A 的两个 score=1 peer 因 seq 先10后11，第一个 total=3 而非 RANGE 默认的5；seq13 的 prev=NULL 证明 LAG 不跳过 NULL，last_two=5 证明 ROWS 下界为当前及前一行。C 的全 NULL frame 是 NULL。改为 preceding(0)，各行 SUM 恰为本行 v（NULL 保持 NULL），LAG/rn不变；改 N=4294967295 时等于 cumulative，不溢出 N+1；换成 DESC NULLS FIRST 时 B 的 seq22 在21前，NULL placement 独立于方向，按新总序重算而非倒转旧函数列。

定向验收由实现者完成，按现有证据复用、不展开全仓矩阵：

1. **C03/A05/A09：** 四单元分别真实公开声明、schema/analyze/explain、batch、stream final、导出再加载并对上述手算值断言。bounded_source 给相同9成员 H，正常 EOF 得同表；不误把窗口外2行纳入该对照。无 partition 的独立输入 x=[3,1]/[2]（各有唯一 seq，v=x）：显式 bounded final 按 x 排为1,2,3，rn=1,2,3、prev=NULL,1,2、total=1,3,6；普通无宏 SQL 仍各批算 rn=1,2 / 1。空 H 恰一个精确 schema 空 final。
2. **C11/C13：** 上述窗口0/10/NULL、W9不闭/W10闭、先水位再域内行、负时间平移、复合/NULL partition、primary ties/方向/NULL placement、N0/N大于组长、全NULL/零。SUM signed/unsigned widening 各一例；[i64::MAX,1] cumulative 在第二输出 overflow 整个 final 失败，[MAX,1,-1] 仍不能被最终抵消掩盖；preceding(0) 对 [MAX,1] 成功。SQL 缺frame/RANGE/不同order、非法LAG/函数及 strict wire 负例定位准确，读取行0。
3. **late/事务：** max_rows=9，W10成功 final 后一个 Data 跨两个 RecordBatch 含域内(ts=0,seq50)、(ts=9,seq51)及ts10/NULL。Error整个Data失败，状态/metrics不动；Drop无新输出、accepted仍9、metrics=2/1/Some(10)，不能被max_rows阻止。closed Drop checkpoint恢复，W12后ts1新成员使metrics=3/2/Some(11)，不重复final或重复应用旧totals。重复seq（含不同partition/跨Data）、错误schema/时间、取消、metric overflow、sum overflow、预算/emit失败不部分安装或假闭合。
4. **预算/恢复：** 复用 copy_visible_v2 外部slice/owner证据，增加analytic新列/宽state/字符串LAG/空结果/capture并存的真实角色账本和B/B-1；rawowner在首await前释放。未闭合D1 checkpoint→恢复→D2/W10与上表相同；closed恢复不重发。每个新identity/layout/政策/坏计数/digest/预算负例在安装/Source.open前失败。改一个frame/name/order/sequence也不复用checkpoint。
5. **L01–L03/A06：** 同job跨Data与空Data保留H；直接同compiled finite plan、同Runtime、暖缓存、无reset，依次完整H、单成员(seq不同,v=9)、typed empty得到完整表、rn1/prevNULL/total9、空；取消/失败后再用仍独立。普通新job fresh，for_mode共享声明不共享state；有限执行不产生窗口late counters/stream根目录。
6. **A08/A09/组合：** Rust外部consumer、Python根包/符号导出/类型/stub及native schema seam可用；SQL和DF导出同义配置，局部late策略持久化、不被全局默认覆盖；真实mixed DAG/fan-out不丢节点且只有一个owner。native/project/reload/convenience/暖cache验证Unknown/Unbounded、错time/sequence来源、snapshot/嵌套/多源等启动前拒绝；无宏旧OVER、快照7/latest6或拒绝、旧布局与mode身份不回退。

## Rationale, rejected alternatives and handoff

选择独立 over 关系和 OverSpec，而不是给普通 SQL/SortSpec 加隐式历史：保留旧每批解释，并能在真实 Source 绑定处检查 cut/闭合。选择 mandatory reproducible sequence，而不是弱 ties/first-arrival：ROW_NUMBER/LAG/ROWS 的值附着到具体行，unordered 最终关系也会受 ties 影响。选择两个 ROWS frame、固定 LAG，而不是时间 RANGE/任意 offset/default：小域足以交付 C03 的跨批函数和边界，NULL/peer/方向含义明确。选择单次final、完整H，而不是边到边发再修正/按frame淘汰：不引入retraction/替换协议或未证明的事件时间排序。

整数SUM检查溢出是新域的明确决定，不默借DataFusion wrapping；采用新native evaluator而不是Python求值或两份SQL/DF算法。窄scalar域及owned normalization复用已合并排序的真实约束，不把外部parent大小或整个process纳入未经证明的预算。独立state_kind/layout和r11避免将旧节点的可编码状态误认为新语义证明。此合同也不承诺Polars全面兼容或外部新版本行为。

**没有待用户/协调器重新选择的公共合同 blocker。** 已确认的实施缺口是新关系/公开配置与AST/schema seam、native partition/ROWS/checked-SUM evaluator、强制准入与Source证明接线、analytic增量峰值计量、capture/restore/inventory/r11/cache、绑定/exports/stubs/docs/examples及上述定向证据；它们是本片需实现的工作，不是现有能力。若实现发现某个具体新kernel分配不能预付或恢复无法满足原子性，带源码与生存期证据回报，不能暗降本合同。

本轮仅必要源码/合同核对及文档自检。Python围栏作语法/签名检查，Rust/SQL为未编译执行的声明种子；产品实现、测试、独立主审、PR及合并均未完成。交协调器安排实现和dal-reviewer独立聚焦主审，只有实际覆盖缺口才追加tester，不另起完整spec/critic阶段、不由本角色扩派或启动实现。DAL-521继续`in_progress`。
