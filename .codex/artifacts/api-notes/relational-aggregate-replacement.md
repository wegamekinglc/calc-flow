# Relational Aggregate and Replacement — API Note

状态：DAL-514 / S1b 局部合同冻结，2026-11-11，可进入实现。**以下新增签名、支持域、预算与 layout 是待实现合同，不是当前已交付能力；本轮仅核对源码与文档，未运行产品测试。**

基线：共享 checkout 的分支 `agent/dal-orchestrator/27a25600495f`，HEAD `c4f1749c0595ed3df0fe7eaf4c6331c6093dcfab`，开始时工作树干净；该提交是前序 DAL-513 / PR #595。本文只补 [G1 note](latest-relation-update-contract.md) 明确留给 S1b 的决定，不替代它。依据为 DAL-514 描述、触发评论 `01a1ca80-ec77-7285-a5ab-727f4bbc1e33`、父 DAL-498 的 API/生命周期讨论及开发计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`（已通过 CLI 下载核对）。

## Audiences and scope

Rust 用户从 `calc_flow::dataframe` 及 crate root 构造聚合关系；Python 用户从根包 `calc_flow` 的 `TableExpr` / `ColumnExpr` 构造同义关系。两者都可继续接 SQL、投影、过滤及共享输出，使用现有 Program、batch plan、owned stream 和 managed Checkpoint。SQL 仍是独立的一等声明入口。

本片新增两语言 DataFrame 聚合，以及显式 latest 下支持名单内普通单 alias SQL 对完整快照的替换消费。旧默认、Program 更新选择、whole-Batch 事件、Source/Sink/Runner 协议和 snapshot_sql 限制沿 G1。普通 append 源仍表示追加事件，不能靠选择 latest 将每个源批次变成全关系。一般 delta/retraction、多输入增量 JOIN、S2/S3 关系扩展和 batch 查询中途恢复不在本片。

## Surface today and evidence

以下是本 HEAD 的事实。G1 note 的 “Surface today” 描述的是其旧基线，不能据此否认前序已实现的选择和 gate。

| Source                                                                                                                              | Existing capability / remaining gap                                                                                                                                                  |
|-------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/src/dataframe.rs:75`、`:305`、`:338`                                                                              | RelationKind 只有 Input/Filter/Sql/Project；公开列、select/with_columns/filter/sql，没有 aggregate。portable_type 已支持多种 Arrow 标量类型。                                        |
| `crates/calc-flow/src/program/lower.rs:315`、`crates/calc-flow/src/lib.rs:94`                                                       | typed SQL 用 SqlOperator 推导 schema 并导出 OperatorSpec::Sql；声明可跨接口组合，Rust root 已导出 TableExpr/ColumnExpr/NamedExpr。                                                   |
| `crates/calc-flow/src/program.rs`、`crates/calc-flow/src/config.rs:124`                                                             | G1 的 UpdateSemantics、Program/builder 选择、project-v3 graph 字段和旧默认省略编码已经存在。                                                                                         |
| `crates/calc-flow/src/query_analysis/sql.rs`、`crates/calc-flow/src/query_analysis/updates.rs:90`                                   | 已有单一 Rust SQL 分类与 latest 准入；完整快照到普通 aggregate 仍以 append-only mismatch 拒绝。现有 Aggregate 证明未完整限制 DISTINCT/函数参数，不能直接充当 replacement allowlist。 |
| `crates/calc-flow/src/operator/sql.rs:1207`、`:1345`                                                                                | 普通 SQL 以 retained/compact 路径累计 append，emit 成功后提交候选；不能用 reset 或重复累计处理 replacement。                                                                         |
| `crates/calc-flow/src/operator/sql/restore.rs:174`、`crates/calc-flow/src/pipeline/stream.rs:1906`                                  | 普通 aggregate 的 Operator state_version=2，实际 payload layouts=3/4/5；不是将 state_version 与 layout 当成一个编号。                                                                |
| `crates/calc-flow/src/operator/snapshot_sql/mod.rs:266`、`crates/calc-flow/src/operator/snapshot_sql/checkpoint.rs`                 | 现有完整替换、初始化、候选/metadata/segment 预算、准备恢复后再安装可复用其内部机制；该公开 Operator 的 stream-only、输出直达 Sink 等限制保持。                                       |
| `python/calc_flow/symbolic/expr.py`、`python/calc_flow/symbolic/lower/sql.py`                                                       | 已有 immutable 表/列节点及 SQL lowering，没有本片的 AggregateExpr 和 aggregate 方法。                                                                                                |
| `python/calc_flow/runtime.py:635`、`crates/calc-flow/src/pipeline/stream.rs:106`                                                    | 已有 sql_state_budget、sql_compact_state_budget、sql_snapshot_state_budget；尚无普通 replacement aggregate 的专用预算。                                                              |
| `examples/32_sql_expression_pipeline.py`、`crates/calc-flow/examples/32_sql_expression_pipeline.rs`、`docs/relational-frontends.md` | 已有独立可用的 mixed 前端示例；增量补充聚合说明，保留位置和用途。                                                                                                                    |

另核对了 `.codex/agents/cf-api-designer.toml`、AGENTS.md、`docs/introduction.md`、两端 example 32 的前序合同及相邻 SQL/relational 测试。dtype 依据为 Cargo.lock 锁定的 DataFusion **55.0.0** 的本地 `datafusion-functions-aggregate/src/{sum,count,average,min_max}.rs`：SUM 签名的 signed/unsigned/float coercion、COUNT Int64/非 nullable、AVG Float64；没有把机器上另一份 55.2.0 当本仓库证据。本片不重查 Polars 版本，不新增兼容承诺。

## Proposed DataFrame surface

选定单一 `aggregate` 方法，必选的命名聚合在前，分组键在后。无分组即 global aggregate；不同时新增 `agg` 别名、GroupBy owner 或聚合标量算术系统。Rust 与 Python 的方法名一致，名字与 SQL 的 SUM/COUNT/MIN/MAX/AVG 对应。

下列 Rust 为签名清单，省略方法体。新类型均 immutable/Clone/Debug，内部字段私有；与现有窗口的 AggregateSpec/AggregateFunction 无关。

~~~rust
// dataframe module and crate-root re-exports.
pub struct AggregateExpr { /* private */ }
pub struct NamedAggregate { /* private */ }

pub fn count_rows() -> AggregateExpr;

impl ColumnExpr {
    pub fn sum(&self) -> AggregateExpr;
    pub fn count(&self) -> AggregateExpr;
    pub fn min(&self) -> AggregateExpr;
    pub fn max(&self) -> AggregateExpr;
    pub fn mean(&self) -> AggregateExpr;
}
impl AggregateExpr {
    pub fn alias(&self, name: &str) -> Result<NamedAggregate>;
}
impl TableExpr {
    pub fn aggregate(
        &self,
        aggregates: Vec<NamedAggregate>,
        group_by: Vec<String>,
    ) -> Result<Self>;
}
~~~

~~~python
from collections.abc import Mapping, Sequence

# AggregateExpr and count_rows are exported by calc_flow and calc_flow.symbolic.
class AggregateExpr:
    # Immutable declaration; no public constructor, row-local operators or truth value.
    ...

def count_rows() -> AggregateExpr: ...

class ColumnExpr:
    def sum(self) -> AggregateExpr: ...
    def count(self) -> AggregateExpr: ...
    def min(self) -> AggregateExpr: ...
    def max(self) -> AggregateExpr: ...
    def mean(self) -> AggregateExpr: ...

class TableExpr:
    def aggregate(
        self,
        aggregates: Mapping[str, AggregateExpr],
        /,
        *,
        group_by: Sequence[str] = (),
    ) -> TableExpr: ...
~~~

Python 按调用时 mapping 迭代顺序复制为 immutable named entries，防止之后修改原 mapping/sequence 改变声明。Rust 保持 Vec 顺序；不使用 BTreeMap 作为有序输出列参数。必需至少一个聚合，group_by 可为空；拒绝 str/bytes 充当键序列、空名字、重复键、聚合名冲突及聚合名与分组键冲突。允许一个源列以不同结果名重复聚合。所有名字严格按现有 quote/lineage 规则处理，不把用户列名插成未转义 SQL。

首片聚合 operand 限定为输入关系上已存在的**直接列引用**；`count_rows()` 没有 operand，在 aggregate 时绑定该输入。`(t["x"] + 1).sum()`、literal aggregate、嵌套聚合及另一关系的列拒绝；先 `with_columns(y=...)`，再对新关系的 y 聚合即可。未知 SQL 派生列在 planning 检查，不因无静态 known_names 提前误报。AggregateExpr 不可传入 select/filter/with_columns，不改变 ColumnExpr 的 row-local 含义；聚合后 scalar 算术/过滤通过结果 TableExpr 的列继续声明。

aggregate 返回新 row lineage，清除 event/entity/sequence 与 temporal ordering 证明，行为同 SQL；分组输出无指定行序，按多重集比较。列序为 group_by 声明顺序，然后 aggregates 声明顺序。schema 必须由同一 native SQL planner 推导并逐列核对，不根据观察到的首批数据猜 dtype 或 nullable。

## Dtype, NULL and finite relation contract

以下为 DataFrame 新入口的共同域，也是 replacement aggregate 的类型白名单；不是削减 legacy SQL 原支持域。numeric 在本文仅指 Int8/16/32/64、UInt8/16/32/64、Float32/64。COUNT 的直接列可为现有 portable_type 中任一类型；MIN/MAX 首片只开放 numeric。分组键只开放 Boolean、上述有/无符号整数、Utf8/LargeUtf8，可 nullable；浮点、日期时间等其他分组键暂不在新增域。它们可在旧 SQL 路径继续按既有准入运行。

| DataFrame                     | SQL equivalent      | Output Arrow dtype / nullable                                                 |
|-------------------------------|---------------------|-------------------------------------------------------------------------------|
| `count_rows()`                | `COUNT(*)`          | Int64，false；计行，含 NULL 行。                                              |
| `c.count()`                   | `COUNT(c)`          | Int64，false；仅计该列非 NULL 值。                                            |
| signed integer `c.sum()`      | `SUM(c)`            | Int64，true，包括 Int64 输入。                                                |
| unsigned integer `c.sum()`    | `SUM(c)`            | UInt64，true，包括 UInt64 输入。                                              |
| Float32/Float64 `c.sum()`     | `SUM(c)`            | Float64，true。                                                               |
| numeric `c.mean()`            | `AVG(c)`            | Float64，true。                                                               |
| numeric `c.min()` / `c.max()` | `MIN(c)` / `MAX(c)` | 与 operand 同 dtype，true；不按 SUM widening。                                |
| group key                     | 直接列 `GROUP BY`   | 保留该 key dtype 和 planner 的原输入 nullability；不因此批无 NULL 改 schema。 |

无分组时，一次完整有限输入总有一行 aggregate 结果；空输入或全 NULL 输入的 SUM/MIN/MAX/AVG 为 typed NULL，COUNT(c)=0；COUNT(*) 对空输入=0，对全 NULL 的 n 行=n。分组时，0 行输入产生带精确 schema 的0行结果；全 NULL key 的非空输入形成一个 NULL 组，值按相同规则聚合。重复行保留计数。NULL predicate 仍只保留 true。NaN/Inf 与 NULL 分开；数值精度、NaN/Inf、已检测 overflow 和执行错误沿锁定版本的 native SQL/既有 SQL 内核，不新加 fill-null、饱和、Decimal、Int128 或 Python 大整数归约。浮点不承诺不同批切分/并行归约位级一致。

独立小表种子：Int32 `[]` → SUM=NULL:Int64、COUNT(*)=0:Int64；`[NULL,NULL]` → SUM=NULL、COUNT(c)=0、COUNT(*)=2；`[1,NULL,3]` → SUM=4、COUNT(c)=2、COUNT(*)=3、MIN=1:Int32、MAX=3:Int32、AVG=2.0:Float64。UInt32 `[1,3]` → SUM=4:UInt64；Float32 `[1,3]` → SUM=4.0:Float64。对照 expected schema 的 nullable，不能只核数值或让 DF 和 SQL 互相当唯一 oracle。SUM 空值与某些 Polars 表达式的空和0有意不同，沿现有计划的 SQL 合同。

## Restricted replacement allowlist

选择仍只有 `legacy_append` / `latest_relation`，沿 G1 Program/builder/project 字段和 mandatory gate。实际 stream 编译根据已证明的**输入边事实**一次确定每个普通 aggregate 的消费方式，不根据第一批行或临时 runtime 值切换。不把 Program 选择直接等同于所有 SQL 都 replacement。

| Input and declaration                                          | Selected execution / facts                                                                                                                                |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 真正 append → DataFrame aggregate / 普通单 alias aggregate SQL | 两种选择下均沿原累计内核；history=accepted_history，input=append，output=complete_snapshot，finality=revisable_snapshot。                                 |
| complete_snapshot → 白名单 aggregate，legacy_append            | 保持追加快照事件的累计内核；SUM 链末值7，不给它伪造 latest 原始历史事实。                                                                                 |
| complete_snapshot → 白名单 aggregate，latest_relation          | 新 replacement 路径；input=complete_replacement，history=latest_relation，output=complete_snapshot，finality=revisable_snapshot。每事件只查询该完整关系。 |
| complete_snapshot → G1 已证明的单输入 row-local DF/SQL         | 保持 G1 支持和完整/空事件传播，可继续接本片 aggregate；输出仍 complete_snapshot。                                                                         |
| 有限 batch aggregate                                           | 每次 execute 只对本次输入运行 native finite SQL；无跨 execute replacement 状态，不要求新 stream budget。                                                  |
| unsupported / unknown                                          | 读取 Source 前按 G1 gate 拒绝，不 fallback legacy/另一 lifecycle，不自动 collect 中间表。                                                                 |

**白名单 SQL 形状固定如下，必须同时满足 AST 与已规划类型/关系证明：**

1. 一个普通 read-only SELECT，恰好引用一次唯一输入 alias 的一个关系；允许通常的列限定名/表别名。无 JOIN、子查询、CTE、递归、set operation 或多次 self-reference。alias 名仍任意现有合法 SQL identifier，不强制叫 input。
2. 至少一个 SUM/COUNT/MIN/MAX/AVG；参数为一个直接输入列，COUNT 另允许 `*`。无 DISTINCT、aggregate FILTER、OVER、WITHIN GROUP、函数内 ORDER BY、多参数、嵌套聚合或其他函数；不能仅依据函数名授予支持。
3. 可无分组，或按上述白名单类型的直接列分组；无 GROUP BY ALL、ROLLUP/CUBE/GROUPING SETS、ordinal 或表达式 key。projection 只含 group key 和这些 aggregate，可有显式结果 alias；所有输出名唯一。没有 arithmetic finalizer、star projection 或未分组普通列。SQL 的合法 SELECT 列序仍由 query 保留，不强行重排成 DF 顺序。
4. 本节点无 WHERE/HAVING、ORDER BY、LIMIT/OFFSET/FETCH、QUALIFY 等额外关系变换。过滤可拆为已支持的上游/下游 row-local 节点；不为本片扩展 HAVING/ORDER/LIMIT 的 replacement 证明。旧 SQL 原查询在 append 路径保持既有前缀快照语义。
5. exact 输入/输出 schema、普通表 Batch、完整事件边界已证明；无 UDF/provider/volatile/clock。输出可接 G1 row-local、另一个获准 replacement aggregate 或命名 Sink；不借此放宽 snapshot_sql 的输出路线。

已知普通 aggregate 不在白名单但真实消费者仅 append 时，保持 unsupported + `update_interpretation_mismatch`，定位 offending alias edge，并在 reason 说明不满足哪条白名单；未知 SQL/类型/边界仍按 G1 的 unknown codes。现有 DISTINCT/top-k/OVER 等按批查询不因本片自动变 global。rolling/window/union/join/ASOF 等 append-only 消费者仍拒绝快照；终端输出和多级 projection/filter 不洗掉更新事实。

一个 Data Batch 的全部 RecordBatch 共同完成一次替换、一次 SQL 计算、一次结果 emit。收到 schema 正确的0行 Data 是真实初始化/清空，global COUNT/SUM 发一行0/NULL，grouped aggregate 发一个0行完整结果；无 Data 不初始化、不发结果、不重算。EOF 保留已提交状态且不额外重复 emit；Idle/Watermark/Barrier 不伪造 Data。输出或输入超过 edge 限制整事件失败，不能切分替换来绕过预算。

## Budget entry and accounting

新增 **`StreamRuntimeConfig.sql_replacement_state_budget: Option<StateBudget>`**，Rust 默认 None；Python 同名 `StateBudget | None = None` 放在现有 dataclass 字段末尾。它是每个获准 replacement aggregate Operator 的独立限额，不是整图共享预算。StateBudget 仍要求 max_rows/max_bytes 在1至最大 JSON-safe integer 内，两个上限均为 inclusive。

存在该类 stream consumer 时，**启动必须显式给出非 None 的预算**；None 意味着拒绝，不能解释为无限容量或借用 snapshot_sql budget。没有 replacement consumer 的旧图及 append aggregate 不因此新增必填项。保留原 Program.stream(..., config=...) / execute 的 config 参数，Rust 经现有 StreamingRunner config 传入；不新增 Program 上的语义开关或 SQL kwargs，不占用 SQL table alias。

~~~rust
let config = StreamRuntimeConfig {
    sql_replacement_state_budget: Some(StateBudget::new(1_000, 8 << 20)?),
    ..StreamRuntimeConfig::default()
};
~~~

~~~python
config = cf.StreamRuntimeConfig(
    sql_replacement_state_budget=cf.StateBudget(1_000, 8 * 1024 * 1024)
)
~~~

compile/analyze 可以在无 Runner config 时证明结构支持，reason 要写明运行需预算；不把 supported 报告当运行许可。实际 config 存在的 job 准备阶段必须在 Source open/poll/next、输入 iterator 遍历/转换、bindings factory 和 job/state root 创建前检查缺失/非法预算及不兼容 compact 要求，包括暖缓存、严格/异步、显式 Runner 和 project reload 路径。实际数据大小仍在事件处理时校验。

Python `_native()` 字典使用成对 `sql_replacement_state_max_rows` / `sql_replacement_state_max_bytes`，不得只接收其中一个；PyO3 的 config parser 和 `_native.pyi` 同步。没有新 primitive runtime class。该可选字段加入 native runtime-config hash 仅在 Some 时序列化，默认旧 hash 不变；不进入声明 fingerprint 或改变更新选择。project-v3 的 StreamRunOptions 目前没有其他 SQL state budgets，本片也不把此运行参数硬塞入 graph；导出再编译后，Runner 仍显式提供预算。

**计量边界固定为逻辑行数 + 保守持有字节，非 RSS：**

- max_rows 分别限制已提交最新输入、待替换完整输入和准备好的完整结果的行数，即检查三者的最大值；不累计历史、不计 RecordBatch 个数。global 空输入的一行结果仍计1。旧/新输入同时存活的成本由 max_bytes 另计，不能在 commit 前“释放”旧状态额度。
- 一个持有 Batch 的 backing charge 复用 snapshot_sql 的 `backing_bytes` 规则：`512 + schema IPC bytes + ΣRecordBatch(256 + Σarray.get_array_memory_size())`。包含 Arrow backing 与容器，不使用仅可见 slice 的 `Batch.estimated_bytes` 规避大 backing。空 Batch 的 schema/descriptor 仍收费。
- metadata 采用现有 `sql/metadata.rs::reserve_with/encode` 的确定性预付上界，包含 source、sequence、attributes 与编码持有；不能只按数据列字节收费。该 helper 与 backing charge 可内部提取共享，不能更改旧路径计量。
- max_bytes 同时承担旧 committed relation、候选 relation、准备好的输出 relation、尚存活 checkpoint capture 的关系/metadata 及 schema/control/IPC segments 的已预留字节。不同角色保守分别计费，即使输入重复引用相同 Arrow backing 也不凭指针身份减免费用；同一 capture 对象的只读 clone 共用同一付费 lease，不重复支付同一个 capture。独立 IPC/schema/control segment 的结算 charge 为 byte capacity + 256，编码前的保守预留沿现有 snapshot capture/SQL IPC helpers；metadata 维持前述预付上界，不再重复计其 segment。不能提前退款。
- 编码/解码临时区与 DataFusion 查询临时工作内存沿既有运行时 memory-pool reservations 独立约束，准备输出本身仍计入上述新预算；edge 中已转移出去的输出由 edge 计量。新预算不是整个进程、所有队列或 DataFusion 的总内存保证。不为了可见上限保留历史版本、无界隐藏候选或长期保留查询 scratch。
- checked arithmetic，增加候选/输出/capture 前先预留；无法预付则报预算错误。失败或取消释放其 lease，仍活着的编码 worker/capture 继续持有 lease。恢复与 checkpoint 准备同样受预算约束，不能用 restore 绕过行/字节限额。

旧 `sql_state_budget` 在 append aggregate 上仍限制累计接受输入；在 replacement consumer 上若配置，限制**本次 latest input** 的原有 rows/retained-input bytes，不能累计被替换版本。它与新预算同时生效，不能替代新预算。`sql_compact_state_budget` 的既有“每 aggregate 必须 compact”要求保留；新路径是有限 SQL 重算而非旧 compact，故配置它且存在 replacement aggregate 时启动前明确拒绝，不能忽略或伪装 compact。`sql_snapshot_state_budget` 仍仅属于 snapshot_sql。

## Candidate commit and checkpoint identity

实现采用普通 SqlOperator 的一个**编译时选定的 replacement 分支**，复用 SQL planner/executor 和可共享的 snapshot relation/capture helpers；不要求公开新 Operator，不改变公开 StreamOutput，也不将普通 SQL 改写成公开 snapshot_sql 以绕过它的限制。

单事件顺序固定为：验证 alias/schema/取消 → 检查整输入行数/预算并持有旧 committed → 准备完整候选 → 对候选而不是旧+新历史运行 native finite SQL → 验输出精确 schema、整结果行/字节与 edge 限制 → 检查取消 → 等待 `output.emit` 成功 → 在同一不再 await 的提交段安装候选及 initialized=true、使旧 capture 缓存失效。失败、查询错误、取消或 emit 拒绝在提交点之前均不得改变旧关系、initialized、checkpoint 可见 generation 或留下失败候选。已经成功的 emit 后，commit 不再插入可能失败的取消检查；取消先于提交被观察到则不 emit、不 commit。

这是单 Operator 候选边界，不是整图或外部 Sink 事务。fan-out 部分 enqueue 成功、Sink 失败或取消竞态仍沿 G1 的成功前缀、job failure/replay/source cut/per-output delivery 合同；不宣称所有外部副作用回滚或新增 ExactlyOnce。

冻结新状态身份如下：

| Identity component            | New replacement branch                                                                                 | Existing branch                                                            |
|-------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| native Operator configuration | 仅选中节点增加 `replacement_policy="single_alias_v1"`，由持久 graph choice + 证明重新导出。            | 原 configuration bytes 完全不变。                                          |
| OperatorCheckpointCapability  | CheckpointedStateful，**state_version=3**。                                                            | 普通 append aggregate 继续2，row-local stateless 不变。                    |
| payload header                | **state_kind="sql_replacement"、state_layout=6、state_accounting=1**。                                 | compact3、retained4、ordered5 不重编码；snapshot_sql 的独立 layout1 不变。 |
| initialization                | 明确布尔 initialized；false 无 relation segment，true 必有完整 relation/schema/metadata，允许 rows=0。 | 不把旧空 payload 或0行累计状态猜成新 initialized。                         |
| retained data                 | 仅保存最后一次成功提交的完整输入及其 metadata；捕获 schema/control/数据 checksum 和行/字节 ledger。    | 不用旧 compact accumulator 或历史 retained payload 冒充最新关系。          |

严格 control/restore identity 还包含 canonical query/alias、精确输入和输出 schema digests、replacement_policy、DataFusion 版本及已有 relevant native runtime/UDF identity；外围继续校验 Program/native mode/update choice、connector/source identity、epoch/source cut 和现有 managed manifest。新 layout 的 uninitialized checkpoint 也写明确 header，不能直接用缺省空 snapshot 在新旧分支间通行。新 binary 只在选中的新分支解释 layout6，旧分支拒绝6；新分支拒绝3/4/5和 snapshot_sql1，未知 kind/layout/accounting/version/checksum 一律拒绝。

恢复先准备、校验身份/预算/取消，所有成功后安装 committed state；失败恢复不覆盖在用关系且在 Source 获取前失败。初始化后的空关系与从未初始化恢复后仍不同；下一 replacement 完整替换而非合并，恢复过程不重新 emit 已捕获结果。新 job 不 restore 就 fresh，新的 batch execute 无此 durable 状态。

原 G1 的省略 legacy 字段 canonical bytes、声明/native mode、cache/session/revision、旧 checkpoints 均保留。仅新 aggregate 声明增加自己的结构 digest 域与内容；不同 DF/SQL 构造不承诺相同 hash。同一声明 for_mode 保持 fingerprint，所有 immutable Program copy 保留 update choice。仅新 replacement native 分支加配置和 state_version 身份，不能全局升级所有 SQL。预算是 runtime tuning，不通过不同预算暗中切换 state/layout；兼容身份下收紧预算可因恢复所需容量不足失败。

QueryAnalysis wire 字段不增加，schema_version 保持2；因支持规则变化，**analysis_revision 升3** 并进入证明缓存键，旧报告不充当新支持证明。若能力清单列 SQL layouts，应如实列出 replacement6 及其选择/支持限制，保留3/4/5的原条件；不要把 capability/report revision 写入 native 执行身份。

## Common lowering and error projection

两端新增 aggregate 声明都持有 data-only 的 input、group key 顺序、命名 reducer 顺序、函数和直接列 operand；不携 Python callable/SQL callback。Rust `RelationKind` 和 Python symbolic node 可各沿现有序列化，但必须生成同义的安全 quoted SQL：分组列投影 + `SUM/COUNT/MIN/MAX/AVG`、`FROM input`、可选 GROUP BY。Rust Program lower 与 Python lower 都进入原 OperatorSpec::Sql / SqlOperator，之后由同一 Rust AST+类型证明决定 replacement；不能让 Python 单独发 supported 或实现一套逐批聚合。

| Layer               | Implementation mapping                                                                                                                                                                   |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Public construction | `dataframe.rs` 新 AggregateExpr/NamedAggregate/count_rows/aggregate；`lib.rs` re-export；Python `symbolic/expr.py`、exports/types 中同义构造。                                           |
| Planning/lowering   | `program/lower.rs`、Python symbolic lower：新增声明的 lineage/schema/name 校验、安全生成 SQL 与 source path 映射；不新增并行 table engine。                                              |
| Proof and gate      | `query_analysis/sql.rs` 的专用 replacement 证明供高层与低层共用，`updates.rs` 接入新事实；Program / builder / project / runner 都使用相同受限证明。                                      |
| Execution           | `operator/sql.rs` 中选中分支调用现有 DataFusion finite SQL；普通 append 路径原样。复用/提取 snapshot_sql 的 relation、metadata、budget、capture/prepare-install 工具而不放宽其公有路线。 |
| Runtime/recovery    | `pipeline/stream.rs`、checkpoint capability/restore 路径接新配置、版本及布局；PyO3 `continuous.rs` 和 Python runtime config/native seam 同步。                                           |
| User delivery       | native stubs/errors、`docs/relational-frontends.md`、streaming guide、两端 example32 增量展示新聚合/预算与拒绝；不存在的 dal-public/C++/Excel/component-boundaries 不构成本仓库障碍。    |

DF aggregate canonical 路径为 `P.aggregate.value` 输入、`P.aggregate.group_by[i]`、`P.aggregate.aggregates.<name>` 及其 `.value`；特殊名字继续用 JSON bracket 段。SQL 消费边沿 `P.sql.tables.<alias>`；Python 使用现有 role 到 logical path 投影，不能把生成 SQL 文本位置伪造成用户 span。低层 project 使用原物理节点/边路径。共享节点/消费边的排序、去重、选择首因规则沿 G1。

| Offending input / condition                                  | Error contract / example                                                                                                                                                                                               |
|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 空聚合、重复结果名/键、与 group key 重名                     | 构造或 planning 的 invalid_literal / duplicate_name，定位具体 aggregate/group_by；Python 类型错 TypeError，值错 ValueError。                                                                                           |
| 错 lineage、未知列、不是直接列、SUM(字符串)、MIN(非 numeric) | 沿已有 lineage/schema/unsupported_type 分层；如 `outputs.result.aggregate.aggregates.total.value: unsupported_type: SUM requires a direct numeric column; found utf8`。                                                |
| 已知 aggregate 不在 replacement 白名单                       | QueryUpdateSemantics / CompileError.query_analysis，edge 上 `update_interpretation_mismatch`，说明具体 DISTINCT/HAVING/operand 等限制。                                                                                |
| 缺少 exact schema / event boundary 或 opaque 语义            | G1 的 update_event_boundary_unproved / update_semantics_unproved；unknown 不授予 supported。                                                                                                                           |
| 缺新预算 / 不兼容 compact 要求                               | 配置/启动错误，字段 `sql_replacement_state_budget` 或 `sql_compact_state_budget`，带 offending node/alias；例如 `sql_replacement_state_budget: required for complete replacement at outputs.result.sql.tables.input`。 |
| 运行超 rows/bytes、edge 限额、计算失败或取消                 | 原 Execution/Cancelled 分层；预算消息必须含 node/alias、限制字段、required 与 limit；失败前候选不提交。                                                                                                                |
| 旧 layout、identity、checksum 或恢复预算不符                 | 原 CheckpointError 分层，包含 node 及 expected/actual kind/layout 或 required/limit；Source 前拒绝，不转换旧快照。                                                                                                     |

## Typical examples

以下是待实现调用的示例种子，语法自检不等于运行通过。Python 同时展示 DF 与 SQL，手工预期为同一 job `[1]`、`[2,3]` 的 latest 输出1、6；把选择改成 legacy，末值应为7。

~~~python
import calc_flow as cf
import pyarrow as pa

source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
first = source.aggregate({"total": source["x"].sum()})
projected = first.select(total=first["total"])
visible = projected.filter(projected["total"] >= cf.lit(0))
result = cf.sql("SELECT SUM(total) AS value FROM s", s=visible)
program = cf.Program(
    "replacement", mode="stream", outputs={"result": result}
).with_update_semantics("latest_relation")
config = cf.StreamRuntimeConfig(
    sql_replacement_state_budget=cf.StateBudget(1_000, 8 * 1024 * 1024)
)

async def observe():
    seen = []
    data = [pa.table({"x": [1]}), pa.table({"x": [2, 3]})]
    async with program.stream({"events": data}, config=config) as events:
        async for event in events:
            seen.append(event.table.to_pydict())
    assert seen == [{"value": [1]}, {"value": [6]}]

summary = source.aggregate(
    {"total": source["x"].sum(), "rows": cf.count_rows()}, group_by=()
)
finite = cf.Program("finite", mode="batch", outputs={"result": summary})
assert finite.collect({"events": pa.table({"x": [1, None, 3]})})[
    "result"
].to_pydict() == {"total": [4], "rows": [3]}
~~~

Rust 的相同声明如下，schema/compile/runner 生命周期沿现有 API；不引入另一个 runner 或 caller-owned 状态。

~~~rust
use std::{collections::BTreeMap, sync::Arc};
use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{CompileMode, InputOptions, Program, Result, UpdateSemantics, table_input};

fn declaration() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let source = table_input("events", schema, InputOptions::default())?;
    let first = source.aggregate(
        vec![source.column("x")?.sum().alias("total")?], vec![],
    )?;
    let result = first.sql("SELECT SUM(total) AS value FROM input")?;
    Ok(Program::new(
        "replacement", CompileMode::Stream,
        BTreeMap::from([("result".into(), result)]),
    )?.with_update_semantics(UpdateSemantics::LatestRelation))
}
~~~

## Why this shape and compatibility

单 `aggregate` + 命名 reducer + optional 分组保持常用调用短、参数顺序清楚，也不会与 Python 的结果名 group_by 或 SQL alias/config 冲突。独立 AggregateExpr 保留 scalar/aggregate 的类型边界；拒绝用 `ColumnExpr.sum()` 返回普通 row-local ColumnExpr 后再让 select 猜上下文。`mean` 唯一映射 AVG，不另加 avg 别名；COUNT(*) 单独用 count_rows，避免把 count(column) 的 NULL 含义改掉。

白名单先开放直接列、普通分组及五类 reducer，足以实现 SUM→SUM=6 与共同聚合语义；WHERE/HAVING 等可拆现有 row-local 边。拒绝把“SQL planner 可执行”当作全部 replacement/portability 已证明，也不重写旧 SQL 以迎合 DF/Polars。新路径重算当前有限关系，优于本片另造通用 retractable accumulator/delta 协议。

新预算独立命名且显式必需，避免旧用户突然被新默认上限拦截，也不暗改 snapshot_sql 或 compact 预算。代价明确：Rust 公有 StreamRuntimeConfig 的完整 struct literal 须补 `sql_replacement_state_budget: None` 或使用 `..Default::default()`；字段新增不是完全 source-compatible。仓内与外部 consumer fixture 随实现迁移。Python 新字段放末尾保留原位置参数含义，所有旧 None 配置的编码/身份不变。新 aggregate 是加法 API；现有 SQL、Program 方法签名不变。

旧 latest 中已获准的 row-local/累计终端图不因新配置或 layout 全局变身份；前序被拒绝的白名单边现在成为有预算的可执行域。旧 append checkpoints 仍按3/4/5及 state_version2恢复，仍可得到7；latest replacement 用独立 configuration/state_version/layout，双向实际 restore 必须拒绝误配。旧 binary 对新 layout 或 graph 字段的拒绝是正确边界，不提供剥字段降级。

## Acceptance mapping and handoff

下表是实现/独立验证必须取得的证据，本轮未执行。范围严格对应 DAL-514；相邻旧功能只选受影响回归，不新增 spec、critic、性能或简化阶段。

| Acceptance                | Minimum independent observable evidence                                                                                                                                                                                                                               |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A04 / C02 / O01           | Rust/Python 分别构造 DF→SQL→DF→SQL 与 SQL→DF aggregate→SQL，共享输出、无中间 collect；`[1]`、`[2,3]` 上 latest 1/6、legacy 1/7，夹 projection/filter 同结果。不是只读取最后快照修饰成6。                                                                              |
| A05 / C04                 | 对上述 typed 空/全 NULL/重复小表，核精确 Arrow dtype、nullable、COUNT(*)/COUNT(c)、SUM widening、MIN/MAX/AVG、NULL group；DF 与对应 SQL 都核手算 oracle。组行序按多重集，输出列序精确。                                                                               |
| A06 / C04 / L01/L02/L03   | 同 job 跨批累计 producer 1/6；同一个已编译暖 batch plan 无 reset 两次 execute 为1/5；新 job fresh；完整 H=[1,2,3] 的独立有限执行=6。便利方法每次新 plan 不能替代同 plan 隔离证据。                                                                                    |
| C12 / O01                 | 无 Data 无结果；首次空 replacement 初始化；非空→空 replacement 清除旧值/旧组；一个含多个 RecordBatch 的 Batch 只输出一次合计，EOF/控制事件不重复更新。snapshot_sql 原初始化/输出直达 Sink 限制不退化。                                                                |
| C13 / O01 budget          | 输入行、输出行、state bytes 与 edge bytes 各自有精确 inclusive 边界和不足一单位的拒绝；byte ledger 必须覆盖旧+新候选、prepared output、metadata、大 backing slice、活 capture。以实现的 accounting-v1 ledger 固定 expected charge，不能只用一个巨大通过值。           |
| C13 / O01 atomicity       | 成功提交 R1 后，分别注入超预算、查询失败、取消和 emit 拒绝；状态/checkpoint 仍R1，失败候选未初始化/未提交。首次失败仍未初始化；同 Operator 重试合法R2成功。checkpoint 编码失败/取消不安装或泄漏候选；不把 Sink 外部副作用宣称全回滚。                                 |
| C14 / A06 / O01 recovery  | 至少新 layout6 的 checkpoint→恢复→下一 replacement，与未恢复的同逻辑 alias 历史一致；另外验 initialized=false 与 initialized=true/0行的区别，未知 layout/accounting 及 legacy↔replacement 实际恢复拒绝；旧3/4/5恢复与清单条件保持。                                   |
| A09 / C13                 | 每类白名单外 SQL（DISTINCT、HAVING/ORDER/LIMIT、CTE/JOIN、函数参数/UDF等）有准确拒绝；缺预算、compact冲突、schema/opaque 未证明在 Source/iterator/bindings/job 前拒绝。暖缓存、低层 builder/project reload/显式 Runner 不可绕过；batch 合法域不受 stream 的条件误拒。 |
| A08                       | 外部 Rust public consumer 与 Python 根包可独立构造/分析/执行；新导出、annotations、native seam/stubs/errors、docs/examples 同步；data-only project roundtrip 保留 update choice、schema与共用 SQL 下沉，重载显式 budget后取得同义结果。                               |
| Identity / limited checks | legacy project/native fingerprint 与旧 checkpoint fixtures 不改；新分支身份可辨；记录本 checkout 的实际源码SHA、匹配 Python extension 来源，按触及路径做定向编译/类型/格式/示例检查。不得转抄前序测试为本 SHA 证据，全量/coverage交CI。                               |

实现可复用 `crates/calc-flow/tests/{relational_frontend,relational_program,query_history_portability,snapshot_sql_checkpoint_contract}.rs`、SQL 邻近候选/restore tests 及 Python `test_relational_frontend.py`、`test_query_history_portability.py`、`test_stream_results.py`、`test_symbolic_compile_cache.py` 的受影响小集合；独立 tester 优先验证预算/更新/恢复原子性，不要求全跑所有列出的模块。

## Open questions and delivery boundary

**无阻塞实现的未决公共合同，可进入 implementer → 独立 tester 原子性验证 → reviewer。** 函数/签名、共同类型域、replacement 白名单、显式预算入口/计量、提交点、新 state/layout 与恢复身份均在本文选定；内部 helper 的文件拆分不属于需要再审批的公共选择。若现有执行路径无法做到已承诺的候选提交、完整空事件或保留旧身份，须报告具体源码位置与失败证据，不能临时放宽 gate 或静默改用旧追加。

本轮只新增此 note，保留其他 artifacts；文档结构/路径/示例语法/表格/空白与写入范围自检，未改产品/测试/配置，未提交、推送、建 PR 或扩派。设计交付不等于 DAL-514 已合并，issue 保持 in_progress；后续实现、验证和主审继续复用协调器指定的工作树。
