# Bounded Global DISTINCT — API Note

状态：DAL-519 / S3-1 / G3 **修订版 v2：外部 Arrow ingress 归一化与预算边界**，2026-11-12，可交 implementer。**取代初版附件 `01a1d056-36e4-7e2e-9c95-cf8209f3d96f` 中关于完整 ingress backing 计费、accounting/layout=1 及 r8 准入的条款。** 下列新增入口、SQL 方言、Source 证明、状态和诊断均待实现；本文不是产品已支持或测试通过的声明。

基线：`be12fd45a18daa5d2f6415993141a8842e114b1f`，分支 `agent/dal-orchestrator/d83ca711d94c`，HEAD 与本地 origin/main 一致。初版开始时工作树干净；本修订开始 tracked diff 为零，原 note 与 analysis 未跟踪且均保留。前序 PR #602 已合并由交接和父 issue 确认。本轮只修订本文并向原 analysis 追加决策，未重新查询远端 CI。

修订输入：触发评论 `01a1d06a-103e-75fe-8e90-8b614fb08ae8` 与现有 [accounting evidence](../analysis/bounded-global-distinct.md)。实现者在相同锁定版本观察到：1行非零 offset 输入的 native array charge 为112 B、FFI capacity 为16 B，却仍持有8,388,608 B父 buffer；删除原 Python 引用后仍存活，删除 Rust owner 才释放。**初版把 C Data 可见容量当完整父分配证明，这一前提错误。** 本轮复用该 RED 证据和日志，不重跑复现或产品测试；analysis 的历史正文保持，决策追加在末尾。

**用户可依赖的新保证：** DISTINCT 总是在本节点接纳时复制所需的可见值，只保留自身创建、所有权可验证且已预留的缓冲。`max_bytes` 限制本节点自有存储及下文列明的同时存活项；**不限制临时借读的外部 ingress 父分配，即使原生入站 envelope 正在为本次调用保活它**。该未知父分配没有被计为0、112 B或某个附加费，而是本修订明确缩窄后的保证范围之外。调用者、原桥接、Source/graph/queue 可能继续持有原输入；本政策不提供整个执行/job/process 的持有字节上限。代价是本特性多一次受预算的复制；同一可见内容的大 slice 和紧凑输入在归一化后使用相同计量规则。四单元的公共方法、结果语义、历史/闭合与旧 SQL 默认保持。

## Audiences, authority and scope

Rust 用户通过公开 TableExpr / SQL / Program 声明、schema、分析、执行和导出使用本能力；Python 用户从 calc_flow 根包获得同义声明及现有 owned stream 生命周期。DataFrame / SQL 是两类接口，batch / stream 是独立生命周期。四单元必须落到同一个 native DISTINCT 语义，不能用 Python set 或单独 batch SQL oracle 充作实现。

依据：本 issue 与触发线程 `01a1d047-a03d-735a-ade7-73dfbb766861`、父 DAL-498 的四单元和生命周期讨论、经 Multica CLI 下载读取的开发计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`，以及 [frontend](first-class-relational-frontends.md)、[query history](query-history-portability.md)、[latest](latest-relation-update-contract.md)、[replacement](relational-aggregate-replacement.md)、[finite window](finite-event-window.md) notes。历史 note 的候选和旧基线不替代本 SHA 的源码事实。

本片选择两个明确的历史域：**一个有界 Source 的完整接受历史；一个显式固定 UTC 半开时间区间的接受历史**。后者是一个选定窗口，不是无限序列 tumbling/hopping 窗口上的逐窗 API；不增加几何、offset、epoch 切片或窗口 aggregate。两者均 final-only，每个节点最多产出一次最终关系。没有 provisional、增量 insert/update、撤回、DISTINCT ON/keep、排序/top-k/OVER、replacement DISTINCT、一般多源或 batch 中途恢复。

遵循 AGENTS.md、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md`。实际边界为 Rust core → PyO3 → Python adapters；本仓库没有 dal-public、C++/Excel binding、methodology 或 `.codex/references/component-boundaries.md`，不补造这些组件。保留现有 core/Python 例子位置和用途，新例另放 `examples/bounded_global_distinct.py` 与 `crates/calc-flow/examples/bounded_global_distinct.rs`。

## Surface today: verified source evidence

下表均为基线已有事实。Rust 路径相对 `crates/calc-flow/src/`；Python 路径写出完整前缀。

| Evidence                                                                      | Existing contract / gap                                                                                       |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `dataframe.rs:75`、`sql.rs:60`、`program/lower.rs:23`                           | 已有 typed relation、SQL 宏识别与共同 lowering；RelationKind 没有 distinct，普通 SQL 不会自动成为 global relation。                 |
| `query_analysis/sql.rs:145`                                                   | DISTINCT 被识别为 per-batch，带过滤等形状可为未证明；不能把旧 SQL 改标签当新能力。                                                         |
| `operator/sql.rs:1236`、`:1316`、`:1400`、`:1436`                                | 普通非 aggregate 使用当前 Batch；finite 调同一 native SQL executor；stream on_end 不补 final DISTINCT。                      |
| `continuous.rs:285`、`:348`、`:567`                                             | SourceCapabilities 有 schema/replay/delivery/单批限额/水位，没有 boundedness；StreamSource 有默认可扩展方法，SourceBinding 拥有连接器。 |
| `runtime/streaming/operator_task.rs:5003`                                     | 既有断言明确 sender drop 是 edge failure，不是正常 EOF；不把“读不动了”当历史闭合。                                                     |
| `operator/window.rs:2015`、`:2042`、`program/lower/window.rs:15`                | 真实水位关闭 end<=W，正常 end 排空；有原样/纯重命名时间列的局部 SQL 证明，可复用时间坐标边界。                                                      |
| `operator/state_budget.rs:32`、`operator/snapshot_sql/mod.rs:484`              | StateBudget 正整数、JSON-safe、inclusive；已有 backing helper，但 v2 不把它当外部 C Data 父分配的权威计量。                            |
| `operator/sql/metadata.rs:32`、`operator/snapshot_sql/checkpoint.rs`           | 已有 metadata 预付和准备恢复后安装机制；新状态仍需独立身份，不能复用旧 SQL snapshot payload。                                                |
| `pipeline/batch.rs:556`、`program.rs:51`、`lib.rs:94`                           | finite execute 的运行上下文隔离；已有 Program、CompileMode、数据类导出，声明不拥有行或 job。                                             |
| `python/calc_flow/runtime.py:295,397,430,458`                                 | SourceCapabilities 为 frozen 数据类；StreamSource Protocol；from_arrow 的材料有限，但当前没有给 global DISTINCT 的证明字段。          |
| `python/calc_flow/_stream_inputs.py:145,296,333`                              | 普通 async iterable 只有 schema/progress descriptor；materialized sequence 可先合并，不保证保留原始 Data 事件边界。                 |
| `query_analysis.rs:484`、`python/calc_flow/symbolic/query_analysis.py:248,613` | 当前报告为 schema 2 / revision 7，strict/latest gate 要求匹配 r7；本片需要同步提升。                                              |
| `crates/calc-flow-python/src/config.rs:944`、`python/calc_flow/_native.pyi:16` | 已有共享 native SQL/window 描述 seam；新增 DISTINCT 也必须由 Rust 判 AST/schema。                                            |

另已读取两端 `finite_event_window`、example 32、相关 exports、Python TableExpr/SQL adapter 与 native stub。这里只核对源码和现有断言，未执行产品测试。

## Frozen public signatures and data

Rust 新增 DistinctSpec / DistinctHistory，定义归属 operator 配置并由 crate root 导出；没有 Default，预算必须显式传入。配置为 owned、Clone/Debug/Eq、严格 serde/JsonSchema 数据，不含 runtime、行或回调。私有字段有下列只读 getters；枚举 wire 值为 snake_case。

v2 保留以下签名，两个构造器均固定选择 `ingress_policy="copy_visible_v2"`；没有 no-copy/信任大小/跳过归一化选项。方法 rustdoc、Python docstring、支持表和 explain 必须直接说明上面的借读 ingress 例外，不能继续写“max_bytes 覆盖全部输入 backing”。

```rust
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DistinctHistory {
    BoundedSource,
    ClosedWindow {
        event_time_column: String,
        start_micros: i64,
        end_micros: i64,
    },
}
pub struct DistinctSpec { /* private fields */ }
impl DistinctSpec {
    pub fn bounded_source(state_budget: StateBudget) -> Self;
    pub fn closed_window(
        event_time_column: &str, start: EventTime, end: EventTime,
        state_budget: StateBudget,
    ) -> Result<Self>;
    pub fn history(&self) -> &DistinctHistory;
    pub fn state_budget(&self) -> StateBudget;
}
impl TableExpr {
    pub fn distinct(&self, columns: Vec<String>, spec: DistinctSpec) -> Result<Self>;
}
```

Python 从 calc_flow 与 calc_flow.symbolic 导出 DistinctSpec；沿 `_compat.dataclass(frozen=True, slots=True, init=False)`，不提高 Python 3.9 下限。无公开任意字段构造器；只用两个不同历史域的构造器。声明调用复制列序列和配置，不受调用者后续修改影响。

```python
class DistinctSpec:
    @classmethod
    def bounded_source(cls, state_budget: StateBudget, /) -> DistinctSpec: ...
    @classmethod
    def closed_window(
        cls, event_time: str, /, *, start_micros: int, end_micros: int,
        state_budget: StateBudget,
    ) -> DistinctSpec: ...

class TableExpr:
    def distinct(
        self, columns: Sequence[str], /, *, spec: DistinctSpec,
    ) -> TableExpr: ...
```

`columns` 是**最终关系的全部列和去重键**，按此顺序输出；语义为投影这些列后 DISTINCT，不保留非键列，也不挑 first/last representative。必须非空、名称唯一、精确存在于本节点输入；拒绝 str/bytes 冒充序列。全部列去重时显式传全部列名；不增加 subset/keep/maintain_order 同义选项。别名、表达式键先用已有 select/with_columns 声明；窗口时间列在本节点输入中可存在而不在 columns 中。

closed_window 要求非空时间列、`i64::MIN <= start < end <= i64::MAX`，负时间合法；Python 必须是真正 int，拒绝 bool/float。这里名字表示**等待该选定窗口闭合**，不是声称配置对象已经证明未来闭合。Program/for_mode/schema/analyze/compile/execute/collect/stream/to_project 入口保持，不加 flush/finalize、公开 EOF、水位注入或另一套 Runner。

## Four-unit SQL spelling and common lowering

两语言保留现有 SQL 签名；TableExpr.sql 仍绑定 input：

```rust
pub fn sql(query: &str, tables: BTreeMap<String, TableExpr>) -> Result<TableExpr>;
```

```python
def sql(query: str, /, **tables: TableExpr) -> TableExpr: ...
```

冻结两个**新增 calc-flow 方言宏**，不是 DataFusion 内建函数或普通 SQL 默认的变化：

```sql
SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 1000, 8388608)
SELECT DISTINCT x FROM CF_DISTINCT_WINDOW('a', 'ts', 0, 10, 1000, 8388608)
```

BOUNDED 参数是 `(input_alias_string, max_rows, max_bytes)`；WINDOW 是 `(input_alias_string, event_time_column_string, start_micros, end_micros, max_rows, max_bytes)`。这是 SQL 配置编码；公共宿主方法仍使用小配置对象。名字不带 schema、大小写不敏感；字符串解码后精确匹配 alias/时间列，数值只接纳范围内十进制整数 literal，window 两边界允许负号。预算同 StateBudget，时间边界同上；不用 SQL float/interval/表达式、NULL、参数占位或隐式 cast。

支持域固定为单顶层 SELECT DISTINCT、恰好一个宏 FROM、tables 恰好一个被引用一次的 alias；SELECT 为一个或多个直接列引用（可限定名/quoted identifier），或单个 `*`。`*` 经精确 schema 展开且必须非空。输出名按输入列名，顺序按 SELECT，不准重复列；重命名在前后独立 row-local stage 完成。没有 DISTINCT ON、表达式/AS、WHERE/HAVING/QUALIFY、GROUP BY/aggregate、OVER、CTE/子查询、JOIN/set operation、ORDER BY、LIMIT/OFFSET/FETCH、UDF/provider/volatile。需要过滤时先声明前一关系；窗口 membership 总在键投影之前判断。

Rust 原生 AST/schema seam 一次识别 alias、columns 和 DistinctSpec，供两端构造、schema、analysis、compile、export 使用。Python 不维护 parser、类型或去重实现。出现任一 CF_DISTINCT_* 名但形状不符必须指出违反项并拒绝，不 fallback 每批 SQL。**没有这两个宏的 `SELECT DISTINCT`、旧 `kind=sql` 项目和 SqlOperator 继续原行为。** 低层 SqlOperator 不新增宏解释器；typed SQL 宏导出实际 distinct 节点。

共同关系 IR 新增一个 distinct(input, ordered columns, spec) 声明，四单元均降到同一 native 节点、native 键投影和 DataFusion 55 DISTINCT 内核；finite 和 stream-final 均复用此内核、精确 schema 和受预算的 collector。不要求跨语法构造的 fingerprint 相同。可抽取现有 native helpers，不在 Python 重算、不物化中间表、不在 finite 中启动持久 stream job。

最小可组合路径为一个 exact-schema input → 已证明 deterministic 单输入 row-local DF/SQL 串 → 一个 distinct → 已证明 row-local DF/SQL 串 → 命名输出。允许独立 distinct、共享 fan-out 和无关合法输出；同声明在一个 job/一次 finite execute 只有一个 owner。distinct 路径内 aggregate/snapshot、union、多源/join/ASOF、rolling/cross-section、既有 window aggregate、嵌套 distinct、external/array 均不在新域；无关分支的既有支持不收紧。时间域沿现有 `program/lower/window.rs` 的原样/纯重命名证明穿过上游 SQL，不能按同名/dtype 猜时间来源。

## Bounded-source proof and actual closure

当前没有有界证明，故本片需要以下小增量，**不能直接假设现有 SourceCapabilities 已提供它**：

```rust
// crate-root export; no change to existing SourceCapabilities struct literals.
pub enum SourceBoundedness { Bounded, Unbounded, Unknown }
// Add this default method to the existing StreamSource trait.
fn boundedness(&self) -> SourceBoundedness { SourceBoundedness::Unknown }
```

Python 根包导出同名 str Enum（BOUNDED/UNBOUNDED/UNKNOWN，值 bounded/unbounded/unknown），SourceCapabilities 最后追加默认字段 `boundedness: SourceBoundedness = SourceBoundedness.UNKNOWN`。Python StreamSource Protocol 不增加必选方法；PyO3 从 capability 新字段投影到 Rust trait 方法。Rust 默认方法避免破坏已有 SourceCapabilities struct literal；旧连接器无需修改仍可运行旧图，新 DISTINCT 不能把默认 Unknown 当 Bounded。public SourceAdapter、内部 Source descriptor 和原生 Arrow/Python adapters 必须传播同一份冻结的事实，不能在任一转接层又落回默认 Unknown；不通过先打开连接器补证明。

每次 job preflight 缓存同一连接器的描述：Bounded 是连接器的权威承诺——本次逻辑输入有固定有限终点，恢复时继续同一 cut，而非重新观察增长数据；它不是总行数上限、无丢失、可重放或 exactly-once 证明。Unbounded 是明确无有限 cut，Unknown 是未声明；这两者区分诊断。纯数据构造的 native Arrow source/SourceBinding.from_arrow 给 Bounded；完整材料的 convenience input 和已有材料 list/tuple 路径给相同证明。普通 AsyncIterable/iterator 不尝试 next/耗尽来推断，保持 Unknown；第三方有界连接器可显式实现合同。文件尾、单批上限、replayable、source history archive 或 `__len__` 单独都不是该证明。

编译不拥有 Source，故可对 admitted distinct 结构给条件性 supported，reason 必须列出 binding 义务。plan 保存该路径需要的证明；**所有实际启动入口**在 source.open/next、async iterator 获取/遍历、sink.open、job/state root 创建之前检查真实 descriptor。包括显式 Runner、Python convenience/async、暖缓存和 project reload；无 portability/latest opt-in 也强制检查。可读取无 I/O 的 descriptor，不靠启动 job 探测。bounded_source 只接受 Bounded，Unknown/Unbounded 拒绝。历史 replay/delivery/schema gate 仍独立生效。

| Domain / observation                  | What proves admission and what closes it                                                                                     |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| bounded_source                        | preflight 的 Bounded 证明允许保存历史；全部上游 Data 处理完毕后的真实正常 EOF 才闭合。证明本身不触发 final。                                                     |
| closed_window(start,end)              | 显式区间、时间坐标来源及可用闭合路径允许保存区间内历史；有效输入水位 W>=end 或真实正常 EOF 闭合。源可以无界，但须有可信水位路径。                                                      |
| window 的 closure route                | Bounded 源可只靠 EOF；否则须经既有 preflight 证明的 source-native 或 runtime-generated 水位策略。Unknown/unbounded 且无水位路径在启动前 finality_unproven。 |
| W<end、Data 空批、Idle、Barrier、checkpoint | 不是闭合；不发 provisional/final，不 reset。仅配置窗口但当前尚未闭合时保持 pending，不能伪造 finality。                                                     |
| 取消、错误、sender drop、连接中断                | 失败，不等同于正常 EOF；不得输出成功 final。                                                                                                  |

这里只有一个上游 Source 的路径，不发明多源 cut。正常 EOF 表示 runtime 已按 FIFO 接收并处理全部合法输入且不会再有 Data；不是 i64::MAX 水位。bounded_source 模式下，即使一个 Unknown 连接器后来恰好结束，也不能倒推通过它本应失败的预检。window 模式下，正常 EOF 能终结已声明区间；无需伪造 end 水位，也不自动证明整个 Source 是有界类别。

窗口 membership 用 `start<=EventTime(ts)<end`，NULL 时间不进入 H；范围外行不进入 H。有效水位由原 runtime/lateness 协议提供；不再次手工加减 allowed_lateness。保留窗口 assignment 的迟到定义：只有所属区间已经关闭的输入才对该区间迟到，W<end 不提前丢掉已落在区间内的行。关闭后区间内行按现有 error/drop 选择处理（drop 不再输出），区间外行忽略；不支持 side-output 选择，须启动前拒绝。没有窗口边界列自动加入结果。

闭合时计算并验证完整 DISTINCT 结果，再 emit **一个完整 table Batch**，可以内含多个 RecordBatch；0行历史也必须发一个有精确 schema 的空 final，不能把“从未初始化”和“空最终关系”混淆。emit 成功后标记 closed，清理已不需的历史，重复 end/后续正常 end 不再输出。窗口在水位提前闭合后仍遵守剩余 Source/job 生命周期，不自行伪造 Source EOF。输出太大不能截断或分割成不完整关系来绕过 edge 限额。

## Finite execution, NULL/schema/order and updates

finite process 把本次 input Batch 的所有 RecordBatch 作为完整 H，一次全量准入后运行同一 distinct finalizer；window 域先按同一区间筛选，再投影 keys。无需 Source descriptor/水位/Checkpoint，因为 finite 的本次调用已经给出完整输入。只在所有步骤成功后返回结果；不模拟逐条 stream 控制消息。每次 execute 拥有 fresh history、closed、sequence 与 query workspace；同 plan、同 Runtime、暖缓存、无 reset 的后续成功/失败/取消后执行都不得继承状态。内置 batch snapshot 为 NULL，拒绝非 NULL restore，reset 不承担清理先决条件。

新增 keys 域选定为 Boolean、Int8/16/32/64、UInt8/16/32/64、Utf8/LargeUtf8、Date32/64、Timestamp[us] naive/UTC，均可 nullable。Float32/64（含 NaN/Inf/signed-zero representative）、Decimal、nested、dictionary、extension、binary 和其他时间类型作为 keys 首片准确拒绝；这不缩减旧 SQL DISTINCT 的类型域。window 的时间列另允许 timestamp[ms] naive、timestamp[us] naive/UTC，按现有 EventTime 的 checked import；可不作为 key。时间坐标算术/DATE_BIN/cast 不能伪装成 passthrough。

同 dtype 的键按 native SQL DISTINCT tuple 等价，NULL 与 NULL 属于同一 distinct 值；不是 SQL `NULL = NULL` predicate。多个键中每个位置相同（包括 NULL）才归一；没有跨 dtype/NULL 和0/空字符串合并。输出列名/列序/dtype/nullable 按选定输入字段精确投影，经 native planner 核对；空结果保持同 schema，不观察首批值来收窄 nullable。沿已有 frontend/Port 规范化 metadata，调用者 Arrow/Batch metadata 不被修改。声明缺列在读取前定位；运行中 schema 漂移整 Data 失败。

输出产生新 row lineage，清除原 entity/event_time/sequence/temporal ordering。**行序未规定**，不承诺 first-seen、字典序、稳定跨切批/恢复排列或 SQL 输入顺序；只按无序关系比较键值与重数（最终每个 tuple 一行）。后接排序属 DAL-520，不在本片隐式补 ORDER BY。

输入解释恒 append；stream 输出为 append/group_final，history 为 accepted_history 或 event_window。batch facts 为 current_batch/append/per_batch，指本次完整 H 内的最终结果，并不承诺与每个 stream Data 边界相等；两 mode ordering 均 unspecified。bounded_source 虽然只发一次完整关系，仍不是 revisable complete_snapshot；后续 row-local 保留此事实。显式 latest 下真正 append 可以运行；complete_snapshot 经任何 row-local 转接到本节点都在 offending edge 报 update_interpretation_mismatch，不按首批/空快照/finite wrapper 猜测安全。不支持 replacement、撤回或从最后快照反推历史；legacy 新路径也不接纳 stateful upstream。旧已合法图、旧快照链7和已获准 latest 链6/拒绝均保持。

## Required budget and failure atomicity

每个 DistinctSpec 必带现有 StateBudget，max_rows/max_bytes 都在 `[1,9007199254740991]`，上限 inclusive；无隐式无限值、spill/truncation/近似去重。预算是节点配置，两个生命周期相同生效，既有 sql_state/sql_compact/sql_replacement/sql_snapshot budgets 不替代它。shared owner 只收费一次，不同节点各有独立限额。

选择保留**域内投影后的完整输入历史，含重复 tuple**，最终才调 native DISTINCT。max_rows 限制累计接纳的域内历史行数（重复也计）、候选历史及最终结果各自的最大值，不是 unique-key 数、RecordBatch 数或 Source 原始范围外行数。例 `[1]/[1]` 需要 max_rows>=2，即使最终只有1行；window 以 membership 后的行计数。0行也有 schema/control/metadata 字节收费。状态优化以后若改变计量必须另设 accounting identity，不能静默把此合同换成 distinct-key 数。

### v2 policy: borrowed ingress, mandatory owned normalization

本节**替代初版整个 max_bytes/backing 计量及 Data 事务段落**，包括初版“整个 ingress Batch 的暂存不能免费”的保证。那个保证对无父分配大小信息的 C Data 无法成立；本片选择公开缩窄到本节点自有存储，不继续以旧 helper、固定倍率/附加费或调用者自报数字掩盖未知量。

归一化边界是共同 native distinct 节点的 ingress，处于既有桥接/上游 row-local 运算之后、任何 distinct retained state/DataFusion final 查询之前。Rust/Python、DF/SQL、finite/stream、低层 node/project reload 均经过同一个边界；不改 `Batch.from_pyarrow`、`SourceBinding.from_arrow`、旧 SQL 或整个 graph 的默认零拷贝行为。有限 executor 当前保留 inputs/produced_values（`pipeline/batch.rs:556,650`）；ArrowSource 也保留可复用材料（`crates/calc-flow-python/src/arrow_source.rs:19,56`）。因此不能声称 node 复制后这些上游 owner 立即释放。

在同步接纳阶段，节点仅借读当前 raw Batch 的 schema、选定 keys、窗口时间和必要 metadata，不把 raw Batch/ArrayRef/Buffer/SchemaRef/metadata Arc、Python capsule/PyObject、C Stream reader 或其闭包装进候选、缓存、状态、checkpoint。Rust `&Batch` 借用不延长上游 Arc 的生命周期；stream 接口拥有的当前 Batch 在归一化和借读结束后显式放开。不得在节点第一次内部 await、emit、异步查询或 checkpoint 工作期间继续持有 raw ingress 的强引用；取消/失败在该借读段内观察或退出时同样释放节点取得的临时引用。排队、driver、Source 或调用者仍有的引用在上述保证范围之外；不是节点替它们证明已计量。

**所有输入一律复制需要的可见值。** 不通过 Rust/Python来源、offset=0、capacity、强引用计数、private_data、replay/Bounded 声明或用户传来的 bytes 值选择零拷贝捷径。内部 provenance 只能由本节点分配路径创建，不能从公共 Batch/JSON 导入或让调用者伪造；本片不新增大小断言/owner-token API。输入拥有未知 backing 本身不是拒绝理由；若支持 dtype/合法 Arrow 表示无法用安全逐值读取重建，则该具体列以 `input_normalization_unsupported` 拒绝，不能整体拒绝 Python/Arrow。

对 first-version 支持的标量 keys 可用安全 typed reader 加新建 Arrow buffer 重建：整数/日期/时间只复制选定位置的值，Boolean 重新打包 bits，Utf8/LargeUtf8 重建从0开始的 offsets 和非 NULL 字符串字节，NULL 的无效 payload 置零/省略。window membership 在借读原时间列时判断，时间未作为 key 就不保存它。新数组 offset=0，不含原值 buffer、null bitmap、offset buffer 或 foreign release owner。**`clone`、`slice`、`project`、`to_data`、普通 concat/take 或 `to_pyarrow` 不能单独作为已复制证明**；允许 kernels 仅当已证明该具体调用分配新存储且没有 optimized alias 分支。

需要随规范化数据保留的 schema/metadata 值也必须深复制为节点所有的表示，内容、字段 metadata 和 Batch.metadata 沿既有规范化合同；不得只 clone 外部 SchemaRef/FieldRef/Arc<JsonMap> 后按新值长度收费。无需保留的原值只借读。包括空数据的 schema/metadata、源码名/属性、长 key/value、嵌套 JSON、容器 spare capacity 和编码临时对象都不能豁免；原输入不修改。managed source cursor/cut 仍归 runtime，不随 key history 保存一个 raw Batch 来间接保活它。

### Allocation-before-reservation is forbidden

顺序固定为**借读验证/计数 → checked 计算有限需求 → 预留全部下一阶段的已知 charge → 分配/逐值复制 → 验证新存储及实际容量 → 放开 raw 引用 → 后续查询/提交**。计数阶段不 collect 行、拼接字符串、构造全行索引或调用会分配整表的 encode/copy；可两次遍历重算 membership，并周期检查取消。受预算的 scratch 如果确需创建，也先 reserve。不能先分配/复制再以最终数组的大小补记账。

设本 Data 域内被接纳行数为 N，`c64(n)=64*ceil(n/64)`（checked，c64(0)=0）。新 value buffer 的已知请求分别是 fixed width `N*width`、Boolean `ceil(N/8)`；有 NULL 时 validity 为 `ceil(N/8)`；字符串 offsets 为 `(N+1)*offset_width`，values 为扫描得到的被接纳非 NULL 字符串 UTF-8 总字节数。先分别为 c64 后的布局预留，再用所锁 Arrow 的显式容量 fresh buffer 构造；offset/remainder bits 与 NULL payload 规范化，不复制无关父区段。所有计数、alignment、isize/offset 范围在分配前校验。零行字符串 terminal offset、数组/Batch/Vec 描述符仍收费。

此选择的可实现依据是本地锁定的 `arrow-buffer 59.3.2`：`buffer/mutable.rs:139` 的 with_capacity 先按64字节取整、以该 Layout 分配；`buffer/immutable.rs:560` 将新 MutableBuffer 的拥有权移动到 Buffer。`reserve` 可能倍增，故不能让普通 builder 在未预付时隐式增长。实现可用安全的显式 fresh buffer 和 array constructors，不新增 unsafe。复制后比对自己的 capacity/provenance 是实现不变量核对，**不是从外部 C Data 反推出父分配的依据**。不可预付的实现路径应准确拒绝该路径；不能在超出预付后谎称仍有严格峰值保证。

schema/metadata 与 IPC 另有预留，不能在复制大字符串/map 或 schema IPC 编码后才计长度。`operator/sql/ipc.rs:127` 的 `schema_bytes` 实际测的是内存中 Schema/Field/metadata capacity，**不是初版误写的 `IPC(schema).len`**。可复用它核对已深复制的规范 schema，但须先按已扫描字段/名称/metadata 的大小和容器上界预付构造；`operator/sql/metadata.rs:30,51,91` 提供从借读内容计算的 encode/decode 预付模式。编码 bytes、解码后的自有 metadata 和 schema 临时对象同时存活时全部累加。固定 schema/容器描述符收费仅支付已知结构，不能代替未知外部 data allocation。checkpoint writer 的增长可复用 `operator/sql/ipc.rs:153` 的先预留旧+新容量再增长模式。

### Exact max_bytes coverage and release

本政策的 runtime charge 是 **M(t)=所有仍由本节点持有的角色 lease 的确定性 logical charge 之和**；每个预留点要求 `M(t)+next_charge<=max_bytes`。不同角色保守分别收费，共享已归一化 buffer 也不能靠 pointer 合并缩短 lease；无需发明完整跨语言 owner registry。包含如下同时存活项：

| Item / lifetime                                  | Covered by max_bytes under copy_visible_v2                                                          |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 原始 ingress / foreign 父 buffer 与原 schema/metadata | **不覆盖**借读 owner，包括 node 同步读取时为它保活的临时 raw envelope；未知量没有被测出。未复制原输入绝不进 retained/candidate。            |
| 归一化目标、选取 scratch、新 schema/metadata/编码            | **覆盖**，从分配前预留直到该角色释放；raw 借读还在时目标和旧 committed/capture 同样计入 M。                                        |
| committed、candidate history 及容器                  | **覆盖**，含重复 tuple、全部自有容量、描述符、schema/metadata；旧+新同时存活时两者收费，提交后只释放已不持有的角色。                             |
| native query workspace/result 与最终 compact 输出     | **覆盖**节点所用 DataFusion reservations、结果 backing 和输出归一化目标；alias 到自有 H 的结果保守按完整父容量计，不能仅计 selected rows。 |
| checkpoint captures/segments、control/IPC/scratch | **覆盖**，捕获/旧状态/新 Data 同时在场时累加；lease 随实际活 capture 到最后释放，提交/发布不提前免账。                                   |
| 成功移交后 solely-owned-by-downstream 的输出             | 移交结束后不再是本节点预算；等待 emit 和本节点仍持有它的提交段均计费。下游独立复制/队列/Sink、caller、上游图/Source 属各自边界。                       |

自有 Batch 的基础 charge 保留 `512 + owned_schema_charge + sum(256 + owned_array_charge)`，其中 array 的完整新布局容量和已知数组描述符可用 native size helper 核对；**该 helper 仅用于 provenance 已证明的自有数组**。额外 Vec/String/Box/schema/metadata、编码和工作空间按可验证容量/预付上界计。旧 helper 不升级成通用外部分配计量器；进程 allocator bookkeeping/实际 RSS 与 DataFusion 不进入既有 MemoryPool 的私有开销仍不在此 logical budget。DataFusion pool 只负责其登记的 allocations，不能覆盖归一化目标、Arrow 输出或 metadata 的其他存储。

最终 native DISTINCT 只读规范化 H；emit 前再将最终结果复制成独立 compact 的自有输出，防止1行最终结果经 zero-copy slice 间接保留整个 H。源结果、H、输出副本、schema/metadata 和复制 scratch 的同时峰值均计费。此后输出拥有权可以独立移交下游，不借 raw ingress owner，也不要求修改旧 graph 的全局 buffer ABI。输入/输出 edge 限额、SQL result cap、取消仍独立生效；edge 的 visible-byte charge 不是本政策的父分配证明。

Data 事务：借读并验证/计数整个 Data → 为新目标/metadata/scratch 与候选预留，同时保留旧 committed/captures → 完成强制归一化并放开节点 raw 引用 → 验证完整候选 → 同一无 await 提交段安装自有 history/accepted_rows，invalidate capture。失败不安装部分 RecordBatch、不前推 initialized/generation/sequence；释放本节点 raw、目标和临时 leases，不把异常/闭包/缓存留成 raw owner。

闭合事务：保留旧自有 H/水位/closed → 对自有 H 运行 native DISTINCT → 对结果和 compact 输出依次预付/复制/验证 → 验 edge 和取消 → 等待 output.emit 成功 → 无 await、无新的可能失败取消检查地安装 closed/已处理水位/sequence、释放 H 及本节点输出引用。失败前全部自有结果和查询 leases 释放，保留旧已提交 H/closed；没有 raw ingress 被带到这段异步工作。checkpoint prepare/restore 同样先计量/校验候选再安装，restore 不读取或保存 foreign ingress。

这只是单 Operator 状态提交保证，不是 fan-out/Sink 的全图回滚。部分 enqueue、外部副作用及恢复重送沿现有 source cut/per-output delivery/成功前缀协议；不新增 ExactlyOnce。窗口先前已发出的最终关系不因后来的其他事件错误而回滚。

## Analysis, export, checkpoint and compatibility identities

QueryAnalysis 提升为 **schema_version=2 / analysis_revision=9**，取代初版未发布的 r8 合同；新增规则继续使用现有 history/update/finality/order 枚举。Rust producer、Python parser、strict/latest gate、缓存 gate 和 binding/stub 同步。当前已有的旧 r1–r7 可展示，但不能当本片准入证明；旧设计 r8 也不作为 v2 执行凭据。analyze/explain 不读行；结构 supported 的 reason 必须公开 `copy_visible_v2`、owned-only peak/borrowed-ingress 例外、预算与 binding/closure/接受集合前提。实际绑定拒绝走既有 preflight error，保留 path/code/status，不把动态大小失败假称编译可证明。

新增 capability `distinct@1`：required table input/output、modes=(batch,stream)、finality=group_final_append_only、requires_datafusion=true、stateful=true、microbatch_invariant=true（相同接受集合/控制轨迹、预算成功下的无序关系）、checkpoint_support=checkpointed_stateful、state_version=1、state_layouts=(2,)。`requires_watermark=false` 表示没有无条件水位要求；窗口特定 closure gate 仍必须执行，不能据此放过无界且无进度的 binding。只扩展清单，不改 capability wire 字段；有限临时状态和 stream 恢复、v2 借读例外均写进支持表。

bounded_source 对 `final_relation/accepted_history/all_sources_eof/unordered` 可给条件性证明：同一完整 H，stream 接纳相同 tuples、真实 EOF、全部预算成功；finite 对显式 H 计算。closed_window 对 `final_relation/event_window/closed_window/unordered` **仅当该输出路径能唯一解析到本片同一固定区间**给证明；finite 同样筛选该区间，不再有 finite-window note 中“任意窗口集合/中途前缀未选择”的歧义。跨多个不同区间/混合截止点、无具体区间的旧窗口输出仍 finality_unproven/unknown。窗口到 all_sources_eof 的比较也要求同一区间的接受集合（尤其 late drop 后）；不把所有原始行自动当相同 H。per_batch 和 declared 行序请求仍拒绝，ordering 未规定；portability 不验证两个实际输入相等或承诺未来进度。

冻结导出为 project-v3 新 variant `OperatorSpec::Distinct { columns, spec }`，wire：

```json
{
  "kind": "distinct",
  "columns": ["x"],
  "spec": {
    "ingress_policy": "copy_visible_v2",
    "history": {"kind": "bounded_source"},
    "state_budget": {"max_rows": 1000, "max_bytes": 8388608}
  }
}
```

window history 精确编码 `{"kind":"closed_window","event_time_column":"ts","start_micros":0,"end_micros":10}`。必需的 table input/output ports 带 exact schema。新增必填 ingress_policy 只接受 copy_visible_v2；缺失、未知或请求旧政策均在导入/编译前拒绝，不补成默认旧条款。两个 DF 构造器和两个 SQL 宏自动生成它，没有额外用户参数。禁止 unknown fields/version；columns 顺序、history、budget、policy 都参与配置/声明 identity。SQL 宏导出同一数据 variant，不留下 parser/callback/globals。project format 仍3，生成 JSON schema 只增加该 variant/配置定义；逻辑/物理 binding、共享 owner、全图其他分支和更新选择 round-trip 保持。

v2 native 配置语义为 `distinct_v2`、state_version=1、**state_layout=2、accounting_version=2、ingress_policy=copy_visible_v2**；关系值/终态语义不变，新的所有权/计量合同必须独立标识，不占用 SQL layouts3/4/5/6 或 window layouts。Checkpoint payload 包含精确 spec/columns/schema identity、policy、含重复的规范化投影历史、accepted_rows、closed、最后已提交水位、output sequence 和必要自有 metadata；初始空历史也有可恢复状态。只写入自有数据的 IPC/控制信息，不写 foreign pointer、进程 owner token 或“自报父大小”。restore 解码必须在预留后创建新的自有表示，不能相信 payload 声称 owned 就绕过分配/计量。绑定有界/水位路径的证明作为准入依赖冻结/复核，不能恢复时把 Bounded 换 Unknown 或改变 finite cut。源自身历史/cursor 身份仍由现有恢复协议验证。

初版 v1/layout1/accounting1 尚无交付功能/checkpoint，原型已撤回；不做把“可能保留 foreign owner”的旧状态自动升级成 v2 的迁移。缺 policy、layout/accounting=1 或任何不匹配在分配大解码目标、状态安装和 Source open 前拒绝。新 policy 纳入声明、prepare/cache、配置与恢复身份，即使不同输入的可见值相同也不能把旧计量证明复用到新合同。已有旧 SQL/桥接项目和 checkpoint 字节不因本改动重写。

恢复须同时校验 graph/config hash、mode/update semantics、native engine/Arrow、state/layout/accounting、schema/columns、domain/cut 和实际 Source 能力；计数/行数/closed/segments 不一致、未知字段/布局、超预算均在安装和 Source open 前拒绝。未闭合时 retained_rows=accepted_rows、output_sequence=0；闭合后 retained_rows=0、output_sequence=1，accepted_rows 保留总计数且仍<=max_rows，不能拿清理后的0行伪装成空历史。历史 schema 不借 nullable coercion 修补。closed=true 的恢复不重发一个额外 final；未提交 sink 的 replay 仍按原 delivery 协议。state_version/layout 相同不替代 engine 身份；预算在 spec 中，改预算即新配置，不能静默用改预算的图接旧 checkpoint。

stream 的未闭合 history 跨 Data/空批和 checkpoint 保留，新 job fresh；显式 durable restart 才延续。finite 的 NULL snapshot 与 stream state 不互认。旧图完全不含新配置/默认 SourceBoundedness=Unknown 时，其 declaration/plan/config/cache/checkpoint bytes 与身份不应变化；新字段仅在确有新证明时进入新路径身份，不批量重写旧 manifests。for_mode 共享声明不共享 plan/state。默认 SQL、旧 update semantics、既有 Source/Sink/managed manifest version 均保持。

## Error contract and independent acceptance seeds

P 为 canonical relation path，DF suffix `.distinct`、SQL `.sql.distinct`；columns 用 `.columns[i]`，配置用 `.spec.history` / `.spec.state_budget`，绑定用 `stream.inputs.<logical_name>.boundedness|watermark_policy`，导入项目仍用 graph.nodes/edges 的现有物理路径。枚举字面量错误不是 unknown 查询。

| Violation                                  | Code/status and required diagnostic                                                                    |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------|
| columns/spec/预算/边界 host 类型错误               | Python TypeError/ValueError 或 Rust InvalidArgument；精确参数、实际类型/值及范围；拒绝 bool 作为 integer。                  |
| 空/重复列、未知列、keys dtype 不支持                   | invalid_literal / duplicate_name / unresolved_type / unsupported_type，unsupported；列名、索引和允许域。           |
| 窗口 start>=end、缺时间列/来源改写                    | invalid_literal / unresolved_type / capability_mismatch，unsupported；半开边界或必须原样/纯重命名的约束。                 |
| 宏越界、SQL ORDER/LIMIT/ON 等                   | capability_mismatch/unsupported；指出具体 clause 和准入子集，不 fallback legacy。                                   |
| BoundedSource 绑定 Unbounded / Unknown       | source_unbounded/unsupported 或 source_boundedness_unproved/unknown；说明该输入需要固定有限 cut，正常 EOF 的未来可能性不足以准入。 |
| 窗口无 closure route / 请求未指定可证明区间             | finality_unproven/unknown；给出 end 和缺少的有效水位/有界 EOF 路径，或未选择的窗口域。                                          |
| complete_snapshot ingress / unknown update | update_interpretation_mismatch/unsupported 或既有 update_*_unproved/unknown；定位真实消费边，包含 row-local 中转。      |
| 声明之外的上游/后处理形状、side-output late policy      | capability_mismatch/unsupported；指出越界节点/策略。                                                             |
| declared ordering / per_batch portability  | row_order_unproven/unknown 或 history_scope_mismatch/unsupported；最终关系无规定序，不是每个输入批的结果。                   |
| 数据预算/schema/时间转换/查询/emit/取消失败              | 既有 execution/stream typed error，保留 cause、节点和输入约束；不提交候选，不产生伪 final。                                     |
| restore 身份/计数/layout/预算不符                  | 既有 CheckpointMismatch/状态错误；指出不兼容项，Source open=0。                                                       |

v2 新增 `input_normalization_unsupported/unsupported`，定位具体 `.distinct.columns[i]` 的无法安全重建表示；未知父大小本身不是拒绝条件。spec 缺少/改变 ingress_policy 为 `invalid_literal/unsupported`，明确必须 copy_visible_v2；缺字段的旧草案不默认获得新保证。预算错误必须带 node、normalization/schema/metadata/query/output/capture 角色、required/limit 与 policy，不能描述成“foreign owner 已测得不足”。

沿既有声明/CompileError/QueryPortability/QueryUpdateSemantics/StreamingRuntimeError/CheckpointError 分层；不新增同义异常。可分析错误保留 QueryDiagnostic 的三态与不可变 query_analysis；机器依 code，message 可改进。unknown 与 unsupported 都不通过强制准入。

实现阶段只做与本片相关的定向验证，以下 expected 为独立字面量，不能用另一 frontend 或新 lowerer 生成：

1. **C03/A05/A06：** 真正两个 Data 事件 `x=[1]`、`x=[1]`，旧 `SELECT DISTINCT x FROM input` 仍是 `[1]`、`[1]`；新的 bounded DF/SQL 在 EOF 前0事件、EOF 后恰1个 `{x:[1]}`，finite 显式 H=[1,1] 为一行。不要用会合并材料 sequence 的 convenience 路径证明两个旧 Data 边界。
2. **NULL/schema：** H=[1,NULL,1,NULL,2] → 无序 {1,NULL,2}，nullable Int64 不变；keys=(a,b) 的 `(1,NULL),(1,NULL),(1,2),(NULL,2),(NULL,2)` → 三行。typed empty/多个空 RecordBatch/Source 无 Data 正常 EOF 均恰1个0行 final；无键列拒绝；float/nested/未知列及运行中 schema 漂移各有具体诊断。
3. **窗口：** `[0,10)`，ts/q=`(-1,9),(0,1),(9,1),(10,2),(NULL,3)`，distinct keys=[q] → {1}；W=9 无输出，W=10 恰输出一次，EOF 不重复；end/start 本身、负区间 `[-10,0)`、窗口空历史都核对。未闭合先 checkpoint 再恢复→有效 W/EOF，与不中断轨迹相同；W<end 的区间内 Data 仍接纳，闭合后的区间内 Data 按 error/drop 处理。
4. **预算/C13（v2 取代初版 seed 4）：** H=[1,1] 在充足 bytes、max_rows=2 成功，=1 在第二 Data 提交前失败，旧自有 H=[1] 保持；`allows(R,B)` inclusive。从真实 v2 lease schedule 得到每阶段 M 与下一已知 charge，覆盖归一化目标+旧 H/capture、schema/metadata 深复制、query result+compact output 的同时峰值；在同一路径恰 B 成功、B-1 分配前拒绝。B 是本节点自有峰值，不是112 B、8 MiB父分配或可见 nbytes。用分配/lease hook 验证拒绝时目标尚未分配，不只验证最后返回 budget error；预算常数不得从未计量 raw owner 大小反推。
5. **L01–L03：** 同 job 跨批/空批保留 H；同 compiled finite plan 无 reset 暖缓存依次 H=[1,1] → [2] → typed empty，结果 {1}→{2}→0行；失败/取消后重用无旧 H/closed。新 stream job fresh；真实 replayable Bounded source 的 durable checkpoint 后补下一 Data 再 EOF，与不中断相同；终态恢复不额外 final。
6. **读取前拒绝/A09：** counting Source/iterator/Sink 验 unsupported SQL/形状、unbounded、unknown、无闭合路径、snapshot 经 row-local、非法预算/policy、restore mismatch 在 open/next/iterator/job/state-root 前失败。包含无 request 默认入口、strict/latest、缓存命中、Runner 和导出再加载；不只测试 analyzer 报告。
7. **四单元/A08：** 每语言独立从 DF 和 SQL 两宏构造，两 mode schema/analysis/export/执行一致，mixed DF→SQL→distinct→DF→SQL、共享 fan-out 和独立合法输出保持。导出再加载恢复同一物理节点/ports/bindings；新 spec/columns/budget/cut 改动不能误命中 cache/checkpoint。同步 exports/rustdoc、PyO3/stubs、capabilities modes/state inventory、支持表和两个独立例子；完整回归/覆盖率交 CI。
8. **原切片反例 → 归一化与释放：** 复用已有 probe 的源码/lockfile/RED 日志，不把其 exit101 当新功能失败。新增最小 feature 测试将同一1行 offset=1、8 MiB父 buffer 的输入喂给共同 normalizer：保持 owned H 活着，删除所有 Python raw/导入 raw/借读引用后，原父分配须已释放，值仍为1；不必等 H/输出析构。另以 B-1、取消、列/schema 错误验证没有 raw owner 存入失败候选/异常/plan cache。Rust 的普通大 backing slice、Python C Data、Nullable Boolean/整数/字符串（非零 offset、长字符串、NULL、空）均查新缓冲不别名原 buffers、无残留 offset 和 foreign release owner；不以小 capacity 自证。保留 SourceBinding/调用者输入映射时父 owner 仍活是公开排除范围，释放检查必须分别清理这些外部引用，不能把它们算成节点泄漏或声称节点已经覆盖其内存。

schema/metadata 的独立负例要含0行表、长 field/schema metadata、含大 spare capacity 的源属性 String/Vec；新表示保留内容、按自有容量收费，失败不保留原 schema/metadata Arc。IPC编码/解码+自有 metadata 同时活着纳入 B/B-1。四单元各须到共同 normalization seam 的证据，至少一个公开 finite 和一个真正两 Data 的 stream 路径验证 raw owner 生命周期；仅测试 helper 不替代入口接线。恢复测试在自有 H、活 capture 与新归一化目标同时存在时核 charge；layout/accounting1、缺/变 policy、未知版本在 Source open 前拒绝，v2恢复后继续不重新引用 foreign owner。

## Required implementation delta for v2

以下是恢复 implementer 所需的确切增量；原有限原型 GREEN 不覆盖它们：

1. 新增 distinct 专用的私有计数/预留/逐值复制 seam，输出携已证明 provenance 的自有数据和 lease；支持首片全部标量类型/nullable/empty/窗口筛选，禁止用普通 Batch 或 JSON 构造“owned”凭据。原公共签名保持，不新增 caller size/token/copy 开关。
2. 共同 native node 的 finite/stream ingress 和最终 emit 前接通该 seam；两语言 DF/SQL lowerer、低层 project 都不能旁路。当前 raw 只在同步借读段活跃，不把原数据扔给异步查询/worker；源和 graph 的其他持有范围如实公开。
3. 新增归一化目标/schema/metadata/容器/输出副本的预付 charge，复用已核对的 metadata/IPC reservation 模式；将它们与旧 H、candidate、DataFusion reservations、活 captures 统一到节点 ledger。清理失败/cancel/emit 和缓存，避免错误对象/闭包保留 raw owner。
4. Spec 严格 serde/schema、新 policy 常量、declaration/config/cache identity、layout2/accounting2/checkpoint control 与恢复候选预检同步；state_version1 的关系含义保持。capability inventory 只列实际 layout2，QueryAnalysis/gates 改 r9，原 SQL/桥接旧身份不漂移。
5. 按上述新增8与修订4做定向 RED→GREEN、所有权释放/分配顺序/四单元接线验证，并续接原 issue 的 EOF、budget、checkpoint、同 plan 隔离。复用当前切片复现，不重开产品矩阵；必要的新断言要证明独立 owned H 存活时父分配已释放。
6. Rust/Python public docs/docstrings/examples/explain 同步写明 copy_visible_v2、自有峰值和临时 ingress 未知分配的公开例外。旧 `Batch.from_pyarrow`/ArrowSource/SQL 的通用行为不在本片扩改，不宣称顺带修复 legacy snapshot_sql 计量。

## Typical declaration seeds

下面为**待实现后的**公开调用种子，本轮仅检查 Python 语法与现有入口签名。材料输入展示短 happy path；两个真实 Data 的兼容验证仍按上节，不借材料合并伪造切批证据。

```python
import pyarrow as pa
import calc_flow as cf

events = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
spec = cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 * 1024 * 1024))
df = events.distinct(["x"], spec=spec)
sq = cf.sql(
    "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 2, 8388608)",
    a=events,
)
program = cf.Program("unique", mode="batch", outputs={"df": df, "sql": sq})
data = pa.table({"x": [1, 1]})
assert all(t.to_pydict() == {"x": [1]} for t in program.collect({"events": data}).values())

async def observe():
    seen = []
    async with program.for_mode("stream").stream(
        {"events": cf.SourceBinding.from_arrow(data)}
    ) as outputs:
        async for event in outputs:
            seen.append((event.name, event.table.to_pydict()))
    assert sorted(seen) == [("df", {"x": [1]}), ("sql", {"x": [1]})]
```

Rust 从独立 core 构造两个声明并复用 Program/plan；无需 Python helper 或手写 graph：

```rust
use std::{collections::BTreeMap, sync::Arc};
use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    CompileMode, DistinctSpec, InputOptions, Program, Result, StateBudget, table_input,
};

fn declaration() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let events = table_input("events", schema, InputOptions::default())?;
    let spec = DistinctSpec::bounded_source(StateBudget::new(2, 8 * 1024 * 1024)?);
    let df = events.distinct(vec!["x".into()], spec)?;
    let sq = calc_flow::sql::sql(
        "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 2, 8388608)",
        BTreeMap::from([("a".into(), events)]),
    )?;
    Program::new("unique", CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sq)]))
}
```

窗口调用只换 spec：Rust `DistinctSpec::closed_window("ts", EventTime::from_micros(0), EventTime::from_micros(10), budget)?`；Python `cf.DistinctSpec.closed_window("ts", start_micros=0, end_micros=10, state_budget=budget)`，随后对有 ts/q 的关系调用 `distinct(["q"], spec=spec)`。两者对应上面的 WINDOW SQL，时间列不必进入去重 keys。

## Rationale, open questions and handoff

选择 columns + DistinctSpec，使 SQL 的投影后 DISTINCT 与 DF 相同，避免 subset/keep 的非键代表行歧义。两个明确 domain 比仅有 global=True 或等到 EOF 猜有界性可检查；固定单窗口让 cut 在 finite/stream 中都可执行，不把前序窗口 aggregate 的输出批误当原始窗口历史。单 final 输出避免先承诺 replacement/retraction 协议。保留原生 DISTINCT 和保守 history budget，避免 Python 内核、浮点代表行与“只限制最终 unique 行数”的隐藏状态增长。

v2 对外部 backing 的选项评估已完成：可验证的 allocator/owner ledger 可以在未来提供零拷贝路径，但 C Data 目前没有完整父分配大小与可认证的释放链，现有公共 Batch 也没有可信 provenance；用户自报 bytes、Python get_total_buffer_size 或 native capacity 都不能认证任意 foreign owner。现在新增跨桥接 ABI/全局分配注册表会扩大旧接口与多路径恢复范围；本片不选。拒绝所有 Python/Arrow、或在通用 from_pyarrow/Source 中全局 eager-copy，都违反现有接口/本片边界；只在后续投影里复制但照旧承诺 ingress 已全计量，仍被当前反例推翻。

因此选择 **特性局部、无旁路的 owned normalization + 公开的临时借读例外**。这是保证的明确缩窄：unknown ingress 的大小未被获知/限制，只保证不会经本节点自有历史/输出继续保留；为复制创建的每个目标和其他已列角色仍在分配前预付。没有声称“预算小于外部父分配就一定拒绝”，也没有声称原始 Python 表在 active SourceBinding/graph 尚持有它时会释放。额外复制/峰值成本与新的 identity 是用户影响，必须随支持表公开。

不选择默认全历史 SQL、无界 seen-set、以 timeout/预算溢出伪造 EOF、窗口未闭合就取快照、MAX watermark、batch 启动 StreamJob 或跨 execute 保存 H。它们分别改变旧语义、缺乏有限状态承诺、改变截止域或破坏生命周期。

**本轮内部合同缺口已作决策，无需用户重新授权或选择。** 本文 v2 取代错误的外部 backing 覆盖承诺，并冻结所有权/复制/计量、layout/accounting2 与 r9。真实实施工作是所列六项增量和验证，尚未证明产品满足新政策；由协调器恢复 implementer，再独立 review。更宽 keys（特别浮点代表值）、可认证零拷贝 owner seam、多窗口/多源 cut、replacement/增量输出、top-k/OVER 和 spill 都是后续范围。若实现发现受控新分配无法先 reserve，带具体分配/生存期证据回报，不能暗中扩大借读例外到 retained data。

本修订仅做必要源码核对及文档自检：围栏、Python 片段 AST、相对链接、表格/JSON、policy/版本一致性、analysis 原正文前缀保留与写入范围/空白；具体结果见交付评论。复用原复现日志，Rust/SQL 种子未编译执行，不声称归一化、产品测试或性能已通过。只修订本 note 并向原 analysis 追加决策，没有产品修改、提交/推送、PR 或扩派。交协调器安排 implementer → 独立 reviewer → 修正及门禁后合并；DAL-519 保持 in_progress。
