# Finite Event Window — API Note

状态：DAL-516 / S2b 局部公共合同冻结，2026-11-12，可交 implementer。**本文规定的新增 Rust 关系方法、SQL 方言、finite 执行及诊断尚待实现；本轮没有产品测试或发布声明。**

基线：`142e74d0ac1a6fd6cef85d79c1a81d2de543100d`，分支 `agent/dal-orchestrator/446905411b20`，开始时工作树干净。前序 PR #599 的合并由触发评论确认；本轮核对实际 HEAD 和源码，不重新验 CI。输入为本 issue / 触发评论 `01a1cd1e-2d4e-71ef-ac18-d90a9b58cbe2`、父 DAL-498 的生命周期与四单元讨论、经 Multica CLI 下载的计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`，以及现有 [event-window spec](../specs/symbolic-event-window-aggregation.md) 和 [finite-union note](finite-union.md)。

## Audiences and scope

Rust 用户从 `calc_flow::TableExpr`、`calc_flow::sql::sql`、Program 和既有 plan/Runner 使用窗口；Python 用户从根包 `calc_flow` 的 TableExpr、sql、window 和 Program 使用同义能力。DataFrame/SQL 是两类声明接口，batch/stream 是独立生命周期。

本片只为既有固定 UTC tumbling/hopping、count/sum/min/max/avg 增加完整输入的有限最终化，并补齐四单元声明、组合和导出。保留旧 SQL 默认、stream 水位/迟到/EOF/恢复、batch execute 隔离。没有 session/calendar window、非零 offset、early trigger、撤回、窗口 replacement、自定义 aggregate、一般 OVER、batch 中途恢复或持久 stream job 适配。父计划 O02 提到 offset 是要求明确支持域，不构成扩展旧 WindowSpec 的授权。

遵循仓库 `AGENTS.md`、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md`。本仓库实际边界是 Rust core → PyO3 → Python adapters，没有 dal-public、C++/Excel 绑定或 `.codex/references/component-boundaries.md`；不补造这些结构。保留 `examples/36_symbolic_event_window.py`、`crates/calc-flow/examples/36_symbolic_event_window.rs` 的窗口 stream/共享示例主题，保留 core `windowed_streaming.rs` 和 example 32 的既有用途。

## Surface today: evidence at this HEAD

| Evidence                                                                     | Verified fact / remaining gap                                                                                                        |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `operator/window.rs:99–208,1106`                                             | WindowGeometry/WindowSpec、duration 构造与 ordered group/aggregate builder、WindowAggregateOperator::new 已公开；无 offset 字段。                 |
| `operator/window.rs:1903–1966`                                               | process_data 只累计；on_watermark 严格推进并关闭 end<=watermark 的组；on_end 排空剩余 accumulators，重复 end 无输出。                                         |
| `operator/window.rs:2539–2611,4727–4863`                                     | assignment 使用整数 Euclidean floor、检查有符号 EventTime 边界；原生输出 schema/类型矩阵已存在。                                                              |
| `pipeline/batch.rs:24–50`、`config.rs:1423,2318`                              | batch compiled variant、project 构造与 validation 都拒绝 window；只改 capability 不足以交付。                                                        |
| `dataframe.rs:75`、`program/lower.rs`、`lib.rs:94–123`                         | Rust 已有 typed relations/aggregate/union，导出 WindowSpec 等低层类型；没有关系窗口节点或方法。                                                             |
| `symbolic/ops.py:1746–1820`、`windows.py:15`、根包 exports                       | Python window.tumbling/hopping 与 WindowAggregate/five helpers 已可声明；非空 aggregates 是 @2，省略/None 是旧 @1。                                 |
| `symbolic/analyzer.py:1657–1816`、`lower/event_windows.py`                    | Python @2 已有 stream lowering、共享 owner、精确 schema 和时间坐标检查；仍拒绝 batch，窗口路径未接入 SQL。                                                       |
| `sql.rs`、`sql/union.rs`、PyO3 `config.rs:762–785`、`_native.pyi:13,341`        | SQL 返回同一关系；union 已有共享 native AST/schema seam。没有窗口 SQL 识别/规划 seam，普通 SQL 的同名聚合不是窗口证明。                                                 |
| `query_analysis.rs:484–485,1024–1072`、`query_analysis/updates.rs:94–135`     | 报告为 schema 2/revision 4；window 仍 stream-only，stream 为 event_window/append/group_final，latest 的 append-only 名单已含 window。              |
| `operator/window.rs:48–49,1991–2008,4181–4196` 与 `capabilities.py:1145–1161` | configuration/state version 为 1；当前 segment metadata 写 layout 2、恢复接受 1/2；Python capability 仍列 `(1,)`，analyzer 也硬编码该旧清单。这是已核实的局部元数据缺口。 |

上表 core 路径相对 `crates/calc-flow/src/`，symbolic 路径相对 `python/calc_flow/`。静态核对的既有断言包括 `tests/window_tumbling.rs` 的负时间、空窗口、NULL 与溢出事务，`tests/window_hopping.rs` 的逐 assignment 迟到，`tests/window_compile.rs` 的类型/schema/几何，及 `python/tests/test_symbolic_event_window_analysis.py`。没有将阅读断言记为测试通过。

## Frozen public signatures

Rust **新增**一个 immutable 关系方法，复用已导出的纯数据配置，避免几何、分组、聚合堆成位置参数：

```rust
impl TableExpr {
    pub fn window(&self, spec: WindowSpec) -> Result<Self>;
}
```

WindowSpec::tumbling/hopping、group_by、aggregate 及 AggregateFunction/AggregateSpec/WindowGeometry 的现有签名不改。新 TableExpr::window 要求至少一个 aggregate；低层 WindowAggregateOperator 原有可接受空 aggregate 的域保持。方法拥有 spec，返回新关系，不保存 rows/runtime；按 input identity 与有序完整 spec 计算新关系身份，不能只按 geometry 共享。

Python **保留现有签名、扩展执行生命周期**；这里的 None 仍指旧 @1，不能改成默认空聚合或自动升级：

```python
class WindowNamespace:
    def tumbling(
        self, value: TableExpr, /, *, event_time: str, size_micros: int,
        group_by: Sequence[str] = (),
        aggregates: Sequence[WindowAggregate] | None = None,
    ) -> TableExpr: ...
    def hopping(
        self, value: TableExpr, /, *, event_time: str, size_micros: int,
        slide_micros: int, group_by: Sequence[str] = (),
        aggregates: Sequence[WindowAggregate] | None = None,
    ) -> TableExpr: ...
```

`window.count/sum/min/max/avg(column: str, /, *, output: str) -> WindowAggregate` 不改；count 必须提供列。新执行域只接纳非空 aggregates 的 @2。group/aggregate sequence 复制为 tuple，顺序、别名和重复聚合重数均保留。旧 @1 的 bytes/digest/构造域与执行拒绝保持，未知版本 fail closed。不新增 Python WindowSpec、另一套 aggregate class 或 TableExpr.window 别名。

两语言 SQL 签名与 TableExpr.sql(query) 不变，后者仍绑定 alias `input`：

```rust
pub fn sql(query: &str, tables: BTreeMap<String, TableExpr>) -> Result<TableExpr>;
```

```python
def sql(query: str, /, **tables: TableExpr) -> TableExpr: ...
```

Program 的 mode/engine、for_mode、with_update_semantics、collect_schema/analyze/explain、compile/execute/collect/stream/to_project 沿现有合同。不增加 finalize/flush、公开控制消息注入、window runner 或有限执行选项。Rust 复用现有 root exports；Python 根包与 symbolic 的 window/WindowAggregate 必须保持可发现。若新增私有 native seam，PyO3 注册、Runtime 适配和 `_native.pyi` 同步；私有 helper 不计作公共 SQL 单元。

## SQL spelling and shared declaration boundary

冻结一个明确的 calc-flow 方言，**新增而非声称 DataFusion 已内建支持**：

```sql
SELECT window_start, window_end, symbol,
       COUNT(quantity) AS n, SUM(quantity) AS volume
FROM CF_TUMBLE('trades', 'ts', 10)
GROUP BY window_start, window_end, symbol
```

CF_TUMBLE 参数依次为 `(input_alias_string, event_time_column_string, size_micros_integer)`；CF_HOP 为 `(input_alias_string, event_time_column_string, size_micros_integer, slide_micros_integer)`。函数名单一、不带 schema、大小写不敏感；前两项为普通 SQL string literal，按解码后的精确名称查 alias/字段，不做 SQL identifier 大小写猜测。数值只接纳十进制无符号整数字面量 `[1,2^64-1]`，不接受 interval、float、表达式、NULL、命名参数或 offset。两者不是运行时 UDF，也不产生中间展开行。

公共 SQL 子集的完整约束：

- 一个顶层 SELECT、一个上述 FROM 调用、恰好一个显式 tables alias 且被该调用引用一次。没有额外 FROM alias、CTE、子查询、JOIN、set operation、WHERE/HAVING/QUALIFY、DISTINCT、OVER、ORDER BY、LIMIT/OFFSET/FETCH。需要过滤/派生列时先构造前一关系，需要结果排序/派生时后接合法关系；不把它们暗塞进窗口识别器。
- SELECT 恰为两边界、有序 group 字段、一个或多个命名聚合；GROUP BY 恰为两边界与相同顺序的 group 字段。无分组时也写 `GROUP BY window_start, window_end`。边界/分组不重命名，aggregate 必须 `AS output`；结果调整在后续 select/sql 中完成。
- aggregate 仅 COUNT/SUM/MIN/MAX/AVG 的单个直接列参数；字段允许现有 quoted identifier 转义。没有 COUNT(*)、DISTINCT aggregate、FILTER、aggregate 内 ORDER BY、任意表达式或嵌套 aggregate。不使用普通 SQL aggregate 的隐式 coercion、COUNT Int64 或 nullable 规则。
- 两语言走**同一 Rust AST 识别和 schema 验证**，产生 data-only `(alias, WindowSpec, 精确 input/output schema, 字段映射/诊断)` 描述，再降到与 DF 相同的 window 节点。Python 不写 regex/parser、membership、类型推导或行聚合。构造、collect_schema、analyze、compile 与 export 必须走一致的识别结果；先识别宏，再走普通 SQL aggregate/多 alias gate。

出现 CF_TUMBLE/CF_HOP 却超出此形状，明确拒绝并定位违反项，不退成每批 SQL 或普通累计 snapshot。未使用这两个方言名的原有 SQL 继续按原支持域和诊断运行：DATE_BIN/GROUP BY、OVER、INTERVAL、其他 TUMBLE/HOP 拼写均不因此获得公共事件窗口证明。低层 SqlOperator/已保存 `kind=sql` 项目不自动解释该方言；typed frontend 导出实际 `kind=window`。用一个 SQL 包装函数返回手接低层图，或把整条 SQL 留给另一个 batch 算法，不算 SQL 一等支持。

## Supported paths, schema and mixed DAG

四单元公共路径为 exact-schema table input → 任意有限串已证明的 deterministic 单输入 row-local DF/SQL → 一个窗口 → 已证明的 row-local DF/SQL → 命名输出。每条窗口路径最多一个窗口。允许独立窗口、窗口外合法输出、相同声明 fan-out；每个唯一声明同 job 只有一个状态 owner，同 finite execute 只计算一次。不同语法的 DF/SQL 窗口不要求 hash 相同或跨语法 CSE。

SQL row-local 仅采用既有 native 证明域：单直接 alias、无聚合/排序/去重/limit/子查询/volatile/provider/UDF；投影、已支持标量和过滤可组合。窗口时间列须经 native AST 字段映射证明为原输入时间列的原样传递或纯重命名；WHERE 可以筛行，不能改变时间坐标。这个**新增的局部 SQL passthrough 证明**必须接通 Python 现有来源检查和两语言 lowering；不能因为 SQL 新 lineage 就一律拒绝 A04，也不能因为同名/同 dtype 就放行时间算术、DATE_BIN 或 cast。此证明不恢复 SQL 的 entity/sequence/temporal ordering；窗口本来也不要求 rolling 排序。

窗口路径上的 union、多源、普通 relation aggregate、另一个 event window、rolling/cross-section、join/ASOF、array attachment、external stateful provider 继续不在新公共域。独立的 union 等输出可以与窗口共存，但不能让整个 Program 被窗口专用 lowerer 截断或误接线。低层原已合法的 stream graph 组合保持原支持，不用新 frontend 白名单收紧它们。

| Role          | Four-unit public input domain                                                               | Output / nullable                        |
|---------------|---------------------------------------------------------------------------------------------|------------------------------------------|
| event_time    | timestamp[ms] naive、timestamp[us] naive/UTC                                                 | 两边界 timestamp[us, UTC]，均 false；输入可以 NULL |
| group_by      | bool、全部 signed/unsigned 整数、float32/64、string/large_string、date32/64、timestamp[us] naive/UTC | 按声明顺序保留精确 dtype/nullable                 |
| count(column) | 既有 public portable/project-v3 可表达字段                                                         | uint64 / false，只数非 NULL                  |
| sum           | int8/16/32/64；uint8/16/32/64；float32/64                                                     | 分别 int64、uint64、float64 / true           |
| avg           | 上述数值类型                                                                                      | float64 / true，算术平均                      |
| min/max       | 与 group_by 相同的 ordered scalar 域                                                             | 保留输入 dtype / true                        |

time32[s]/time64[us] 仅能作 count 输入，ms timestamp 不能作 group/min/max。naive 时间按既有 UTC 坐标解释。低层原生支持的 second/nanosecond timestamp 或更宽 count 类型继续留在低层，不借本片扩全局 public Arrow 类型语法。

输出严格为 `window_start, window_end, group_by..., aggregate_outputs...`。从原生窗口端口取得权威 schema；前后 row-local 片段逐 stage 用 native planner 确认 nullable。所有字段/metadata 的规范化沿既有 frontend/Port 合同，不能根据本批 NULL 分布缩窄 nullable 或改调用者 Arrow 数据。typed empty 也具有完整结果 schema。

两个保留边界名不得作为 group 或 aggregate output；group 名、所有输出名唯一；输出可以覆盖已消失的普通输入名。同 function/column 的不同 output 合法。窗口产生独立 row lineage，清除原 entity/event_time/sequence 证明；原始列与窗口列混用、伪造展示名、数组 attachment 均准确拒绝。无状态后处理只引用新结果 lineage。

导出仍是 project-v3 `OperatorSpec::Window { spec }` / `{"kind":"window","spec":...}`，required table `input` 与 `output` 均有 exact schema；同名逻辑输入/输出继续使用现有物理 binding 映射。SQL 宏导出后不需要 parser、Python globals 或 callback。共享分支、几何、字段顺序与聚合别名必须在 round-trip 后保留。没有新 project 字段，生成 JSON schema 不应漂移。

## Finite finalization, real stream progress and comparison

BatchOperator::process 接收本次 `input` Batch 的**全部 RecordBatches，按既有物理行顺序**作为完整 H。复用既有 membership、group key、五种 accumulator、overflow 和 output schema 内核，在本次调用拥有的新 accumulator 中处理 H；没有 watermark，也没有 late assignment。全部数据接纳后排空所有非空窗口，生成一个 `output` Batch（可含多个 RecordBatches），只返回完整成功结果。空 H/全部 NULL 时间都返回 typed empty，不补无数据窗口，即使无 group 也不产生全局 aggregate 的一行 0/NULL。

复用 drain/finalize 语义不等于在 batch 内启动 StreamingRunner、创建 StreamJobContext/托管 checkpoint 或伪造 `i64::MAX` 水位。允许抽取内部纯内核和有限 collector，但不调用持久 stream 生命周期来执行 finite。接通 BatchOperator 实现、CompiledBatchOperator、builder 与 project 构造/validation、native binding 和 Python lowering，不能只删 mode guard。

stream 合同保持：Data 只累计；真实 watermark W 关闭 `end<=W`；迟到按 assignment 的 end 判断，hopping 一行可部分丢弃；NULL 时间整体忽略并计数。输入全部结束后 runner 调用 on_end 排空剩余组，再发送下游 EOF。EOF 是生命周期结束，不是一个最大 EventTime。空 Data 不 reset，不作为 EOF。stream 已在水位输出的组不得在 end 再输出；无水位但合法 SourceBinding 正常 EOF 也须终结剩余组。源是否满足所选进度/恢复能力仍走原 preflight，不新增公开控制注入。

每次有限最终化和每次 stream closure 内保留原生 WindowKey(start,end,stable_group_key) 顺序；普通新 finite Batch metadata 使用既有内置算子的 node source/sequence=0 约定，不把 stream sequence/checkpoint cursor 带过来。跨模式不比较 chunk 大小、Batch metadata 或分支调度。分组 NULL/NaN/signed zero 与浮点/补偿累加均沿原生规则；不通过排序输入来声称提升浮点结合律。

严格比较明确为：相同 H、append 输入、同一配置，stream 轨迹中 H 的所有合法 assignments 都被接纳，到 all_sources_eof 后拼接窗口的 append-final 输出，与一次 finite H 比较 schema、NULL、重复数及值。确定性精确小整数 oracle 必须完全相等；浮点遵守既有内核精度/NaN 比较合同，不新增任意切批 bitwise 相等承诺。若水位已使部分 hopping assignments 丢弃，batch 重放原始行不再是相同输入语义；不能仅过滤“迟到整行”伪造 H。

分析使用现有 `final_relation / accepted_history / all_sources_eof / unordered` 表达上述**有前提的全输入终态**；window 本地 history 仍为 event_window，不把它改成 current_batch 或 latest_relation。理由文本必须列出完整 H、无被丢弃 assignment、真实 EOF 前提；这是条件性语义证明，不验证未来 source 轨迹。上游 row-local 节点不必都标为 event_window 才能证明全图。

`final_relation / event_window / closed_window` 仍表示实际窗口关闭域，不将其悄悄解释成 EOF。当前 request 没有具体水位/窗口集合，而 finite 默认返回 H 的全部窗口；对任意 stream 中途关闭前缀，本片保持 `finality_unproven/unknown`，严格编译拒绝。执行对照可另按相同真实 W 选择 `end<=W` 的输出作 closure 回归，但这不授予全关系相等证明。per_batch 与声明全局行序也不因 finite 可用而自动 supported；后者沿 row_order_unproven。无需新增 request 字段/合法 tuple。

CompiledBatchOperator 对 window 的 snapshot 为 NULL、非 NULL restore 拒绝、reset 为 no-op，沿内置 finite 约定。已编译同 plan 暖缓存无 reset 的后续 execute，成功/失败/取消后都不继承 accumulator、ended、watermark、sequence 或输出行。缓存只存不可变配置/计划/schema。stream 同 job 仍跨批保留，普通新 job fresh；显式恢复沿原协议继续，不接受 finite 状态。

不新增窗口预算参数或固定内存承诺。finite 临时状态沿既有资源/取消机制、低层已有 StateBudget 约束与完整输出检查；不借用 SQL replacement budget，不截断行绕过限制。失败不返回本节点部分 finite 结果，下次可 fresh 执行；不扩大成整图外部副作用事务/ExactlyOnce。stream 保留已有预算、chunking、delivery 与取消合同。

## Explicit latest admission

窗口消费解释恒为 **append**，输出恒为 **append/group_final**，没有保存/撤销每个 alias 最新完整关系的能力。Program 的更新选择与 portability 请求的 history 是不同维度。

| Producer / explicit choice                                                        | Required behavior                                                                                                              |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| 已证明 append → window，legacy_append 或 latest_relation                               | 支持域内可以运行；latest opt-in 不把普通事件强制变成 replacement，也不改变窗口输出为 snapshot。                                                              |
| complete_snapshot → window，latest_relation                                        | 在启动前于该 ingress 报 update_interpretation_mismatch/unsupported；即使本次 snapshot 为空、目前只有一次事件或执行的是 batch wrapper，也不能绕过同声明的 stream 边准入。 |
| complete_snapshot 经 project/filter/单 alias row-local SQL → window，latest_relation | 仍按上项拒绝；wrapper 不得洗掉 producer update，缓存命中也重新 gate。                                                                              |
| 更新类型/完整事件边界未证明，latest_relation                                                    | update_semantics_unproved 或 update_event_boundary_unproved/unknown；mandatory supported-only gate 拒绝，不猜首批。                      |
| complete_snapshot → window，旧 legacy graph                                         | 原已合法的低层 stream 仍追加每个 snapshot 的行；不静默修成 replacement。新公共窗口路径本就限制 stateful upstream，不扩大其结构域。                                      |
| 请求严格 final_relation，存在上述 snapshot→append 边                                        | 无论是否显式 latest，都定位 update_interpretation_mismatch；不只检查下游是 aggregate/union 的旧分支。                                                 |

latest 校验必须覆盖窗口本身、SQL 宏降出的内部边、转接节点与每个共享消费边，在 source.open、迭代绑定、job/state root 创建前执行。窗口输出再交已有普通 aggregate 是 append 输入，不能凭此放开当前公共窗口后处理的 stateful 白名单；低层组合仍保留其既有语义。旧 SQL 快照追加链结果 7、前序已准入 latest 链结果 6/不支持时拒绝、replacement budget 与恢复身份均不改。

## Errors, analysis revision and compatibility

沿 QueryAnalysis/QueryDiagnostic、CompileError.query_analysis，unknown 不能当 supported。本片扩展真实双生命周期/SQL/时间字段/边证明，报告输出 **schema_version=2、analysis_revision=5**；Rust producer、Python parser、mandatory latest gate、缓存 gate 与绑定/stub 说明同步。保留旧 (1,1)/(2,2)/(2,3)/(2,4) 展示，旧报告不构成本片准入证明；不新增 wire 字段/enum 或 capability schema 版本。

令 P 为 canonical 逻辑关系 path。Python DF 保留 `P.window_tumbling` / `P.window_hopping`，Rust 新方法用 `P.window`；公共 suffix 为 `.event_time`、`.geometry.size_micros|slide_micros`、`.group_by[i]`、`.aggregates[i].function|column|output`、`.value`。已存在 Python 几何构造路径保持原样，不批量迁移旧错误。SQL 宏用 `P.sql.window` 加同一 suffix、alias 问题用 `P.sql.tables.<alias>`；不是 identifier 的名称用现有 JSON string 下标规则。重载项目继续使用 graph.nodes/edges 的既有物理路径。

| Violation                                            | Code/status and actionable message                                                                                               |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Python 非 TableExpr/非 WindowAggregate、bool/float size | TypeError 或既有严格构造错误；指出 value/aggregates[i]/size 的实际类型，不能 bool 当 int。                                                             |
| 0、负数、>u64、非整倍数、overlap>1024                          | invalid_literal/unsupported；指出 size/slide 数值、正整数范围、整倍数或 overlap 上限。Rust Duration 非整微秒同样拒绝。                                       |
| 缺时间/分组/聚合列                                           | unresolved_type/unsupported；精确字段名、输入、用途及索引。                                                                                      |
| 不支持 dtype/timezone/aggregate                         | unsupported_type/unsupported；实际 dtype 与上表要求，例如 sum(bool) 不能隐式 cast。                                                              |
| 空/重复/保留输出、空 aggregate sequence                       | duplicate_name 或 invalid_literal/unsupported；具体冲突名/索引/约束。                                                                        |
| offset、lateness/trigger 等额外窗口字段                      | Python 未知 keyword TypeError、严格 project unknown-field/format 错误；SQL 多余参数 invalid_literal；明确 epoch 对齐、无 offset 参数。0 offset 参数也不接纳。 |
| SQL 宏形状/COUNT(*)/alias 不符                            | capability_mismatch 或 invalid_literal/unsupported；定位 P.sql.window 或 alias，并给出本片受限语法，不假称 SQL 全局非法。                                |
| 时间改写/来源不明、窗口路径越界                                     | capability_mismatch/unsupported；指出 event_time 或违反边界的 operand，说明必须 passthrough/单窗口 row-local。跨 lineage 用 schema_mismatch。         |
| snapshot→window、更新证明缺失                               | 按上一节逐边 update_interpretation_mismatch/unsupported 或 update_*_unproved/unknown。                                                   |
| 缺 capability、mode/layout 清单不兼容                       | capability_mismatch/unsupported；指出 window@1、所需生命周期及实际清单，不用表达式能力替代窗口能力。                                                           |
| EventTime assignment 越界、聚合溢出、超长 group key、budget、取消  | 现有 runtime typed error/exception；指出 operator/column/constraint，保留 cause；数据相关错误不谎称 compile 可证明。                                   |

声明期沿 Rust InvalidArgument / Python TypeError、ValueError；分析/编译沿现有 Compile、QueryPortability、QueryUpdateSemantics 分层，保留 path/code 与不可变 query_analysis。诊断仍按 path/code/status 排序去重。新 finite mode 获准前，未支持形状继续准确 unsupported_mode。

capability `window@1` 的 modes 增至 batch/stream，finality/stateful/requires_watermark/checkpoint 描述仍解释 **stream** 能力：batch 是本次内临时状态，不要求源提供水位或有恢复能力。analyzer 按 mode 检查，不把 requires_watermark 套到 finite。**同步修正已核实的 inventory 为 state_version=1、state_layouts=(1,2)**，并更新现有 `(1,)` 相等检查/相关快照；这只对齐 native 已接受/已生成布局，不新增布局、字节格式或放宽未知 layout。未完成该对齐不得宣称 capability 准确。

兼容边界：Python @1/@2 canonical bytes/digest、已有 @2 stream 拓扑/node ID/配置 hash、state version、manifest/project version、checkpoint bytes/source cut 不应因 finite 路径改变。若抽取内核，保留 stream checkpoint emission/restore/输出顺序；native 当前写 layout2、读取1/2 的事实不回退。新 Rust window relation 有自身稳定身份；新 SQL 宏此前没有合法 native stream checkpoint，不需迁移旧 SqlOperator state。旧普通 SQL/union lowering 不重写，准备缓存只为新方言增加私有策略标识（如 finite_event_window_v1），保留 mode/schema/update_semantics/registrations 等既有身份。for_mode 共享声明，不能共享不同 mode plan/state；不同 DF/SQL 声明不要求相同 fingerprint。

## Independent expectations: C11 / O02

以下为**独立的字面量预期**，实现测试不得调用新 lowerer、窗口 assignment helper 或另一个 frontend 的结果生成 oracle。native 手工 WindowSpec 对照是附加证据；独立预期仍须单独写出。时间单位全部为 Unix epoch 相对微秒。

数学定义：offset 固定为 0，窗口起点是整数格点 `k*slide`，仅当 `start<=t<start+size` 归属该窗。tumbling 取 slide=size。负数采用数学 floor，不能向 0 截断；边界通过精确整数计算，再检查 start/end 可由 i64 EventTime 表示。

| t    | tumbling size=10: [start,end) | hopping size=10, slide=5: all memberships |
|------|-------------------------------|-------------------------------------------|
| -11  | [-20,-10)                     | [-20,-10), [-15,-5)                       |
| -10  | [-10,0)                       | [-15,-5), [-10,0)                         |
| -1   | [-10,0)                       | [-10,0), [-5,5)                           |
| 0    | [0,10)                        | [-5,5), [0,10)                            |
| 4    | [0,10)                        | [-5,5), [0,10)                            |
| 5    | [0,10)                        | [0,10), [5,15)                            |
| 9    | [0,10)                        | [0,10), [5,15)                            |
| 10   | [10,20)                       | [5,15), [10,20)                           |
| NULL | 无 assignment                  | 无 assignment                              |

size=slide 的 hopping 与 tumbling membership 相同。size=10/slide=3、slide>size、0/负值、1025-overlap 拒绝；size=1024/slide=1 是允许边界。无 offset 字段即 epoch/0 原点；`offset_micros=1` 或 `0` 的 keyword/JSON、CF_HOP 第五参数都应拒绝，不测试未实现的平移结果。overflow 独立反例：t=i64::MAX,size=1 的 end 越界；t=i64::MIN,size=10 的 floor start 越界；两者须报错，不能 wrap、饱和或漏掉 assignment。

聚合 fixture：ts 为 nullable timestamp[us,UTC]，g 为 nullable string，q 为 nullable int8；以 `(ts,g,q)` 的字面量行按此序输入：`(-1,A,2),(0,A,3),(9,A,NULL),(10,A,5),(NULL,A,99),(2,B,NULL),(3,NULL,4)`。size=10，无水位先到达，EOF 后预期为下表；表顺序仅便于阅读，不用它替代 stable_group_key 的原生排序断言。

| [start,end) | g    | count(q): uint64! | sum(q): int64? | min(q): int8? | max(q): int8? | avg(q): float64? |
|-------------|------|-------------------|----------------|---------------|---------------|------------------|
| [-10,0)     | A    | 1                 | 2              | 2             | 2             | 2.0              |
| [0,10)      | A    | 1                 | 3              | 3             | 3             | 3.0              |
| [0,10)      | B    | 0                 | NULL           | NULL          | NULL          | NULL             |
| [0,10)      | NULL | 1                 | 4              | 4             | 4             | 4.0              |
| [10,20)     | A    | 1                 | 5              | 5             | 5             | 5.0              |

`!` 表示 nullable=false，`?` 为 true；边界均 timestamp[us,UTC]!，g 仍 string?。NULL ts 行不贡献任何 aggregate，null-time 行数=1；有效时间的 NULL q 仍建立 B 组。原始 count_rows()/COUNT(*) 的计行含义不能偷换为这里的 count(q)。typed empty、仅 NULL ts、无 group 的 empty 均为 0 行；已有有效时间组而 q 全 NULL 则是 count=0、其余 NULL。

另用单组 q:uint8 `[1,NULL,2]` 预期 sum:uint64?=3、count:uint64!=2、avg:float64?=1.5；q:float32 `[1,NULL,2]` 预期 sum:float64?=3.0、min/max 保留 float32?。这些期望直接来自输入字面量和本 note 类型表，不从 DataFusion 普通 SUM/COUNT schema 反推。补 int64 `[MAX,1]`、uint64 `[MAX,1]` 的 sum 溢出错误，验证下次执行 fresh。

stream closure 独立轨迹：HOP(size=10,slide=5)，先 t=0,value=1，再 W=5，只输出 `[-5,5):1`；再 t=0,value=2 时仅 `[-5,5)` assignment 迟到，`[0,10)` 累计为3；正常 EOF 无需 W=10，恰再输出 `[0,10):3`，无重复。batch 原始两行会得两个窗口各3，**不应**与这条含丢弃 assignment 的轨迹宣称等价。无中途水位的同两行至 EOF 则两个窗口均3，与 finite 相符。

## Typical declaration seeds and implementation acceptance

下面是**待本片实现后**的公开调用种子，本轮只静态检查。Python 沿现有 namespace，SQL 独立表达同一行为；后接 SQL 展示 mixed DAG，不 collect 中间数据：

```python
import pyarrow as pa
import calc_flow as cf

def declaration():
    trades = cf.table_input("trades", schema=pa.schema([
        pa.field("ts", pa.timestamp("us", tz="UTC")),
        pa.field("symbol", pa.string()),
        pa.field("quantity", pa.int8()),
    ]))
    df = cf.window.tumbling(
        trades, event_time="ts", size_micros=10, group_by=["symbol"],
        aggregates=[cf.window.count("quantity", output="n"),
                    cf.window.sum("quantity", output="volume")],
    )
    sq = cf.sql(
        "SELECT window_start, window_end, symbol, "
        "COUNT(quantity) AS n, SUM(quantity) AS volume "
        "FROM CF_TUMBLE('trades', 'ts', 10) "
        "GROUP BY window_start, window_end, symbol",
        trades=trades,
    )
    post = df.sql("SELECT * FROM input WHERE n > 0")
    batch = cf.Program("bars", mode="batch", outputs={"df": df, "sql": sq, "post": post})
    return batch, batch.for_mode("stream")
```

Rust 同样是关系声明；实际 execute/Runner 绑定沿前序 Program 接口：

```rust
use calc_flow::{AggregateFunction, InputOptions, Result, TableExpr, WindowSpec, table_input};
use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn declarations() -> Result<(TableExpr, TableExpr)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), true),
        Field::new("quantity", DataType::Int8, true),
    ]));
    let input = table_input("trades", schema, InputOptions::default())?;
    let spec = WindowSpec::hopping("ts", Duration::from_micros(10), Duration::from_micros(5))?
        .aggregate(AggregateFunction::Count, "quantity", "n")?
        .aggregate(AggregateFunction::Sum, "quantity", "volume")?;
    let df = input.window(spec)?;
    let sq = calc_flow::sql::sql(
        "SELECT window_start, window_end, COUNT(quantity) AS n, SUM(quantity) AS volume \
         FROM CF_HOP('a', 'ts', 10, 5) GROUP BY window_start, window_end",
        BTreeMap::from([("a".into(), input)]),
    )?;
    Ok((df, sq))
}
```

实施完成需要以下定向证据，不用只跑 convenience collect 代替 plan 隔离：

1. **四单元 / A05–A06/A08–A09 / C11/O02：** 分别从公共 Rust/Python DF/SQL 构造 tumbling 与 hopping，collect_schema/analyze/compile/export 显示真实 window 节点，batch/stream 与上面独立字面量/schema 预期一致。覆盖负时间、半开边界、nullable group/time、count 非 NULL、三类 sum 提升、typed empty；未支持 SQL/类型/offset 准确拒绝。
2. **A04/A08：** 每语言 DF→row-local SQL→window→DF→SQL 与 SQL-window→DF→SQL，无中间物化；时间原样/重命名通过，时间算术拒绝。两个输出共享同一个窗口，另有独立合法输出；window 策略不吞 union/aggregate 等独立分支。导出再加载，schema/bindings/owner 重数与结果保持。
3. **L01–L03/C13：** 直接复用同一 compiled finite plan、同 Runtime、暖缓存、无 reset，依次完整 fixture → `(1,A,8)` → typed empty，应为本表 → 单个 `[0,10)` A count1/sum8 → 0行；插入 schema/overflow/取消失败后再执行无旧值/ended。stream 两批跨批累加、空批不 reset、新 job fresh。
4. **EOF/水位与恢复：** 单批/逐行/多 RecordBatch 在相同控制轨迹位置比较；无水位 EOF 排空、W=end 恰关闭、hopping 部分迟到按上表。stream 未闭合窗口 durable checkpoint→新 runner 恢复→EOF 对照不中断结果；复用原有 fixture，检查真实 owner 与配置/layout/fingerprint 负例，不把内存 restore 充作 durable restart。
5. **latest/诊断：** append-only window 在显式 latest 中可用；complete_snapshot 直达及经 row-local 中转到窗口逐边预拒绝，含空快照与暖缓存。高层结构域先拒绝的情况不足以验证低层 update gate，另用实际 schema 合法低层 graph 验证，source.open/job/state root 计数为0。r5 parser/gate 对齐、EOF 条件性证明、closed prefix 未知及 legacy 7/latest6 保留。
6. **必要交付面：** Rust exports/rustdoc、Python exports/注解、私有 PyO3/stub、capability layout `(1,2)`、文档支持表与独立 `examples/finite_event_window.py` / `crates/calc-flow/examples/finite_event_window.rs` 同步。只检查受影响测试/编译/格式/示例，完整回归/覆盖率交 CI，不默认性能或增加全专员串行阶段。

## Rationale, open questions and handoff

复用 WindowSpec 的一个 Rust 方法比另造长参数 API 更容易与现有配置、错误和 project 对齐；Python 现有 namespace 已够用，不再增加第二套入口。SQL 选择带 CF_ 的显式窄方言，避开“普通 DATE_BIN/OVER 就等价于事件窗口”的歧义，也避免用户猜测 TUMBLE/INTERVAL/offset 的外部方言含义。拒绝在 Python 展开 hopping 行或用普通 SQL batch 重算替代原生节点，否则很容易漂移 count dtype、NULL/空窗口与迟到合同。

不选择 MAX watermark、临时持久 stream job、batch 跨次 accumulator 或把窗口当 replacement 消费者。它们分别混淆数值边界/EOF、所有权、执行隔离或更新解释。closed_window 前缀保留准确 unknown，不通过改标签消除真实比较前提。layout inventory 以当前源码生成/恢复能力为准，不追随旧 spec 的 `(1,)` 断言。

**无需要用户决定的 API 阻塞。** 已选择的入口、SQL 子集、零 offset、finite/EOF、latest、schema、导出与版本边界可直接实现。真实实现缺口为 finite adapter/各层 mode gate、Rust 关系窗口、native SQL 描述及局部 passthrough 证明、mixed lowerer 接入、r5 分析与 layout inventory 对齐；这些不是已完成产品能力，也不要求新增外部依赖或另开设计轮。更宽 SQL、非零 offset、窗口 replacement、stream 中途前缀的显式 cut 选择均留后续，不阻本片。

文档自检通过：7 个代码块围栏配对、3 个 Python 片段 AST 解析、2 个相对链接解析、6 张表格格式、9 行 membership/5 组聚合字面量的独立整数与 NULL 算术核对、溢出反例与空白检查。没有导入 calc_flow 或执行产品代码；Rust 种子仅按公开声明核对，未编译或声称 rustfmt 通过。仓库仅新增本文，生成 project schema 无差异。

本轮仅写本文并做源码/契约和 note 自检，未修改产品、建分支/PR、运行产品测试、提交/推送或扩派。交协调器按实现/定向验证与文档 → 独立 dal-reviewer 推进，仅发现真实覆盖缺口时再加 tester。DAL-516 尚待实现与 PR 实际合并，保持 in_progress。
