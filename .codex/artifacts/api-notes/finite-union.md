# Finite Union — API Note

状态：DAL-515 / S2a 局部公共合同冻结，2026-11-12，可交 implementer。**本文新增签名、SQL lowering、有限执行和诊断规则均待实现；没有产品运行或测试通过的声明。**

基线：`a5a5f4d8972e5eba47147589d695bf0ecbb20994`，分支 `agent/dal-orchestrator/e18e31f2354e`；开始时工作树干净。该 HEAD 包含前序 DAL-514 / PR #597。依据为本 issue、触发评论 `01a1cc42-9586-76ba-9f09-361bfcba0995`、父 DAL-498 的生命周期/四单元讨论，以及通过 Multica CLI 下载核对的开发计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58` 的 S2a/G2、C05/L02、A04–A06/A08–A09。

## Audiences and scope

Rust 用户从公开 `calc_flow::TableExpr`、`calc_flow::sql::sql`、Program 和既有 plan/Runner 使用 union；Python 用户从根包 `calc_flow` 的 TableExpr、sql、Program 使用同义能力。两语言各有 DataFrame/SQL 一等声明入口；batch/stream 是独立维度。

本片只增加**同 schema 的表行追加 union**：同一 union 节点双生命周期编译，保留重复、NULL 和每源 FIFO，按相同完整输入比较有限终态。没有去重、按名对齐、隐式类型提升、latest 快照合并、一般多输入 SQL 增量执行或跨源总序。snapshot_sql finite、window/join/ASOF、S3 全历史 DISTINCT/top-k/OVER、external 强制隔离及 batch 中途恢复继续按原范围排期。

沿用 [frontend note](first-class-relational-frontends.md)、[update note](latest-relation-update-contract.md)、[aggregate note](relational-aggregate-replacement.md) 的已合并合同，实际能力以本 HEAD 源码为准。它们标注“待实现”的旧时态不代表今天仍缺。按 `AGENTS.md`、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md` 使用 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint；本仓库边界为 Rust core → PyO3 → Python adapters，没有 C++/Excel/dal-public 或 component-boundaries 文件需要补造。

## Surface today: verified at the baseline

| Source                                                                  | Existing capability / gap                                                                                                                       |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/src/dataframe.rs`、`dataframe/aggregate.rs`、`lib.rs` | 已导出 TableExpr/ColumnExpr、aggregate/count_rows 及 typed row-local 变换；RelationKind 没有 Union，TableExpr 没有 union_all。                  |
| `crates/calc-flow/src/sql.rs`、`program/lower.rs`                       | typed SQL 返回可组合 TableExpr，schema 由 native planner 推导；当前 SQL 声明整体降为 SqlOperator，没有 set-operation → union lowering。         |
| `crates/calc-flow/src/operator/union.rs`                                | 已公开 UnionOperator::new(name, Vec<Port>)，至少两个唯一端口，同 kind/schema/schema-presence/required；stream 原样转发 Batch，无行状态。        |
| `crates/calc-flow/src/pipeline/batch.rs`、`config.rs`                   | CompiledBatchOperator、compile_batch、project 构造及校验均仍拒绝 union；只改一处 mode metadata 不足以交付 finite。                              |
| `crates/calc-flow/src/config.rs`、`schemas/project-v3.schema.json`      | 已有 OperatorSpec::Union / `{"kind":"union"}` 和显式输入/输出 Port；本片不需要新 project operator 或 format。                                   |
| `python/calc_flow/symbolic/{expr,sql,analyzer}.py`                      | 已有 aggregate/sql/Program；无 union primitive 或 TableExpr.union_all；普通 SQL stream 仍先拒绝多 alias。                                       |
| `python/calc_flow/symbolic/lower/{sql,schema,program}.py`               | 已有 SQL/aggregate/shared DAG lowering、native schema seam；尚无 union boundary。                                                               |
| `crates/calc-flow-python/src/config.rs`、`python/calc_flow/_native.pyi` | 已有 _infer_sql_schema 和 _analyze_query_graph；没有提供 SQL UNION ALL 分支拆解的 shared seam，不能用“整个 SQL schema 可推导”当 union 证明。    |
| `crates/calc-flow/src/query_analysis.rs`、`query_analysis/updates.rs`   | 报告 schema_version=2、analysis_revision=3；union 被列为 stream-only，stream 本地事实 current_batch/append/per_row_final，latest 已拒绝快照边。 |
| `python/calc_flow/symbolic/query_analysis.py`                           | 解析支持 (1,1)/(2,2)/(2,3)，latest gate 固定要求 (2,3)；新增 union 证明时必须同步升级 gate，不能只升级报告生产者。                              |

已读相邻证据：`tests/stream_compile.rs` 的 union 构造/stream 准入/batch 拒绝，`tests/stream_operator.rs` 的 FIFO/共享 payload/未知 ingress，`tests/query_history_portability.rs` 的 stream-only 报告，`tests/continuous_public.rs` 和 `tests/project_connector_compile.rs` 的运行/项目入口。这里只静态读断言，没有重跑。

两语言 `examples/32_sql_expression_pipeline` 已含前序 mixed DAG/aggregate；保留其位置与用途。`crates/calc-flow/examples/sql_plan_reuse_probe.rs` 是已有 UNION ALL + GROUP BY 的 batch SQL 性能探针，证明 SQL batch 能做 set operation，不证明 UnionOperator 已支持 batch；不把它改成新前端示例，也不运行性能。

## Proposed public surface

新增一个 immutable 二元方法；链式调用表达多路 union，不新增 concat/union 别名、配置结构或全局 SQLContext。下列为签名清单，省略实现体。

~~~rust
impl TableExpr {
    pub fn union_all(&self, other: &TableExpr) -> Result<Self>;
}
~~~

~~~python
class TableExpr:
    def union_all(self, other: TableExpr, /) -> TableExpr: ...
~~~

Rust 保持现有 TableExpr 的 Clone/Debug 与私有字段；Python 返回新 immutable 节点，不捕获 rows/runtime。只检查 host 类型与可静态验证的声明条件；涉及 SQL 派生 schema 的比较在 collect_schema/analyze/compile 规划时完成，不执行样本行。Python 非 TableExpr 参数抛 TypeError。允许 `t.union_all(t)`：共享上游计算一次，但同一结果连接两个不同 ingress，行出现两遍；不能用子节点 digest 去重输入重数。

新增方法在 Rust dataframe module 与 crate root 已导出的 TableExpr 上可发现，在 Python 根包/symbolic 的同一个 TableExpr 上有真实注解与文档。没有新的 native Python union class；私有 native SQL 规划 seam 若需扩充，应同时更新 PyO3 和 `_native.pyi`，不把内部 seam 当用户入口。

现有 SQL 签名**不变**：

~~~rust
pub fn sql(query: &str, tables: BTreeMap<String, TableExpr>) -> Result<TableExpr>;
~~~

~~~python
def sql(query: str, /, **tables: TableExpr) -> TableExpr: ...
~~~

Program 的 mode/engine、for_mode、with_update_semantics、collect/stream/execute/schema/analyze/export 签名均沿现有合同。没有 union-specific runner、finalize/flush 调用、预算参数或更新开关。

UnionOperator 保留原构造签名，新增既有 BatchOperator trait 实现：`async fn process(&mut self, inputs: &BTreeMap<String, Batch>, context: &BatchOperatorContext<'_>) -> Result<BTreeMap<String, Batch>>`，返回单一 output。CompiledBatchOperator、两类 builder/project 校验和查询事实必须接通这个同类型实现；不通过 external wrapper 或临时 SQL 替代它。

## Common declaration and SQL subset

DataFrame union 降到既有 `OperatorSpec::Union` / `NodeOperator::Union`，生成 table required=true 的 `input_0`、`input_1` 和单一 `output`；端口名只在新 frontend 内约定，低层用户原有端口名不变。二元链保留声明树即可；不要求优化成一个 n 元节点。共享节点只计算一次，保留每条消费边。

SQL 公共子集为**顶层纯 UNION ALL 树**，至少两个 arm、至少两个不同的显式 SQL alias；每个 alias 恰好使用一次，所有传入 alias 都被使用。arm 的 textual 顺序是新 ingress 的声明顺序，不是 BTreeMap/kwargs 的 alias 排序。允许括号构成等价 UNION ALL 树。

每个 arm 是前序已证明的确定性单 alias row-local SELECT：可投影现有列/既有 row-local 标量、显式 AS、过滤以及普通表限定名；允许 SELECT *。每个 arm 必须只有一个直接 FROM alias、一次关系引用。arm 及外层均无 CTE、子查询、JOIN、aggregate、DISTINCT、OVER、ORDER BY、LIMIT/OFFSET/FETCH、set BY NAME、UDF/provider/volatile/clock。UNION、UNION DISTINCT、INTERSECT、EXCEPT 不在本子集。**必须在各 arm 独立规划后、set operation 隐式 coercion 之前比较 schema。**

典型 SQL：
`SELECT x FROM a UNION ALL SELECT x FROM b`；
三 alias 可继续 `UNION ALL SELECT x FROM c`。两个不同 alias 可以绑定同一 TableExpr，等价于 DF 的重复 ingress。单 alias 重复引用（包括 TableExpr.sql 的 `input UNION ALL input`）本片保留旧 SQL 路径，不授予新的公共 union 证明；需要同图多重输入时用 DF 方法，或给 sql 两个显式 alias。

native AST/规划层产生一个共用 union 描述：有序的 arm query/alias/规划 schema，随后两语言适配成相同 union 节点。每个 arm 用既有单 alias SQL/row-local 内核，再接 UnionOperator；后续 DF/SQL、aggregate、fan-out 走同一 DAG。Python 不写另一份 SQL parser、regex split、类型推导或 union 行计算。私有 helper 的文件/函数名由 implementer 选择；descriptor 必须 data-only，不能携 rows/callables/运行 owner。

先做此识别，再应用“普通多 alias stream SQL 拒绝”的旧 gate；只有该子集获准。把原 SQL 整体交给每批执行、在 Python concat 两份 collect 结果，或 stream 路径仍保留多 alias SqlOperator，都不算四单元完成。

**旧 SQL 的能力保留：** 不在上述公共子集的查询仍走既有 SQL 行为和诊断，包括 batch 的 coercion/首 arm 命名、复杂查询及单 alias per-batch SQL。union arm schema 不同的 SQL 可以继续合法 batch 执行，但不能获得 shared-union 支持或绕过旧 stream gate；报告必须保留实际 batch schema，并定位其不能进入 exact-schema 公共子集的原因。DF union 则直接 schema_mismatch。严格 portability/latest 对未证明形状沿现有 supported-only gate 拒绝，不自动转模式、物化中间结果或削减 SQL。

## Exact schema, lineage and ordering

公共 union 的每个输入与输出有同一个**规范化的精确表 schema**：列数、列序、名字、Arrow dtype（包括 time unit/timezone）、nullable 全部相同。输出取这个公共 schema，不重命名、不重排、不扩宽类型、不把 nullable=false/true 合并为 true。NULL/重复是值，不影响 schema；不能因某批没有 NULL 缩窄 nullable。只覆盖前序 frontend 已支持、可按 project-v3 表达的类型，不借 union 扩展 Arrow 类型域。

高层 schema 按现有 `schema_fields` / Python Field 规范化：schema/field 用户 metadata 不属于其声明身份，入口仅重包共享 Arrow buffers，不修改调用者。低层 `Port::validate` 仍按实际 Arrow Schema equality 验证，UnionOperator::new 的 schema/required 一致性不放松；不能为通过 union 修改全局 schema equality。低层已有 schema-less/table、array 或统一 optional Port 的 stream 转发保持原支持；**本片新增 finite 准入仅 table + 显式相同 schema + 所有 ingress required=true**，其他形状 finite 准确拒绝，避免发明 external payload 合并或缺省输入语义。

union 产生新 row lineage；从任一 arm 取得的列不能直接用于结果 projection，改从 union 结果取列。清除 entity/event_time/sequence 与 temporal ordering 证明，SQL arm 的 ORDER BY 也不能为 union 提供跨源序。不要把 union 当 RowLocal 从第一个 child 继承 identity、快照标志或排序。

保留每 ingress 内 Batch、RecordBatch、行的 FIFO；不同 ingress 的交错不承诺总序、左右先后、时间归并或稳定 ties。有限实现可以按端口声明顺序收集，以保持可重复实现，但它不构成跨 mode 的全局排序承诺。全局 row order/tie facts 为 unspecified，temporal identity=None。声明有序比较遇 union 应报 row_order_unproven，而不是排序后假装原声明有序。

## Updates and finite terminal relation

union 始终是 append-only 消费者，不新增 replacement 状态或 alias 最新表缓存。

| Actual input / choice                           | Behavior and admission                                                                                                                         |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| 已证明 append，legacy_append 或 latest_relation | 按行追加；输出 append/per_row_final，边解释 append；latest 不把普通 source 批次变成全关系。                                                    |
| complete_snapshot → union，legacy_append        | 保留快照事件逐批追加的旧解释；输出是这些事件的行多重集，边保留 producer_update=complete_snapshot / consumer_interpretation=append。            |
| 任一 complete_snapshot → union，latest_relation | stream 启动前 unsupported + update_interpretation_mismatch，定位该 ingress；不因为另一路是 append 或已实现 replacement aggregate 就放行。      |
| unknown 更新/边界，latest 或显式严格比较        | unknown + update_semantics_unproved / update_event_boundary_unproved，沿原 gate 拒绝；不把合并多个输入当成丢弃 provenance 的理由。             |
| 同 graph batch aggregate → union                | 本次 execute 的有限 aggregate 结果可追加；stream 完整快照那一侧仍准确拒绝，报告允许 batch supported / stream unsupported，不强求全图双端支持。 |
| snapshot_sql → union                            | 保留 snapshot_sql 输出必须直达 Sink 的限制；union 不成为绕过该限制的路由。                                                                     |

batch 一次 execute 为每个 required ingress 提供一个 Batch，Batch 内全部 RecordBatch 是该源本次完整输入 H。缺少绑定是错误，schema 正确的0行 Batch 是合法输入；每源可有任意批切分，比较时 H 必须相同。有限 union 不复用 stream job，也不向调用者暴露控制注入：先验证所有输入，再产生一个 output Batch，包含本次各输入的全部行；任意一侧空不删其他侧，两侧空也返回带公共 schema 的0行结果。若没有可复用的 RecordBatch，必须构造 typed empty RecordBatch，不能调用 Batch::table(vec![])。

有限 output 使用本 Operator 名、sequence=0、空 attributes 的新 BatchMetadata，不把多 source provenance 压成某一输入的 source/cursor；它不是重放位置或全局行序。stream 原有转发 Batch 的 metadata/cursor 行为不变。终态数值/schema 对照不比较 envelope metadata、批数或 sink 事件数。

stream 每个已接受 Data 原样按原 union 转发；无 Data 的源 EOF 不需要伪造空 Data。EOF/Idle/Watermark/Barrier 不生成额外行；只有所有相关源 EOF、原有 FIFO 数据排空且 job 成功后，才把该输出的**全部 append 行拼接为终态**。不是“取最后一个 Batch”，不是先到的源 EOF 关闭整个 union。stream 零 Data 产生零行历史；有限端显式传 typed empty，二者终态多重集相同，消息数量可以不同。控制对齐、checkpoint/source cut 和逐输出 delivery 沿既有 runtime，不重设计。

公共比较请求使用现有
`PortabilityRequest(final_relation, accepted_history, all_sources_eof, ordering=unordered)`
（Python 用具名字段；Rust 用现有 new/with_ordering）。union 本地 history 可保持 current_batch，因为它不存历史；accepted_history 是截止 EOF 后由所有 append 输出归约的观察范围。必须新增这种组合证明，不能只把 union 从 stream_only 列表移除；不在节点伪造 retained state 或 complete_snapshot。纯 append union 请求 history=latest_relation 应报 history_scope_mismatch；Program 的 latest 更新选择与请求的 history 仍是两个独立维度。

严格 final_relation 遇 snapshot→union 的 legacy 边，不能证明“把原源 H 跑一次 batch”等价；即便 downstream 只是 projection/终端，也应定位 update_interpretation_mismatch，不只在下游又有 aggregate 时才检查。此诊断不修改旧 legacy 的实际执行。上游 per-batch DISTINCT/top-k/OVER、未知时序或非确定性等仍须各自证明，不因末端 union 本身可追加就放行全图。

finite UnionOperator 无 query accumulator；CompiledBatchOperator 的 snapshot 为 null、非 null restore 拒绝、reset 为 no-op，沿现有 stateless 内置约定。复用同一已编译 plan、同 runtime、暖缓存，换输入/全空输入/失败后重试均只含本次行。缓存只存只读计划/精确 schema，不存输入 Batch、结果行或 owner。stream 同 job 的上游有状态算子继续跨批保留状态，新 job fresh，显式恢复按原合法 checkpoint 续接。

新增 finite 仅追加已存在的 immutable RecordBatch，不引入 union state budget 配置；沿原执行取消、输出限制和 stream edge budget 校验完整 envelope。超过现有限制或取消仍返回原层的错误，不截断行、切片绕过预算或提交缓存结果。无新增全图事务/ExactlyOnce 承诺。

## Diagnostics, versions and compatibility

沿现有 QueryAnalysis/QueryDiagnostic 与 Python CompileError.query_analysis。新 union 本地事实、SQL arm 拆解、snapshot 边义务改变分析规则，输出 **schema_version=2、analysis_revision=4**；不新增报告字段、枚举或 capability schema 版本。Rust producer、Python parser、latest gate 和绑定/stub 的 revision 说明同步；旧 (1,1)/(2,2)/(2,3) 仍可按既有方式展示，不能代替本次 r4 新 union 准入证明。运行入口命中缓存也用当前规则重新 gate。

令 P 为该关系的 canonical 逻辑 path。新增 DF ingress path 固定为 `P.union_all.inputs[0]` / `P.union_all.inputs[1]`，字段错误后接 `.schema[i].name|data_type|nullable`。SQL 保留 `P.sql.tables.<alias>` 的用户 alias，arm 查询问题用 `P.sql.arms[i]`；嵌套 union 逐级追加，非 identifier 名沿 JSON string 下标规则。重载 project 仍使用已有 graph node/edge 物理地址，不虚构原逻辑 alias。

| Input violation / diagnostic                               | Status / required message content                                                                                                      |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Python other 不是 TableExpr                                | TypeError；union_all.other requires TableExpr, got <type>。                                                                            |
| DF 列数/列名/顺序/dtype/nullable 不同                      | schema_mismatch，unsupported；指出 input index、field index/name、expected/found，需 exact schema；不是泛泛的 invalid union。          |
| SQL arm 不满足共同 schema                                  | schema_mismatch，unsupported 的 shared/stream 诊断，定位 arm alias/field；batch legacy planning/execution 仍按原能力；不能报双端支持。 |
| complete_snapshot → union，latest 或严格终态比较           | update_interpretation_mismatch，unsupported；完整快照需要 replacement，此 union ingress 只追加。                                       |
| 缺 schema/更新/完整事件证明                                | unresolved_type 或既有 update_*_unproved，unknown；说明缺失证明，不能靠首批推断。                                                      |
| 不在 SQL union 识别域的声明                                | 保留旧执行范围；严格请求用 sql_semantics_unproven 等真实原因；旧多 alias stream 拒绝仍 unsupported_mode，不把合法 SQL 报成语法错误。   |
| 无全局排序证明                                             | row_order_unproven，unknown；说明 per-ingress FIFO 不证明跨源总序或 ties。                                                             |
| finite 低层 union 使用 array / schema-less / optional Port | unsupported_mode，unsupported；消息指出 node/input、batch、table + exact schema + required=true 约束；原 stream 支持不变。             |
| runtime missing input / schema / budget / cancellation     | 沿原 typed error/exception；包含实际 endpoint 和约束，不吞 source 因果链；先验证所有 finite 输入，不返回部分结果。                     |

构造错误走 Rust InvalidArgument / Python TypeError、ValueError；规划/编译失败沿 Compile 等现有层次，message 保留 `{path}: {code}: {message}`。latest mandatory gate 用 QueryUpdateSemantics，显式 portability gate 用 QueryPortability；Python 都保留不可变 query_analysis。诊断仍按 (path, code, status) 排序去重，逐消费边报告；unknown 不能当 supported。规则改变之外的旧诊断地址不批量迁移。

保留以下身份界限：

1. 既有低层 union stream 的配置、端口、stream fingerprint、stateless checkpoint capability、payload/layout/source cut 不变；仅为 table exact-schema、全部 required 的 union 增加 batch 生命周期。Project format=3、OperatorSpec::Union 无新字段，生成 schema 不应漂移。
2. 既有 SQL 声明文本/alias/Program digest 编码不改；DF 新 primitive 有自己的稳定有序身份。for_mode 复用同声明 fingerprint，编译 cache/native/project/runtime 仍保留 mode、schema、update_semantics、registration 和原恢复身份。不要求 DF 与 SQL 两种写法 hash 相同。
3. **获准的多 alias SQL 公共子集会产生新的 native union 拓扑**，与旧单 SqlOperator 的 native fingerprint 不同，这是有意的编译产物变化，不能声称字节相同。原多 alias 查询没有合法 append-stream checkpoint，因而不需要把旧 SQL state 恢复到新 union。新导出记录实际 union 拓扑；新版不得自动把旧保存的 `kind=sql` 项目重写为 union，也不接纳 fingerprint 不同的 checkpoint。
4. 准备缓存为采用新 SQL union lowering 的条目增加私有策略标识（如 finite_union_v1），防止混用旧整体 SQL 准备结果；不全局改旧 SQL key，不删除 mode。公共子集之外及单 alias SQL 的原 lowering/默认/恢复路径保持。
5. 旧 SQL snapshot 追加链7、获准 replacement 链6、sql_replacement_state_budget 准入与原恢复格式继续有效。aggregate→union 只因 union 的 append 限制拒绝 latest，不回退或放宽前序 aggregate 白名单/预算。

## Typical examples and implementation acceptance

以下是**待实现后的调用种子**。Python 示例语法可静态解析，本轮不导入包或执行。

~~~python
import asyncio
import pyarrow as pa
import calc_flow as cf

async def example():
    schema = pa.schema([pa.field("x", pa.int64(), nullable=True)])
    a = cf.table_input("a", schema=schema)
    b = cf.table_input("b", schema=schema)
    df = a.union_all(b)
    sq = cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=a, b=b)
    batch = cf.Program("union", mode="batch", outputs={"df": df, "sq": sq})
    stream = batch.for_mode("stream")
    assert stream.fingerprint == batch.fingerprint
    av = pa.Table.from_arrays([pa.array([1, None, 1], type=pa.int64())], schema=schema)
    bv = pa.Table.from_arrays([pa.array([1, 2], type=pa.int64())], schema=schema)
    tables = await batch.collect_async({"a": av, "b": bv})
    for table in tables.values():
        values = table["x"].to_pylist()
        assert values.count(1) == 3 and values.count(None) == 1 and values.count(2) == 1
    seen = {"df": [], "sq": []}
    async with stream.stream({"a": [av], "b": [bv]}) as events:
        async for event in events:
            seen[event.name].extend(event.table["x"].to_pylist())
    for values in seen.values():
        assert values.count(1) == 3 and values.count(None) == 1 and values.count(2) == 1
~~~

Rust 声明种子如下；schema/analyze/compile/execute/Runner 用前序已公开的 Program 接口，不手写用户物理端口：

~~~rust
use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{CompileMode, InputOptions, Program, Result, table_input};
use std::{collections::BTreeMap, sync::Arc};

fn declaration() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let b = table_input("b", schema, InputOptions::default())?;
    let df = a.union_all(&b)?;
    let sq = calc_flow::sql::sql(
        "SELECT x FROM a UNION ALL SELECT x FROM b",
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    Program::new(
        "union", CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sq".into(), sq)]),
    )
}
~~~

implementer 的最小证据集沿 C05/L02、A04–A06/A08–A09，不扩成全类型矩阵：

- 四单元分别用公共 DF/SQL 声明，schema/analyze/export 中实际有 union，双端编译执行；按独立 oracle 验 [1,NULL,1] + [1,2] 为多重集 [1,1,1,2,NULL]，schema 仍 nullable Int64。两侧 typed empty、单侧空、多 RecordBatch、stream 某源无 Data 分别覆盖必要边界。
- 一份直接编译的 finite plan 无 reset 连续执行：上述输入 → [9]+[] → []+[]，预期五行 → [9] → typed empty；夹入一次 schema/取消失败后再执行不残留旧行。convenience collect 每次新建 plan 不能代替这项证据。
- stream 多批/交错验证各源 FIFO 和重复重数，t.union_all(t) 与两个 alias 绑定同节点都计算上游一次、消费两次；union 前后加现有 row-local DF/SQL 和一个共享命名输出，不 collect 中间表。
- schema 的列名/顺序、Int32/Int64、nullable 差异准确拒绝 DF/shared-union；SQL legacy batch 的既有 coercion 另验仍合法。已知 complete_snapshot→union latest 预拒绝，含空快照/另一路 append；断言 Source/binding/job/state root 尚未触碰，暖缓存也不绕过。legacy 快照追加路径保留。
- 验最终多重集请求 supported、声明全局序 unknown、复杂 set SQL 不获虚假证明、batch/stream 不支持形状定位正确；r4 parser/gate 对齐，不只改 hardcoded expected revision。
- data-only project 往返仍保留 union 和 schema/端口重数；新 batch project 可编译，原 stream union、控制对齐/恢复及旧 SQL 项目/默认身份复用直接受影响回归。四单元 exports/注解/必要 native stub、文档支持表及公开示例同步。

保留 example 32 的原主题；若增加 union 示例，放在本仓库 examples 与 crates/calc-flow/examples 的对应独立程序，不搬动原 core 示例或改性能探针用途。先定向 RED/实现/必要验证，再交 dal-reviewer 独立主审；本轮不执行这些产品验收、创建 PR 或扩派。

## Why this shape, open questions and self-check

二元 union_all 与 SQL UNION ALL 同名同义，保留 immutable TableExpr 和原 plan/Runner；链式调用足够表达多路，不需要 concat 模式、参数包或另一套关系 API。拒绝隐式 coercion/按名对齐，保证既有 UnionOperator 的 exact Port 合同可直接复用；SQL 更宽能力保留在 legacy batch，不冒充公共子集。

拒绝让 union 保存每 alias 最新快照：这会新增多输入初始化、空替换、预算、候选原子提交和 checkpoint 合同，已经超出 S2a append-final 片。拒绝把整个 UNION SQL 留在独立 batch 内核充当适配完成，或为了支持它削弱 stream 多 alias 的一般拒绝。有限执行只适配既有 union 节点，stream FIFO/控制/恢复和 stateful 上游保持原合同。

**本片无需要用户回答的接口阻塞。** 以上入口、SQL 子集、schema/update/finality/diagnostic/identity 边界已选定；内部 helper/文件布局、二元链是否做保序 flatten 属 implementer 选择。多输入 replacement union、单 alias 重复 set-query 的公共证明、更宽 SQL arms 和类型对齐均为后续增量，本片不承诺。

本轮自检仅覆盖源码事实、前序/父计划引用、现有与新增时态、签名/参数、诊断与 revision、身份兼容、Markdown 表格/链接/空白及 Python 示例语法；Rust 示例按现有签名静态核对，当前工具链无可用 rustfmt，未声称 Rust 格式或编译检查通过。未运行产品测试、native build、性能或 CI；只写本文，DAL-515 仍在实现阶段，保持 in_progress，附 note 交协调器接续。
