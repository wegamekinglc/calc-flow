# Finite Inner Join — API Note

状态：DAL-517 / S2c 局部 API 合同冻结，2026-11-12，可交 implementer。**本文的新 Rust 关系方法、SQL 方言、原节点 finite 执行及诊断规则尚待实现；未运行产品测试，不表示本片已交付。**

精确基线：`3c4b919eb389544d8b8080d3571e49479da68b2c`，分支 `agent/dal-orchestrator/390cc43d485c`，开始时工作树干净。HEAD 与触发评论确认的前序 PR #600 merge SHA 相同；本 checkout 没有本地 `main` ref，不把它写成已独立核验的 ref。依据：DAL-517、触发评论 `01a1ce5a-771c-7972-b86a-bc1375b00e64`、父 DAL-498 的生命周期/四单元讨论，以及通过 Multica CLI 下载核对的计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`（R01–R08、S2c/G2、C05–C07/O02、A04–A06/A08–A09、L01–L03）。

## Audiences and scope

Rust 用户从公开 TableExpr、`calc_flow::sql::sql`、Program、BatchExecutionPlan 和 StreamingRunner 使用 bounded inner；Python 用户从根包 `calc_flow.table.stream_join`、sql、Program 使用同义关系声明。DataFrame/SQL 是接口类别，batch/stream 是独立生命周期。声明不持有 rows、plan、job 或状态 owner。

本片复用现有 inner 的闭区间、等值 key、重复 pair、NULL、排序及身份内核，使**同一个 `StreamJoinOperator` / `kind=stream_join` 节点**获得 finite 生命周期，并补齐四单元声明、mixed DAG、分析、导出和例子。一般 SQL JOIN、outer/semi/anti/cross、ASOF、replacement join、snapshot_sql finite、全局排序、batch 查询中途恢复不在本片。

承接已合并的 [关系前端合同](first-class-relational-frontends.md)、[finite union](finite-union.md)、[finite event window](finite-event-window.md) 和 [symbolic relational DAG](symbolic-relational-dag.md)；旧 note 的“待实现”时态以本 HEAD 源码校正。遵循 `AGENTS.md`、`.codex/agents/cf-api-designer.toml` 和 `docs/introduction.md` 的 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint 词汇。实际依赖是 Rust core → PyO3 → Python adapters；仓库没有 dal-public、Excel/C++ binding 或 `.codex/references/component-boundaries.md`，不补造它们。

## Surface today: verified source facts

下表 Rust 路径相对 `crates/calc-flow/src/`，Python 路径相对 `python/calc_flow/`。行号仅定位本基线，不是实现后的固定行号。

| Evidence                                                                      | Existing behavior / gap                                                                                                                    |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `lib.rs:120`、`operator/join.rs:2405–2690,3581`                                | 已导出 StreamJoinOperator/Spec/Type、JoinTimeBounds/Limits；spec 只有 inner，operator 编译两个精确 table schema。                                         |
| `operator/join.rs:2467,6180–6241,6343`、`operator/join/admission_masks.rs:227` | bounds 用 i128 闭区间比较；key 类型相同；NULL time/key 丢弃；时间换算失败报错。原生时间支持域比公共 symbolic 宽。                                                              |
| `operator/join.rs:5982,5153,5193`                                             | 每次 ingress 按 incoming position、opposite event_time/row_id 发 pair；Data 即匹配；on_end 清理，不补发匹配。                                                 |
| `pipeline/batch.rs:24–50,58–145`、`config.rs:1520,2444`                        | compiled batch、project 构造和 project validation 都拒绝 stream_join；尚无 BatchOperator 实现。                                                         |
| `dataframe.rs:75`、`program/lower.rs:51`、`sql.rs:35`                           | Rust 关系已有 aggregate/union/window，无 join variant/方法或 SQL join 识别。                                                                           |
| `symbolic/ops.py:1547`、`join_spec.py`                                         | Python 已有 table.stream_join；bounds/limits 必填；省略 output ordering 是 @1，完整三项是 @2。                                                             |
| `symbolic/analyzer.py:1263–1599`                                              | 仍拒绝 batch；公共时间列要求 non-null timestamp[us, UTC]；嵌套消费要求完整 v2 输出排序。                                                                            |
| `symbolic/lower/strategies.py:504,532,848`                                    | 已有唯一 join 收集、原生 spec 生成及 relational DAG 路径；旧 @1/@2 physical identity 必须保留。                                                                 |
| `symbolic/lower/sql.py:160`、`symbolic/event_window_sql.py:37`                 | 通用 SQL/union/window 图已有 materialized memo 与 native passthrough seam；不能仅让 join 专用策略截获并吞掉 mixed 分支。                                          |
| `query_analysis.rs:485,1067,1102`、`query_analysis/updates.rs:132`             | 当前 schema=2/revision=5；join 仍 stream-only，history=accepted_history、append、finality=unknown；latest 已把它列为 append-only consumer。              |
| `capabilities.py:1125`、`symbolic/lower/program.py:481`                        | stream_join@1 modes 仅 stream，state_version=1/layouts=(1,2,3)，finality=unproven、microbatch_invariant=False；watermark/checkpoint 检查须按生命周期解释。 |

已静态读现有 stream_join 的边界/状态/属性测试、`python/tests/test_symbolic_relational_dag_acceptance.py`、raw validation、两语言 example 12/13。阅读断言不是测试通过；本轮不重验 PR #600 的 CI。

## Frozen public signatures

Rust **新增**一个 immutable 方法和一个小型纯数据配置；复用既有 StreamJoinSpec，不复制 bounds/limits/keys 到十一项位置参数，也不把 InputOptions 的“input”含义借作输出合同。

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JoinOutputOrdering {
    pub entity_by: Vec<String>,
    pub event_time: String,
    pub sequence_by: Vec<String>,
}

impl TableExpr {
    pub fn inner_join(
        &self, right: &TableExpr, spec: StreamJoinSpec,
        output_ordering: Option<JoinOutputOrdering>,
    ) -> Result<Self>;
}
```

这是签名清单，省略实现体；JoinOutputOrdering 放在 dataframe 公共面并从 crate root 重导出。`None` 表示终端无排序声明，Some 必须三项完整；验证失败返回现有 Result 错误。struct 本身无 runtime/serde project 扩展，无 Default 去伪造证明。已有 StreamJoinSpec::inner 的六项必填参数、with_prefixes、JoinTimeBounds/Limits 的构造器和原生类型名保持；不因类型名含 Stream 而复制一套 BatchJoinSpec。

Python **保持现有签名、canonical attrs 和版本选择**，只扩 finite 编译/lowering；不增加第二个 join 别名或 Python spec class：

```python
class TableNamespace:
    def stream_join(
        self, left: TableExpr, right: TableExpr, /, *,
        left_keys: Sequence[str], right_keys: Sequence[str],
        left_event_time: str, right_event_time: str,
        bounds: JoinTimeBounds, limits: JoinStateLimits,
        left_prefix: str = "left", right_prefix: str = "right",
        output_entity_by: Sequence[str] = (),
        output_event_time: str | None = None,
        output_sequence_by: Sequence[str] = (),
    ) -> TableExpr: ...
```

所有 caller sequence 防御性复制，返回新声明；省略三项继续生成原 @1 bytes/digest，提供完整三项继续生成原 @2，部分提供仍 ValueError。Rust 新关系 identity 包含有序左右子节点、完整 spec 和可选 ordering；不同 DF/SQL 语法不要求同 hash。

两语言 `sql(query, explicit aliases)`、TableExpr.sql、Program mode/engine、for_mode、schema/analyze/compile/execute/collect/stream/to_project 均保持原签名。没有新 runner、flush/finalize、reset 义务或公开控制注入。必要私有 SQL seam 由 Rust AST/验证提供 data-only 描述，PyO3、Runtime 适配与 `_native.pyi` 同步；私有 helper 不是 SQL 公共入口。

## SQL supported spelling and lowering

冻结显式 calc-flow 方言，**新能力，非声称 DataFusion 已内建**：

```sql
SELECT * FROM CF_INNER_JOIN('a', 'b', '{"spec":{"join_type":"inner","left_keys":["k"],"right_keys":["k"],"left_event_time":"ts","right_event_time":"ts","bounds":{"before_micros":2,"after_micros":3},"limits":{"max_state_rows_per_side":100,"max_state_bytes_per_side":1048576,"max_matches_per_input_batch":1000},"left_prefix":"left","right_prefix":"right"}}')
```

三个必填参数依次为 left alias string、right alias string、configuration JSON string。第三项严格对象，只允许必填 `spec` 和可选 `output_ordering`。spec **复用原 StreamJoinSpec 的 raw 验证、必填项、prefix 默认、整数域与未知字段拒绝**。output_ordering 省略即无证明；出现时必须是含非空 entity_by、event_time、sequence_by 的完整对象，字段名与 Rust 配置相同，不接受 null/部分字段/未知字段。它是 frontend 元数据，不塞入 native StreamJoinSpec。对象 key 顺序无语义，数组顺序有语义；规范化配置后计算新方言声明身份。

支持域明确为一个顶层 `SELECT *`、一个 FROM 调用、恰好两个显式 tables alias，二者不同且各用一次，无额外未用绑定。alias 字面量按解码后精确名称查找；沿现有 alias 唯一/归一化规则。两个 alias 可以绑定同一 TableExpr，此时保留两个 ingress 的消费重数。函数名不带 schema、大小写不敏感；参数只接普通 SQL string literal，JSON 里的字符串按 JSON 解码，不执行表达式/UDF/代码。

无 FROM 附加 alias、CTE、子查询、JOIN ON/USING、WHERE、投影改名、aggregate、DISTINCT、OVER、set operation、ORDER BY、LIMIT/OFFSET/FETCH。过滤/投影和额外列在前后独立 DF/row-local SQL 关系中声明；本片不实现另一个宽 SQL parser。普通 `a INNER JOIN b ON ...`、BETWEEN/INTERVAL 算式不获原生 bounded join 证明，仍保留旧 batch SQL 能力及旧 stream gate。出现 CF_INNER_JOIN 却不合该语法必须准确拒绝，不回退每批 SQL。

Rust AST 识别器先于普通 SQL/多 alias gate 执行，解析配置并用 StreamJoinOperator 对精确 schema 规划；两语言取得相同的 `(left_alias,right_alias,spec,ordering,input/output schemas,diagnostics)` 描述，降成与 DF 相同的原节点。Python 不写 regex/parser、匹配/排序或类型推导。这个 SQL relation 可以继续取列、进入下一次 SQL/DF join、共享命名输出、分析和导出；只提供一个字符串包装函数但留下整条 SqlOperator 不算 SQL 单元完成。

三源嵌套通过两次 sql 声明组合：第一次 configuration 带合法 output_ordering，第二次把该 relation 绑定为一个 alias、第三源绑定为另一个；不靠未支持的嵌套 SQL 子查询。比如第一层选 `entity_by=["left__k"]`、`event_time="left__ts"`、`sequence_by=["left__seq","right__seq"]`，下一层 spec 用 `left_keys=["left__k"]`、`left_event_time="left__ts"`。最后一层无状态后处理可不声明 ordering。

低层 SqlOperator 和已保存 `kind=sql` 项目不自动解释该方言；新高层方言导出真实 `kind=stream_join`，没有 parser/JSON string 或 callback 的运行依赖。新语法从未有合法旧 stream checkpoint，不需迁移它；旧普通 SQL 不重写。

## Bounds, NULL, schema and ordering

数学定义：对每个左行 l、右行 r，仅当所有成对 key 均非 NULL 且类型相同、值相等、两时间有效，并满足 `l.t - before <= r.t <= l.t + after` 时输出一次 pair。两端含等号，0/0 只匹配相同时间；重复行有独立物理身份，m 个左副本与 n 个右副本输出 m*n，不能按 payload/key/digest 去重。时间差用原生 i128 算术，不 wrap、饱和或用浮点比较。

bounds 是 `[0,9007199254740991]` 的整数微秒；limits 三项各为 `[1,9007199254740991]`。Python timedelta 必须 exact 类型；limits 不接受 bool；Rust Duration 拒绝亚微秒。SQL JSON raw 验证同样不能把 bool/小数/负数强转整数。prefix 沿原两个不同 portable identifiers，不新增 suffix 参数。

| Layer / field       | Supported domain and NULL contract                                                                                                                                                                                                         |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 四单元公共 event_time    | 保留 Python 当前的 **non-null timestamp[us, UTC]**。Rust 新方法/SQL 子集同域；nullable、naive/ms/ns 时间在此公共域准确拒绝，不能因有限端可换算就自动获双端证明。                                                                                                                        |
| 四单元公共 equality keys | 成对 Arrow dtype 必须相同，采用现有 portable 类型与 native supported_key_type 的交集：bool、signed/unsigned integers、string/large_string、date32/64、timestamp[us] naive/UTC。NULL key 不匹配，任一 composite key 分量 NULL 即丢弃该行；不支持 float/decimal/coercion/null-equal。 |
| 原生低层 time           | 保持既有 Arrow timestamp naive/UTC 及单位换算域，NULL time 整行丢弃、不保留不匹配；不可表示的微秒报原 typed conversion failure。新 BatchOperator 与旧 stream 内核同义。                                                                                                             |
| payload / output    | 公共面沿既有 project-v3 portable 字段域；原生更宽 payload 沿原计费准入。输出先全部左字段再全部右字段，分别 `prefix__name`；保留各字段 dtype/nullable，不做 outer 式 nullable 扩张或同名 key 合并。                                                                                                 |

**C06 的 NULL time 必须分层验收：** 原生同节点的 batch/stream 验 NULL time 丢弃；四单元公开入口验 nullable time 的准确拒绝，同时用 non-null 时间的共同支持域完成真实计算交付。不能声称公共 NULL time 已支持，也不能为通过此项删除既有排序拒绝。本片不扩 nullable temporal identity；若未来要公开可空时间需另定 Source progress/identity 合同，不阻本片现有子集。

公共 schema/field metadata 规范化沿现有 frontend；低层 Port 保持精确 Arrow schema 验证。输出 schema 从原生端口取得，不从当前数据的 NULL 分布推导，不修改调用者 Arrow buffers。未知字段、列序、dtype、nullable 不匹配按对应 ingress 报错。缺绑定是错误，typed empty 有效；空侧或无匹配是完整结果 schema 的 0 行，不是 None，也不是 null 补行。输出产生新 row lineage，原输入列不能直接混入 joined select。

v1 不携带输出 temporal identity；它可作终端/无状态后处理，继续禁止在无证明时作为下一层 join 或 rolling/cross-section 的输入。v2/新 Rust Some/SQL output_ordering 的规则完全沿现有 analyzer：entity_by 恰为 prefixed 左 join keys（原顺序、字段非 NULL 且类型合法），event_time 恰为 prefixed 左或右 join 时间（non-null timestamp[us,UTC]），sequence_by 恰为 prefixed 左输入全部 sequence keys再接右输入全部 sequence keys（原顺序、类型及非 NULL 约束）。source 的 temporal metadata/输入排序要求沿当前 stream analyzer；不根据本批数据猜 identity。

下一层选择的事件时间必须等于上层已声明的输出时间；字段真实存在、prefix 和序列完整性逐层核对。ordering metadata 命名现有身份，**不会排序、缓冲或生成新唯一序列**。原生每次 ingress 的 `(incoming position, opposite time, opposite row_id)` 顺序保持；跨源交错不承诺左右先后、全局时间序、跨模式完全相同的物理行序。严格 global ordering 仍 row_order_unproven，v2 不能冒充全局 ORDER BY。

## Mixed DAG, nesting and one owner

共同成功域为精确 table source、已证明 deterministic row-local DF/SQL、bounded inner 链、row-local 后处理和命名输出；允许三源嵌套、独立 join、无关合法输出及 fan-out。嵌套中间节点遵循上节 v2；终端 v1 不升级、不在 batch 自动补 identity 来掩盖 stream 拒绝。已有合法的 post-join rolling/cross-section stream 路径保持其既有准入，不借本片扩大其他 stateful 类别。

为 A04 接通现有 `sql_passthrough_columns` 的 **join 局部**字段映射：确定性单直接 alias、无聚合/排序/去重/limit/子查询/volatile/provider/UDF 的 SELECT，可以过滤，并原样传递或重命名全部必需的 entity/event_time/sequence 字段。native 证明成立才映射 temporal metadata，保留 join ancestry；丢列、cast、时间算术、换 sequence 或来源未知，不能保持证明，嵌套/temporal consumer 准确拒绝。普通不在 join 证明路径上的 SQL 继续新 lineage、清除 ordering，不能全局改成自动继承。

DF→SQL→join→DF→SQL 和 SQL-join→DF→SQL 都不物化中间表。共有 union/window/aggregate 的**独立输出分支**不得被 join 策略丢失或错误接线；它们在 join 路径中的新组合不纳入本片公共证明。既有低层合法 stream graph 不被新 frontend 白名单收紧。SQL 与 join 路径的交集证明必须显式存在，不能凭名字含 SELECT 宣称 row-local。

每个唯一声明/旧 digest 在同一 stream job 一个 mutable owner、同次 finite execute 计算一次；所有消费边共享不可变输出 Batch。不同 spec/ordering 不合并；左右消费重数不去重，`join(t,t)` 也保留左右角色。两条不同语法但同义声明不要求跨语言或跨 DF/SQL CSE。新拓扑可采用通用 memo；已有 Python @1/@2 stream 路径必须保留旧 `cf_stream_join_<digest16>`、来源/输出 bindings 和配置 hash，不因换 lowerer 就让旧 checkpoint 失效。已有 source 名冲突的 deterministic fallback 也保留。

导出 project-v3 仍为 `OperatorSpec::StreamJoin { spec }`、required exact-schema `left`,`right`（此顺序）和 `output`。output_ordering 只参与 frontend 分析/身份，不新增 native spec 字段。导出再加载用已验证物理拓扑及既有 schema/config 身份；不声称能还原原 Python logical alias。共享 owner 和边重数必须往返保持，生成 project JSON schema 不应漂移。

## Finite lifecycle, limits and terminal comparison

一次 BatchOperator::process 收到本次完整 H_left/H_right：每个 required ingress 一个 Batch，包含该侧全部 RecordBatches，按物理 RecordBatch/行顺序。先验证两侧完整 envelope/schema；finite 的确定 ingress 次序为完整 left，再完整 right，**在两侧 Data 都处理完前不标记任一侧 EOF/Ended、不做进度驱逐**。沿现有 admission/key/contains_pair/probe/gather/row-id 和计费逻辑计算；复用内核不等于调用持久 stream 回调。

所有临时行、索引、dictionary、row-id/counter、输出 sequence、progress/ended 和候选只属于这一次调用。实现可抽取 context-independent 内部操作与 collector；不得启动 StreamingRunner、创建临时 StreamJobContext/managed checkpoint 或伪造 MAX watermark。**join 不像窗口需要 end 才补结果：每个 pair 在后到的一侧处理时即产生，finite 结束只完成本次收集并释放临时状态。**

成功后返回单个 output Batch（可含多个 RecordBatches）；无匹配也要 typed empty RecordBatch，不能 Batch::table(vec![])。新 finite envelope 沿内置约定使用 node name、sequence=0、空 attributes；不冒充某一源 cursor、stream emission sequence 或恢复位置。出现错误/取消不返回部分 output，也不把候选写入 plan；不新增跨节点事务承诺。

原三项 limits 保持必填，finite 不默认无限制：无 watermark 驱逐时，每侧本次全部 admitted retained rows/逻辑 bytes 受原 side 限制；NULL/drop 行按原计费规则。每个完整 ingress Batch 是一个 match-limit 单元，不能按 RecordBatch 或内部 chunk 重置。确定 left→right 下，right 的本次完整 envelope 产生全部匹配，故本次总 pair 数受 max_matches_per_input_batch；两侧即使无匹配仍须按原 retained-state admission 检查，不以空侧短路绕过。逻辑 bytes 不是 RSS 保证，输出/执行取消和既有收集限制继续生效。stream 按真实每次 ingress Data envelope 检查原 limit，不把 finite 的全输入范围偷偷改成 stream 全 job 上限。

因 budget 单元不同，相同 H 在切成小 stream Data 后可能成功、单 finite Batch 可能超限；只有**两端均获准且成功**时比较数值终态，不承诺预算错误跨切批完全一致，不切片绕过任何一端预算。

stream 保持同 job 跨批状态、真实 ingress progress/late/eviction、FIFO/Barrier/EOF/checkpoint 与逐输出 delivery；空 Data 不 reset 或结束。某侧先 EOF 不可抛弃另一侧之后仍能匹配的已留数据，只有原 runtime 的安全 progress 驱逐规则能释放。全部相关 source EOF、排空已接受 Data 后 on_end 清理；无额外 unmatched/pair 输出，不把 EOF 当最大时间。恢复继续按原 source cut 和状态身份延续。

有限终态与 stream 对照的前提是同 spec、同完整 H、append 输入，且 stream 轨迹没有因 late/progress 驱逐丢掉 finite H 的合法匹配候选，预算均满足、所有相关 source 正常 EOF。终态归约是**拼接全部 append pair 输出**，比较精确 schema/NULL/重复数/值；不是只取最后一个 Batch，不比较 chunk、metadata 或跨源全局行序。对含 lateness/eviction 的轨迹不能只把“接纳行”拼起来就当全笛卡尔 bounds 匹配已经等价；候选同时存活这一前提也必须说明。

使用既有 `final_relation / accepted_history / all_sources_eof / unordered` 请求证明此条件性共同终态，join stream 本地 history 仍 accepted_history、update=append；可将已证明 inner 的每个输出标 per_row_final，但不能把所有 stream_join/ASOF 一并标为相同 finality。非 EOF cut、per_batch、latest_relation history、未证明上游历史或 global ordering 均按实际义务拒绝/unknown。分析不预测未来 source 调度，不虚称验证了无丢弃轨迹。

CompiledBatchOperator 的新 join 分支采用内置 finite 约定：snapshot=NULL，非 NULL restore 拒绝，reset 无跨次状态可清。batch plan/Runtime/暖缓存只留只读 spec/schema/模板，不能持有输入/结果/owner。无需调用 reset，连续执行完整 fixture→任一空侧→全空→新数据，以及 schema/budget/取消失败后的重试，都只包含当次行。普通新 stream job fresh；只有显式合法 checkpoint 恢复延续历史。for_mode 共享声明，不共享异 mode plan/state。

## Updates, diagnostics and compatibility

join 是 append-only consumer；不是 latest 两表关系维护。已证明 append 输入在 legacy_append/latest_relation 下都可进入支持域。complete_snapshot→join 的 legacy 解释仍追加每个快照事件；显式 latest 在任一 ingress 预拒绝 update_interpretation_mismatch，含空快照、row-local/SQL 中转及共享消费边；未知更新/完整事件边界报 update_semantics_unproved/update_event_boundary_unproved。按**所选生命周期**检查：本次 batch aggregate 的有限结果不是 stream replacement 事件，不无故阻止原合法低层 finite 图；其 stream 分支仍准确拒绝。

预检在 source.open、迭代 binding、job/state root 获取前完成，暖缓存不能绕过。严格 final_relation 遇 legacy snapshot→append join 边也不能证明“原源 H 执行一次”的关系等价，需定位该边；旧非严格执行不改。旧 SQL 快照链 7、已支持 latest 链 6/其余拒绝、budget、snapshot_sql 直达 sink 和恢复身份保持。

本片新增真实 lifecycle/SQL/组合证明，分析输出 **schema_version=2、analysis_revision=6**。Rust producer、Python parser、mandatory latest gate、cache gate 和 binding/stub 说明同步；保留旧 (1,1)/(2,2)/(2,3)/(2,4)/(2,5) 展示，但旧报告不构成本片 r6 新 join 准入依据。不增加 QueryAnalysis wire 字段、project/manifest/capability schema 版本。

stream_join@1 capability modes 增至 batch/stream；state_version=1、layouts=(1,2,3) 保持。stateful/requires_watermark/checkpoint/replay 描述仍指 stream，batch 是单次临时状态，不要求 source 提供 watermark/replay。保留 microbatch_invariant=False 和现有保守 finality 清单，不能因条件性 EOF 多重集证明改成无条件在线轨迹不变。

令 P 为 canonical relation path：Python 沿 `P.stream_join`，Rust 新关系用 `P.inner_join`，SQL 用 `P.sql.inner_join`；suffix 为 `.left|right`、`.left_keys[i]|right_keys[i]`、`.left_event_time|right_event_time`、`.bounds.before_micros|after_micros`、`.limits.<field>`、`.output_ordering.<field>`。已有 Python output_entity_by/output_event_time/output_sequence_by 错误路径保留；SQL alias 用 `P.sql.tables.<alias>`，不合法 identifier 沿 JSON string 下标。项目重载仍用 graph.nodes/edges 物理地址。解析错误须映射到 SQL configuration 子字段，不能只回“invalid JSON”。

| Violation                                 | Error/status and useful message                                                                                                    |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Python 非 TableExpr/bounds/limits、非法序列     | TypeError/ValueError 沿旧声明边界；指出实际类型或 keys 输入；不接受 string 冒充序列。                                                                       |
| bounds/limits 越界或精度错误                     | 沿现有 invalid_time_bound/invalid_join_limit 等 raw code；指出具体值、字段、整数范围/精确微秒要求；SQL 不吞掉原字段。                                              |
| key 缺失/空/重复/长度不同、dtype 不同                 | unresolved_type/invalid_literal/schema_mismatch 或原 native compile；指出 side/index/field、两侧 dtype 和相同支持类型约束。                          |
| 公共 event_time nullable/非 us UTC           | ordering_required/unsupported；如 `left_event_time 'ts' requires non-null timestamp[us, UTC], found ...`；不误称 native NULL time 会报错。   |
| partial/v1 nested/v2 非 canonical identity | 沿旧 ValueError 或 ordering_required；列出缺失三项、预期 prefix keys/event_time/完整 sequence；不自动补元数据。                                            |
| SQL 方言形状/config/alias 越界                  | invalid_literal/capability_mismatch/unsupported；定位 sql.inner_join/config 子项或 alias，给出窄语法；普通更宽 SQL 不被全局宣告非法。                        |
| 缺 binding/schema/端口错误                     | 原 InvalidArgument/Port 校验/Compile 层错误；明确 required left/right、预期与实际 schema。                                                         |
| limit、time conversion、计数溢出、取消             | 原 typed native failure/cause；指出 operator/side/limit；失败无本次部分结果，下次 fresh。                                                            |
| snapshot/update/opaque/history/order 证明不足 | 对应 update_*、history_scope_mismatch、query_semantics_unproven/finality_unproven、row_order_unproven，unsupported/unknown 不当 supported。 |

分析/编译保留 Rust InvalidArgument/Compile/QueryPortability/QueryUpdateSemantics 和 Python CompileError.query_analysis 分层，诊断按 path/code/status 稳定排序去重。不批量重命名旧错误或为统一文字改变 structured code。

兼容要求：旧 Python @1/@2 bytes/digest、原 stream node IDs/topology/config hash、native row-id 和 emission order、state version/layout、checkpoint bytes/manifest/source cut 均不因新增 finite 路径改变。若内部抽取内核必须用受影响 stream fixture 验证，不能靠 schema/version 数字相同推定兼容。旧低层 stream 项目无需迁移；新 Rust 关系/SQL 方言各有新声明身份。为新 SQL 方言的 prepared-cache 增加私有策略标识（例如 finite_inner_join_v1），保留 mode/schema/update_semantics/registrations 等已有 cache key；不改所有普通 SQL 的策略身份。

## Typical declarations

以下种子在本片实现后可用；本轮仅语法/签名静态核对。Python 保留旧入口，完整 metadata 使这一共同声明可独立 batch/stream 编译，SQL 不手写 ports：

```python
from datetime import timedelta
import json
import calc_flow as cf

def declarations():
    schema = [cf.Field("k", "int64", nullable=False),
              cf.Field("ts", "timestamp[us, UTC]", nullable=False),
              cf.Field("seq", "uint64", nullable=False)]
    a = cf.table_input("a", schema=schema, entity_by=["k"],
                       event_time="ts", sequence_by=["seq"])
    b = cf.table_input("b", schema=schema, entity_by=["k"],
                       event_time="ts", sequence_by=["seq"])
    df = cf.table.stream_join(
        a, b, left_keys=["k"], right_keys=["k"],
        left_event_time="ts", right_event_time="ts",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta(microseconds=3)),
        limits=cf.JoinStateLimits(100, 1048576, 1000),
    )
    config = {"spec": {"join_type": "inner", "left_keys": ["k"],
              "right_keys": ["k"], "left_event_time": "ts", "right_event_time": "ts",
              "bounds": {"before_micros": 2, "after_micros": 3},
              "limits": {"max_state_rows_per_side": 100,
                         "max_state_bytes_per_side": 1048576,
                         "max_matches_per_input_batch": 1000}}}
    literal = json.dumps(config).replace("'", "''")
    sq = cf.sql(f"SELECT * FROM CF_INNER_JOIN('a', 'b', '{literal}')", a=a, b=b)
    post = df.sql("SELECT * FROM input WHERE left__k = 1")
    batch = cf.Program("join", mode="batch", outputs={"df": df, "sql": sq, "post": post})
    return batch, batch.for_mode("stream")
```

Rust 方法接在已声明 a/b 上；外部用户只依赖公开 crate，不需要仓内 helper：

```rust
use calc_flow::{JoinOutputOrdering, JoinStateLimits, JoinTimeBounds,
                Result, StreamJoinSpec, TableExpr};
use std::time::Duration;

fn join(a: &TableExpr, b: &TableExpr) -> Result<TableExpr> {
    let spec = StreamJoinSpec::inner(
        ["k"], ["k"], "ts", "ts",
        JoinTimeBounds::new(Duration::from_micros(2), Duration::from_micros(3))?,
        JoinStateLimits::new(100, 1_048_576, 1000)?,
    )?;
    a.inner_join(b, spec, Some(JoinOutputOrdering {
        entity_by: vec!["left__k".into()],
        event_time: "left__ts".into(),
        sequence_by: vec!["left__seq".into(), "right__seq".into()],
    }))
}
```

Rust SQL 同样调用现有 `calc_flow::sql::sql(query, BTreeMap::from([("a".into(), a), ("b".into(), b)]))`，query 为上面的方言；Program::new/for_mode 与 compile 的真实接口沿现有关系前端。最终 shipped 例子应包含实际输入和 execute/stream，单个程序控制在 20–50 行；这里不把只返回关系当完整运行例。

## Independent acceptance and implementation handoff

O02 独立 oracle（时间均为整数微秒，键 A）：两左行 `(id=L1,t=10),(id=L2,t=10)`；右行 `(R7,7),(R8a,8),(R8b,8),(R13,13),(R14,14)`；before=2、after=3。预期恰六个 pair：每个 Li 分别匹配 R8a、R8b、R13；R7/R14 均无匹配。去掉 id 看 payload 时 `(10,8)` 重数4、`(10,13)` 重数2。添加 NULL key、不同 key 不改变结果；原生另加 NULL time 行也不改变结果。0/0 的左10与右9/10/11恰一匹配；左-10、before2/after3时右-12/-7匹配、-13/-6不匹配。i64 两端加 bounds 用数学整数检查，不能因边界算术溢出丢掉仍可表示的合法候选。

输出 schema 保持左后右及原 nullable；上述公共 fixture 用 non-null UTC us 时间，NULL key 用 nullable key、独立合法的 source entity identity，终端无 v2。不从新 lowerer 或另一个 frontend 生成 oracle；手接原生图/DataFusion 只能是附加对照。

1. **四单元/C05–C06/O02/A05/A09：** 公共 Rust/Python DF/SQL 各自构造、schema/analyze、双端 compile/execute、export 显示实际 stream_join；六 pair、含等号/边外、重复、NULL key、typed empty/无匹配与多 RecordBatch。NULL time 原生 drop 与公共拒绝分别验。普通宽 SQL、非法 JSON/bounds/limit/keys/time/prefix 明确限制，不能只测试包装函数。
2. **C07/A04/A08：** 复用同一 first join，两个派生输出、另一个独立 join/无关输出；三源第二层使用完整 v2 identity。在 first join 前后放合法 DF/SQL passthrough；时间改写、v1 nested、partial/noncanonical ordering 准确拒绝。实际导出节点计数、source 读取计数或现有 owner 观察证明每唯一 join 一个 owner，重载保留所有输出，不 collect 中间表。
3. **L02/C13：** 直接复用同一 finite plan、同 Runtime、暖缓存、无 reset：六 pair→左空/右原表→左原表/右空→全空→新 singleton，预期6→0→0→0→仅新 pair。夹入 schema、side-state-limit、完整 Batch match-limit 和取消失败后再用空侧，不能残留 rows/IDs/ended/结果。convenience collect 每次新 plan 不能替代此证据。
4. **L01/L03/C06–C07：** 同 stream job 左右多批/空批/反向 ingress 交错、某侧先 EOF，正常 EOF 拼接全部输出多重集；定向保留 native 单次 emission order。新增 finite 不借 reset 改 stream；普通新 job fresh。保留 v1 identity fixtures、v2 ordering 拒绝、嵌套和 fan-out；checkpoint→真实 managed restart→下一输入与不中断结果一致，含 fingerprint/layout/config 负例及 terminal recovery 原有断言，恢复只指 stream。
5. **更新/诊断/A06/A09：** append latest 可用；直达和 row-local 中转 snapshot→join latest 逐 ingress 预拒绝（含空快照、缓存命中、source/job/state root 计数0）；legacy 不改，严格 final relation 不虚假放行。r6 parser/gate/cache、条件性 EOF 支持、未知 history/order 与旧7/latest6 回归。超限对照明确每 Data envelope 与 finite 全 Batch 的不同预算域。
6. **必要交付面/A08：** Rust exports/rustdoc、Python docstring/注解、必要私有 native/stub、capabilities/mode gate、`docs/relational-frontends.md` 支持矩阵、两语言独立 finite_inner_join 例子同步。保留 example 12/13 的 stream/嵌套主题及 core 既有位置，不搬动、不改作有限例子。新程序分别放 `examples/finite_inner_join.py`、`crates/calc-flow/examples/finite_inner_join.rs`。

实施缺口明确：BatchOperator 与 compiled/project 两层准入；Rust typed join/ordering；native SQL AST/config/schema 描述；Python batch/mixed SQL lowerer 与其 memo/alias/identity 兼容；join 局部 passthrough；r6 分析及逐边 gate。只改 capability modes 或删 analyzer guard 不算交付。按触发路线由 implementer 做代码/定向验证/小文档，再 tester 独立核对边界/嵌套/隔离，reviewer 主审，有证据 blocker 修复后由协调器推进。本轮不扩派、不增加 spec/critic/性能阶段。

## Rationale, compatibility limits and open questions

一个 Rust 方法加可选输出配置能直接复用现有 spec，避免新增长参数族；Python 既有命名虽带 stream，保留它可避免两个等价构造入口和旧 digest 改动。SQL 选三参数显式方言加严格纯数据配置，让必填 state/match limits 和排序声明有确定位置；拒绝猜测普通 ON/BETWEEN 的无限 bounds、默认预算或把 output ordering 当 ORDER BY。JSON 较长是配置显式性的代价，prefix 默认保留；不为缩短例子隐藏必填 limits。

拒绝 SQL 文本重写成独立 batch 算法、Python 执行 join、MAX watermark、先结束一侧再喂另一侧、把 stream owner 缓存在 batch plan，分别避免匹配/NULL/计费漂移、EOF 丢候选和跨次状态泄漏。v2 元数据和原生 emission order 保持原义，不把有限模式的确定 ingress 次序提升为跨模式总序。

**没有需要用户回答的 API blocker。** 上述签名、SQL 支持域、nullable 时间的分层限制、嵌套/共享、finite 完成/预算/隔离与身份边界已选定，可直接实施。公共 nullable time、更宽 SQL ON/CTE/subquery、join 路径中的 union/window/aggregate 新组合、replacement 与显式中途 cut 均为清楚的未支持范围，不冒称四单元已覆盖。若实现无法保留旧 @1/@2 stream 拓扑/identity，或出现原生 finite 内核无法共用的实证，应报告具体兼容 blocker，不能静默换算法/迁移 checkpoint。

本轮仅写本文；自检覆盖本地源码符号/路径、相对 note 链接、签名参数、Python 种子语法、SQL configuration JSON、O02 整数多重集、Markdown 围栏/表格/空白与写入范围。未编译 Rust、导入 calc_flow、运行产品测试/性能/CI，也未提交、推送、建 PR 或更改产品。checkout 保留供后续角色；DAL-517 整体仍待实现及 PR 实际合并，保持 in_progress。
