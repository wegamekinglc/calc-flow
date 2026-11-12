# Finite ASOF Join — API Note

状态：DAL-518 / S2d 局部 API 合同冻结，2026-11-12，可交 implementer；无待用户决策项。**新增公开表达、finite 生命周期和诊断尚待实现。本文只有静态源码核对与文档自检，不是产品验收通过记录。**

精确基线：`909463f0298084fc7629596215aa7d7c6b78ed07`，分支 `agent/dal-orchestrator/937ec660b6b0`，开始时工作树干净，HEAD 与 origin/main 一致。前序 PR #601 合并事实由 DAL-498 / DAL-518 交接确认，本轮未重复检查其 CI。依据为本子任务及触发线程 `01a1cf4c-30a7-70bc-ae1d-7de1dc8d4567`、父任务四单元/生命周期讨论和经 Multica CLI 下载核对的计划 v2 附件 `01a1bc0b-3060-7c9d-ad57-b23a24daca58`（R01–R08、S2d、C08/O02、A04–A06/A08–A09、C13、L01–L03）。

## Audiences and decision boundary

Rust 用户需要 typed DataFrame、SQL、Program、BatchExecutionPlan / StreamingRunner；Python 用户需要根包 DataFrame、SQL、Program 和既有 collect / owned stream 结果。DataFrame/SQL 是公开表达类别，batch/stream 是独立执行维度。声明不持有输入 rows、运行 plan、job 或可变 owner。

承接已合并的 [finite inner join](finite-inner-join.md) 和 [原 ASOF 合同](bounded-backward-asof-join.md)，仅决定 backward、bounded、left-preserving ASOF 的同节点有限执行、四单元入口和必要组合/身份兼容。原 note 的历史版本数字、stream-only 结论以本 HEAD 源码为准；不恢复其早期 layout reader 或 Studio 开发范围。forward/nearest、无界 tolerance、一般 outer/semi/anti、replacement ASOF、snapshot_sql finite 和 batch 查询中途恢复不在本片。

遵循 `AGENTS.md`、`.codex/agents/cf-api-designer.toml`、`docs/introduction.md` 的 Batch、Port、Operator、Pipeline、Source、Sink、Runner、Checkpoint 词汇。实际组件为 Rust core → PyO3 → Python adapters；本仓库没有 `dal-public/src/`、Excel/C++ binding 或 `.codex/references/component-boundaries.md`，沿现有 owning component 的例子设计，不补造其他组件。

## Surface today and evidence

Rust 路径相对 `crates/calc-flow/src/`；Python 路径相对 `python/calc_flow/`。行号定位本基线。

| Source                                                                                                         | Verified behavior / gap                                                                                                       |
|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `lib.rs:94,104–121`、`operator/asof/spec.rs:34,82,154`                                                          | 已公开 TableExpr、AsofJoinSide、AsofStateLimits、StreamAsofJoinSpec / Operator；spec 为两侧纯数据、tolerance、limits，late policy 默认 error。   |
| `dataframe.rs:75,915–968`、`program/lower.rs:50,329`                                                            | 已有 inner_join / JoinOutputOrdering / 关系 lowering，没有 ASOF 关系 variant 或方法。                                                      |
| `sql.rs:28–62`、`sql/inner_join.rs:82–260`                                                                      | 已有 CF_INNER_JOIN 的 Rust AST、严格配置和 alias 检查；没有 ASOF SQL 方言。                                                                    |
| `config/asof.rs:13–72`、`pipeline/batch.rs:24–61`                                                               | project 构造/验证及 compiled batch 分支拒绝 ASOF；ASOF 尚未实现 BatchOperator。                                                              |
| `operator/asof/mod.rs:104,281,472–588`                                                                         | 已有原生 admission、schema/spec 编译、stream process_data / progress / on_end；Data 入库，progress/EOF 才最终化。                              |
| `operator/asof/schema.rs:6–145`、`operator/asof/state/right.rs:716,761,1197`、`operator/asof/finalize.rs:59–137` | 严格 non-null identity schema；原生 typed candidate 与 i128 tolerance 边界；按左侧 canonical 顺序排空，equal frontier 不闭合。                     |
| `symbolic/expr.py:679`、`symbolic/ops.py:1525`、`symbolic/asof.py:101–216`                                       | Python 已有 namespace / fluent 入口；从 operand metadata 解析 spec，输出身份自动从左侧派生。                                                       |
| `symbolic/asof_analysis.py:58–104,215–263`                                                                     | 仍只接 stream；要求当前 metadata 与 spec 一致；允许 row-local 和已声明 join 边界，拒绝其他 stateful 路径。                                                |
| `symbolic/lower/asof.py:16–52`、`lower/strategies.py:532–543`                                                   | 已有唯一 ASOF 节点收集及原生 stream_asof_join lowering；旧 stream 拓扑/identity 是兼容基线。                                                       |
| `symbolic/lower/sql.py:160–247`、`symbolic/inner_join_sql.py:60–169`                                            | 通用 SQL 图 memo 已接 inner；局部 SQL temporal passthrough 和 ASOF 原时间识别可复用，但尚未接 ASOF SQL/mixed 图。                                     |
| `query_analysis.rs:486,1075–1120`、`query_analysis/updates.rs:121–136`                                          | schema_version=2 / analysis_revision=6；ASOF stream-only、accepted_history / append / unknown finality，已是 append-only consumer。 |
| `capabilities.py:1105–1128`、`symbolic/asof_analysis.py:42–99`                                                  | stream_asof_join@1：stream only、group_final_append_only、microbatch_invariant=True、state_version=3、state_layouts=(10,)。         |
| `crates/calc-flow-python/src/config.rs:908,984`、`_native.pyi:14`                                               | 已有 inner SQL 私有解析 seam；ASOF 需要同类原生描述及 binding/stub，不能只加 Python 字符串包装。                                                         |
| `docs/asof-join-guide.md:10,80,355–405`、`examples/35_stream_asof_join.py`                                      | 已文档化 stream 的闭合、组合和恢复；明确不支持 batch，需随实现更正；当前 native 仅 layout/accounting 10，semantic state 3。                                   |

静态参考已有 `crates/calc-flow/tests/stream_asof_join_{properties,boundaries,state,validation}.rs`、`python/tests/test_symbolic_stream_asof_join.py`、`test_asof_stream_results.py` 及 finite inner 的两语言例子/验收。阅读断言不等于运行通过。

## Frozen public signatures: four first-class cells

Rust **新增**一个 immutable TableExpr 方法；复用现有小型配置，不复制十余个位置参数，不增加输出排序配置：

```rust
impl TableExpr {
    pub fn asof_join(&self, right: &TableExpr, spec: StreamAsofJoinSpec) -> Result<Self>;
}
```

这是签名清单，省略实现体。既有构造器保持：

```rust
AsofJoinSide::new(keys, event_time, sequence_by, prefix) -> Result<AsofJoinSide>
AsofStateLimits::new(max_state_rows, max_state_bytes) -> Result<AsofStateLimits>
StreamAsofJoinSpec::new(left, right, tolerance, limits) -> Result<StreamAsofJoinSpec>
StreamAsofJoinSpec::with_late_policy(self, policy) -> StreamAsofJoinSpec
```

Rust/SQL 的显式 spec 两侧 event_time 和 sequence_by 必须与当前 operand temporal metadata 完全一致；keys 可以显式覆盖 entity_by，与 Python keys= 相同。不能凭 spec 给无证明 SQL 输出补 metadata。输出 entity_by 为 prefixed left effective keys、event_time 为 prefixed left time、sequence_by 为 prefixed left sequence。无 JoinOutputOrdering 参数，右侧 nullable sequence 不能参与输出身份。

Python **保持现有 public signature、默认、primitive stream_asof_join@1 和 canonical spec attrs**，仅扩大 finite 准入/编译和必要组合：

```python
class _AsofJoinOptions(TypedDict, total=False):
    keys: tuple[Sequence[str], Sequence[str]] | None
    late_policy: Literal["error", "drop"]
    prefixes: tuple[str, str]

# cf.table namespace
def stream_asof_join(
    left: TableExpr, right: TableExpr, /, *,
    tolerance: timedelta, limits: AsofStateLimits,
    **options: Unpack[_AsofJoinOptions],
) -> TableExpr: ...

# TableExpr method
def stream_asof_join(
    self, right: TableExpr, /, *,
    tolerance: timedelta, limits: AsofStateLimits,
    **options: Unpack[_AsofJoinOptions],
) -> TableExpr: ...
```

keys=None 取两侧 entity_by；optional defaults 仍是 keys=None、late_policy="error"、prefixes=("left","right")。未知 option 继续 TypeError，caller sequences 防御性复制。namespace 和 fluent 是同一声明路径，不增加 batch_asof_join 或 Python asof_join 别名。高级 `PipelineBuilder.stream_asof_join(name, *, left_schema, right_schema, spec)` 保持签名并获 batch 准入。

四单元分别为 Rust DataFrame 的 asof_join、Rust `calc_flow::sql::sql(query, BTreeMap<String, TableExpr>)`、Python 上述 DataFrame 入口、Python `cf.sql(query, /, **tables)`。两语言 SQL 均返回可继续组合的 TableExpr，真正 lower 到同一个 `kind=stream_asof_join`。TableExpr.sql、Program、mode/engine、for_mode、schema/analyze、compile/execute、collect/collect_async、stream、to_project 签名不变。增加 `BatchOperator for StreamAsofJoinOperator`，使用既有 required exact-schema left/right 和 output ports；没有新 runner、公开 flush/finalize/reset 义务或控制注入 API。

## SQL supported spelling and validation

新增 calc-flow 窄方言，遵循 finite inner 的三个必填参数及 AST 验证方式：

```sql
SELECT * FROM CF_ASOF_JOIN('a', 'b', '{"spec":{"left":{"keys":["k"],"event_time":"ts","sequence_by":["seq"],"prefix":"left"},"right":{"keys":["k"],"event_time":"ts","sequence_by":["seq"],"prefix":"right"},"tolerance_micros":2,"limits":{"max_state_rows":100,"max_state_bytes":1048576},"late_policy":"error"}}')
```

依次为 left alias string、right alias string、configuration JSON string。配置对象**只允许必填 spec**；其内容恰为原 project-v3 StreamAsofJoinSpec，复用 raw 验证和 nested field paths：left/right 的 keys、event_time、sequence_by、prefix，tolerance_micros、limits 两项、late_policy 均必填，所有层拒绝未知字段。raw 没有构造器默认；不接受 output_ordering、direction 或独立 output identity。对象 key 顺序无语义，数组顺序有语义。

tolerance_micros 是 [0,9007199254740991] 的整数；两个 limits 各为 [1,9007199254740991]。不得把 bool、浮点 token、负数或溢出值强转整数后再验证。Rust Duration 要精确整数微秒，Python timedelta 使用既有整数转换。前缀沿既有 distinct portable identifier 与 collision 检查。

支持域为一个顶层 SELECT *、一个无附加 alias 的 FROM 调用、恰好两个显式绑定且名称不同的 tables aliases，分别引用一次，无额外未用绑定；函数名不带 schema、大小写不敏感，三个参数只接普通 SQL string literal。aliases 沿已有唯一/归一化规则，解码后精确查找；两个不同 aliases 可以绑定同一 TableExpr，保留左右角色重数。

不接 CTE、子查询、JOIN ON/USING、WHERE、投影改名、aggregate、DISTINCT、OVER、set operation、ORDER BY、LIMIT/OFFSET/FETCH 或 FROM modifiers。前后过滤/投影另建 row-local relation，三源嵌套用两次 sql 调用组合。不宣称 DataFusion 内建 CF_ASOF_JOIN 或普通 ASOF JOIN 自动成为受证明的原节点；旧普通 SQL 的 batch 能力和 stream gate 保留。识别到该函数却形状/config 越界时必须拒绝，不 fallback 到每批 SqlOperator。

Rust AST/schema 规划是权威解析/类型来源；以现有 inner seam 为例提供 data-only 描述供 PyO3、Python Runtime adapter 与 _native.pyi 使用。Python 不写第二套 SQL parser、matching、typed tie 或 dtype 推断。私有 helper 不是 SQL 公共入口。原生低层 SqlOperator / 已保存 kind=sql 项目不自动重解释新方言；高层导出真实 ASOF 节点，不保留 SQL 文本的运行依赖。

## Matching, NULL and output identity

对每个合法左行 l，在同 effective key 的右行中，选 inclusive 区间 [l.t−tolerance,l.t] 内最大的 **(right_time, typed right_sequence_tuple)**。相同时间按 typed lexicographic sequence 比较；整数按数值，不按十进制字符串，UTF-8 按原生字节序，组合按字段声明顺序。sequence 类型不要求左右相同。tolerance=0 只接相同时间，距离恰好 tolerance 有效，超出一微秒无效，future right 不候选；时间差用原生 i128，不 wrap/饱和/浮点。

每个合法左身份恰好一条输出；无匹配仍保留左行并将所有右字段置 NULL。相同 key/time、不同 sequence 的左行是独立事件，不按 payload 去重。身份为 (side,effective keys,event_time,sequence_tuple)，同侧重复身份即使 payload 不同也按原 native duplicate 错误拒绝；不同 ingress 的相同值不互相判重。

| Field/layer          | Frozen constraint                                                                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 四单元 event_time       | non-null timestamp[us, UTC]，必须是当前有证明的原时间列或纯 rename；不扩大 nullable、ms/ns 或 naive 时间。                                                                                                   |
| 四单元 keys             | 非空、列存在、列表内唯一、成对 dtype 完全相同、schema non-null；沿 portable 字段域与 native key allowlist 的交集（bool、各宽度整数、string/large_string、date32/64、portable microsecond timestamp）。无隐式 cast 或 null-equal。 |
| sequence             | 每侧非空、列存在、唯一、non-null；int/uint 8/16/32/64、string/large_string。拒绝 float、bool、timestamp 等非原 sequence 域。                                                                                |
| payload/output       | 原 native flat Arrow allowlist 保持，前端仍用既有较窄 portable vocabulary；nullable payload 有效，不扩大 nested/view/dictionary 支持。                                                                    |
| actual NULL identity | schema nullable 先在分析/编译拒绝；坏数据中的 NULL key/time/sequence 走 asof_invalid_input / 更早 envelope schema 校验，不套用 inner 的 NULL-drop。                                                          |

输出仍为全部左字段后全部右字段，prefix__name，保留 dtype/field metadata 和左 nullable，**全部右字段 nullable**，不合并 keys、不加 match flag。right event_time 原本 non-null，其输出 NULL 可表示无匹配，不能拿任意 nullable payload 判定是否匹配。schema 取 native ports，不看本次行值推断；声明/Arrow buffers 不变，joined relation 有自己的 row lineage。空左为精确 typed empty，空右为每左行一条 unmatched，双方空也必须返回完整 schema。

canonical ASOF 排序为 (left_time,left_key_tuple,left_sequence_tuple)，沿原 typed comparator；不承诺 physical RecordBatch/chunk 边界、输入 metadata 或跨输出调度次序。下游 metadata 只是 identity/时间证明，不表示 select 排序操作。

## Finite lifecycle and terminal comparison

一次 BatchOperator::process 接收本次完整 H_left/H_right，每个 required ingress 一个 Batch，含该侧所有 RecordBatches；缺 binding 是错误，typed empty 是有效关系。先校验两侧 envelope/schema，再按**完整 left → 完整 right** 准入；在两侧数据都完成准入前不设置任何 Ended/EOF、不做 progress finalization 或 eviction。所有 RecordBatches 共享本次侧身份集合，跨 RecordBatch 的重复同样拒绝，不能按 chunk 重置。

两侧准入成功后以明确 finite completion 最终化全部 pending left，复用原 native admission、typed identity/candidate、gather、排序与计费内核。**ASOF 与 inner 不同：Data 本身不能代表答案已最终化；finite 结束必须执行每左行一次的排空。** EOF 独立于最大 EventTime，必须覆盖 i64::MAX 左时间，不伪造 MAX watermark；空 Data 不等于 EOF。可抽取无 stream lifecycle 依赖的内部操作/collector，不启动 StreamingRunner、临时 managed job / StreamJobContext、checkpoint/source/sink 或控制流协议。

候选、identity-only entries、typed encodings、pending/output cursor、progress/terminal、counter/output sequence 和 retirement 所有权均属于这一次调用；成功、失败或取消都释放。可缓存的 compiled operator/Runtime/plan 仅持只读 spec/schema/模板，不携上述状态。有限 compiled 分支 snapshot=NULL、非 NULL restore 拒绝、reset 无跨次状态可清，沿既有内置 finite 约定；不要求用户调用 reset。

保留必填 max_state_rows/max_state_bytes 的原**每算子总计费**，不是每侧各一份；有限两侧全部已准入身份/候选及原索引/持有内存按当前 native 规则计费，不因无匹配、空对侧或 dominated payload 提前跳过输入验证/重复检查/预算。没有中途 progress 驱逐，故本次完整输入要在既有总预算内。transient workspace ceiling 沿 max_state_bytes；内部 bounded 输出 chunk 不得重置身份/状态预算或规避 allocation 前预检，不引入无限预算或 match-count 新参数。现有 output 收集/资源上限继续生效；流式 edge 行/字节预算仍由 stream context 管理，batch 不要求水位源或 stream edge 配置。

成功返回单个 output Batch（可含多个 RecordBatches）；0 行也含 typed empty RecordBatch，不能用无 schema 的 Batch::table(vec![])。finite envelope 用 node name、sequence=0、空 attributes，不冒充 source cursor / stream emission sequence。错误/取消不返回本次部分结果，不把候选留入 plan；不扩展为全图外部副作用事务承诺。下次重试 fresh。

stream 保留同 job 跨批状态与原 schema → late policy → on-time duplicate → budget → atomic admission 顺序；late_policy=error/drop 与既有失败计数边界不变。左 t 只有在**两侧各自 watermark>t 或该侧正常 EOF**时可最终化；equal watermark 和 idle 不闭合，单侧 EOF 不补全另一侧。先排空安全左前缀再转发最多 C−1 微秒的 output frontier，i64::MIN 不下溢；双 EOF 排空含 i64::MAX 后，终态恢复不再重复输出。保持当前 source/runtime-first 错误分层和 managed delivery/source-cut 合同。

比较前提：同 spec、同合法完整 append 历史 H、stream 无 late 丢弃/拒绝导致 H 改变、两端均满足自身预算且成功、所有相关 source 正常 EOF。把 stream 所有 append 输出拼接成完整终态，与 finite 的精确 schema/NULL/每左身份唯一值比较；独立 oracle 手工给出匹配，不能只比较两条共用内核路径。验 canonical ASOF 顺序，但不比较 chunk/envelope/中间事件；严格 portability 请求可用 existing final_relation / accepted_history / all_sources_eof / unordered。不同水位/切批可改变峰值预算或 late 结果，不承诺无条件轨迹或预算错误等价；更小 stream 切批成功不能证明完整 finite 输入也应成功。

## Nested composition, shared owners and export

支持四单元的 source → row-local DF/SQL → ASOF → row-local DF/SQL，以及 ASOF→ASOF、inner→ASOF、ASOF→inner、三源嵌套、独立 joins、无关合法输出及 fan-out。保留已合法的 post-ASOF rolling/cross-section stream 路径原准入；union/window/aggregate 可作独立输出分支，不因 ASOF lowerer 被丢失，新 join 祖先路径不加入这些类别或 replacement/snapshot。既有合法低层 stream 图不被新 frontend 白名单额外收紧。

嵌套必须逐列保留真实 proof：ASOF 的左派生 event/entity/sequence 三项经 source/row-local 传递到下一层。不能拿 nullable 的 right__ts/right__seq 当输出 temporal identity，也不能把缺失列补回来。inner 作上游时仍必须完整 v2 / JoinOutputOrdering，ASOF 左侧由 prefixed effective left keys/time/sequence 自动提供合法 proof，下一层指定的时间须匹配此事实。把内层 nullable 右字段选作下一层 key/sequence 按原 non-null 约束拒绝；不能 blanket 拒绝所有 ASOF nested，或 blanket 放开任意字段。

为 A04 扩用现有 **join 局部** `sql_passthrough_columns` 证明：确定性单直接 alias 的 row-local SELECT/WHERE 可原样传递/纯 rename 全部必需 entity/event_time/sequence；native 证明后映射 metadata 并保留 join ancestry。丢列、cast、时间算术、sequence 换值、aggregate/sort/distinct/limit/子查询/volatile/UDF/provider 等无证明形状不获 temporal identity。仅获准 join 的祖先路径可用；普通 SQL 仍新 lineage/无 ordering。CF_ASOF_JOIN 关系本身是 ASOF 边界，可以作为下一次 sql 的 alias，无需嵌套 SQL 文本或 collect 中间表。

每个 canonical ASOF 声明在一次 stream job 一个 mutable owner，同次 finite execute 计算一次，共享不可变 output 给所有消费边；不同 spec/左右顺序不合并，两个 aliases 绑定同一源仍保留 left/right 两条 ingress。不能按输出名字重复创建 owner，或按 source digest 合并 ingress。相同声明复用身份，DF 与 SQL 不同语法/语言不要求相同 hash 或跨语法 CSE。SQL/ASOF 新图接通 memo 时必须保留无关分支。

导出仍为 project-v3 `OperatorSpec::StreamAsofJoin { spec }`，required exact-schema left/right（此序）及 output。自动 output identity 是 frontend fact，不加 native spec 字段；project 无 rows/callback/globals/运行 state，重载按物理 binding 与原 schema/config 校验，owner/边重数往返保持，不承诺重建 Python logical aliases 或 declaration objects。不因新增 finite 更改 project JSON Schema 格式。

## Updates, diagnostics and compatibility

ASOF 仍是 append-only consumer。已证明 append 在 legacy_append/latest_relation 下均可进入支持域；complete_snapshot→ASOF 的旧 legacy 解释仍追加快照事件，显式 latest 在任一 ingress 按原 update_interpretation_mismatch 预拒绝，经过 row-local/SQL 中转、空快照或 shared edge 也不能丢失事实。未知更新/事件边界继续 update_semantics_unproved / update_event_boundary_unproved。选定生命周期判事实，不能把 stream replacement 规则机械套在合法的单次 finite 结果上，也不扩大本片 frontend 祖先路径。

严格 final_relation 对 legacy snapshot 追加链不能证明“原 H 只计算一次”的关系等价，应定位边；旧非严格执行不改。旧 SQL 默认链 7、支持域显式 latest 6/其余拒绝、既有预算和 checkpoint 语义不变。预检先于 source.open/迭代输入/job 或 state root 获取，暖 cache 不能绕过。

新增 ASOF finite/SQL/最终化证明，analysis **schema_version=2、analysis_revision=7**，不新增 wire 字段。Rust producer、Python parser、mandatory latest/strict gate、cache gate 和 binding/stub 同步；旧 (1,1)/(2,2)…(2,6) 仍可展示，不作为 r7 新能力准入证据。ASOF stream local facts 保持 accepted_history / append，可将已证明 ASOF 产出标 group_final，并以 all_sources_eof 证明共同终态；未知上游/opaque SQL、非 EOF cut、history_scope/order 不足仍准确 unsupported/unknown，不能仅看 capability 名字推断可移植。

stream_asof_join@1 modes 增为 batch/stream，**保留** finality=group_final_append_only、microbatch_invariant=True、semantic state_version=3、native state_layouts=(10,)、deterministic/replay_safe 及原 ports。stateful/requires_watermark/checkpoint 等按 stream 生命周期解释；batch 使用本次临时状态，不要求用户提供 watermark/replay。原 native layout/accounting 10、source_replay version 3、manifest/source-cut/engine 校验和旧 reader 拒绝域不变；不按早期 note 的 1/2/3–9 扩 reader。

保留已有 Python primitive @1 canonical bytes/digest、原 ASOF/inner node IDs/topology、source/output bindings、配置 hash/fingerprint、typed comparator、输出顺序、stream checkpoint bytes/terminal/恢复身份；切换 lowerer 不能令原合法 stream checkpoint 失效。新 Rust/SQL 声明可有各自新身份。新增 SQL prepared-cache 私有策略标识（如 finite_asof_join_v1），保留既有 mode/schema/update_semantics/registration 等 key，不改变普通 SQL 策略；for_mode 共享声明、不共享异 mode plan/state。若抽取共享内核，使用最小受影响 stream capture→restore→下一更新及 terminal 不重发 fixture 验证，版本号相同不能替代行为/字节证据。

令 P 为 canonical relation path：Python 既有 P.stream_asof_join；Rust 新方法 P.asof_join；SQL P.sql.asof_join。配置路径后接 spec.left/right.keys[i]、event_time、sequence_by[i]、prefix、tolerance_micros、limits.<field>、late_policy；既有 Python 已有字段路径不批量改名。SQL alias 错误定位 tables.<alias>，raw project 仍 graph.nodes[i].operator.spec。配置解析不可只报 invalid JSON 而吞掉具体字段。

| Violation                                          | Error boundary and useful constraint                                                                                                                  |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python 非 TableExpr、非法 options/sequence、bool limits | 沿旧 TypeError/ValueError；指出 side/keys/limits、实际类型和必需类型；不用 string 冒充列序列。                                                                                |
| tolerance/limit 类型、范围或精度错误                         | 沿原 raw invalid_type/out_of_range 等分层；具体 spec 字段、整数范围或精确微秒；SQL 保留 nested path。                                                                         |
| 缺列、nullable identity、dtype/metadata 不一致            | 原 symbolic ordering_required/type_mismatch 或 native missing_field/nullable_identity_field/incompatible_key_type 等；指出 side/field 与要求，不自动 cast/补 proof。 |
| 实际 NULL / 同侧 duplicate / late                      | 原 asof_invalid_input / asof_duplicate_identity / asof_late_row；不暴露 identity 值/payload，不把 source/runtime-first 错误转换成 ASOF reason。                      |
| state/workspace/output/checked arithmetic 超限       | 原 asof_state_limit_exceeded / asof_workspace_limit_exceeded / asof_output_limit_exceeded / asof_counter_overflow；指出 operator/limit，失败本次无部分返回。         |
| SQL shape/config/alias 越界                          | 原 invalid_literal/capability_mismatch 等框架，定位 sql.asof_join 子字段并给窄语法；不全面拒绝更宽普通 batch SQL。                                                              |
| 缺 binding、错误端口/envelope/schema                     | 原 InvalidArgument/Compile/typed validation；明确 required left/right 和预期/实际 schema。                                                                      |
| replacement、未知 history/finality/ordering           | 原 update_*、history_scope_mismatch、finality_unproven、row_order_unproven 等对应 obligation；无法证明不标 supported。                                               |
| 取消 / incompatible checkpoint                       | 沿原取消及 checkpoint mismatch 分层，finite 下次 fresh；stream 未提交 chunk 不冒充已输出。                                                                                 |

Rust 保留 InvalidArgument/Compile/QueryPortability/QueryUpdateSemantics/OperatorReason 等分层；Python 保留 CompileError.query_analysis、既有 stream failure reason 投影。不要为统一文字批量重命名 structured code，诊断按既有 path/code/status 稳定规则去重。

## Typical declarations

以下是实现后的种子，依赖项名/调用已静态核对，未执行。Python DataFrame 常见路径保持简短；SQL 配置对应上节显式 full spec。

```python
from datetime import timedelta
import json
import calc_flow as cf

schema = [cf.Field("k", "int64", nullable=False),
          cf.Field("ts", "timestamp[us, UTC]", nullable=False),
          cf.Field("seq", "uint64", nullable=False)]
def source(name):
    return cf.table_input(name, schema=schema, entity_by=["k"],
                          event_time="ts", sequence_by=["seq"])

a, b = source("a"), source("b")
df = a.stream_asof_join(b, tolerance=timedelta(microseconds=2),
                       limits=cf.AsofStateLimits(100, 1048576))
spec = {"left": {"keys": ["k"], "event_time": "ts",
                 "sequence_by": ["seq"], "prefix": "left"},
        "right": {"keys": ["k"], "event_time": "ts",
                  "sequence_by": ["seq"], "prefix": "right"},
        "tolerance_micros": 2,
        "limits": {"max_state_rows": 100, "max_state_bytes": 1048576},
        "late_policy": "error"}
literal = json.dumps({"spec": spec}).replace("'", "''")
sq = cf.sql(f"SELECT * FROM CF_ASOF_JOIN('a', 'b', '{literal}')", a=a, b=b)
post = df.sql("SELECT * FROM input WHERE left__k = 1")
batch = cf.Program("asof", mode="batch", outputs={"df": df, "sql": sq, "post": post})
stream = batch.for_mode("stream")
```

Rust 在已声明 a/b 上构造同义关系，不依赖仓库私有 helper：

```rust
use calc_flow::{AsofJoinSide, AsofStateLimits, Result, StreamAsofJoinSpec, TableExpr};
use std::time::Duration;

fn matched(a: &TableExpr, b: &TableExpr) -> Result<TableExpr> {
    let left = AsofJoinSide::new(
        vec!["k".into()], "ts".into(), vec!["seq".into()], "left".into())?;
    let right = AsofJoinSide::new(
        vec!["k".into()], "ts".into(), vec!["seq".into()], "right".into())?;
    let spec = StreamAsofJoinSpec::new(
        left, right, Duration::from_micros(2), AsofStateLimits::new(100, 1_048_576)?)?;
    a.asof_join(b, spec)
}
```

## Acceptance mapping and handoff

以下是 implementer / reviewer 的交付标准，本轮未执行这些产品测试；不要求新增固定 tester/critic/spec 阶段。

| Acceptance                     | Concrete evidence required in DAL-518                                                                                                                                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C08 / O02                      | 字面小表：left(k=1,t=10,seq=1)、tolerance=2；right t=8 → 匹配，t=7 或 11 → unmatched，t=8/9 → 9，t=10 → 10；同 right t=9 的 uint seq=2/10 选 10，再验 signed/UTF-8/composite tuple；tolerance=0、负时间、i64 两端。独立写 expected payload/右 nullable，不用另一个调用当 oracle。 |
| C08 / A05 / C13                | NULL payload 保留；nullable key/time/sequence declaration 拒绝、actual NULL 按真实层拒绝；两侧同值不互判重，同侧跨 RecordBatch/Batch duplicate 原子拒绝。左不同 sequence 各输出一次，右空全 unmatched、左空 typed empty。                                                            |
| C08 / L01 / L03                | 真实 stream equal frontier 无输出，单侧超越/idle/单侧 EOF 不早发，dual closure 一次输出；双 EOF 含最大时间每左身份一次，terminal restore 不再 flush，恢复后下一更新延续同一历史。不要用 batch MAX watermark 代替此证据。                                                                           |
| L02 / A06 / C13                | 直接同一 compiled BatchExecutionPlan 和暖 Runtime，无 reset 连续执行完整 fixture→右空→左空→全空→换 keys/右 payload；相同 plan 的 schema/duplicate/budget/取消失败后再执行，结果仅当次输入、无旧右候选/terminal/counter。仅多次 convenience collect 不足。                                     |
| A04 / A06（C07 组合复用）            | 两语言 DF→SQL→ASOF→DF→SQL、SQL-ASOF→DF→SQL；三源 ASOF 嵌套、inner↔ASOF 的获准身份子集、独立 join/无关 union/window/aggregate 输出、共享 ASOF 两输出一个 owner、同源双 ingress 重数。v1 inner/丢列/nullable 右身份准确拒绝。                                                             |
| A04–A06 / A09                  | 四单元 schema/analyze/batch/stream 真正执行同支持域；相同 H 双端成功后全 append 终态对手工 oracle，验证 history/finality/ordering、latest 两侧逐边 gate 与旧 legacy 解释。opaque SQL/更宽方言/unsupported ancestor 不 fallback。                                                   |
| A08                            | Rust public rustdoc / 外部调用编译；Python root exports、PyO3/Runtime adapter、_native.pyi 和 types 同步；四单元 project-v3 导出重载能运行，边重数/owner 保持，logical/physical binding 文档明确。                                                                        |
| A09 / C13 / compatibility      | r7 producer/parser/strict/latest/cache gate 同步，capability 准入按生命周期；raw config nested paths/整数域/未知字段；原 ASOF/inner identity、checkpoint/layout/source-replay 拒绝域和旧 SQL 7/latest 6或拒绝不退化。                                                   |
| L01–L03 / shared-kernel change | 最小相关 stream capture→restore→下一批及 terminal fixture；同 Runtime 普通新 job fresh。复用有效既有证据，改了的路径定向复核，不跑性能或以全矩阵替代证据。                                                                                                                            |

实现连同小文档同步 `docs/asof-join-guide.md`、`docs/relational-frontends.md`、受影响 API/batch/project 文档的 stream-only 断言、capabilities 和例子。保留 `examples/35_stream_asof_join.py` 的真实 watermark/EOF/owned stream 演示目的；新增有限例子可参照 `examples/finite_inner_join.py`、`crates/calc-flow/examples/finite_inner_join.rs`，不把原 stream 例子改成 batch。产品例子 20–50 行覆盖正常 finite ASOF、SQL 和一次明确 expected；复杂矩阵留定向测试。

## Rationale, rejected alternatives and open questions

采用现有 StreamAsofJoinSpec，避免新 BatchAsofSpec 漂移；Rust 方法与 Python 沿用名字不同是兼容选择，不因 Stream 前缀创建平行执行模型。左派生 identity 已足够，拒绝新增 output_ordering 或使用右 sequence，避免 unmatched 破坏 non-null 证明。显式窄 SQL 方言给四单元真正同节点支持，避免猜测普通 JOIN / LATERAL / ORDER BY LIMIT 的 bounds 和 tie 语义。原内核的 run-local finite 适配保留预算与错误，不采用独立 DataFusion all-pairs ASOF、Python per-batch matching、MAX watermark、persistent stream job 或用户 reset 制造一致。

无阻塞开放问题；配置 helper 的私有命名、内部有限 context/collector 抽取方式由 implementer 在上述行为合同内决定。若实现证据表明必须改变旧身份/恢复含义或扩支持域，先带具体冲突回本任务，不静默修改合同。本轮只交付本 note，产品实现、定向验证、小文档和独立主审由既定路线继续。
