# 有界 backward ASOF Join - Specification

## Source

- 用户请求：2026-09-09 的有界 backward ASOF 设计，以及后续直接基于
  `feature/python-expression-api-refactor` 实施至提交 PR 的授权。
- 基线：PR #259，`eda1583751abbd1ca4d246fcb8ee6b70f57d9b09`；不再等待它合并。
- 背景：[原开发计划](../../../docs/plans/2026-09-09-bounded-backward-asof-join.md)、
  [Introduction](../../../docs/introduction.md)、
  [streaming guide](../../../docs/streaming-guide.md)、
  [runtime envelope](../../../docs/runtime-envelope.md)。
- 交接：[API note](../api-notes/bounded-backward-asof-join.md) 冻结签名、wire、
  收费表及错误/status 字段；本规格冻结可观察语义和验收，不指定数据结构或查询算法。

## Problem Statement

交易流需要关联事件发生时刻同 key 的最近历史报价。现有 bounded inner Join
输出区间内所有匹配，没有 pending-left 最终决策；下游即时过滤不能保证选到最终最近值。
本项增加一个原生有状态流算子，经同一 Python 表表达式和 project-v3 图执行，
保留未匹配左行，并在事件时间封闭后追加最终结果。

## Goals

- 在有限 inclusive tolerance 内按确定的右侧时间/sequence 全序选择至多一条右行。
- 每条被接受左行最终输出一条；合法乱序、Batch 切分和恢复不改变逻辑结果。
- 精确校验身份、时间、资源及恢复协议；输出等待、迟到和资源错误可观测。
- 接入 Rust、PyO3、Python 表达式/advanced builder、capability、项目和 Studio 合同。
- 保留旧 inner Join 的配置字节、fingerprint、schema、capability 和 checkpoint 读取行为。

## Non-Goals

- forward/nearest、无界历史、动态数据库 lookup、通用 outer Join、更新/撤回、batch ASOF。
- Python 匹配引擎、新 table backend、新公开控制消息注入或新的外部 exactly-once 保证。
- 改写旧 inner Join 合同、扩展现有 event-window/matrix 的未证明组合、增加 ASOF 编辑器页面。
- 吞吐或延迟承诺；资源场景只验证有界性、释放和失败行为。

## Functional Requirements

### 声明、数据与选择

- **FR1 - 独立身份。** 新原生 kind、capability 和 primitive 为
  `stream_asof_join@1`，具有独立 state identity/layout；项目格式和 managed manifest
  继续为 v3。旧 `stream_join` 不增加新默认字段，不借用其状态 magic 或版本。
- **FR2 - 严格声明。** 两个 required table 输入名为 `left`、`right`，一个输出名为
  `output`。每侧声明 keys、event time、sequence 列表和字段前缀；全算子声明
  tolerance、late policy、正数 rows/bytes 限额。JSON 全层级拒绝未知字段；Python
  声明不可变、防御性复制容器，不保存数据、callable 或 import path。
- **FR3 - Schema。** 双侧编译期 schema 必须精确、字段名非空且唯一；运行期 Batch
  必须匹配。key 列表非空、等长、位置对应且精确 Arrow 类型相同，不隐式 cast。
  keys、event time 和 sequence 字段声明均 non-null，实际数据出现 null 同样拒绝。
  其他 payload 字段可以 nullable；所有被接受的 payload 类型必须有明确状态收费和恢复编码。
  v1 native 仅接受 API note 列明的 flat Arrow 类型；嵌套、字典、run-end 和 view 类型
  在 schema 校验期返回 `invalid_type`。Python/project 保留既有 portable 类型子集，
  不因 native payload 支持而扩展项目格式的类型词汇。
- **FR4 - 类型。** key 支持 Boolean、Int8/16/32/64、UInt8/16/32/64、Utf8、
  LargeUtf8、Date32/64 和精确类型匹配的 Timestamp；不支持浮点、字典或嵌套 key。
  event time 仅为 `timestamp[us, UTC]`。sequence 非空，逐列支持上述整数类型、
  Utf8/LargeUtf8；不要求连续，不从到达计数、物理行号或 Batch metadata 生成。
  左右 sequence 无需相同类型，因为不跨侧比较。所有列引用必须存在且各列表不得重复。
- **FR5 - 全序与 identity。** sequence 按有类型的逐字段字典序比较；整数按数值，
  字符串按 UTF-8 二进制顺序，不依赖 locale。完整 identity 为
  `(side, key_tuple, event_time, sequence_tuple)`；同 identity 不同 payload 也是重复。
  同 key/time 不同 sequence 是合法候选。输入必须提供可重放的稳定身份。
- **FR6 - 匹配。** 对左行时间 `t` 和 tolerance `T`，只考虑同 key 且
  `t - T <= right_time <= t` 的右行，取最大的 `(right_time, right_sequence)`。
  一条右行可供多条左行使用。`T=0` 只匹配同时间；数学运算不得在 i64 边界 wrap，
  区间与可表示时间域相交。EOF 是独立状态，不以最大时间值代替。
- **FR7 - 输出。** 每条被接受左行恰好一条逻辑最终结果；无候选时右字段全 null。
  输出按输入字段顺序先左后右，使用 `<prefix>__<field>`，保留类型和字段 metadata。
  前缀为非空且不同的 portable ASCII identifiers，最终字段名必须唯一；左字段
  nullability 不变，所有右字段 nullable。
  输出 event time、entity 和 sequence 从左侧声明经前缀映射派生，不能拼接 nullable
  的右 sequence。规范结果顺序为 `(left_time, left_key_tuple, left_sequence_tuple)`；
  相同时间组封闭后整体参与排序。物理 Batch 分组不属于稳定合同。

### 水位、迟到与资源

- **FR8 - 严格 finality。** `closed_s(t)` 当且仅当该侧 EOF，或已接受水位严格
  大于 `t`。只有两侧均 closed 才输出；相等水位不足。每次 ingress watermark/EOF
  变化都必须重新判定，包括没有新聚合 watermark emission 的情况；输出经过有界 collector。
- **FR9 - 安全下游进度。** 未结束侧（包括 idle）参与最小水位 `C`；任一未结束侧
  没有水位则阻塞。先输出全部 `t<C`，再最多发送 `C-1us`，保持单调；
  `C=i64::MIN` 不发送下溢水位。双 EOF 时输出全部剩余左行后传递 EOF，不发送 sentinel。
  输出/水位顺序必须使下游 rolling/cross-section 不将之后合法的 `t=C` 行判迟到。
- **FR10 - Idle/reactivation。** idle 不提升或清除本侧时间边界，不视为 EOF。
  ASOF 在双 EOF 前抑制 output idle，即使当前没有 pending-left；reactivation
  保留旧水位和身份约束。重复水位不重复输出，水位回退及 EOF 后数据沿用明确协议错误。
- **FR11 - 迟到和重复优先级。** 校验顺序为 schema/type/null、本侧 late、准时
  identity 重复、预算、提交。`row_time < own_watermark` 为 late；等号仍准时。
  `late_policy` 为 `error` 或 `drop`，默认 `error`；两侧分别计数。准时完整 identity
  重复报错；越过本侧边界的历史 identity 按 late policy 处理，不承诺永久 duplicate 检测。
  不另加 allowed-lateness 参数，不修改源本身的水位策略。
- **FR12 - 正确回收。** 不得删除仍可影响 pending 或未来合法左行的右候选。
  设 `P` 为最早 pending-left 时间（空为正无穷），`L` 为未来合法左行时间下界
  （无左水位为负无穷，左 EOF 为正无穷）；完成本轮最终输出后，仅在
  `right_time + T < min(L,P)` 时允许按此保守条件回收右 payload。等号必须保留。
  单侧 EOF 不允许清空仍有用历史。payload 不再需要而本侧尚未严格越过其时间时，
  必须保留仍能拒绝准时重复的 identity-only 条目；身份仅在本侧已越过时间或 EOF 后可回收。
- **FR13 - 预算。** `max_state_rows`、`max_state_bytes` 是全算子总额度，均为
  JSON 安全整数域中的正整数。rows 计每条存活逻辑身份一次，包含 pending-left、
  retained-right 和 identity-only，不能按左右各自获得一份额度。带版本的 bytes 收费
  覆盖 payload、identity/key/sequence、索引、待发结果/游标及所持 checkpoint segments；
  共享持有、编码副本和 Arrow backing allocation 不可漏计。
- **FR14 - 工作区和失败。** 短期 admission、排序、输出物化、编码/恢复工作区有
  独立正数上限，首版上限取 `max_state_bytes`；输出另受现有 edge rows/bytes 约束。
  64 MiB state 额度不等于进程 RSS 上限。必须先验证并预留，再原子接纳一个输入 Batch；
  Batch 后半段的重复/资源错误不能留下部分接纳或对应输出。不得用裁剪必要候选或改选
  次优匹配满足限额。单行输出超限、工作区超限和计数器溢出明确失败，不 wrap；
  先前成功交给 sink 的输出不因此撤回。背压、取消及失败必须释放算子拥有的缓冲和任务。

### 状态、错误与观察

- **FR15 - 一致快照。** snapshot/restore 包含 pending-left、右历史、存活 identity、
  逻辑计数器、输出序号及终态，保留原始选择语义。各 ingress 水位/活动/EOF 与已发
  output frontier 由 runtime wrapper 按既有 ownership 保存；恢复时与算子状态交叉校验。
  配置、schema、排序、late policy、额度、state layout 纳入 fingerprint/兼容检查。
- **FR16 - 恢复原子性。** managed v3 manifest、source cursor、barrier alignment 和
  sink commit 构成同一切点。最终输出与 barrier 按 FIFO 串行形成一致状态：左行仍 pending，
  或其结果在 barrier 之前且状态已记录输出，不得半提交。恢复拒绝错误 kind/version/layout、
  schema/config、segment/checksum、重复 identity、排序、资源收费、序号和进度矛盾；
  先完整验证再替换，失败不保留半恢复状态。恢复不能信任快照自报的资源字节。
- **FR17 - 生命周期与交付。** 在无候选、已有临时候选、输出后 checkpoint 发布前后
  恢复，逻辑结果均与连续执行相同。已发布终态恢复不得重新最终输出。
  `reset` 只清理算子自有内存，不删除共享 checkpoint、重置 source cursor 或操作 sink 事务。
  普通 at-least-once sink 的恢复重放仍可重复；外部 exactly-once 只按现有逐输出证明声明。
- **FR18 - 结构化诊断。** 配置/type/null、duplicate identity、late、state rows/bytes、
  workspace/output row 资源错误、overflow、checkpoint mismatch 保持可区分 reason。
  reason、node 采用既有结构化错误字段；side、字段路径和必要额度信息由现有 validation
  issues 或安全诊断文本保留，经 PyO3 和 Studio 正确传递。status/log 不携带完整输入
  payload。既有 reason 的含义不变；错误分类不得依赖消息文本解析，不增设通用错误协议。
- **FR19 - 确定指标。** 暴露左右 accepted/late/duplicate、pending-left、retained-right、
  identity-only、charged bytes、matched/unmatched/emitted-left、eviction/resource failures、
  两侧进度和 output frontier。成功终态满足
  `emitted_left = matched_left + unmatched_left = accepted_left` 且 `pending_left=0`。
  同输入控制轨迹和恢复切点的逻辑计数确定；跨调度/切分不要求峰值或 batch 计数相同。
  失败 admission 不改变已接纳数据、gauges 和逻辑输出计数；失败尝试计数与逻辑计数区分，
  具体递增/回滚规则及版本化收费表由 API note 单独持有。

### #259 Python、symbolic 与合同

- **FR20 - Python 表达式。** 正常入口由 root `cf.table` 和 `TableExpr` 提供薄声明，
  event-time/sequence 来自两侧已声明元数据；`keys=None` 从 entity 元数据推导，
  `keys=(left_names,right_names)` 允许显式选择不同关联键。输出 entity 为有效左关联键。
  不凭数据推断或丢失 non-null 约束。高级 `PipelineBuilder` 接受显式不可变完整 spec，
  保留独立选择 key/time/sequence 的能力。`timedelta` 到微秒只用整数运算，不经过浮点。
  Rust、Python、raw project 入口必须得到一致的配置语义与输出 schema。
- **FR21 - 执行入口。** `TableExpr.stream`/`Program.stream` 绑定逻辑输入名，分别
  返回 Arrow 表/命名 `StreamOutput`；两输入水位配置使用逻辑名称映射，不能覆盖已有
  `SourceBinding` 的策略。默认 iterable 保持 #259 非递减到达校验和 `max_seen-1us`
  水位语义；显式策略允许合法乱序。源不具所需水位能力时 preflight 失败。
  普通 iterable 及临时 managed state 不提供 durable restart；显式编译、物理绑定名、
  replayable sources、sink 证明和稳定 managed checkpoint 根用于恢复验收。
  `collect`/`compute`/batch 编译必须明确拒绝 ASOF。
- **FR22 - Capability 与 lowering。** 新 capability 必须证明 stream-only、精确端口、
  `checkpointed_stateful`、state version/layout=1、`group_final_append_only`、
  requires-watermark、deterministic、replay-safe 和合法切分下 microbatch-invariant。
  任一缺失、伪造或未知版本均 fail closed；`requires_datafusion=True`。
  primitive digest 包含全部语义有序参数，不能和 inner 合并；同 digest 多输出共享一个
  物理状态 owner。ASOF 是时间最终性边界，优化不能穿越它改变候选集合或提前执行状态计算。
  不变性适用于等价合法数据/控制轨迹的成功运行；固定有限额度不保证任意切分均可成功。
- **FR23 - 组合与 explain。** 下表逐项成功或给出稳定的 analysis/compile error，
  不能静默退化、增设无水位状态副本，或继承不存在的排序证明。explain 展示 backward、
  inclusive tolerance、sequence、late、严格双水位封闭、左输出 ordering、状态限额、
  owner 数量和交付前提；说明 ASOF 链的保守 frontier 可能增加等待。
- **FR24 - 跨表面合同。** 更新 Rust exports、Python exports/stubs、project-v3 schema、
  OpenAPI、生成 TypeScript 及实际手写 status/error 消费者。Studio 通用导入、查看、
  保存新 kind 往返无损；不要求新增专用编辑器。生成文件二次生成无漂移；旧 inner
  子定义和序列化向量保持。文档示例使用 #259 优先 Python API，匹配计算全部进入原生运行时。
  新 ASOF status 的整数 counters/gauges/watermarks 及结构化额度诊断，在 Rust/Python
  保持整数，在 Studio wire 统一使用规范十进制字符串；不存在的水位仍为 null，bool
  仍为 bool。不经过 JavaScript number 或浮点中转，以语义无损覆盖完整 i64/u64 域。
  该投影仅适用于新 ASOF status；project 配置中的安全整数 tolerance/limits 和旧 inner
  status 表示均不改变。

## Composition Contract

| 组合                                                       | 验收结果                                                                    |
|----------------------------------------------------------|-------------------------------------------------------------------------|
| 两个原始表或保留排序的 row-local 转换 → ASOF                          | 支持；字段/schema/双侧 source watermark lineage 必须可证明                          |
| ASOF → 投影/filter/with-columns/单 alias stream SQL         | 支持；SQL 后输出遵守既有新 lineage、无继承 temporal ordering 规则                        |
| 同 ASOF fan-out、多独立 ASOF、旁路/独立 event-window               | 支持；共享 digest 一个 owner，独立分支不错误共享状态                                       |
| ASOF → ASOF                                              | 支持；选用左派生的非空 key/time/sequence，保留每层严格 finality                           |
| ASOF → rolling/cross-section                             | 支持；满足消费者既有类型/ordering 条件，不能越过边界提前计算                                     |
| 旧 inner → ASOF                                           | 完整非空 ordering/schema/水位 lineage 下支持，重复由 admission 拒绝                    |
| ASOF → 旧 inner                                           | 支持其既有合法输入路径；inner 输出仍遵守旧 inner 合同                                       |
| event-window → ASOF，ASOF → event-window                  | 拒绝；#258 未提供该串联所需时间/行身份和降低证明                                             |
| SQL → ASOF 或 SQL → temporal stage                        | 拒绝；#259 SQL 输出不继承时间排序和 source watermark lineage                         |
| ASOF 跨 matrix attachment/不兼容 lineage 组合                  | 按既有边界明确拒绝，不把右 nullable 列声明成 non-null                                    |

## Non-Functional Requirements

- **单一执行引擎。** DataFusion 54 仍是表表达式/SQL 的唯一引擎；经 API note 固定的
  Rust/Arrow 状态身份和时间选择边界每条最终左行最多提供一个候选，DataFusion
  执行有界分块的关联、投影和顺序物化。不得先生成所有同 key 区间配对再截取结果，
  不得在 Python 重写匹配，不升级 DataFusion 作为前置依赖。原型以精确类型 oracle
  验证 key/sequence 相等与全序、无候选补空、分块预算及输出物化一致。
- **资源验证。** 固定 1000 key、总计 100000 输入行、60 秒 tolerance、67108864
  state bytes。记录左右比例、schema/payload 宽度、seed、倾斜、时间跨度、水位轨迹、
  Batch 切分、rows/edge/workspace 额度及源码/依赖版本。分别验证正常推进释放、单侧
  停滞、热 key、宽 payload/bytes 超限；成功轨迹对齐独立 oracle，失败前输出前缀正确。
  同时报告 charged state、保留 Arrow/segment allocation 与 workspace，禁止改写为 RSS/吞吐承诺。
- **兼容性。** 精确基线旧 inner 的规范配置（包括省略默认值）、project/native
  fingerprint、schema、symbolic v1/v2 digest/lowering、capability、状态 metadata/segments
  和 managed 恢复向量必须不变；不能通过重录 golden 消除漂移。项目包装 fingerprint
  与直接 native graph fingerprint 分别比较自身入口，不要求不同入口字面相等。
- **本地验证。** 每项行为先记录 focused RED，再 GREEN；只运行新 Join 与实际受影响
  runtime/Python/合同定向检查。完整回归、跨平台、Rust 90%/Studio 85% coverage 和例行
  性能门禁交 CI。本目标完成条件为提交经过最终 specialist review 的 PR；提交后一次
  非阻塞 CI 快照可以 pending，不能声称通过或 merge-ready，本次没有 merge 授权。

## Inputs and Outputs

| 名称                         | 类型                               | 单位       | 范围 / 约束                                                      |
|----------------------------|----------------------------------|----------|--------------------------------------------------------------|
| left/right                 | immutable table Batch            | 行        | FR3/FR4 exact schema；payload 只读                              |
| side.keys                  | 有序非空列名列表                         | 无        | 两侧等长、对应类型一致，各列表不重复                                           |
| side.event_time            | 列名                               | us       | 非空 timestamp[us, UTC]，物理值为完整 i64 时间域                         |
| side.sequence_by           | 有序非空列名列表                         | 无        | 非空 integer/string typed tuple，全序且输入身份稳定                      |
| side.prefix                | 非空字符串                            | 无        | 左右不同，生成输出名唯一                                                 |
| tolerance_micros           | Rust 整数 / Python timedelta       | us       | 0 至 9007199254740991；bool/float/string 不可冒充整数                |
| late_policy                | 严格枚举                             | 无        | error 或 drop；默认 error                                        |
| limits.max_state_rows      | 正整数                              | 行        | 1 至 9007199254740991；全算子共享                                   |
| limits.max_state_bytes     | 正整数                              | byte     | 1 至 9007199254740991；state 和独立 workspace 各用此上限               |
| output                     | immutable table Batch            | 行        | 每个被接受左 identity 一行；右派生字段全部 nullable                          |

## Acceptance Criteria

- [ ] **AC1 (FR1-4, FR20, FR24)** Rust/Python/raw JSON 一致接受有效声明，拒绝未知字段、
  bool/float/string 数值、超安全域、零额度、缺列、nullable identity、错单位/时区和不支持类型；
  每个失败具有稳定路径。调用者 schema/字段 metadata/容器保持不变。
- [ ] **AC2 (FR5-7)** R=90/100/110、L=105：T=10 选100，T=4 无匹配但保留左行；
  T=0 仅同时间；上下界均包含。另覆盖复合 key、右行复用、空侧/空 Batch 和 nullable payload。
- [ ] **AC3 (FR4-7)** 独立枚举 oracle 与固定种子合法乱序/分批测试逐行逐字段及规范顺序相同；
  负整数/整数边界、UTF-8/复合 sequence 同时间选值不依赖到达顺序，完整重复明确拒绝。
- [ ] **AC4 (FR6, FR8-10)** 在 i64 MIN/MAX、T 安全上界和 EOF 下无 wrap/sentinel；
  水位等于 t 不输出，两侧严格越过才一次输出，单侧 EOF 仍等另一侧。边界算子测试使用
  能表达该进度的显式策略，不改变默认源 watermark 的既有下溢校验。
- [ ] **AC5 (FR8-10)** 真实 operator-task 测试覆盖无聚合 emission 的 ingress 推进、
  idle/reactivation 有/无 pending、重复水位、单/双 EOF；下游观察到先数据后安全 frontier。
- [ ] **AC6 (FR9-10, FR23)** C=105 后合法 L@105，双侧推进106；ASOF→rolling/CS
  不丢该行。ASOF→多输入下游在 idle/reactivation 两类 pending 轨迹均不提前封闭。
- [ ] **AC7 (FR11)** time=own_watermark 准时；旧 identity 在本侧水位之前按 drop/error
  计 late，本侧尚未越过的重复（即使 payload 已回收）仍为 duplicate；EOF 后数据拒绝。
- [ ] **AC8 (FR12)** 左 wm=1000、pending L=105、右 wm=90、T=10 时保留 R=100；
  回收等号、单侧 EOF、无 pending、identity-only 回收条件均有正反例。
- [ ] **AC9 (FR13-14)** rows、bytes、workspace、单行 edge 超限分别可触发；Batch
  后半段失败不部分入状态。小 slice 不隐藏大 backing buffer；snapshot/restore 工作区受限；
  finalization 背压/取消后释放资源，所有已输出匹配仍正确。
- [ ] **AC10 (FR15-17)** 在无候选、临时候选、collector finalization、输出已写但 manifest
  未发布、发布后和终态发布后，用真实 managed runtime 恢复并对齐连续执行；终态不重复 flush。
  普通 sink 允许物理重放的切点与可证明 exactly-once 路由的预期分别验证。
- [ ] **AC11 (FR15-16)** 腐败/错版本/schema/config、缺 segment/坏 checksum、重复身份、
  非法排序、伪造资源收费、序号/进度矛盾分别拒绝；失败 restore 不改变此前合法状态。
- [ ] **AC12 (FR17-19, FR24)** reset 不删共享 checkpoint 或操作 source/sink；逻辑计数
  恢复后与相同控制轨迹相同，终态等式成立；错误及 status 经 PyO3/Studio 语义无损且
  无 payload。新增 backend/TypeScript 合同向量覆盖 i64 MIN/MAX 水位、u64 计数极值、
  null 水位和 bool；原生/Python 整数精确映射规范十进制 wire 字符串并往返保持数值，
  不允许指数形式、前导加号、冗余前导零、负零或浮点舍入；旧 inner 向量不变。
- [ ] **AC13 (FR20-21)** root cf.table、fluent、advanced builder 对同交易/报价向量一致；
  TableExpr/Program.stream 按逻辑名映射两侧水位，显式乱序策略有效，不覆盖 SourceBinding；
  await/async-context 取消和 cleanup 有定向验收。batch/collect/compute 拒绝 ASOF。
- [ ] **AC14 (FR22-23)** 缺失/逐项伪造 capability 失败；相同 ASOF digest fan-out 仅一个
  state owner。组合表每行至少一个真实 compile/execution 成功向量或明确拒绝向量，
  optimizer 不穿越 finality，explain 展示实际合同。mixed inner/ASOF 两方向覆盖乱序、
  stable sequence 和重复身份；不以 inner 未预证唯一性为由拒绝合法输入。
- [ ] **AC15 (FR1, FR24, Compatibility)** 旧 inner baseline 向量不变，旧状态定向恢复通过；
  新 project/schema/OpenAPI/TS 往返一致，Studio 通用保存保留 ASOF 全部字段，二次生成无 diff。
- [ ] **AC16 (Resource verification)** 四条固定资源轨迹有可复现参数、oracle/前缀证据、
  收费和 allocation/workspace/释放观察；总输入不超过原定义的十万行，不冒充速度保证。
- [ ] **AC17 (Verification)** 新原生 `stream_asof_join_validation`、
  `stream_asof_join_properties`、`stream_asof_join_state`、boundaries/resources/
  restore_corruption 集成目标，以及 `operator_task`/`runner` 内的 `asof_tests`
  定向运行；Python `test_stream_asof_join_builder.py`、
  `test_symbolic_stream_asof_join.py`、`test_asof_stream_results.py` 和实际受影响
  stream/Studio 合同目标
  按阶段定向通过；旧
  `stream_join_validation`/`stream_join_state` 及 Python 原有 Join 兼容检查通过。
  changed-module format/lint、必要合同生成及 `git diff --check` 通过；完整矩阵明确归 CI。
- [ ] **AC18 (Delivery)** 更新交易/报价示例和相关 Python/streaming/project 文档，说明
  watermark 等待、late、资源额度、源/sink 交付边界；最终 specialist review 阻塞项关闭，
  以指定 feature 基线提交 PR，描述列明定向证据、基线和一次 CI 状态快照。

## Open Questions

本规格不保留需要重新询问用户的语义问题。API note 给出版本化收费表、完整 reason/status
字段与 DataFusion/原生状态物化分工，连同本规格经 critic 检查。原型与实现必须落实
FR1-FR24 和 AC1-AC18，不得因现有 lowering 分派缺少新 kind 而缩小已批准组合。
