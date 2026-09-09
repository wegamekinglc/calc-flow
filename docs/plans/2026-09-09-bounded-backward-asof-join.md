# 有界 backward ASOF Join 开发计划

日期：2026-09-09。实施基线：PR #259 的 `feature/python-expression-api-refactor`，
commit `eda1583751abbd1ca4d246fcb8ee6b70f57d9b09`。
状态：原生、Python 和跨表面实现、资源轨迹及 managed recovery 定向验收已完成。
最终 FR/AC 审计和 specialist review 为
[Approve](../../.codex/artifacts/reviews/bounded-backward-asof-join.md)，
该审阅记录列出测试证据与局限；PR 发布和一次 CI 快照由实际交付记录确认。
完整回归、跨平台、覆盖率和常规性能门禁仍交 CI，本文件不宣称这些门禁已通过。
用户已授权直接在该分支基础上实施至提交独立 PR，不再等待 #259 合并；未授权合并。

提交前核查：#259 已合并为 `main` 的 `06c0223`。该提交与原实施基线
`eda1583751abbd1ca4d246fcb8ee6b70f57d9b09` 的代码树相同；新 PR 以 `main` 为目标，
只包含 ASOF 改动。最终提交 SHA、审阅和定向验证结果在交付记录中单独列明。
原计划基线 `464a480` 仅作为历史分析来源，不用作实现或兼容验收基线。

实施合同由 [正式规格](../../.codex/artifacts/specs/bounded-backward-asof-join.md)
和 [API note](../../.codex/artifacts/api-notes/bounded-backward-asof-join.md) 共同定义。
下文给出已冻结的语义、实施分工和退出门禁；具体 wire、公开签名、计费和支持组合按这两份合同。
[Critic 结论](../../.codex/artifacts/critiques/bounded-backward-asof-join.md) 为 proceed with caveats，
C2 要求覆盖所有工作区所有权，C3 要求多轮 compact/restore 与 wrapper progress 一致性。
实际使用方法见 [ASOF guide](../asof-join-guide.md)。
最终交付需覆盖 FR1–24/AC1–18，不能只完成原生 happy path 或旧 builder 入口。

### 对 #259 的必要调整

- 正常 Python 入口使用 root `cf.table.stream_asof_join` 和薄的
  `TableExpr.stream_asof_join`；两者共享现有 symbolic IR。时间/sequence 从输入声明推导，
  `keys=None` 使用 entity 字段，也支持显式 `keys=(left_names, right_names)`。
- 通过 `TableExpr.stream`/`Program.stream` 执行，原始输入使用逻辑名称绑定；
  source watermark 策略同样按逻辑名称映射，不覆盖 `SourceBinding` 的既有策略。
  保留 iterable 默认非递减时间、`max_seen-1us` 水位及无 replay 的交付边界。
- 新 wire 使用独立的嵌套 `left`/`right` side specs；advanced builder 接受不可变完整
  `AsofJoinSpec`。不复制旧 inner 过长签名，不新增复杂度或参数数量豁免。
- 按正式矩阵支持 fan-out、独立/嵌套 ASOF、旁路、双向 mixed inner 和 post-ASOF
  rolling/cross-section。mixed 输入必须保留完整 non-null 时间/身份及水位 lineage；
  实际 identity 唯一性仍由 ASOF admission 验证，不能因缺少静态唯一性证明而拒绝所有 mixed。
- #259 SQL 输出重新建立 lineage 且丢弃 temporal ordering，SQL → ASOF 明确拒绝；
  ASOF → 单 alias row-local stream SQL 支持。#258 event-window 与 ASOF 串联保持拒绝，
  独立 event-window 分支允许。
- 原生 Arrow RowConverter 编码 typed key/sequence，Rust 有序状态索引先选择每左行至多
  一个右候选，再由 DataFusion 按内部 ordinal 加精确 key 等值执行 LEFT JOIN、前缀投影
  和顺序物化。复用一个算子会话，禁用无界 all-pairs 中间结果；工作区有独立同额度上限。
- 提交时若 #259 仍开放，新 PR 的 base 为其 feature 分支；若已合并，先核查实际主分支
  再调整 base，确保 PR diff 仅包含本项。提交后记录一次非阻塞 CI 快照，不等待无授权的合并。

阶段 0 已取得的证据：固定基线上旧 `stream_join_validation` 16 项、
`stream_join_state` 10 项通过；DataFusion 54 小原型验证 signed/UTF-8 typed tuple 全序、
ordinal+key LEFT JOIN 的未匹配补空/右行复用，以及有限 memory-pool 预留失败，共 3 项通过。
原型不等于 ASOF 实现验收，也未证明进程 RSS 上限或吞吐。

## 1. 建议与范围

采用独立的 `StreamAsofJoinOperator`，原生配置 kind 为
`stream_asof_join`，capability 和 symbolic primitive 使用独立的
`stream_asof_join@1`，state/layout/accounting version 均为 1。

第一版固定为 stream、backward、有限 inclusive tolerance、左行保留、最终追加。
每条被接受的逻辑左行产生一条结果；右行可被多条左行复用。没有匹配时，所有右侧输出字段为空。
不增加方向选择、无限历史、动态 lookup、通用 outer Join、更新/撤回、batch ASOF 或新 sink 保证。

优先独立 operator 的原因是两种 Join 的输出时机、输入约束、identity、右侧 nullability、
状态布局和资源模型都不同。给 `StreamJoinType` 增加 `asof` 很容易使旧 inner Join
序列化默认值、schema、指纹或恢复分支发生改变，收益不足以抵消兼容风险。
可以复用已经证明等价的纯校验和 Arrow 工具，不先把旧 Join 改造成通用 Join 框架。

## 2. 实施接入与兼容边界

- [operator/asof/](../../crates/calc-flow/src/operator/asof/) 独立负责 spec/schema、
  typed identity、有序历史、admission/finalize、状态收费、工作区和 checkpoint。
  旧 [operator/join.rs](../../crates/calc-flow/src/operator/join.rs) 继续输出全部区间 inner
  匹配，其 timestamp/null/schema/default/checkpoint 合同不被新算子反向收紧。
- [pipeline/stream.rs](../../crates/calc-flow/src/pipeline/stream.rs) 和
  [operator_task.rs](../../crates/calc-flow/src/runtime/streaming/operator_task.rs)
  为 ASOF 增加私有的逐 ingress 进度和有界 collector 分派，覆盖没有新 aggregate
  watermark emission 的进度变化。公共 `StreamOperator` trait 签名保持。
- ASOF 读取真实双侧水位/idle/EOF，先 drain，再更新安全输出 frontier；双 EOF 前抑制
  output idle。runtime wrapper 仍是持久化控制状态的 owner，恢复与 native 状态交叉校验。
- [config/asof.rs](../../crates/calc-flow/src/config/asof.rs)、
  [project_store/asof.rs](../../crates/calc-flow/src/project_store/asof.rs) 处理独立严格声明，
  [runtime/streaming/job/asof.rs](../../crates/calc-flow/src/runtime/streaming/job/asof.rs)
  检查直接 runner 中所有可达源的水位策略；无关禁用水位分支仍合法。
- Python [asof_join_spec.py](../../python/calc_flow/asof_join_spec.py)、
  [symbolic/asof.py](../../python/calc_flow/symbolic/asof.py)、
  [symbolic/asof_analysis.py](../../python/calc_flow/symbolic/asof_analysis.py) 和
  [symbolic/lower/asof.py](../../python/calc_flow/symbolic/lower/asof.py) 负责不可变声明、
  类型/组合证明和 native lowering，不含运行期匹配引擎。独立 capability 不修改 inner 条目。
- PyO3 和 Python `JobStatus` 增加 `stream_asof_joins`；Studio 对新 metrics 的整数和
  watermark micros 使用十进制字符串，nullable watermark 保留 null，bool 保持 bool。
  project-v3、OpenAPI 和生成 TS 记录新 kind/status；SSE 的 `RunEvent` 纳入 OpenAPI。
- 冻结兼容向量位于 [asof-inner-compat-v1](../../tests/fixtures/asof-inner-compat-v1/)，
  对照 base `eda1583`；不得重录 golden 消除差异。普通 inner 的配置、schema、指纹、
  capability、symbolic v1/v2 和 checkpoint reader 仍逐项验证。

## 3. 已冻结语义

### 3.1 输入、类型与 identity

两个 required table 输入固定名为 `left`、`right`，输出固定为 `output`。
双方使用编译期 exact Arrow schema，运行时逐 Batch 检查 schema 和实际 null 值。

key 列表非空、等长且位置对应，类型必须完全一致，不做隐式 cast。
首版 key 支持 Boolean、所有 8/16/32/64 位整数、Utf8/LargeUtf8、Date32/Date64
和精确类型匹配的 Timestamp；
浮点、嵌套、字典等未证明的类型明确拒绝，不能只检查“左右类型相同”。
key、event time、sequence 的字段声明必须 non-null，实际 null 也必须拒绝。
event time 仅接受 `Timestamp(Microsecond, UTC)`，不隐式补时区或截断精度。
Arrow timestamp 的单位和时区属于类型参数，null 有独立有效位图，参见
[Arrow 格式规范](https://arrow.apache.org/docs/format/Columnar.html)。

payload 首版采用明确的 flat Arrow allowlist：null、boolean、整数、Float16/32/64、
date/time/timestamp/duration/interval、decimal、UTF-8、binary 和 fixed-size binary；
支持对应的 large UTF-8/binary。嵌套、字典、run-end 和 view 类型在 schema 校验期拒绝。
这使编码、恢复和物化的临时内存计费覆盖已接受类型；不承诺尚未证明的嵌套展开上限。
Python/project 仍使用既有 portable schema 类型子集，详见已冻结 API note。

双方显式声明非空的 `sequence_by` 列表。首版支持有符号/无符号整数和 UTF-8 字符串，
以有类型的逐字段字典序比较；字段顺序属于配置身份。字符串按确定的二进制 UTF-8 顺序比较，
不使用 locale；不同整数位宽不隐式转换。跨左右两侧 sequence 类型可以不同，因为不互相比较。
如果扩展到其他类型，必须先添加跨 Rust/Python/Arrow 的排序向量。
不能用到达计数、Batch metadata sequence、物理行号或未规范化的 little-endian 字节序作全序。

完整行 identity：`(side, key_tuple, event_time, sequence_tuple)`。
相同 identity 即使 payload 不同也报重复错误；同 key/time 的不同 sequence 是合法竞争候选。
sequence 不要求连续，也不要求物理输入按 sequence 或时间排序；跨 Batch 乱序只需未越过本侧水位。
identity 必须由可重放输入稳定提供，不能在 Python lowering 中重新编号。

### 3.2 匹配与输出

对左行时间 `t` 和容差 `T`：

```text
0 <= T <= 9_007_199_254_740_991
C(l) = {r | key(r)=key(l) and t-T <= time(r) <= t}
answer(l) = argmax_C(l) (time(r), sequence(r))
```

区间两端包含。`T=0` 只接受同时间；未来右行始终不能匹配。
时间比较使用 `i128` 中间值，不先在 `i64` 上计算 `t-T` 或 `r+T`；
边界超出 EventTime 可表示区间时按数学区间与可表示域相交，不能 wrap 或任意截断。
EOF 用独立状态表达，不制造 `i64::MAX` watermark。因此 `t=i64::MAX` 可以在 EOF 时完成。

输出所有左字段，再输出所有右字段，使用显式且不同的前缀；保留类型和字段 metadata。
左字段 nullability 不变，所有右侧派生字段都置 nullable，包括右 key/time/sequence。
右 event time 非空约束使用户可用它是否为 null 判断未匹配，不额外添加匹配布尔列。

输出 event time 固定来自左侧时间。按 `(left_time, left_key_tuple, left_sequence_tuple)`
排序最终左行；相同时间的整组必须在封闭后完整参与排序。
保证逻辑行内容和规范顺序不受合法到达顺序与 Batch 切分影响；不保证物理 Batch 分组相同。
下游 entity/sequence 元数据从左 key 和左 sequence 派生，不能沿用左右 sequence 拼接规则。

### 3.3 水位、迟到、idle 和 EOF

定义 `closed_s(t) = ended_s or (watermark_s exists and watermark_s > t)`。
仅当 `closed_left(t) and closed_right(t)` 时输出左行结果。
水位等于 `t` 时不输出，等于本侧已接受水位的数据仍准时；这是本功能需要锁定的严格边界。

迟到判断使用本侧已经接受且不回退的 watermark：`row_time < watermark_side`。
首版显式 `late_policy="error" | "drop"`，默认 `error`；两侧使用同一策略并分别计数。
idle 不提升、不清除本侧 watermark；reactivation 不清空迟到边界；EOF 后的数据是协议错误。
源自身的 watermark 生成延迟仍存在，但 ASOF 不增加第二个 allowed-lateness 参数。

验证顺序为：schema/非空/类型 → 本侧迟到 → 准时 identity 重复 → 资源预算 → 状态提交。
必须特别冻结“重复拒绝”的范围：本方案对准时可接受区间中的重复报 `duplicate_identity`；
已越过本侧水位的旧 identity 按 late policy 处理，不能再次成为候选。
若要求任意久远重放都必须区别为 duplicate error，则需要无界 seen-set 或额外上游唯一性证明，
与当前有限状态、drop 迟到策略存在冲突。本项冻结为准时身份拒绝重复，历史重放按本侧 late policy 分类。

输入封闭 frontier 的公式是：未结束侧保留其 watermark，包含 idle；未获得水位的未结束侧阻塞；
结束侧视为已封闭。记有效最小值为 C，先 drain 所有 `t < C` 的左行。
输出 watermark 不能直接照搬 C：现有 [rolling.rs](../../crates/calc-flow/src/operator/rolling.rs)
的 `is_late`/`closing_keys` 在零 lateness 时按 `t <= watermark` 封闭，cross-section 也有相同边界。
若提前转发 C，之后仍合法到来的 `left_time=C` 会被下游误判迟到。
安全转换是发 `C-1` 微秒，`C=i64::MIN` 时不转发；计算仍用宽整数并保持单调。
这使对外 watermark 只覆盖已完成的时间，不修改旧 rolling/inner 的合同；
连续 ASOF 组合可能多等待一次水位推进，示例和 explain 必须注明这个保守边界。
双 EOF 时 drain 全部左行后发 EOF，不发无穷大 sentinel。单侧 EOF 不能丢弃仍被 pending-left 需要的右值。

还需控制 output idle：输入全 idle 不代表 ASOF 已无未封闭结果。
首版在双 EOF 前不透传 output idle；idle 仍记录在本侧状态/status 中。
pending-left 为空也不能直接视为安全：reactivation 后仍可能到来旧水位以上的新左行，
而下游若曾跳过该 idle 输入，其水位可能已超过这些左行。
阶段 0 用 ASOF → 多输入下游的有 pending/无 pending 反例确认该保守规则，
不要只通过算子自身“没有提前输出”的测试。

### 3.4 状态回收与有限资源

状态包含 pending-left、右侧有序历史、仍需拒绝重复的 identity、索引、待发结果/游标、
计数器及 prepared checkpoint segment 的相关内存。不把其中任何一类藏在不受限队列里。

设 `P` 为最早 pending-left 时间，空集合视为正无穷；`L` 为左侧未来可接受的时间下界，
未有 watermark 为负无穷，左 EOF 为正无穷。安全保守回收规则为：

```text
H = min(L, P)
right payload at r can expire only when r + T < H
```

这是充分条件，可先用全局 P，后续有独立证据再按 key 优化。
必须先完成本轮 finalization，再计算 P 和回收；等号不能删除，因为下界包含端点。
例如左 watermark=1000、pending-left=105、右 watermark=90、T=10 时，右值 100 不能回收。
右侧已经 EOF 但左侧尚未 EOF 时，历史仍需为未来合法左行服务。

payload 可以回收但本侧 watermark 尚未严格越过其时间时，重复检测 identity 仍要保留；
该 identity-only 条目也计入 rows/bytes。只有旧 identity 已必然按 late 拒绝，或其本侧已 EOF 时才能释放它。
不因同时间已有更高 sequence 就无账本地忘记旧 identity；首版可以先不做该压缩。

`AsofStateLimits(max_state_rows, max_state_bytes)` 表示每个算子的总额度，
不是左右各自一份；`64 MiB` 对应 `67_108_864` bytes。两者是 JSON 安全域内的正整数，必填。
ASOF 没有匹配爆炸，无需复用 inner 的 `max_matches_per_input_batch`。
输出分块使用既有 edge row/byte budget，并处理单行本身超限的错误。

API note 的 accounting version 1 冻结收费表：存储行和 identity-only 条目如何算 rows，
payload、key/sequence 编码、索引以及待发缓冲如何算 bytes，共享 allocation 如何避免漏计/重复计。
现有 inner 的逻辑字节不等于进程 RSS，不能把其常数直接宣传成真实内存上限。
另外冻结 Arrow 整块 buffer 被小 slice 保留、临时排序/物化和 checkpoint carried segments 的有界策略；
无法计入 state 额度的短期工作区必须有独立有限预算和失败路径。
验收分别观察 charged state、实际保留的 Arrow/segment bytes 和工作区峰值，不宣称 RSS <=64 MiB。

先验证并预留预算，再提交状态；超限返回结构化资源错误，不能裁剪仍影响答案的历史、
不能改选次优匹配。可保证失败 admission 不产生其对应的结果、不部分加入该输入 Batch；
不声称可以撤回之前已成功交给 sink 的输出。水位停滞时允许最终明确失败，不保证无限运行。

### 3.5 恢复、错误和指标

复用 managed v3 manifest、source cursor、barrier alignment 和 sink commit 协议。
ASOF 使用独立 operator/state identity、magic 和 layout version；不复用 `CFJOIN1` 布局。
配置、端口 schema、排序定义、late policy 和 limits 必须进入 fingerprint/恢复兼容性检查。

snapshot 记录 pending-left、右历史/identity、确定性输出序号、逻辑计数和终态。
本侧活动状态、水位和已转发 output frontier 由 runtime wrapper 的既有 ownership 保存；
不另建互相矛盾的第二份进度真相，restore 时校验算子状态和 wrapper 一致。
将 ASOF 加入 `requires_output_frontier_state` 的既有保存/恢复判断，
复用 runtime reserved frontier 字段的 ownership，不修改旧 inner 的字段编码或缺失字段判定。
如暂存 best-candidate 引用，必须可验证地恢复引用，或从同一快照重算；禁止悬空引用。

原生 `checkpoint` capture 共享已准备的单个 `asof-state-v1` segment 并记录 metadata，
不在同步 capture 中全量编码。当前表示在 async 数据/进度 handler 中完整编码和压缩 retained
state；这一步是 O(retained state)，存在预算检查和 cancellation/yield 点。输出分块过程中也
需要准备一致状态，因此 predecessor lookup 的对数复杂度不能代表整个 handler 的复杂度。
prepared segment、逐行 IPC payload 和独立编码副本都纳入 state；临时编码、Arrow/DataFusion
物化和 restore 共用额外 workspace 上限 `max_state_bytes`。64 MiB state 不是 64 MiB RSS，
固定资源轨迹只证明其指定输入下的释放和明确失败，不承诺吞吐或任意切分都成功。
restore 先解码、验证 identity 唯一性、排序、schema、charge、epoch、版本和终态，再原子替换状态。
错误 snapshot 不留下半恢复算子；单侧 EOF、双侧 EOF、idle 的 checkpoint 都有定向用例。
`reset` 只清除算子自有内存，不删除共享 checkpoint、不重置源游标、不提交/撤回 sink 事务。

错误至少区分配置非法、类型/null 约束、重复 identity、late、state rows/bytes 超限、
单行输出超限、计数器溢出和 checkpoint mismatch。复用现有 `OperatorReason` 和
`ProjectValidation` 的结构，确需新增字段时同步 PyO3/Studio。错误携带 node、side、字段路径、
稳定 reason code 及预算信息，不把完整行情 payload 放进 status/log。

公开指标：左右 accepted/late/duplicate 行数、pending-left、retained-right、identity-only rows、
charged state bytes、matched/unmatched/emitted-left、evicted-right、resource failures、
本侧进度与 output frontier。成功终态满足：

```text
emitted_left_rows == matched_rows + unmatched_rows
emitted_left_rows == left.accepted_rows
pending_left_rows == 0
```

同一输入控制轨迹和恢复切点的计数必须确定；跨 Batch/调度改变时，峰值状态和 affected-batches
指标可以不同，不能把这些也作为分批不变性断言。计数器 checked overflow 必须失败，不能 wrap。
恢复带回已提交切点的逻辑计数，不把 sink 发生的物理重放次数伪装成逻辑输出次数。

## 4. API 合同

Rust 新增 `StreamAsofJoinSpec`、`AsofStateLimits`、`StreamAsofJoinOperator`，
构造器表达固定 backward left-preserving 语义；不添加暂不支持的 direction/join_type 参数。
精确签名在阶段 0 冻结，并提供 rustdoc。Python 配置只负责不可变声明和参数校验。

正式 API note 和已实现配置使用以下独立 wire：

```json
{
  "kind": "stream_asof_join",
  "spec": {
    "left": {
      "keys": ["symbol"],
      "event_time": "trade_time",
      "sequence_by": ["trade_sequence"],
      "prefix": "trade"
    },
    "right": {
      "keys": ["symbol"],
      "event_time": "quote_time",
      "sequence_by": ["venue_id", "quote_sequence"],
      "prefix": "quote"
    },
    "tolerance_micros": 60000000,
    "late_policy": "error",
    "limits": {
      "max_state_rows": 100000,
      "max_state_bytes": 67108864
    }
  }
}
```

所有声明 deny unknown fields；布尔、浮点、字符串不能冒充微秒和限额整数。
项目格式仍为 v3，新 kind 的含义固定在版本化原生 operator 身份中。
当前代码的 version 约定优先，不为方便给全部旧 OperatorSpec 新加 version 字段。

正常入口是 `cf.table.stream_asof_join(left, right, /, *, tolerance, limits, **options)`，
以及同义 fluent 方法。typed `Unpack[_AsofJoinOptions]` 限定可选 `keys`、`late_policy`、
`prefixes` 关键字，默认分别为 None、"error" 和 ("left", "right")；未知关键字拒绝。
用户仍直接传这些关键字，不传字典参数；签名细节以正式 API note 为准。
高级入口为 `PipelineBuilder.stream_asof_join(name, *, left_schema, right_schema, spec)`。
两者共享新建的 `asof_join_spec.py`；公共参数 `tolerance: timedelta`，
使用整数运算转成 `tolerance_micros`，禁止经过 `total_seconds()` 浮点换算。
keys/sequence 列表防御性复制；limits 使用 frozen/slots 容器。
root exports、`_native.pyi`、公开 Python 类型和 rust exports 同步；仅增加实际需要的 PyO3 入口。

capability 为 stream-only、两个 table 输入、checkpointed stateful、
requires_watermark、deterministic、replay_safe，独立 state version/layout。
`finality` 使用现有 `group_final_append_only`，明确事件时间封闭后追加；
不能标成允许 row-local 提前执行的 `per_row_final`。
`microbatch_invariant=True` 只表示成功执行且具有相同合法控制语义时逻辑结果不受切分影响，
其含义必须与现有 optimizer 一致，资源限额不承诺所有任意切分都同样成功。
ASOF lowering 的 capability gate 必须逐项验证准确 finality、支持的 state layout、端口和 stream-only
等事实，不能复制当前 inner gate 后只把 kind 改名；缺失或伪造任一必需事实都 fail closed。

symbolic 必须分析类型、右 nullable、左输出 ordering、双源 watermark lineage 和模式；
缺失/伪造 capability、batch、未支持类型在分析/编译阶段失败。
新 primitive 的 digest 包含全部有序参数，不能和 inner Join CSE 合并；
同一 ASOF digest 的多个输出共享一个物理状态 owner。
将 ASOF 作为 temporal/finality boundary，不允许 filter/projection/rolling 优化穿越后改变候选集合。
复用现有 relational DAG 编排，覆盖独立/嵌套 Join 和旁路输出；未证明的组合明确拒绝，不能静默退化。
不以支持 ASOF 为由扩展已被禁止的 matrix attachment 或 symbolic event-window 组合。

explain 明确展示 backward、闭区间、tolerance、左右 sequence、late policy、
双水位严格封闭、输出 event time、state 额度、state owner 数量及 delivery 前提。

## 5. 原生实现结构与运行时接入

已新增独立模块 `operator/asof/`，按 spec/schema、state/index、process/finalize、checkpoint 分工，
避免把全部功能继续堆入现有大型 `join.rs`。纯状态转换容易用小 oracle 单独验证。

数据结构是 key 到 `(right_time, typed_sequence)` 有序索引，
以及按 `(left_time, key, left_sequence)` 排序的 pending-left 索引。
最终关闭左行时做同 key 的 predecessor lookup，并检查 tolerance，避免生成全部区间匹配后截取一行。
保留右历史而不只存“每 key 最新一条”：后到的左行可以需要较旧的报价。

Rust 索引属于流算子的状态和时间选择，不增加第二个通用表引擎。
key 相等语义必须与 DataFusion 在支持类型上相同，使用现有 DataFusion/Arrow 表执行与物化边界；
阶段 0 必须明确 key normalization/equality、query 和输出物化分别由哪一层执行，
用有限工作区原型证明满足 sole-table-engine 约束；仅声称编码等价不足以替代这项设计审阅。
原型同时证明 typed key encoding、排序和 null 规则，不引入 Python 匹配或新的 SQL backend。
禁止照搬旧 `matched_pairs` 的全量同 key 配对再过滤路径；先用时间索引缩小到有限候选，
再执行经过审阅的批量/分段 DataFusion 处理，临时内存同样有正数上限。
不能把升级 DataFusion 或等待上游 ASOF SQL 支持作为本项必要依赖。
单次索引插入/查询为对应有序索引的对数操作；完整 handler 还包含状态 clone/编码、
输出物化和一致快照准备，按上文 O(retained state) 限制评估，不作整体吞吐承诺。

运行时新增 crate-private 的逐 ingress progress 输出分派，传入 context 和 collector，
让 ASOF 在每次相关 watermark/EOF 转换后 drain，再推进自己的 output frontier。
保留公共 `StreamOperator` trait 既有方法签名，不要求所有 external provider 增加方法。
不能只在 `apply_progress_emissions` 产生新聚合 watermark 时执行：idle/reactivation 后
单侧水位的有效推进可能没有新的聚合 emission。ASOF 必须仍能完成 finalization。

输出通过有界 collector 分块，背压和取消点覆盖大批 finalization/编码工作。
barrier capture 与 drain 不能并发形成半个状态切点：或者 pending 仍在 snapshot 内，
或者结果已按 FIFO 排在 barrier 之前，且 snapshot 已记录移除及输出序号。
若 collector 中途失败，作业失败并从已提交 checkpoint 恢复，不在同一失败状态上继续匹配。

## 6. 分阶段任务与退出门禁

以下阶段保留交付退出门禁，阶段标题的人日为规划估计，不是已耗时或验证结果。

### ASOF-00：语义、API 和兼容基线（2–3 人日）

- 在隔离 worktree 下使用共同 slug `bounded-backward-asof-join`，形成
  `.codex/artifacts/specs/`、`api-notes/`、`critiques/` 的小型 spec/API/阻塞审阅。
  该审阅已完成；最终语义以批准的 spec/API 为准，实施证据仍需分别关闭 C2/C3。
- 冻结 identity/late 优先级、支持类型、严格水位、output idle、GC、收费/工作区和异常原子性。
- 列出 direct/fan-out/nested/independent/mixed/post-stateful 的支持/拒绝矩阵；
  后续仅承诺批准的组合，其他组合必须显式分析失败，不默认扩展整个 relational DAG 的能力。
- 固定旧 inner 的规范配置字节、native fingerprint、symbolic v1/v2 digest 和 lowering、
  output schema、state metadata/segments、managed 恢复向量。保存配置默认值省略场景。
- 对逐 ingress collector 接口、DataFusion/key/index 分工和工作区预算做小型可丢弃原型；不发布 capability。
- 退出：每条 blocking finding 有合同决定和反例测试映射，独立 spec/API critique 无未解决阻塞。

### ASOF-01：纯匹配状态与 oracle（3–4 人日）

- 在新原生模块内完成 spec/schema、typed identity、admission、有序历史、pending-left 和 predecessor。
- 加独立枚举 oracle：过滤同 key 和闭区间，再按 typed tuple 取最大，未匹配补空。
- 先记录 focused RED，再实现 GREEN；覆盖 tolerance=0、边界、乱序、并列、null 和重复。
- 加固定种子的性质测试，改变 Batch 切分、Batch 内排列及合法左右交错。
- 退出：原生逻辑行/顺序与 oracle 相同，输入不可变，整数边界无溢出；尚不对外宣布完整 ASOF。

### ASOF-02：finality、资源和原生运行时（4–6 人日）

- 完成逐侧进度输出 hook、output frontier/idle 策略、EOF drain 和安全 GC。
- 实现总 rows/bytes 收费、identity-only 回收、分块输出、受控工作区、背压和取消。
- 把缺 watermark 的 source route 纳入 preflight；用真实 operator-task harness 测试，不能只调用方法。
- 加 ASOF → downstream stateful operator 的错误提前封闭反例；回归旧 Join 的控制分派。
- 退出：双侧封闭前零输出，封闭后每条已接受左行一次；超限明确失败且不改选候选。

### ASOF-03：checkpoint、恢复与交付证明（4–5 人日）

- 独立 state identity/layout、单 prepared segment 编解码、进度 wrapper 和终态校验。
- 覆盖 barrier alignment、idle/单 EOF/双 EOF 恢复、corrupt snapshot、资源重验和失败 restore 原子性。
- 在真实 managed checkpoint 发布和 sink 故障切点验证；至少有普通 at-least-once 与
  可证明 exactly-once 路由的不同预期。源 cursor 和 sink ack 必须使用同一切点。
- 退出：已提交切点恢复后的逻辑结果等于连续执行；已提交终态恢复不重新 final-flush；旧 inner 恢复不变。

### ASOF-04：Rust/Python 配置、错误与 status（3–4 人日）

- 更新 `operator/mod.rs`、`lib.rs`、`pipeline/{mod,batch,stream}.rs`、`config.rs`；
  batch 编译、raw import、typed validation、exact schema 和 fingerprint 都覆盖新 kind。
- 新增 Python immutable spec/builder/exports，PyO3 状态/错误桥接以及 public stubs。
- 接入 `StreamingFailureReason` 的实际定义位置和所有消费者，不把结构化错误退化成文本匹配。
- 原生恢复就绪后才添加独立 capability；旧 `stream_join@1` capability 保持不变。
- 退出：Python/raw JSON/Rust 编译得到一致的严格合同和 schema，错误路径稳定。
  同一规范项目经等价 project 入口的指纹相同；direct native graph 的指纹独立验证。
  现有 `with_project_fingerprint` 会把规范项目加入 graph 指纹，不能要求两类入口无条件字面相等。

### ASOF-05：symbolic、生成合同与 Studio（4–6 人日）

- 更新 `symbolic/{ops,analyzer,optimizer}.py`、`lower/{program,strategies}.py`，
  证明每 digest 单 owner、左输出 lineage、已批准组合的 finality 阻隔，以及未批准组合的明确拒绝。
- 更新项目 JSON Schema、OpenAPI、TypeScript 生成文件；核对旧 inner 子定义无漂移。
- 更新 Studio models/run-manager/status/error 和手写 TS 类型；验证项目导入、保存和状态显示兼容。
  首版不要求新增可视化 ASOF 编辑器，但已有通用编辑/往返保存不能损坏新配置。
- 增量生成后再生成一次必须零 diff；contracts 变更仅在新 kind、类型和受影响 status/error 范围内。
- 退出：native、builder、symbolic 同一交易/报价向量一致，capability spoofing 失败，Studio round-trip 无丢字段。

### ASOF-06：示例、资源验证和交付审阅（2–3 人日）

- 新增交易/报价示例，用逻辑 source watermark 展示等待与 finalization，给出 late drop/error 的区别。
  文件为 `examples/22_stream_asof_join.py`，复用逻辑名称与 SourceProvidedWatermarks。
- 更新 streaming/symbolic/API/project 文档与适用 changelog；注明普通 sink 恢复可重放。
- 执行第 8 节的资源验证，提交可复现数据生成参数与证据，不能把结果改写为吞吐承诺。
- 运行实际受影响的 focused tests、contract checks 和最终 specialist review；完整矩阵交 CI。
- 退出：完整 AC 证据、兼容向量、合同和最终 specialist review 就绪；提交 PR 并准确报告一次 CI 快照。
  完整 coverage/跨平台门禁仍由 CI 证明，pending 不等于 green；本次不合并。

依赖顺序是 00 → 01 → 02 → 03 → 04 → 05 → 06；00 后可并行编写合同适配和测试设计，
但不能让 wrapper/capability 在原生恢复未完成时对外承诺可运行能力。
建议 4–5 个 review 单元：合同；原生状态/控制；恢复；公共与 symbolic/合同；文档和证据。
拆分边界以主分支始终自洽为准，未完整的 native 代码保持内部，不暴露半成品 operator。

估算合计 22–31 人日，另预留约 25% 风险，即约 28–39 人日；这是当前源码分析后的规划估计，
不是交付承诺。最不确定的是进度/idle 下游证明、checkpoint 工作区上限及 mixed relational DAG。
阶段 0 完成后重新估算，不把“优先级 3”解释为已有前两项的技术依赖。

## 7. 测试矩阵与兼容验收

- **匹配**：右 90/100/110，左 105；T=10 取 100，T=4 未匹配，T=0 只取同时间。
  另测正好 t-T/t、不同 key、复合 key、空左/空右/空 Batch、payload 自身为 null。
- **排序**：相同 right_time 不同 sequence，整数负值/边界、UTF-8、复合 sequence、sequence 缺号；
  变更分批、批内顺序和合法交错后规范结果一致。测试不能只比较数量。
- **拒绝**：nullable schema、实际 null、错时区/单位、unsupported key/sequence、bool/float 微秒、
  JSON 上界+1、rows/bytes 为零、unknown fields、同 identity 不同 payload。
- **finality**：`W_L=t`、`W_R=t`、一侧 `t+1`、两侧 `t+1`；无初始 watermark；
  repeated watermark、idle/reactivation、单 EOF、双 EOF、最大 EventTime。直接验证输出和下游 frontier 顺序。
  增加 `C=105` 后合法输入 L@105、双方推进到 106 的 ASOF → rolling/cross-section 用例，
  确认输出 watermark 转换使该行不被误判迟到；`i64::MIN` 转换和 terminal flush 单独测试。
- **GC**：上文 pending=105 反例；tolerance 下界等号；右 EOF 后左继续；
  payload 过期但 identity 未过期；热 key 和完全无匹配；限额不能诱发错误候选。
- **原子性/取消**：Batch 后半段才出现重复或预算错误，先前行不部分入状态；
  finalization 期间 output 阻塞/取消/失败，作业与 checkpoint 状态一致，缓冲和 task 被释放。
- **恢复切点**：左行尚无候选；已有候选但水位未封闭；封闭但待 collector；
  输出已经写入而 checkpoint 未发布；checkpoint 已发布；terminal checkpoint 已发布。
  最后两类区分逻辑输出与普通 sink 物理重放，不能只做 snapshot round-trip。
- **恢复拒绝**：错 kind/version/layout/schema/config、缺 segment、坏 checksum、伪造 charge、
  duplicate identity、非法排序、损坏 output sequence、前沿与 pending-left 矛盾。
- **组合**：按阶段 0 的矩阵验证共享 fan-out、独立 ASOF、ASOF 链、与 inner 混合、旁路输出、
  post-ASOF rolling/cross-section 的批准成功或明确拒绝；右 nullable 不被优化器错误标 non-null。
- **旧 inner**：保持原有配置字节/默认值、fingerprint、输出 schema、错误向量、capability、
  symbolic v1/v2 declaration/lowering、state layout 和恢复方向测试。不可通过重录 golden“修好”漂移。

原生 integration 文件为 `stream_asof_join_{validation,properties,state,resources}.rs`；
真实任务/managed 故障测试放在 `runtime/streaming/{operator_task,runner}/tests/asof_tests.rs`。
Python 分别使用 `test_stream_asof_join_builder.py`、`test_symbolic_stream_asof_join.py` 和
`test_asof_stream_results.py`；旧 inner 向量使用独立 compatibility 测试，不建共享 conftest。
每一项行为改动先执行并记录最小 RED，再 GREEN，最后必要重构。

## 8. 验证范围、资源场景与完成定义

本次已从计划阶段进入用户授权的实施阶段。使用独立 worktree 保存源码/构建/测试证据，
保留主工作树全部无关修改；不运行默认完整回归或本地吞吐基准。

实施阶段按改动选择以下已有定向测试目标；命令是验证入口，不代表本文件证明执行通过：

```bash
# 单个原生目标，按阶段选择对应文件；不要每一步重复整个组合。
cargo test -p calc-flow --test stream_asof_join_properties
cargo test -p calc-flow --lib asof_tests
cargo test -p calc-flow --test stream_asof_join_state

# 共享运行时改动后，对旧 inner 做必要的定向兼容验证。
cargo test -p calc-flow --test stream_join_validation --test stream_join_state

# 使用对齐的 managed Python/native 构建，只测实际修改的模块。
uv run pytest python/tests/test_stream_asof_join_builder.py -q
uv run pytest python/tests/test_symbolic_stream_asof_join.py -q
uv run pytest python/tests/test_asof_stream_results.py -q

# 已有 project schema 入口只写 stdout，先导出供审阅，再更新规范文件。
mkdir -p target/asof-contracts
cargo run -p calc-flow --example export_schema > target/asof-contracts/project-v3.schema.json
diff -u schemas/project-v3.schema.json target/asof-contracts/project-v3.schema.json
```

PyO3 只修改 status/错误时使用实际受影响的 inline tests，不为此默认运行完整 Python/native 矩阵。
Python 使用定向 ruff；Rust 做必要模块的 fmt/clippy/compile/rustdoc。
Studio 只执行所改 models/validation/jobs-v3 的定向测试及必需的 TypeScript 类型/构建检查；
仅在实际修改浏览器流程时加对应 Vitest/Playwright，不默认跑整站 E2E。

合同生成使用现有 `export_schema` 和 `web-ui` 的 OpenAPI/TypeScript 导出链路。
`npm run sync:api` 当前在脚本内部把 uv cache 写死为 `/tmp/calc-flow-web-uv-cache`；
受限 worktree 中应执行同等的 `export_openapi.py` 和 `npm run generate:api` 两步，
为前一步设置指向仓库 `target/uv-cache` 的绝对 `UV_CACHE_DIR`，不能依赖外层变量覆盖 npm 内部赋值。
其余构建/cache/临时结果也放 `target/`；checked-in schema/OpenAPI/TS 仍更新各自规范路径。
在审阅并应用预期生成修改后检查重生成无漂移和 whitespace。
完整 Rust/Python/Studio、跨平台、Rust 90% 与 backend 85% coverage 及既有性能门禁交 GitHub CI。
局部测试通过不代表这些 CI 门禁通过。

资源验证固定 1000 key、总计 10 万输入行、60 秒 tolerance、64 MiB 总 state 字节上限。
必须注明左右比例、字段类型/宽度、key 倾斜、seed、事件时间跨度、水位轨迹、Batch 切分、
state rows 上限、edge/工作区预算和确切源码/依赖版本；“10 万行”不能在报告中偷偷变成每侧 10 万。

至少四条独立轨迹：正常双水位推进应正确释放；一侧暂停应等待并最终明确资源失败；
单热 key 检查候选/identity 堆积；加宽 payload 或收紧限额确保 bytes 错误可触发。
正常成功场景与 oracle 对齐；失败场景检查错误、已输出前缀正确、无错误替代匹配，
并在结束/取消后检查 live state、保留 Arrow buffers、checkpoint allocation 的归属和释放。
某些数据宽度和轨迹可能无法在 64 MiB 下成功，预期必须按固定 fixture 事先定义。
不把此场景视为吞吐或延迟基准；如果后来需要速度承诺，另开配对性能任务。

最终交付需要合同/API 阻塞项关闭、原生与 Python/symbolic 定向证据、旧 inner 兼容向量、
生成合同同步、交易/报价示例、资源验证和最终 specialist review。
本次授权包含实现、commit、push 和提交新 PR；未授权 merge。需要合并时所有 required checks
必须在最终提交上通过；本次提交后的 pending CI 应如实报告，不作为任务阻塞或通过证据。
算子逻辑每左行一次与 sink 外部 exactly-once 是不同的验收，源与 sink 的证明仍然决定交付级别。
