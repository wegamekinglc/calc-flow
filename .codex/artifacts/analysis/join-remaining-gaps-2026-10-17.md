# Join / interval_join 剩余差距分析与修改方案（2026-10-17）

前置报告：[join-interval-benchmark-perf-2026-10-16.md](join-interval-benchmark-perf-2026-10-16.md)。
本文基于新的 `main`（07dc169e，已含 #460 DataFusion 55 / Polars 2.0），叠加三个
待合并修复 #463/#464/#465（本地组合分支，冲突已解）。原始数据、脚本、profile 在
`target/analysis-2026-10-16/`（`kit/ab_rounds.sh`、`profiles/comb-*.json.gz`、
`profiles/old-join.json.gz`）。

## 1. 更正后的同机数据

空闲机器（load < 3），4 核绑核（`taskset -c 0,2,4,6`），同一套 harness（新 main），
流式为两轮交替测量。旧版 `dac0eb1a` 是 #427 之前的 DF54 构建；批处理为 DF55。

| 1M 行             | #427 之前 | main（DF55） | 三修复组合     | calc-flow-sql | DataFusion 55 | Polars 2.0 |
|-------------------|-----------|--------------|----------------|---------------|---------------|------------|
| stream join       | 82–89 ms  | 476–479 ms   | **279–280 ms** | 6.1 ms        | 8.1 ms        | 9.6 ms     |
| stream interval   | 1615–1716 | 4711–4901 ms | **1650–1788**  | 338 ms        | 346 ms        | 522 ms     |
| stream projection | 7.0 ms    | —            | 6.1 ms         | —             | —             | —          |

更正上一轮的两处说法：

- interval_join 组合版**已回到 #427 之前水平**（差 3% 以内）。剩余差距不是回退，
  是流式与批处理的结构差。同机实测是 **约 5 倍**（1.7 s 对 338 ms），不是之前拿
  CI 批处理数字对比得出的“约 3 倍”。
- 流式运行时本身的固定开销很小（1M projection 6 ms），所以流式 join 的 280 ms 中
  约 95% 是 Join 算子自身。#427 之前的 89 ms 也远不是下限。

## 2. 差距拆解

### 2.1 流式 join（1:1，64 行静态维表）

基准特征：维表推送后立刻发出 `Watermark(BASE + span)` 并结束，所以每行 quote
在准入时 `retain=false`，从不进入 state；淘汰几乎为零（profile 中 3.3 ms）。
CPU 占比（`profiles/comb-join`，组合版）：

| 组件                                                    | 占比 | 性质                   |
|---------------------------------------------------------|------|------------------------|
| key identity：`Pending::prepare_rows`（含 hash/equals） | 40%  | 每行查持久字典         |
| `finish_prepared`：`retained_rows` + `PayloadCharges`   | 20%  | 对不保留的行也逐行处理 |
| `project_admitted`：投影拷贝 + 逐行 `quantum.step`      | 9%   | 不保留也整组拷贝       |
| 输出 `materialize` + `native_pairs` + gather            | 15%  | 必要工作               |
| 其余（准入、调度、取消检查）                            | 16%  |                        |

对照 #427 之前（`profiles/old-join`）：当时 key 处理是**按批对整列 interning**
（`intern_probe_keys` / `visit_string_values`，约 15%），没有 `prepare_rows`、
`retained_rows`、投影拷贝这三项。

### 2.2 流式 interval_join（±5 s，双流，行确实保留）

CPU 占比（`profiles/comb-ij`，组合版）：

| 组件                                                          | 占比 | 性质                                   |
|---------------------------------------------------------------|------|----------------------------------------|
| 输出：`materialize` 14% + `ranges` 预检 9% + gather 6%        | 29%  | 物化必要；预检对定长列可降为 O(1)      |
| 淘汰 `evict_opposite`（其中 tombstone 8.5%，`retire` 2.5%）   | 19%  | 逐行堆 pop/push、逐行移除、逐行 retire |
| checkpoint 增量簿记：tombstone 8.5% + `PendingLog::push` 4.6% | 13%  | 计时窗口内从不 checkpoint，仍逐行维护  |
| `finish_prepared` / `retained_rows`                           | 13%  | 逐行构造 `StoredRow`                   |
| key identity `prepare_rows`                                   | 12%  | 同 2.1                                 |
| 时间窗探测 `window_by_id`                                     | 9%   | 每行 4 次二分（计数、收集各 2 次）     |

（tombstone 同时计入“淘汰”和“增量簿记”两行，合计不按行相加。）

### 2.3 批处理 interval_join

DF55 `EXPLAIN ANALYZE` 与改写实验（`kit/interval_variants.py`，结果与 oracle 一致）：

| 写法                                       | 4 分区 | 32 分区 |
|--------------------------------------------|--------|---------|
| 基准 SQL（`Timestamp(us,UTC) + INTERVAL`） | 302 ms | 310 ms  |
| `Duration` 偏移                            | 165 ms | 167 ms  |
| `Int64` 微秒                               | 157 ms | 161 ms  |

瓶颈是 arrow 的 `add_month_day_nano`：每个元素走一次带时区的日历运算，
11M 行约占一半时间。interval 的 months=days=0 时，这等价于整数加法。

## 3. 修改方案

### P0：无语义变化，按常规 implementer → reviewer 流程

**A1. `retained_rows` / `PayloadCharges` 按 retain 位图跳过**
（`join.rs` `retained_rows`，`join/admission_charge.rs` `PayloadCharges::new`）

- 现状：每 64 行块先对每行调用 `rows.row(index)`（虚调用加 `segment_index` 二分）
  读取 retain，外层循环再调用一遍，然后才过滤；这一阶段对不保留的行也要逐块 await。
- 方案：`AdmittedBatch` 已有 `retained` 位图（`admission_masks.rs` 按 64 行块算出）。
  按组判断：位图为全 0 的组整组跳过；否则只遍历置位行（`trailing_zeros` 迭代），
  并复用 #465 加的 `key_offsets` 按组解析 segment。
- 预期：join −18%（约 −50 ms）；interval 约 −4%。

**A2. `project_admitted` 对无保留行的组走零拷贝投影**
（`join/state_projection/admission.rs`）

- 现状：对每个准入组都做 `copy_projected_payload` 独立拷贝，然后逐行
  `quantum.step(8, 0).await` 回填 `StoredRow.record`。
- 约束：不能跳过投影本身，输出 gather 依赖投影后的列序。
- 方案：组内 retain 位全 0 时，用零拷贝的投影 `RecordBatch` 视图替换 payload
  （独立拷贝只是为了让保留行不钉住整个输入批）；回填循环只遍历置位行。
- 预期：join −9%（约 −25 ms）。

**B. key identity 改为“批内列式 interning + 每个不同 key 查一次持久字典”**
（`join/key_dictionary.rs` `Pending::prepare_rows`，`join/typed_key.rs`）

- 现状：#465 之后仍是每行执行“`small()` 判断 → 预付一次协作步 → 规范帧哈希 →
  pending/字典查找 → 比较”，`KeyWriter::parts` 每行被调用 3 次以上。
  基准只有 64 个不同 key，每批却做 64k 次查找。
- 方案（与 #427 之前的 `ProbeKeyInterner`、DataFusion `create_hashes`、
  Polars `HashKeys` 同一思路）：
  1. 按列类型做向量化哈希（单列 Utf8 直接哈希原始字节，基本类型哈希原值，
     多列组合），用批内 `HashTable` 加按类型列比较，为每行分配批内 id，
     同时记下每个不同 key 的代表行。
  2. 对每个代表行调用现有的 `TypedKey` 路径（规范帧哈希加 equals）解析持久 id，
     所以字典内容、重建、恢复、checkpoint identity 都**逐位不变**。
  3. 广播 `ids[row] = resolved[local[row]]`；retain 引用计数按不同 key 汇总计数，
     一次更新。
  4. 协作调度按块计费（例如每 64 行或 4096 字节一次 `flow.step`），
     取消仍在有界工作量内可见。批内哈希表经现有 `scratch_bytes` 预留出资。
- 预期：join −36%（约 −100 ms）；interval −9%。
- 风险点：超大 key（大于 4096 字节）仍走现有分块路径；批内表的内存上限
  要按行数受控。

**C. 时间窗探测缓存边界并单调推进**（`join/native_dictionary.rs` `window_by_id`，
`join/native_lookup.rs` `count_pairs` / `collect_native_pairs`）

- 现状：每行两次 `partition_point`，计数和收集各算一遍，共 4 次二分。
- 方案：计数阶段把每行的 `(lower, upper)` 写入暂存 `Vec`，收集阶段直接复用；
  同一 key 的输入时间单调时，用指针前移（galloping）代替二分，非单调时退回二分。
- 预期：interval −6%。

**F. 定长、无 null 输出的分块按行数计算**（`join/materialization.rs` `ranges`）

- 现状：interval 每 64k 批产生约 704k 输出行，超过 `max_rows=64000`，
  `flat_ranges` 要逐对累加字节。投影后的输出列都是定长、非空（u64/f64）。
- 方案：两侧投影列都是定长，且本批的源数组 `null_count == 0` 时，每行字节数是
  常量，直接按 `min(max_rows, (max_bytes - 固定开销) / row_bytes)` 切块，
  与现有字节模型保持一致；否则走现有路径。
- 预期：interval −8%。

**P0 合计预估**：join 280 ms → **约 100–110 ms**（回到 #427 之前水平）；
interval 1.7 s → 约 1.3 s。

### P1：涉及 checkpoint / 恢复语义，先 spec + critic 再实现

**D. 增量簿记改为 checkpoint 时推导**（`join.rs` `mark_retained_rows`、
`record_tombstones`、`PendingLog`；`checkpoint_v2/writer/encode.rs`、
`encode/rebase/pending.rs`）

- 现状：开启 checkpoint 时，每保留一行就写一条 upsert（克隆 `RowPayload` 和
  key Arc，HashMap 插入加链表），每淘汰一行就在 HashMap 删除或写 tombstone。
  基准的 `checkpoint_interval=24h`，计时窗口内从不 checkpoint，全是纯开销。
- 方案：每侧只维护“上次 capture 时的 `row_id` 水位”。`row_id` 每侧单调，且已持久化。
  - 准入：不写任何记录。
  - 淘汰：只有 `row_id` 低于水位的行（已进入 checkpoint 的行）才记 tombstone，
    用紧凑的 `(row_id, event_time, key)` 列表。
  - capture：upsert 由“水位以上的存活行”从 state 按 chunk 推导；
    tombstone 取自上述列表；然后推进水位。
  - 在两次 checkpoint 之间先保留又被淘汰的行不留任何痕迹，与现有合并语义一致。
- 待 spec 明确的点：
  - v2 delta 格式不变，但 delta 内 upsert/tombstone 的相对顺序会变。要确认恢复
    不依赖交错顺序；新的集合不存在冲突（同一 row 不会同时有 upsert 和 tombstone）。
  - capture 前的资金预留要改为由 state 计数。
  - checkpoint 字节不再与旧实现逐位相同，但恢复结果等价。
- 预期：interval −13%。

**E. 淘汰改为按 run 前缀批量移除**（`join/expiration.rs`、`evict_expired`、
`native_dictionary` `remove_stable`、`key_lifecycle::Retirement`）

- 现状：每行一次堆 pop/push，一次 `remove_bound`（含 native index 的逐行
  `remove_stable`），一次 `retire`（lease 克隆加引用计数递减）。
- 方案（DataFusion SHJ `prune_internal_state` 的思路）：
  - 每个 run 已按 `(time, row_id)` 有序，用 `partition_point` 求出过期前缀，
    整段移除；
  - native index 每个 key 的 run 也按时间有序，同样整段截掉；
  - 字典 retire 按 key 汇总计数，一次递减。
  - 与 D 一起做时，tombstone 顺序不再需要全局归并序，可以不用堆。
- 预期：interval −8%。

**P0+P1 合计预估**：interval 约 0.85–0.95 s，约为批处理（338 ms）的 2.6 倍。
剩余主要是 11M 行输出物化与 gather，这是增量算子固有的成本。

### P2：批处理与更长期项

**G. `timestamp ± interval` 快路径**（新增，与 `datafusion_predicate.rs` 的
`UInt64ModuloPredicate` 同一注册方式，`datafusion.rs:821`）

- 方案：analyzer/optimizer 规则把操作数类型为 `Timestamp(unit, tz) ±
  Interval(MonthDayNano)` 的 `BinaryExpr` 替换为标量 UDF。
  - UDF 按批检查：interval 数组全部满足 months=days=0，且 nanos 能被时间单位
    整除时，走带溢出检查的整数加法；否则整批回退到 arrow 原 kernel。
  - 语义按构造保证相同。基准里 interval 是 `VALUES` 产生的列而不是字面量，
    所以必须用 UDF 处理列，不能只做字面量常量折叠。
  - 测试覆盖：时区与 DST、负偏移、不可整除的 nanos、null、溢出报错类型。
- 预期：calc-flow-sql interval_join 338 ms → 约 180–190 ms。
- 可选：把同样的 months=days=0 快路径提交给 arrow-rs 上游
  （`add_month_day_nano`），DataFusion 会直接受益。

**H. 列式准入进 state（长期）**：用“批切片 + key id 列 + event_time 列 +
row_id 区间”代替逐行 `StoredRow`。这是把流式 join 从约 100 ms 推向 30–40 ms
（批处理 6 ms 加流式固有开销）的必要一步，但会改动 `Rows` trait 和大量调用点，
需要单独立项。

## 4. 建议顺序

1. 先完成 #462–#465 的 rebase 合入。#460 合并后四个 PR 都显示冲突；本地组合分支
   验证过，代码冲突只有 `materialization.rs` 的 `FlatRow.terminal` 一处，其余是
   CHANGELOG。
2. P0 的 A1+A2+B 合为一个 PR（都在准入/key 路径），C 和 F 各一个 PR。
   每个都按“红灯测试 → 实现 → 交替 A/B”的流程。
3. P1 的 D、E 先走 `cf-spec-writer` 加 `cf-critic`（涉及 checkpoint 恢复语义），
   再实现。
4. G 独立进行，与流式工作无冲突。
5. H 视 P0/P1 之后的数据再决定是否立项。
