# 流式 ASOF Join 第二轮加速开发方案

日期：2026-09-30。分析基线：`feature/accelerate-streaming-asof` 的 commit
`3a0ab5e`（`perf: Accelerate streaming ASOF joins` 与 review 修复之后）。
状态：实现完成，交付测量见 PR。本文记录问题定位、必须保持的不变量、分阶段实施项和验证
门禁；它不是性能验收记录。最终交付行为以
[ASOF 指南](../asof-join-guide.md)、[CHANGELOG](../../CHANGELOG.md) 和实际交付
记录为准。

文中耗时是调查线索，不是发布回归基线；不据此承诺达到某个毫秒值或追平 Polars。
性能结论必须按[配对比较合同](../benchmark-suite.md#revision-comparisons-and-regression-gate)
重新测量。

2026-10-01 实施状态：P0–P2 已接入，定向测试与 lint 通过；最终配对测量与审查记录见 PR。
按用户最新范围，直接实现 v3，不保留 v1/v2 checkpoint 恢复与迁移。

right 由 `HashTable<u32>` key 字典定位，批量哈希使用 DataFusion 工具，标量
探测使用相同字节域；保留独立的载荷与 identity-only 时间、sequence 列和 head。
有序驱逐访问 key 与过期前缀；较旧载荷转为身份时使用树索引，不搬移未过期列。
left 保留 Arrow chunk、key 字典、时间列、sequence 列与可选 `u32` 位置；
乱序 chunk 在线程池 lexsort，重叠 chunk 按 `(time, key, sequence)` 归并。
八种整数 sequence 使用实际 1/2/4/8 字节宽度；有序连续输入直接复制 Arrow 值区间。
单个非空 Utf8/LargeUtf8 key 加整数 sequence 使用类型化字符串探测和批次字典，
每个唯一值只编码一次，Arrow take 的临时值缓冲在分配前预留 workspace；
generic sequence 共用每批连续 Arrow binary 缓冲。8 字节批次/行引用指向唯一载荷池，输出 worker 仅持有唯一
Arrow 批次与位置索引。载荷 IPC 在 checkpoint 时按需编码。

v3 index 规范编码 key、共享编码 owner、列、批次引用及容量提示；整数列直接
复制原始 sequence 字节。计费覆盖实际容量、唯一 owner、载荷与预留 index，
接纳和前缀/驱逐投影只处理本次变化。共享 right 列和稀疏载荷池替换在线程池
预备，旧输入与新缓冲预留随 worker 存活，覆盖取消后 reset。sink 接受输出后
同步提交前缀。Rust/Python checkpoint 能力版本均为 3，算子身份仍为
`stream_asof_join@1`。性能结果须以最终源码的配对测量为准，并写入 PR 描述。

接纳仍按验证、身份构建、workspace 和容量投影分阶段执行；固定宽度列按区间计费，
整数 sequence 的 owner 增量复用每 key 计数。驱逐提交按批次和唯一编码 owner
汇总释放；无变化 tick 使用缓存最小时间，有变化 sweep 仍遍历 right bucket 和
过期前缀。PR 单独记录这些实施选择与单遍接纳、严格 `O(evicted)` 和调查耗时目标
之间的差距。

## 1. 问题与测量

### 1.1 工作负载

与 engine suite 的 `asof_join` 相同：`benchmarks/engine_comparison.py::workload`
生成的表同时作为左右两侧输入，64 个 symbol，`tolerance=0`，64,000 行一批，
两侧各在批后推进 watermark。calc-flow 使用 ready-runner 计时边界；Polars 使用
预构建 lazy plan 的 `join_asof(on="event_time", by="symbol", strategy="backward")`
加 `to_arrow()`。复现脚本位于 `target/asof-analysis/repro_gap.py`（未纳入版本库）。

### 1.2 端到端结果

机器：13th Gen Intel Core i9-13900HX，32 逻辑核；Polars 1.44.1。

| 引擎                        | 10k 行/侧 | 100k 行/侧 | 相对 Polars 单线程 |
|-----------------------------|-----------|------------|--------------------|
| Polars `join_asof`，1 线程  | 1.3 ms    | 9.4 ms     | 1x                 |
| Polars `join_asof`，32 线程 | 14.7 ms   | 16.9 ms    | 不作基准           |
| calc-flow stream（ready）   | 18.2 ms   | 222 ms     | 14x / 24x          |

较早一轮记录为 Polars 1 线程 6.6 ms、4 线程 3.9 ms，calc-flow 165–190 ms，
差距为 25–45 倍。Polars 在该规模下多线程反而更慢，因此以单线程为基准。

同一计时边界下 native stream projection 在 100k 行约 2 ms，说明 source、通道、
sink 和 Python 汇总的固定开销可以忽略，差距几乎全部位于
`StreamAsofJoinOperator` 内部：约 2 µs/输出行，Polars 约 0.05 µs/输入行。

本分支已把 10k 行从约 2.4 s 降到约 18 ms，100k 行从约 4.2 s 降到约 0.2 s，
主要消除了逐行 IPC 编码和逐 chunk 全量索引重建。剩余问题主要是行式数据模型的
常数因子，而非渐进复杂度；eviction sweep 仍是 `O(retained)`/tick 的例外。

### 1.3 耗时分解

来源：`target/asof-analysis/current_phase_profile.log` 中一次 196 ms 样本。该
日志在最终 review 修复前用临时 `Instant` 探针采集，只用于判断比例。

| 阶段                                                                | ms   | 占比 |
|---------------------------------------------------------------------|------|------|
| admission 身份构建（RowConverter、Arc 分配、HashSet/BTreeMap 查重） | 68.6 | 35%  |
| admission 载荷 IPC 编码与 SHA-256（`encode_rows`）                  | 16.9 | 9%   |
| admission workspace 预留与其他                                      | 6.6  | 3%   |
| install（逐行插入 BTreeMap）                                        | 22.1 | 11%  |
| inventory 记账                                                      | 5.4  | 3%   |
| 输出候选查找                                                        | 15.1 | 8%   |
| 输出 workspace 估算与 materialize                                   | 9.0  | 5%   |
| 未插桩：prefix 提交、eviction sweep、runtime                        | 52.3 | 27%  |

## 2. 根因

代码位置均在 `crates/calc-flow/src/operator/asof/` 下。

1. **行式、堆分配的身份模型。** `state.rs` 中 `LeftOrder` 为
   `(i64, Arc<Vec<u8>>, Arc<Vec<u8>>)`；left 为 `BTreeMap<LeftOrder, RowPayload>`，
   right 为 `BTreeMap<Encoding, BTreeMap<(i64, Encoding), Option<RowPayload>>>`。
   每行 sequence 都执行 `Arc::new(bytes.to_vec())`（两次分配）；key 驻留使用
   SipHash 加 Vec 线性比较；树比较需要解引用后 memcmp，缓存局部性差；每行
   `RowPayload` 克隆 `Arc<PayloadBatch>`，`State::attach` 每行再查一次 batch 表。
2. **逐行查重。** `admission.rs::admission_identities` 对每行执行
   `State::contains_identity`（BTreeMap 查找）和 `seen: HashSet<LeftOrder>` 插入
   （SipHash，未预分配）。输入按 `(key, time, sequence)` 有序时，查重只需相邻比较。
3. **同一行被遍历约 15 遍。** admission 依次经过 `validate_late_rows`、
   `identity_workspace`、`admission_identities`、`input_workspace`、`encode_rows`、
   `inventory_after_admission` 与 `index_length_after_admission`（right 侧各自再查
   BTreeMap、插 BTreeSet）和 `Admission::install`。输出依次经过
   `count_finalizable_keys`、`finalizable_keys`（克隆 key）、`output_attempt`
   （`state.left[key]` 与 `candidate` 两次查找）、`output_workspace`（逐行逐列
   downcast）、两个 `GatherPlan`（每行查 BTreeMap）、`materialize` 的 owned 克隆、
   `left_prefix_length`、`inventory_after_left_prefix` 和逐个 `remove`/`detach`。
4. **checkpoint 机制位于热路径。** `admission.rs::encode_payload` 对每个接纳批次
   立即 IPC 编码并由 `StateSegment::new` 计算 SHA-256。`finalize.rs::finish_progress`
   在存在可驱逐数据时执行 `state.clone()`、完整 `index_v2::encode`、SHA-256 和
   全量 `checked_inventory`；`state.rs::eviction_pending_at` 每个 tick 扫描全部
   right 行。高频 watermark 的真实流会把该成本放大为 `O(ticks × retained)`。
5. **输出拼装逐行。** `output.rs::GatherPlan` 在 64 个 key 交错时，right 侧连续段
   几乎都只有 1 行，`MutableArrayData::extend` 退化为逐行逐列拷贝。
6. **其他常数项。** 每行 4 次以上 `check_cancelled()` 阻碍向量化，设置 deadline
   时还会每行读取 `Utc::now()`；记账在每一步都执行 checked 算术。

语义上确有 Polars 不承担的成本：双侧 watermark 最终性、按
`(left_time, left_key_tuple, left_sequence_tuple)` 的全局输出顺序、重复身份拒绝、
fail-closed 的有界内存和可 checkpoint 状态。在输入有序的常见情形下，这些都可以
做成线性检查，因此合理目标是 Polars 单线程的 2–3 倍，而不是 20 倍以上。

## 3. 必须保持的不变量

1. **选择语义。** 左行时间 `t`、容差 `T` 只匹配同 key 且
   `t - T <= right_time <= t` 的候选，取最大 `(right_time, right_sequence)`；
   区间两端包含；right 行可服务多个 left 行。
2. **最终性。** left 行只有在两侧 watermark 都严格大于 `t` 或对应侧已结束时才
   输出；输出 watermark 为 `C - 1 µs`，不产生下溢 watermark。
3. **输出契约。** 列顺序、前缀命名、nullability、全局行顺序保持不变；物理批次
   边界不属于结果契约。
4. **admission 原子性。** schema/null、late、重复身份和资源预检失败时不留下部分
   状态；被取消的 future 不暴露半接纳的行；重复身份仍按原 reason 失败。
5. **资源边界。** `max_state_rows`、`max_state_bytes` 和等额 workspace 上限继续
   fail-closed；任何估算都必须是实际分配的上界；不得通过放宽限制、扩大队列或减少
   输出工作换取收益。
6. **eviction 保守性。** right 历史只在 `r + T < min(left_future_bound,
   earliest_pending_left)` 时释放；identity-only 条目的保留规则不变。
7. **checkpoint 与恢复。** 快照确定且可规范化；sink 接受输出后才提交前缀；
   仅支持 v3 快照，拒绝 v1、v2；恢复后的 gauges 与重算 inventory 一致。
8. **公开表面。** Python API、`StreamAsofJoinSpec`、状态 reason 和 status 字段
   不变；DataFusion 仍是 SQL 与表达式引擎。

## 4. 目标与非目标

目标（均为调查目标，需配对测量确认）：

- 阶段 1 后，suite workload 100k 行/侧降到约 100 ms。
- 阶段 2 后，降到 20–30 ms，即不超过 Polars 单线程约 3 倍。
- eviction sweep 与每 tick 检查的成本与驱逐量成正比，而非与保留量成正比。

非目标：追平 Polars；在阶段 1、2 引入并行；改变 ASOF 语义或公开 API；为基准
特化 `tolerance=0` 或自连接。

## 5. 实施方案

每项先写能记录预期失败的聚焦测试，再实施，符合
[code-style](../../.agents/skills/code-style/SKILL.md)。

### 阶段 0：度量基础

- **P0.1 Polars 外部参考。** 在 `scripts/benchmark_suite/catalog.py` 的
  `CAPABILITIES["polars"]` 加入 `asof_join`，在
  `benchmarks/engine_comparison.py::_polars_plan` 实现 backward `join_asof`，
  同步 [benchmark suite](../benchmark-suite.md) 支持矩阵。新增外部 case 属于
  new coverage，不影响已有配对门禁。另在诊断脚本中记录 1 线程参考值。
- **P0.2 端到端 Rust 基准。** 新建 `stream_asof_e2e` target，而不是修改
  `stream_asof_perf`，以免改变现有 target 的 workload fingerprint。用例：
  `admit_settle_100k`（两侧各 100k、64 key、64k 批，计时包含 `process_data` 与
  settlement）、`eviction_ticks`（小批且每批推进 watermark，持续驱逐）、
  `out_of_order_within_watermark` 和 `composite_key`。报告保留测量线程的分配
  总量与峰值；该计数不包含 `spawn_blocking` 输出线程，不能代表算子总内存。
- **P0.3 分段诊断。** 用已有 `tracing` 依赖在批次粒度（不在行粒度）记录
  admission、output、prefix 提交和 sweep 的 debug span，取代临时 `Instant` 探针。
- **P0.4 随机对照测试。** 扩展 `crates/calc-flow/tests/stream_asof_join_properties.rs`，
  用 proptest 生成 watermark 内乱序、重复身份、组合 key、字符串 sequence、
  中途 checkpoint/restore 和取消，并与朴素 oracle 比较完整输出与 status。

### 阶段 1：低风险优化（不改状态格式与 checkpoint wire）

- **P1.1 prefix 一遍提交。** `finalizable_keys`、`output_attempt` 和
  `commit_prefix_output` 合并为一次有序遍历：同时取得 left payload、候选行、
  index 长度差和 inventory 差；sink 接受后用 `BTreeMap::split_off` 一次切除前缀；
  batch 引用计数按批次汇总后递减。取消 key 克隆与重复查找。
- **P1.2 增量 eviction。** 预检阶段只计算待驱逐集合，不克隆 `State`；预检通过后
  同步、不可失败地原地应用；`deferred_index_len` 用算术更新，不再重编码。维护
  right payload 时间直方图（如 `BTreeMap<i64, u32>`）与 identity-only 最小时间，
  使 `eviction_pending` 变为 `O(log n)`；每个 bucket 用 `split_off` 移除前缀。
- **P1.3 延迟载荷 IPC。** `PayloadBatch` 以 `OnceLock<StateSegment>` 缓存编码，
  在 `prepare_checkpoint_async` 的 blocking worker 中生成；admission 改用 IPC
  大小的算术上界记账（schema 消息、对齐后的各 buffer、framing）。增加测试证明
  上界不小于实际编码长度，沿用 `workspace.rs` 中现有上界测试的形式。
- **P1.4 向量化输出拼装。** right 侧改用 `arrow::compute::interleave`，left 侧
  连续段用 `slice` 后 `concat`；只传 `(source, row)` 索引与
  `Vec<Arc<RecordBatch>>`，不再克隆 `RowPayload`；`GatherPlan` 的 BTreeMap 改为
  小型向量或哈希表。workspace 估算按连续段和变长列 offsets 差计算。
- **P1.5 合并 admission 遍历。** late 过滤、身份、workspace、index 长度和
  inventory 增量在每批一次循环内完成；`check_cancelled()` 改为每 1,024–4,096 行
  一次；批量哈希使用 DataFusion 的 hash 工具，并按行数预分配容量。
- **P1.6 无堆 sequence。** 单个整数 sequence 用定长内联编码（例如 `[u8; 9]`），
  其他情况用小缓冲内联；编码字节保持与现有 row format 一致，保证 checkpoint 兼容。

### 阶段 2：列式状态重构

- **P2.1 key 字典。** 以 `datafusion::common::hash_utils::create_hashes` 批量计算
  哈希，用 `hashbrown::HashTable` 把 key 驻留为 `u32`，key 的 row-format 字节只存
  一份。若需将 `hashbrown` 设为直接依赖，版本与 lockfile 中 DataFusion 所用一致
  并通过 `cargo deny`。输出排序比较驻留的 key 字节，不比较到达顺序分配的 id。
- **P2.2 right 按 key 有序列。** 每个 key 一个
  `RightRun { times: Vec<i64>, sequences, refs: Vec<(u32, u32)>, head: usize }`，
  按 `(time, sequence)` 排序。顺序到达时直接追加，一次比较同时完成查重；watermark
  内乱序行用二分插入。eviction 只推进 `head`，按阈值周期性压缩。
- **P2.3 left 保留为 Arrow chunk。** 每个 chunk 保存原批次与 `key_id`、`time`、
  sequence 列；admission 时一次向量化检查是否已按输出顺序排列。finalize 时用
  `partition_point` 截取 `time < C` 的部分，已排序 chunk 做 k 路归并，未证明有序
  的 chunk 才调用 `lexsort_to_indices`。
- **P2.4 候选探测。** 在连续的 `times[head..]` 上做 `partition_point`；left 在
  同一 key 内按时间有序时改用逐 key 归并指针，摊还 `O(1)`。
- **P2.5 类型化快速路径。** 单个 key（Utf8/LargeUtf8 或整数）加单个整数
  sequence 走类型化存储；组合 key 或字符串 sequence 使用每批一个
  `arrow::row::Rows` 连续缓冲区，不再每行一个 `Arc<Vec<u8>>`。
- **P2.6 按批记账。** 行数乘固定常量，加上各列 buffer 与变长列 offsets 差；
  `Vec` 容量增长按实际 capacity 计费，保证计费仍是上界。
- **P2.7 列式 checkpoint index v3。** 编码 key 字典、每 key 数组和 left chunk
  引用，编码接近内存拷贝，长度增量用算术维护。state/layout/accounting version
  升级为 3；仅恢复 v3 快照。fingerprint 与恢复校验规则不放宽。

### 阶段 3：可选扩展

- 按 key 哈希分区并行 admission 与探测，最后按
  `(time, key bytes, sequence)` 归并输出。
- 小 chunk 在当前任务内直接拼装，避免 `spawn_blocking` 往返。

## 6. 实施顺序与预期

| 编号 | 工作项              | 主要文件                                         | 预期收益（100k）   | 主要风险             |
|------|---------------------|--------------------------------------------------|--------------------|----------------------|
| P0   | 度量与对照测试      | `benches/`、`tests/`、`scripts/benchmark_suite/` | 无，建立证据       | fingerprint 变化     |
| P1.1 | prefix 一遍提交     | `finalize.rs`、`finalize/prefix.rs`              | 15–25 ms           | 前缀提交原子性       |
| P1.2 | 增量 eviction       | `state.rs`、`finalize.rs`                        | 15–25 ms 及每 tick | 驱逐保守性           |
| P1.3 | 延迟载荷 IPC        | `admission.rs`、`checkpoint.rs`、`codec.rs`      | 约 15 ms           | 计费上界             |
| P1.4 | 向量化输出拼装      | `output.rs`、`finalize.rs`                       | 5–10 ms            | 空值与类型覆盖       |
| P1.5 | 合并 admission 遍历 | `admission.rs`、`workspace.rs`、`mod.rs`         | 20–30 ms           | 失败路径不留部分状态 |
| P1.6 | 无堆 sequence       | `state.rs`、`admission.rs`                       | 5–10 ms            | 编码兼容             |
| P2   | 列式状态重构        | `state.rs`、`admission.rs`、`checkpoint/`        | 降到 20–30 ms      | v3 恢复与乱序路径    |

各项收益来自 1.3 节的分段比例估算，彼此有重叠，不能简单相加。建议按
P0 → P1.1 → P1.2 → P1.5 → P1.3 → P1.4 → P1.6 → P2 顺序实施，每项独立提交并保留
配对测量证据。

## 7. 验证与退出门禁

1. **正确性。** 现有 `crates/calc-flow/tests/stream_asof_join_*.rs`、
   `stream_asof_inner_compatibility.rs`、模块内单元测试和 P0.4 随机对照测试全部
   通过；输出与 oracle 逐行一致，status 计数一致。
2. **恢复。** 覆盖 v3 快照恢复、旧版本与损坏快照
   拒绝和中途 checkpoint 后继续输出；`stream_asof_join_restore_corruption.rs`
   不放宽。
3. **资源。** `stream_asof_join_resources.rs` 覆盖紧限制下的 fail-closed；新增
   上界测试证明延迟 IPC 与按批计费不低于实际分配。
4. **性能证据。** 引擎与 Rust 基准按两轮十对 AB/BA 配对合同比较；一个 build 的
   结论不迁移到后续 source 或 build。阶段 2 完成后再评估放宽
   `STREAM_ASOF_MAX_ROWS`。
5. **覆盖率与检查。** 保持 Rust 90% 行覆盖率下限；本地只运行受影响模块的格式化、
   lint 与定向测试，完整回归交给 CI。
6. **文档同步。** 行为或计费说明变化时更新
   [ASOF 指南](../asof-join-guide.md#bounded-state-and-workspace)、
   [benchmark suite](../benchmark-suite.md)，必要时更新
   [runtime envelope](../runtime-envelope.md) 与 CHANGELOG。

## 8. 风险与缓解

- **输出顺序。** key id 按到达顺序分配，不能直接用于排序；排序必须比较驻留的
  row-format key 字节，并由 oracle 测试覆盖字符串与整数 key。
- **乱序输入。** 快速路径假设有序；watermark 内乱序必须走二分插入或排序回退，
  由 `out_of_order_within_watermark` 基准与随机测试同时覆盖。
- **计费上界。** 延迟 IPC 与按批计费改变了计费来源，必须有上界测试；上界过松会
  让紧限制场景提前失败，需要在资源测试中同时检查不过度保守。
- **格式恢复。** v3 index 需要确定性编码、fingerprint 校验和容量上界预检；
  旧版本和损坏快照必须拒绝恢复，不得静默重建。
- **收益归因。** 基准噪声较大（同配置样本相差 10–20%），任何单项收益都需要配对
  测量，不能用独立样本或历史二进制作为基线。
