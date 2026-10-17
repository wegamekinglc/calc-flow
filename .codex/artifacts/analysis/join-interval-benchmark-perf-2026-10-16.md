# Benchmark join / interval_join 性能调研（2026-10-16）

范围：`engines/*/{calc-flow-stream,calc-flow-sql}/{join,interval_join}`。
参考实现：DataFusion 55.0.0（tag `55.0.0`）、Polars 2.0.0（tag `py-2.0.0`），
对照基线 DataFusion 54.0.0。原始数据、脚本和 profile 在 git-ignored 的
`target/analysis-2026-10-16/`（`kit/`、`profiles/`、`ci/`）。

## 结论摘要

1. 当前 `main` 上流式 join 慢，主要原因是两次**已确认的回退**，并非一直就有的设计差距：
   - #427（`aaf793a3`，Join key dictionary 交付）让流式 `join` 慢了 **5.1 倍**
     （1M：约 95ms → 约 510ms），interval_join 慢了 **27%**。CI 当时的配对门禁已判定
     `regression`（1M **+596%**，10M **+587%**，run `38718611339`）。
   - #452（`9c811215`，chunked Join state + batch selections）让流式 interval_join
     又慢了 **2.1–2.2 倍**（1M：约 2.4s → 约 5.1s）。CI 从未报出这次回退。
2. CI 门禁有盲区：完整基准只比较 `HEAD` 与第一个父提交。#427 那次运行报了红，
   但下一次运行的 baseline 已经包含回退，于是转为 `no-confirmed-regression`；
   #452 不是任何一次运行的「最后一个合并」，因此从没被单独测过。
3. 即使回到回退之前，流式路径相对批处理引擎仍慢 6–10 倍。原因是结构性的逐行开销：
   逐行 async/预算检查、逐行取消检查、基于堆的逐行淘汰、逐行 tombstone，
   以及逐对的输出字节预检。DataFusion SHJ 和 Polars streaming 都是按批或 morsel
   的粒度处理这些事情。
4. 批处理 interval_join：calc-flow-sql 与 DataFusion 持平，且优于 Polars。但基准 SQL
   用的是带时区 Timestamp 加 `INTERVAL`，单这一步就约占 650ms CPU（总计约 1.35s）。
   改用 `Duration` 偏移，同样结果下快约 2 倍。
5. 升级参考引擎：DataFusion-Python 55 的 batch join 约快 1.6 倍，interval_join 快
   8–10%；Polars 2.0 与 1.44 基本持平。升级不会改变「流式路径是瓶颈」这个结论。

## 1. 现状（CI，`b341912c`，run `38903277250`，4 vCPU runner）

| Rows | Scenario      | Stream ms | Calc Flow SQL ms | DataFusion ms | Polars 32T ms | Stream/SQL |
| ---- | ------------- | --------- | ---------------- | ------------- | ------------- | ---------- |
| 100k | join          | 72.5      | 2.8              | 3.1           | 3.7           | 25.6x      |
| 1M   | join          | 679.0     | 14.4             | 16.8          | 17.0          | 47.1x      |
| 10M  | join          | 7219.5    | 111.8            | 120.9         | 149.1         | 64.6x      |
| 100k | interval_join | 619.8     | 60.3             | 68.0          | 105.9         | 10.3x      |
| 1M   | interval_join | 6578.8    | 733.9            | 784.0         | 1156.0        | 9.0x       |

10-14 的报告（`dac0eb1a`）中，1M 流式 join 为 104ms、interval_join 为 2173ms。
今天的 baseline 和 candidate 都约为 677ms（双方都已包含回退），所以配对门禁显示「无回退」。

## 2. 回退定位（本地 A/B，同一 harness，`taskset -c 0,2,4,6`）

CI release wheel 加上本地 `cargo rustc --release` 的未 strip 构建；每个点 warmup 1 次。

| Build                | Commit                            | join 1M ms   | interval_join 1M ms |
| -------------------- | --------------------------------- | ------------ | ------------------- |
| dac0eb1a (wheel)     | before #427                       | 92-115       | 1976                |
| aaf793a3 (wheel)     | #427 merge                        | 510          | 2506                |
| 718ab62f (local)     | before #452                       | 509          | 2261-2518           |
| 9c811215 (local)     | #452                              | 514          | 4757-5319           |
| 40cddf67 (local)     | before #453                       | 514-540      | 4976                |
| 6ad1e957 (wheel)     | #453                              | -            | 5039                |
| b341912c (wheel)     | current CI candidate              | 514-599      | 5567-5819           |

718ab62f 与 9c811215 交替运行两轮：2518/2261 vs 5319/5086 ms。

### 2.1 #452：interval_join 翻倍

Profile（`profiles/ij-718ab62f` vs `ij-40cddf67`，tokio 线程 CPU）：#452 之后
`materialization::flat_row` 自身占 34%，`Selection::pair` 占 23%；#452 之前两者都不在前列。

- `crates/calc-flow/src/operator/join/materialization.rs:173` `flat_ranges`：
  #452 把 DAL-303（#324）的「按稠密下标的 `Vec<Option<FlatRow>>` 缓存」改回了
  `BTreeMap<usize, FlatRow>`（注释给的理由是 retained chunk 坐标可能有空洞）。
  于是每个匹配对要做两侧的 `entry` 和 `rows[&index]`，也就是 4 次 B 树操作。
- `matches.rs:135` `Selection::pair`：fan-out 情况下（每个输入行匹配 11 行）
  `Incoming::Runs` 每次访问都要做一次 `partition_point` 二分，外面还隔着 `&dyn Matches`
  的虚调用；而 `iter()` 只是 `(0..len).map(pair)`。
- interval_join 每个 64k 输入批产生约 704k 输出行，超过基准的
  `EdgeBudget(max_rows=64000)`，所以 `single_chunk_fits`（`:156`）会失败，
  每一对都要走 `flat_ranges`。流式 `join` 是 1:1 匹配，走快路径，因此不受 #452 影响。

### 2.2 #427：流式 join 慢 5 倍

Profile（`profiles/join-40cddf67`）中，热点组件全部是 #427 新增的文件：

- `key_dictionary.rs:657` `Pending::prepare_rows` 及 `typed_key.rs:78/105`
  `TypedKey::hash/equals`，合计约占 join CPU 的 **44%**。对 1M 行，每行都要经过
  `resolve → hash → find → dictionary.find` 多层 async；每个 key 字节块都要
  `flow.step(...).await` 一次；哈希用的是流式 `CanonicalStream` 按字节喂数据。
  实际只有 64 个不同的 key。折算约 265ns/行，而向量化哈希加查表约 10ns/行。
- `checkpoint_v2/projected_copy.rs`（e884adc3）`copy_column` 约占 21%，其中 **75%**
  是 `GatherStop::check()`。`contiguous_runs`（`:394`）、`slice_runs`、`run_end`（`:425`）
  每行调用一次 `check()`，里面会调 `CancellationToken::is_cancelled()`；
  tokio-util 0.7.18 的实现是 `node.inner.lock().unwrap().is_cancelled`（std Mutex）。
  设置了 deadline 时，每行还要多一次 `Utc::now()`。

### 2.3 回退前的结构性开销（interval_join，718ab62f）

- `evict_opposite`（`join.rs:6952`，由 `on_ingress_progress` 触发）约 31%：
  其中 `expiration::Storage::sift_down`（逐行二叉堆）11%，`record_tombstones`
  （`join.rs:7114`）加 `PendingLog::push`（逐行 checkpoint delta）约 13%。
- key dictionary 约 27%，物化约 11%，`native_pairs` 约 6%。

## 3. 参考实现对照

| Concern            | Calc Flow stream join (now)                  | DataFusion 55                                    | Polars 2.0                                              |
| ------------------ | -------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| Key hashing        | per-row async, byte-stream hasher            | batch `create_hashes` over key columns           | batch `HashKeys` per morsel                             |
| Key equality       | per-row `TypedKey::equals` with flow steps   | `equal_rows_arr`; typed single-column fast path  | hash table probe on row-encoded / typed keys            |
| Yield/cancel       | per row/chunk `flow.step`, per-row `check()` | once per batch via tokio coop budget             | at morsel boundaries                                    |
| State layout       | `StoredRow` per row + chunks                 | one `RecordBatch` buffer + `PruningJoinHashMap`  | build `DataFrame` + partitioned hash tables             |
| Expiration         | per-row heap + per-row tombstones            | sort-order prune: slice buffer front             | n/a (finite inputs)                                     |
| Range predicate    | per-row window over native index             | `apply_join_filter_to_indices` on index arrays   | `FusedPredicate::retain_matches` on morsel-sized chunks |
| Interval output    | per-pair index gather after preflight        | `build_batch_from_indices` (take)                | `range_join`: search_sorted + contiguous subslice copy  |
| Output chunking    | per-pair byte preflight (`flat_ranges`)      | row count (`batch_size`, output coalescer)       | row count (`get_ideal_morsel_size`)                     |

代码位置：

- DataFusion 55 `physical-plan/src/joins/symmetric_hash_join.rs`：
  `join_with_probe_batch`（:1242）、`update_internal_state`（:1485，`concat_batches`
  加 `update_hash`）、`prune_internal_state`（:1555，`input_buffer.slice` 加
  `prune_hash_values`）；`physical-plan/src/coop.rs:144-162`：每次 `poll_next` 调一次
  `poll_proceed`；`joins/utils.rs:2269` `equal_rows_single_col`（55 新增的按类型单列
  等值快路径）。
- Polars 2.0 `polars-stream/src/nodes/joins/range_join.rs:302`
  `compute_and_emit_task`：对已排序的 point 侧做 `search_sorted`（:356），再
  `subslice_extend` / `subslice_extend_repeated`（:391）整段拷贝；
  `equi_join.rs:92` `FusedPredicate::retain_matches`：在 morsel 大小的块上先对候选下标
  向量化求谓词并压缩，再 gather payload；`equi_join.rs:1172`：按 morsel 行数 flush 输出。
  另外 2.0 新增了 streaming `merge_join.rs`（有序输入）和 `runtime_filter.rs`（向扫描
  下推 min/max 和 bloom 过滤，与内存流场景关系不大）。

## 4. 批处理 interval_join

DataFusion 54 `EXPLAIN ANALYZE`（1M，4 分区）：`HashJoinExec` 约 450ms，
`ProjectionExec`（`event_time + INTERVAL`，作用在 11M 行上）约 646ms，
`RepartitionExec` 约 200ms。`Timestamp(us, "UTC") + IntervalMonthDayNano` 会逐元素
做时区感知的日历运算。

`kit/interval_variants.py`，DataFusion-Python 54，均与 oracle 的 11M 对完全一致
（测量时有后台编译，只能看方向）：

| Variant          | 32 partitions ms | 4 partitions ms |
| ---------------- | ---------------- | --------------- |
| suite INTERVAL   | 591              | 429             |
| Duration offsets | 253              | 219             |
| Int64 micros     | 292              | 225             |

当 `months == 0 && days == 0` 时，把 `timestamp ± IntervalMonthDayNano` 字面量改写成
`Duration` 是精确的（不涉及日历分量，与时区无关），可以作为 calc-flow 的逻辑改写规则，
也值得提给 DataFusion 上游。如果改基准 SQL，calc-flow-sql 和 DataFusion 两个后端会
同时受影响，属于基准契约变更，需要同步更新 catalog 和 validator。

## 5. 引擎升级的影响（本地，1M，4 CPU，5 次中位数，有噪声）

| Backend          | join ms           | interval_join ms  |
| ---------------- | ----------------- | ----------------- |
| DataFusion-Py 54 | 11.6 / 16.3       | 356 / 385         |
| DataFusion-Py 55 | 6.8 / 7.4         | 332 / 344         |
| Polars 1.44.2    | 9.1 / 9.2         | 682 / 677         |
| Polars 2.0.0     | 11.1 / 9.5        | 665 / 563         |

calc-flow 目前固定 `datafusion =54.0.0`、`arrow-data =58.3.0`；DataFusion 55 依赖
arrow 59.2，升级要连带迁移 arrow。

## 6. 建议（按优先级）

P0：修复回退

1. 撤回 #452 中 `flat_ranges` 的 `BTreeMap` 缓存：恢复稠密 `Vec`（如有空洞，就按
   chunk 内的稠密局部下标）；`Matches` 提供顺序游标迭代器（沿 runs 前进），不要每次
   `pair(i)` 都二分。预期 interval_join 回到约 2.3–2.5s。
2. 更彻底的做法：flat schema 不再逐对预检输出字节。每侧按列一次性算出行字节（定长
   取常数，变长取 offsets 差），再用前缀和切块；投影后全是定长列时，直接用
   `max_bytes / row_bytes` 换算行数。
3. key dictionary 向量化：整列算哈希（DataFusion `create_hashes` / hash_utils），
   一次遍历 hashbrown 表并做按类型的列比较；把 `flow.step` 和让出 CPU 提到批或
   morsel 粒度。
4. `projected_copy` 中把 `check()` 从逐行改为按 run 或每 N 行（如 4096）检查；
   热路径不要调 `CancellationToken::is_cancelled()`（它会加锁），改读一个由取消回调
   置位的 `AtomicBool`。
5. CI 门禁：除了和第一个父提交比，再和滚动参考（上一个 release tag 或上一次全绿的
   nightly head）比；已确认的回退要有人确认后才能成为新 baseline；涉及性能的 PR 合并时
   至少跑 `engines/1000000/calc-flow-stream/{join,interval_join}`。

P1：缩小与 DataFusion/Polars 的结构差距

6. 淘汰：按时间有序的 chunk 整段淘汰，代替逐行堆（参照 DataFusion
   `prune_internal_state`）；tombstone 记录成 `(chunk, range)`，不再逐行记录 identity。
7. 区间匹配：state 按 key 内 event time 有序，探测时用二分求 `[t-lower, t+upper]`，
   得到连续区间后整段拷贝（Polars `range_join` 的 `subslice_extend`）。ASOF 已有同类
   实现（DAL-342 的 contiguous range copy），可以复用。
8. 时间界谓词改成在候选下标数组上分块向量化求值，然后再 gather（Polars
   `FusedPredicate` / DataFusion `apply_join_filter_to_indices`）。

P2：批处理路径

9. 加 `INTERVAL`（months=days=0）到 `Duration` 的改写规则，或调整基准 SQL 的写法。
10. 评估 DataFusion 55 升级（batch join 约 1.6 倍；需要 arrow 59）。
11. 长期：原生 band join 算子（等值 key 加时间窗，按 key 排序归并加整段拷贝），
    代替现在的「11 次等值 join」展开写法。

## 复现

- A/B：`target/analysis-2026-10-16/kit/ab_case.py`（复用 `EngineCase`）。把 site 目录
  放在 `sys.path` 最前，用 `taskset -c 0,2,4,6` 绑核。
- 构建：`kit/build_sha.sh <sha>`（`git archive` 后用种子 target 做
  `cargo rustc --release`，增量约 8.5 分钟）。
- Profile：`kit/profile_case.sh`（打过补丁的 samply 在 `tools/bin/samply`；
  **taskset 要包在子进程上，不能包在 samply 外层**，否则拿不到 mmap/lib 信息）
  和 `kit/analyze_profile.py`（按 `threadCPUDelta` 加权统计 self/inclusive）。
