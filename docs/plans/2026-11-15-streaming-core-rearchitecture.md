# 流式核心重构：架构修改与实现方案

日期：2026-11-15。状态：执行中；全部阶段在单一集成 PR #619 中交付，P1 已并入，≤20 µs 门槛已移到 P2，见执行记录。

本文记录对 Calc Flow 流式引擎的一次整体 review、由此确定的目标架构，以及分阶段的实现方案。
它是后续各阶段规格和逐任务实施计划的统一入口；各阶段的逐任务计划另行编写，本文不包含。
当前实现契约仍以 [runtime envelope](../runtime-envelope.md)、
[streaming guide](../streaming-guide.md) 和源代码为准；本文描述目标状态，已实施的部分见执行记录。

review 基线为 `main` 的 `39cab2fb`，文中的 `路径:行号` 均对应该提交。
本次 review 只阅读了源码和仓库已有的测量产物，没有构建，也没有新增测量。
文中所有“目标”和“验收门槛”都是待各阶段验收的估计值，不是已取得的结果。

## 执行记录

- 交付方式（2026-11-16 用户决定）：整个方案只保留一个[集成 PR #619](https://github.com/wegamekinglc/calc-flow/pull/619)，
  所有阶段在同一分支上累积，最终按整体重构效果评估。原 #617、#618 已作为被集成 PR 取代而关闭。
- P1 `StreamSession`：原 PR #618 的提交已并入 #619，交付公开的同步 push API，
  包括 Rust `StreamSession`、`SessionWatermark`、`SessionOutput` 和 Python `TableExpr.session`、`Program.session`。
  现有 handler 由适配器在调用线程上按拓扑序跑完，没有 source、sink 和进度任务。
  准入范围是单个动态输入，rolling（不含 RSI/ATR/MACD 指标 layout）、表达式和 union 节点，无 checkpoint。
  规格为 `.codex/artifacts/specs/stream-session.md` 第 2 版，critique 和 review 是
  `.codex/artifacts/critiques/` 与 `.codex/artifacts/reviews/` 下的同名文件。
- P1 测量（`benchmarks/stream-session/p1-20261115-summary.md`，每个用例 30 对 AB/BA）：
  预登记门槛（1 行 append 的 session p50 ≤20 µs）未达到。强制 GC 下为 152.9 µs，
  常规 GC 的诊断运行为 90.5 µs，进程内热循环为 18.6 µs。
  相对 runner 的配对加速：1 行和 99 行 append 为 3.5–5.4 倍，6,400 行 64 实体为 1.9 倍；
  runner 对照的置信区间跨 1.0，没有回退。
  强制 GC 下的阶段归因是 Python 参数处理 18.9 µs、原生调用 126.9 µs、结果装配 7.0 µs；
  剩余成本主要在原生算子 dispatch 内，并随缓存状态大幅变化。
- 按 7.2 的停止规则定位成本后，用户于 2026-11-15 决定：P1 作为严格改进并入整体方案；
  ≤20 µs 门槛移到 P2，与每次调用的分配预算、rolling 和 projection 的同步算子 trait（4.1）一起验收。
- 决定 5：原 PR #617 的提交已并入 #619，让便捷流默认关闭 checkpoint，
  有序事件时间输入在每个推进的数据批之后发出 watermark，不再等 100 ms 定时器；
  显式 `StreamingRunner` 仍默认开启 checkpoint。配对测量已完成并归档在
  `benchmarks/stream-convenience-defaults/`：2 批用例按维护规则 improved（−39.24%），
  32 批用例为 inconclusive（−9.33%）；两项默认变化合并测量，不能分别归因。
- P1 NumPy 路径与单源 runner 内联已实现并处于性能验收阶段（取代草稿 PR #616），结果见后续执行记录。
- P2.1（完成；基线登记 `bb201632`）：五个 rolling dispatch 证据用例和 native 配对控制器已并入；
  10 轮基线在 14.8 秒内完成，原始样本、二进制身份和协议归档于 `benchmarks/rolling-kernel-v2/`。绝对基线为
  G1 6.264 µs、G2 19.577 ns/window-row、G3 452.568 µs；两臂使用同一二进制，1.00x 不是性能结论。
- P2.2（完成；数值实现 `16ae0d03` 经 `8434b862` 集成，精确 oracle `a7511438`）：
  rolling sum/mean、离散度和 pair 输出改用唯一的 compensated sliding 算术，Rust/Python/IR 的数值选项及
  project-v3 对应 enum/property 已删除；整数 sum 保持精确检查，未改递推保持不变。旧 profile marker 和
  先前 kernel identity 的 checkpoint 拒绝恢复。`python/tests/test_rolling_numeric_contract.py` 已加入
  exact `Fraction` oracle、误差界与特殊值分类检查，以及 batch/session 逐位一致性断言。
  本项不包括 P2.3 的 slab、undo log 或 layout 7。
  P2.2 的 10 轮 release 证据保存在 `benchmarks/rolling-kernel-v2/p2-2-08b17812.*` 和
  `p2-2-be094d3f.*`：初版大批量 dual-mean 为 0.83x，按输出关闭无用 dispersion 后仍为 0.89–0.90x。
  本步骤规格没有性能 Gate；该回退未隐藏，留作 P2.8/P2.9 的固定成本、run/full-tick 优化输入，最终 G1–G3 仍须验收。
- P2–P7 规格已全部并入：rolling 数值与内核、checkpoint v4、keyed state 与 join、并行执行与 window pane。
  实现按单一 PR 内的提交序列推进，不再建立分阶段 PR。
- P2.4 部分交付：Float64/Float32/Int64/UInt64 的 lag/delta 已进入 typed plan，保留 null/NaN、periods=0、
  checked arithmetic、溢出错误与 checkpoint 历史恢复；rank/quantile 也有 typed window state 和 ties/invalid-row
  parity，kernel registry 已同步；Utf8 lag 使用 typed `Option<String>` 历史和 `StringBuilder`；Duration
  argmax/argmin/unique_count 使用 ordinal、event-time expiry 和 typed candidates/counts；TimestampMicrosecond lag
  使用 signed `i64` 历史并恢复 UTC/naive 类型。rolling 703 项、symbolic optimizer 11 项和 clippy 通过。
  其余 temporal/nested lag、剩余 typed groups 与 general `ScalarValue` 路径删除仍未完成。
- P2.4 继续补齐 Date32/Date64/Time64Microsecond/DurationMicrosecond lag，复用 signed history 和输出类型恢复；
  rolling 707 项和 clippy 通过。其他 temporal/nested 类型仍需单独验收。
- P2.5 部分交付：MACD 三组件通过 typed group/readout 复用原生 `MacdKernel`/`MacdParameters`，不再创建
  indicator resource pool；RSI/ATR 已加入 typed plan 并通过 parity，但仍保留原 resource/lifecycle 路径。
  rolling 705 项和 clippy 通过。layout 5/6 删除与 RSI/ATR slab persistence 尚未完成。
- P3.1 fake-table 协议/故障 harness 已交付；P3.2 COW pages、dirty lineage、精确 undo 与 layout 7 已通过验证。
  P3.3 v4 object/manifest/partition persistence 已验证；P3.4 coordinator 与 opaque adapter 已有状态机和
  epoch roundtrip。Rust 与 Python `StreamSession` 的 durable construction、restore 和同步 checkpoint 已交付，
  使用 managed storage、`job_kind=session`、canonical opaque fingerprint 与 input/watermark progress；Rust session 25 项、
  Python focused 53 项通过，包含 epoch 1/2 continuation 和 running-loop 拒绝。runner 的 opaque-v4 接线与故障证据见下条；
  性能门槛尚未完成。
- P3.4/P3.6 runner checkpoint v4（`17893f8f`、`e7cd2f76`）：runner 将 source cut、opaque operator snapshot 及其
  ingress/output-frontier progress、sink recovery blocks 发布为 `job_kind=runner` 的 manifest v4。原生恢复选择最高完整
  epoch，校验精确 participants、capabilities、frontier 和 terminal 一致性，明确拒绝 v3 manifest，在输入准入前恢复
  operator/source/sink，并从下一 epoch 继续。Join/ASOF 保留付费恢复准入，包括 checkpoint-preload reservation；
  terminal Join 可使用付费本地 opaque loading。manifest durable 仍是提交点，sink finalization 先于 source acknowledgement，
  本 epoch 的 retention/orphan cleanup 在 durable settlement 后执行；rename/parent-sync 结果不确定时保留 prepared sinks
  并要求恢复，manifest durable 后的 settlement/maintenance 失败保留该恢复真相。非 soak streaming suite 已验证；命名 restart matrix 覆盖
  12 个故障点的 I/O、panic、cancel、restart，包含真实 parent-directory sync 失败、partial sink commit、retention 和
  compaction。进程重启 soak 证据已分类 v4 artifacts 并计入其 state bytes。legacy 协议删除、P6、性能门槛和
  最终 CI 仍未完成。
- P3.6 source history native v4（`ca760192`）：`SourceHistoryManifestEntry` 直接携带 canonical public
  `SourceHistorySegmentHandle`；`archive_file` 将原始字节直接写入 v4 storage，不再经 v3 operator-state staging/conversion。
  handle 的 raw `byte_len` 用于 archive/restore/load 的 storage 与 prepaid buffer budget，读取校验 checksum。后续 manifest
  携带原 handle，retention 保留跨 epoch 引用；restored ended source 将 history 带入 out-of-band final cut 且不 reopen。
  ASOF ordinary/terminal v4 restore 保留 `source_replay` metadata 并经 native history replay；最终 manifest publication
  移到 blocking worker。legacy v3 history 拒绝而非转换。验证通过：source history 6、frozen connectors 12、
  非 soak streaming 697、soak 67。
- P4.1 shared keyed core 与单整数 SUM 已交付；P4.2 typed aggregate state、temporal extrema、8192-group dirty
  snapshot cache 和 rollback/recovery 已通过 327 个 SQL 测试。P4.3 opt-in changelog 已集成，
  SQL 336 项与 Python 5 项测试通过。旧 compact 删除和 P4 测量尚未完成。
- P2.3 已完成 Float64 Rows slab、共享多窗口 history、layout 7 和 pending restore，rolling 694 项测试通过。
  后续同一基线的 paired 测量最初发现 large-batch 回退扩大为 0.29x；批量 counter、readout 定位、逐窗口
  rebuild history、O(1) retained count、checkpoint-off COW elision 和 numeric family 修复后，中间证据
  `p2-2-slab-family-selection` 为 0.68x，随后 first-touch undo 和 checkpoint-off direct recurrent 将
  `p2-2-slab-direct-recurrent` 提升到 1.09x/1.11x（faster），消除了相对基线的回退；绝对 G1–G3 仍未达标。
  独立审查发现的 freeze rollback、mean-only rebuild restore 和 abort dirty-loss 问题已各有回归测试并修复。
  其余 typed groups、P2.4–P2.9、Join 生产接线、并行/SIMD 与 window pane 的剩余工作仍需实现和验收。
- P2.7 已为 admitted ordered graphs 接入同步 Rolling：serial prepare/update/emission、sync adapter 双类型分派、
  checkpoint/restore/reset 和空输出处理，StreamSession 24 项、rolling 705 项和 clippy 通过；其他图继续使用
  async runner。`p2-2-slab-sync-rolling` 的 paired large-batch 为 1.12x/1.13x，未超过 full-tick 的 1.15x；
  G1 5.394 µs、G2 20.882 ns/window-row、G3 399.043 µs，绝对门槛仍未通过。
- P2.9 已交付首个 full-tick tier：严格验证 entity 顺序后，首 tick 完整 first-touch，后续 tick 复用同一
  transaction/partition，非 full-tick 自动回退。rolling 701 和 clippy 通过；`p2-2-slab-full-tick` 的 paired
  large-batch 为 1.15x，G2 19.291 ns/window-row、G3 371.773 µs，绝对门槛仍未通过。
- P6 SIMD 打包（`311d248c`）已交付：独立 `calc-flow-python-v3` 与 portable 版本使用相同版本号和
  `calc_flow` import package，要求在干净环境显式安装且二者互斥；Linux、macOS、Windows 三个 x86-64 target
  各有六个 Python tag 和两个 abi3 tier，不发布 v3 sdist。只有 v3 job 设置 `target-cpu=x86-64-v3`，portable
  与 AArch64 保持不变；artifact/hash、v3 vector inspection、clean-install smoke 和独立发布链路已接入。
  静态测试通过：`python/tests/test_distribution.py` 4 项、`scripts/test_release_workflow.py` 4 项、
  `scripts/test_inspect_v3_wheel.py` 2 项、`scripts/test_release_config.py` 42 项；实际 wheel 与跨平台发布仍由
  release workflow 验收。P6 的分区执行器、统一 morsel 池、I/O 与计算分离及性能门槛仍未完成。
- P7 pane 部分交付（`68b43fe3`）：N=1 hopping 且完整聚合列表仅含 COUNT/MIN/MAX 时启用按 slide 对齐的 pane；
  负时间戳使用数学向下取整，每个有效行只更新一个 `(group, pane)`，关闭时按 pane start 升序合并恰好 k 个 pane。
  COUNT 更新与合并均检查 u64 溢出，MIN/MAX 保持既有全序和浮点位模式；全 NULL 操作数的有效组仍产生 count=0
  和 NULL extrema。保留逐目标窗口的 `end <= watermark` late 判定，pane 在最后可贡献窗口关闭时（含等号）退休。
  逻辑预算仍按开放 `(window bounds, group)` 计数，字节预算覆盖实际 pane/key 与 merge scratch；输入、预算或
  rejected close 不安装部分状态，rejected close 同时恢复输出序列。Layout 3 checkpoint 保存 canonical
  panes/open-windows，兼容恢复后的后续输出与 snapshot bytes 逐位一致。8 项 pane 测试、53 项 window 测试和
  clippy 通过。SUM/AVG 及含它们的混合列表仍走 legacy per-window 路径；其他聚合扩展、P6 分区 dispatch、
  性能验收和最终旧路径/旧 codec 删除仍未完成。
- P5.1 Join run 核心部分交付（`695d301a`）：每个保留 morsel 形成不可变 Arrow run，携带 group ID、event time
  和物理 row ID 列；`(run ID, row offset)` locator 按 `(event_time, row_id)` 合并到 per-key 有序索引，乱序 run
  不重写 payload，probe-only 行不建立持久 run。原生探测复用 shared keyed core 和单调 window cursor，生成保持
  `(incoming admitted position, opposite event_time, opposite row_id)` 顺序的 selection vectors；输出按列通过
  Arrow `take`/`interleave` 物化，覆盖 projection、dictionary 和 nested payload。watermark 仍使用严格
  `row_time + extension < opposite_watermark` 前缀淘汰，EOF 清空对应侧。每侧 rows 计 live retained rows，bytes 覆盖
  retained payload、identity/locator columns、run owners 与 index capacities，并计入 replacement overlap。
  内存 snapshot/restore 校验 locator 覆盖、顺序、row-ID/frontier 和 accounting，后续输出与序列一致且 snapshot fingerprint
  相同。Join 569 项和 clippy 通过。该核心尚未接入生产 `StreamJoinOperator` admission 或 v4 state pages；
  现有 managed Join layouts 1–3 和恢复路径保持不变。生产接线、最终旧 layout 删除和性能门槛仍未完成。
- 2026-11-19 P5.1 整批原子预检（`76630a79`）：`RunState` 可在一个输入批内预检多个 morsel，统一执行 match cap，
  在全部输出物化成功后由调用方显式提交 run/frontier/sequence；输出上限或 match cap 失败不提交半批状态。
  10 项 run 测试通过。此提交不包含生产 admission 接线或 v4 pages，生产 Join 仍保持原路径。
- 2026-11-19 P5.1 run 迁移适配验证：`join::production::RunCore` 复用既有 admission，按最多 4,096 行构造
  Arrow morsel，经 shared-key routing、native probe 和 selection gather 产生输出，并在输出接受后提交 run/frontier。
  match/state limit 与淘汰指标失败保持原子性，legacy capture/restore 仅作为迁移测试边界。该路径只在显式测试中启用，
  公共生产路径尚未切换。Join 576 项和 Clippy 通过；native v4 pages、旧实现删除、统一 checkpoint-on/off 与性能验收未完成。
- 2026-11-19 P3/P5 native 页面与恢复：`state::v4::native` 以 `OperatorV4::Table` partition roots、Arrow/JSON pages
  和 metadata fingerprint 持久化状态，retention 递归保留 root 引用的全部页面。Join 增加 run payload/identity、
  per-key/expiration index 与 shared-key pages；恢复校验 group 域、locator 覆盖、capacity、row-ID/frontier、spec 和
  retained-byte counters，前缀淘汰后的 fingerprint 与后续输出保持一致，缺失页面及篡改 counters 拒绝安装。
  `KeyStore::restore` 保持原始 group ID 与 canonical bytes，重复输入完整回滚。Join 578 项、operator-task 76 项和
  Clippy 通过。统一生产切换、dirty-page reuse、旧实现删除和性能验收仍未完成。
- 2026-11-19 durable ACK 完成边界修复（`f4c88429`）：手动 `trigger_checkpoint` 不再于 sink commit 后提前返回，
  而是等待 manifest、transactional sink commit 与 durable source acknowledgement 全部完成；失败终态返回错误。
  ACK 测试改用 v4 `lineage/manifests/epoch-1.json`，正常、ACK 失败、ACK 挂起及 `continuous_public` 18 项通过。
- P5.2 ASOF columnar 生产迁移（`888d7f93`，已批准）：生产 `StreamAsofJoinOperator` 以 shared-key columnar runs
  替换旧 row/object state。每个 admitted batch 形成不可变 Arrow run，携带 group/time/sequence 与 payload/identity
  liveness；稳定 `(run ID, row offset)` locator 按 per-key `(time, sequence)` 和全局 pending-left
  `(time, key, sequence)` 顺序合并。selection 携带 left/right locators 与 right validity，输出按列通过 Arrow
  `take`/`interleave` 物化，保留 unmatched null 和 direct projection。输出接受后仅提交 pending prefix；
  eviction 淘汰 expired/dominated right-payload prefixes，在 right progress/EOF 后退休 identity-only history，
  并释放空 run。原生 layout/accounting 11 与 `arrow-columnar-59.2.0` checkpoint 保存 canonical shared keys、
  per-run identity/payload 和 control/index-capacity descriptors；restore 校验 inventory、schemas、charges、progress
  和 counters，后续 output sequence 与 snapshot bytes 保持一致。旧 ASOF row/object codecs 和 base/delta
  checkpoint machinery 已删除，较早 native layouts 拒绝恢复并要求新 lineage。133 项 focused 测试通过，
  独立审查已批准。最终 P5 性能验收、P6 和最终 CI 仍未完成。
- P6 统一 morsel 池与 Rolling lanes 迁移（`84d7e23f`、`378736b0`、`1ad671d8`）已交付，
  包含有界票据、取消/异常结算和有序合并；Join/ASOF 迁移、固定分区执行器、I/O 分离和性能门槛仍未完成。
- 2026-11-19 P2.5 继续交付：纯 RSI/ATR/MACD 及其与同一 Float64 Rows numeric 输入的混合节点已迁入事务化 slab，
  recurrence 的 u64/浮点位模式写入 layout 7 keyed family；numeric ring 与额外 keyed 列可共存，旧 numeric 编码不变。
  StreamSession 不再拒绝 indicator，Python RSI+MACD session 与 runner 结果一致；混合 RSI/ATR/MACD+mean 的冷恢复及后续输出
  与不中断运行一致。rank/quantile explain 也已恢复 typed scan 报告。Rolling 688 项、layout/slab focused、optimizer/lowering 52 项、
  Clippy 与 Ruff 通过。indicator 与 EWMA 等尚未迁 slab 的 family 混合时仍保留旧 resource/layout 5/6 路径；其余 family、
  旧子系统删除和性能门槛尚未完成。复杂度门禁仍是此前同一组既有增长项，本步骤没有新增失败位置。
- 2026-11-19 P1 后续实现已落地：NativeArray、原生 array batch 与精确 `numpy.ndarray` 均可进入同步 sequence session，
  单源 Rolling/Expression/NativeSequence 图在 runner source task 内联执行并保留逻辑 task identity、背压、错误归属和 reservation 清理。
  Rust session 26 项、operator fusion 26 项、source 31 项、entity 10 项、非 soak streaming 700 项及 Python 66 项通过；
  真实 rolling_runner 3 项通过。性能预登记：基线 `3638876c`，候选为当前工作区，共 3 例、预算 590 秒（构建另计）。
  NumPy 1 行 rolling mean 取 30 个测量值（5 次预热），候选 p50 ≤5 µs，无可用旧基线；单行 dual-SMA 和 6,400×64
  各取 30 对 AB/BA，评估相对回退，均不扩展案例、样本或为理想结果重试。预算分配为配对测量最多 570 秒，
  NumPy 隔离安装、预热与测量最多 20 秒。配对测量已完成（`target/p1-inline-performance.json`，总计 4.154 秒）：
  单行 dual-SMA 为 0.976x（95% CI 0.934–1.137），6,400×64 为 0.962x（0.881–1.031），两项均为 inconclusive，
  未确认回退。NumPy 初次测量误用相对安装路径，子进程切换目录后加载了开发模块；其 136.003 µs 结果无效且不得用于验收。
  修正路径后，sealed 候选的分段诊断（不作为门槛结果）为 push 5.52 µs、`to_numpy` 3.85 µs，证明输出转换也是固定成本。
  内联启动参数已合并，新增复杂度豁免已删除，复杂度报告回到此前同一组既有债务；配对结果来自内联实现 wheel，
  其后仅合并启动参数，未改热路径。与最终代码一致的 sealed wheel 已正式验收 NumPy：30 个样本的数值全部正确，
  p50 为 9.419 µs（`target/p1-inline-numpy-sealed.json`，1.662 秒），未达到 ≤5 µs。输入准备不计时，
  `_StreamSession.push` 到 `NativeArray.to_numpy()` 计入；该门槛仍未完成，后续优化不得删除输出转换来规避。
- 2026-11-19 NumPy 导出优化：有效 Float32/Float64 直接在 PyO3 层创建只读 NumPy 快照，不再经 Python/Arrow 往返；
  null/fill value 和 Boolean 保留原路径。预登记以 `434b8e4f` 的 9.419 µs 为对照，候选为当前工作区，仍仅测同一
  单行 rolling mean 用例，5 次预热、30 个样本、p50 ≤5 µs，隔离安装及测量预算 20 秒，构建另计，不扩展或重试。
  sealed 复测为 6.613 µs（`target/p1-numpy-direct.json`，1.758 秒），较前值下降 2.806 µs，但仍未达到门槛；
  后续只优化同一输入/输出路径，不改变计时边界；仅在实现变更后重新构建测量，相同构建不为理想结果重试。
  输入现已在 PyO3 层形成拥有副本，sealed 复测为 5.196 µs（`target/p1-numpy-snapshot.json`），仍未达到门槛。
  随后以原生类型描述符取代每次调用的 `__class__`/dtype 字符串属性读取，61 项测试通过。最终 sealed 结果为
  **2.885 µs p50，满足 ≤5 µs**（`target/p1-numpy-native-types.json`）；默认 feature 的 PyO3 Clippy 通过。
  全 feature Clippy 的本机构建在第三方 `rdkafka-sys` 处因缺少 `curl/curl.h` 停止，不能据此宣称全 feature 检查通过。
- 2026-11-19 P2.3 继续交付：独立 EWMA 及 numeric+EWMA 节点以 keyed recurrence 写入 layout 7，复用 `next_ewma`，
  恢复保存有效样本数与浮点位模式，min_periods、null/NaN 和后续输出保持一致。indicator+EWMA 仍走旧 resource/layout 5/6，
  其余 typed families 和旧子系统删除未完成。Rolling 689 项、EWMA 恢复测试和 Clippy 通过，未新增性能结论。
- 2026-11-19 P2.3 继续交付：Float64 cumulative sum/min/max/variance 复用 `CumulativeAccumulator` 的 canonical 列编码进入 layout 7，
  覆盖恢复和后续输出一致性、128 位总量拆列与非法 value 状态拒绝。indicator+cumulative、整数和其他类型仍保留旧路径；
  旧 snapshot fixture 已改为 layout 7 的缺失 segment 契约；Rolling 691 项、三项 slab 恢复用例和 Clippy 通过，未新增性能结论。
- 2026-11-19 P2.3 继续扩展：primitive cumulative 的 signed/unsigned/float 输入已复用 `TypedCumulativeState` 和 `DerivedBuilder`
  进入 layout 7，保持 sum 输出提升、min/max 输入类型、variance 与 null/min_periods 规则。Int8、UInt64、Float32、Int64
  恢复与 128 位溢出回滚通过，Rolling 693 项和 Clippy 通过；其余 family 仍未完成。
- 2026-11-19 P2.3 继续扩展：pure COUNT 的同一输入、多个 Rows 窗口已共享有效位 ring 并写入 layout 7，nested Arrow 输入
  不经过 ScalarValue，恢复与后续输出一致；duration 和混合节点保留旧路径。Rolling 694 项和 Clippy 通过。
- 2026-11-19 P2.3 继续扩展：纯 Float64 lag/delta 使用独立 raw history 组件，保留 NaN/null、共享多周期历史及 O(1) 引用读取；
  entity/keyed、history 与 ring 元数据分开编码，避免 AVG NaN 汇总污染，恢复继续校验 schema、offsets、capacity 和 logical bytes。
  双实体 lag(1)/lag(2)/delta 与恢复测试通过，Rolling 695 项和 Clippy 通过；Float32 同构 lag/delta 也已进入同一 raw history
  路径并通过 storage/恢复检查，Rolling 696 项和 Clippy 通过。Int64/UInt64 raw ring 已保留整数精度与 checked delta，
  Int8/Date32/UTC timestamp（含 >2^53）恢复通过，溢出和下溢不提交状态，Rolling 699 项和 Clippy 通过；其他 family 仍未完成。
- 2026-11-19 当前工作区修复了既有 CI 失败：v4 恢复保留静态输入摘要诊断和安全的错误分类，
  测试改用 `manifests/lineage/manifests/epoch-N.json` 与 opaque `blocks`；typed MACD 不再误入旧 resource worker，
  late-output Rolling 可恢复当前声明 layout，native sequence 保留 operator 上下文，ASOF lifecycle fixture 不再由 worker
  持有 observer schema。完整 Python 套件为 3138 passed、3 skipped（另有 1 条 PyO3 runtime drop warning），
  相关 Rust 测试、Clippy 与 MySQL 测试编译通过。MySQL 容器、跨平台、复杂度与最终 CI 仍需重新验收。
- 2026-11-20 Join 生产执行与恢复统一为 native run/layout 12：stream 和 finite 路径不再编译旧 SQL probe、
  StoredRow/row IPC、三层 key 身份或 v1/v2 writer/restore；这些实现仅保留在显式迁移测试中。旧 DataFusion owned
  query、v2 history/dictionary 诊断及其 feature 接线已删除。公共 `checkpoint()`/`restore()` 与 managed runner
  使用同一 native page 编码，保留的 capture/compaction component 改为验证 stable right pages 和恢复等价。
  Join 579 项、operator-task 77 项（2 ignored）、finite 8 项、native component 5 项、Python 验收/runner/fresh-harness
  共 40 项和全 feature Clippy 通过。当前 capture 仍重建 run/index/key page images，尚未实现 dirty-page/COW 成本界限；
  stable-page witness 不是性能测量，P5 的 1M/interval 门槛、其余 typed families、P6/P7 和最终 CI 仍未完成。
- 2026-11-20 Join clean-page 缓存已接入：run commit/eviction 维护 dirty lineage，未修改的 run identity、
  per-side index 和 append-only key record 跨 epoch 复用，retired run 从缓存删除，control counters 未变时复用 control page。
  连续 epoch 的第二/第三次 capture 分别携带 4/12 个累计 clean objects，部分淘汰与整 run 退休恢复通过；Join 580 项、
  operator-task 77 项（2 ignored）、native component 5 项及全 feature Clippy 通过。此步骤减少页面内容重建，
  但仍装配完整页面列表并在 dirty 时重建 per-side index，端到端 O(dirty pages) capture 与 P5 性能门槛尚未完成。
- 2026-11-20 Join 索引页已拆为 per-key bucket 与独立 expiration：commit/eviction 分别记录 dirty run、group 和
  expiration，单组追加只重建该 bucket，未触及 bucket 继续复用；恢复严格校验页面组身份、entry kind、descriptor、
  顺序和 accounting。连续 epoch 的第二/第三次 capture 分别累计携带 5/15 个 clean objects；两个 key 中仅一组变化时，
  另一 bucket、旧 run、另一侧 expiration 和 key record 均复用。Join 581 项、operator-task 77 项（2 ignored）、
  native component 5 项及全 feature Clippy 通过。完整页面列表仍按全量装配，O(dirty) 增量装配与 P5 验收尚未完成。
- 2026-11-20 Join capture 已改为 previous-lineage 增量页面集合：run/group/expiration/control/key 仅记录
  upsert/remove，页数与 reservation 汇总随变更更新，不再为每次 capture 装配完整页面 Vec。第二次 capture 仅 5 个变更，
  clean 第三次为 0，部分淘汰与整 run 退休分别为 4/5；直接 snapshot 与 manifest 恢复保持等价。Join 581 项、
  operator-task 77 项（2 ignored）、native page-set 3 项、native component 5 项及全 feature Clippy 通过。
  当前直接 snapshot 和 persistence 仍物化完整 partition，端到端成本、持久化增量化和 P5 门槛尚未完成。
- 2026-11-20 Native persistence 已直接应用 upsert/remove 到 previous partition index：clean pages 只保留原 handle，
  changed payloads 与 metadata 才写入，replacement/removal 保持 slot、generation、free-slot 和恢复语义。首次 full capture
  物化 1 次，后续增量 capture 不再增加物化次数；连续 epoch、部分淘汰、整 run 退休、普通 full capture 和 retention 均通过。
  Join 581 项、operator-task 77 项（2 ignored）、native page-set 3 项、native component 5 项及全 feature Clippy 通过。
  直接 snapshot 仍物化全部页面，partition index 仍每 epoch 全量序列化；端到端成本与 P5 门槛仍需验收。
- 2026-11-20 P5 首轮性能验收为 **inconclusive**：候选 `6581fdda` sealed wheel（wheel SHA `aa308dac`，
  loaded native SHA `759dc99`）单独构建耗时 563.654 秒，固定 Join 1M 与 interval 1M、单核、2 次预热 + 10 个正式样本、
  完整 oracle、总预算 600 秒。两次采集均在 source-ready 30 秒等待处超时，未产生有效性能样本或 p50，未执行 interval 计时。
  后续 100 行与 1M 诊断的完整输出 oracle 均通过，但不作为性能证据；按预登记停止规则不再为理想结果重试。
  Join ≤30 ms 与 interval ≤350 ms 均未验证，后续须先解决启动测量不确定性，不能以诊断通过替代门槛结论。
  后续检查 `failure-before-cancel.json` 与 100 行连续 observation 诊断已定位启动原因：每次新建 `EngineCase`
  都从 `sample-1` 开始，第二次 warmup 恢复到第一次 warmup 的 terminal checkpoint，当前 Python sources/sink 从未打开，
  因而 source-ready 超时；单次 1M 成功与 clone 无关，也不是 runner-ready 的产品缺陷。测量脚本已为每个 warmup/正式 observation
  分配独立 root，100 行连续 2 次预热 + 2 次 observation 的完整 oracle 验证通过。下一轮预登记 `212ca55b`，使用独立
  observation roots、2 次预热 + 10 个样本和独立 600 秒预算；这是测量有效性修复，不进行计时驱动的重试。
- 2026-11-20 P5 fresh-observation-root 验收完成，证据归档于 `benchmarks/join-asof/p5-native-runs-20261120/`。
  Join 1M 单核完成 2 次预热 + 10 个正式样本，P50 **424.412 ms**、P95 430.192 ms，10/10 完整 oracle 通过，
  ≤30 ms 门槛未通过且慢于 archived descriptive 78.51 ms，不声称配对加速比。interval 的首个 warmup 在 600 秒等待时
  仅输出 5,376/10,998,080 行，无有效 P50，≤350 ms 未通过。采集总耗时 610.174 秒，超过 600 秒预算；
  wrapper 未在 `acquisition_stop_seconds=570` 阻止进入 interval 的 600 秒等待，协议缺陷已公开记录。
  wheel 构建耗时 1084.473 秒，wheel/loaded native SHA 已归档。后续先修复性能与 interval 输出推进问题，不追加计时样本。
- 2026-11-20 P5 多 morsel 正确性修复（`dfc077eb`）及复测完成，证据归档于 `benchmarks/join-asof/p5-morsel-state-20261120/`。
  prepare 现在预留不同 run ID，并串联前序 prepared side update；空 morsel 保留最近非空累计状态。此前重复 ID 和基于批前状态的更新
  使 interval 仅输出部分匹配。直接状态与生产回归均从 18,928 恢复到 108,080 行，空 morsel、匹配上限、驱逐和恢复共 19 项测试通过。
  1M 单核正式采集完成两例各 2 次预热 + 10 个样本，Join P50 **461.093 ms**、interval P50 **3,699.085 ms**，两例 10/10 完整 oracle
  均通过且正常结束；采集耗时 84.957 秒，在 600 秒预算内。Join ≤30 ms 与 interval ≤350 ms 仍未通过，不将正确性恢复表述为性能达标。
  新 wrapper 将剩余截止时间传入底层 `run_stream`，保留原计时边界，4 项协议测试通过；wheel 构建耗时 557.219 秒，SHA 已归档。
- 2026-11-20 P5 run admission 已改为同步批量模式（`6f15e8fc`），保留取消检查、输出背压和旧路径的协作默认值。
  大输入回归的 Tokio budget 边界从 15,847 降为 0，相关 19 项测试及全 test-target Clippy 通过。复测 Join P50 **449.057 ms**、
  interval P50 **3,574.794 ms**，两例各 10/10 完整 oracle 通过，采集 84.064 秒，在预算内；相对上轮描述性下降约 2.6%/3.4%，
  但不是配对显著性结论，≤30 ms/≤350 ms 仍未达到。证据见 `benchmarks/join-asof/p5-bulk-admission-20261120/`；下一步处理 morsel 间重复合并。
- 2026-11-20 P5 整批状态合并（`e737fb0f`）已交付：先构造不可变 run，再统一合并 bucket/expiration，乱序输入排序回退，
  全部门限和输出序列检查后才原子安装。10,000 行回归的合并访问从 40,480 降为 20,000；Join 模块 585 项测试与 Clippy 通过。
  复测 interval P50 **1,397.678 ms**，较上轮描述性下降 60.9%；Join P50 **459.643 ms**。两例各 10/10 完整 oracle 通过，采集
  47.989 秒，在预算内，但 ≤30 ms/≤350 ms 仍未达到。证据见 `benchmarks/join-asof/p5-batch-merge-20261120/`；下一步处理匹配查找和输出碎片。
- 2026-11-20 P5 probe run 缓存与输出合并（`12d7759c`）已交付：每个 probe 只解析唯一对侧 run，输出按原 EdgeBudget
  使用最多 64,000 行分片。回归的输出批次从 27 降为 3，匹配查找从 108,080 降为 7；Join 585 项测试与 Clippy 通过。
  复测 interval P50 **1,167.624 ms**、Join P50 **450.038 ms**，较上轮描述性下降约 16.5%/2.1%，无配对显著性声明。
  采集 43.949 秒，两例各 10/10 完整 oracle 通过，≤30 ms/≤350 ms 仍未达到。证据见 `benchmarks/join-asof/p5-probe-cache-20261120/`。
- 2026-11-20 P5 dense Arrow 零拷贝（`fcec2dc3`）已交付：同源连续 admitted morsel、identity key 与全保留 run 复用切片，
  稀疏或 schema 转换输入保留 gather。两个 buffer 共享回归及 Join 模块 587 项测试、Clippy 通过。复测 Join P50 **422.432 ms**、
  interval P50 **1,147.609 ms**，描述性下降约 6.1%/1.7%，采集 43.173 秒，完整 oracle 均通过，但门槛仍未达到。
  证据见 `benchmarks/join-asof/p5-dense-runs-20261120/`；下一步以分阶段计时定位剩余成本，不再仅按推测微调。
- 2026-11-20 P5 生产阶段诊断（`42d5ab31`）已完成：在显式 `benchmark-internals` wheel 上记录 admission、route、probe、merge、
  materialize、emit 和 commit，单次 1M Join 完整 oracle 和自然结束通过。`run_route` **460.377 ms**（65.7%）、
  `run_materialize` 140.155 ms（20.0%）、`run_probe` 79.144 ms（11.3%）、`run_admit` 11.400 ms（1.6%）。
  带记录器的 701.569 ms 不与默认 P50 比较，也不作为门槛样本。证据见 `benchmarks/join-asof/p5-stage-profile-20261120/`；
  下一步针对重复 key-run 的 group ID 复用，默认 KeyStore 的所有权行为不变。
- 2026-11-20 P7 SUM/AVG pane 准入被 FR21 数值契约阻止：对 `[-1e-12, 1e19, 100, -1e19, -100, 1e-13]`，普通
  compensated partial merge 与逐行求和的浮点位结果不同。`pane_sum_merge_reassociation_is_not_bit_exact` 已固化该反例，
  因此没有启用 SUM/AVG pane，也没有以近似相等或 async fallback 冒充完成。COUNT/MIN/MAX pane 保持现状；
  9 项 pane 测试及全 feature Clippy 通过。SUM/AVG 需要能保持逐位输出的合并证明与新数值方案，P7 仍未完成。
- 2026-11-20 P6 ASOF managed checkpoint capture/restore 已从旧 gather worker 迁到 job 统一 morsel 池，
  保留独立 workspace reservation、job cancellation/deadline 等待和取消 ticket；放弃 future 后的后台失败继续归入原 operator task。
  新增旧 gather 准入关闭后的 capture/restore 回归及 abandoned failure 诊断回归，ASOF 49 项、operator-task 77 项
  （2 ignored）和全 feature Clippy 通过。SQL/order、Join 迁移测试辅助路径等其他 gather 用户、固定分区执行器、
  剩余 I/O 分离与 P6 性能门槛尚未完成。
- 2026-11-20 P6 SQL incremental output ORDER 已迁入统一 morsel 池：job context 可懒初始化唯一 owner，
  ASOF 与排序共用 managed ticket 的取消/deadline 和 abandoned failure 托管；普通计算错误只返回原 ticket，
  不再污染后续可重试工作，panic 仍按池级故障处理。SortWork 按输入行与 prepaid output bytes 准入，
  reservation 随输出保留到调用方取得结果，旧 gather 准入关闭后的排序仍与 DataFusion oracle 一致。
  SQL 337 项、ASOF 49 项、morsel pool 20 项、Rolling parallel 4 项及全 feature Clippy 通过。
  global record、bounded order、Join 迁移测试辅助路径、固定分区执行器和其余 I/O 分离尚未完成。
- 2026-11-20 P6 bounded order 的流式 FinalWork 已迁入统一 morsel 池，finite finalize 保持原路径；
  排序/analytic 共用原执行函数，按历史行数与 control output bytes 准入，control reservation 随输出释放，
  analytic 的 StateBudgetExceeded 映射及输出拒绝语义不变。旧 GatherOperatorId reservation-owner 接口和过时测试已删除，
  declaration 由 FinalWork 自身持有。旧 gather 关闭后的 on_end 输出、order 11 项、bounded_global_ordering 10 项、
  order_closure 2 项及全 feature Clippy 通过。global record、Join 迁移测试辅助路径、固定分区执行器和其余 I/O 分离尚未完成。
- 2026-11-20 P6 SQL global record 与 FILTER coalescer 已迁入统一 morsel 池，普通 aggregate 与 coalescer
  共用 MorselLease，prepaid credit 随输出释放；旧 gather 准入关闭后的两类查询均与 DataFusion oracle 一致。
  这也是最后一个生产 gather 用户：非 test 构建已不再编译 legacy gather submission/worker/process pool，
  共用停止检查移到独立 stop 模块，runner/supervision 的旧 owner 仅在 test 保留作迁移证据，benchmark 收尾改用 morsel owner。
  SQL 339 项、legacy gather 43 项、supervision 5 项、operator-task 77 项（2 ignored）、order 11 项、ASOF 49 项、
  Rolling parallel 4 项及非 test/test 全 feature Clippy 通过。entity_work、固定分区执行器、其余 I/O 分离与性能门槛尚未完成。
- 2026-11-20 P6 legacy entity_work 已退出生产构建：Rolling 生产只使用统一 morsel 池，runner、supervision、
  operator task 与 StreamOperatorContext 不再创建或携带旧 lane owner；旧 entity_work 仅在 test 编译，保留迁移证据，
  没有 dead-code 豁免或非 test 兼容壳。非 test 全 feature compile 无警告，legacy entity 12 项、真实 morsel runner 10 项、
  operator-task 77 项（2 ignored）、Rolling 699 项、supervision 5 项、session 26 项及全 feature Clippy 通过。
  固定分区执行器、其余 I/O 分离与性能门槛尚未完成。
- 2026-11-20 P6 SQL checkpoint 编码已从 spawn_blocking 迁入 managed morsel：保留原 attempt 取消检查和预付
  reservations，池关闭后拒绝新编码，普通错误仍可重试。懒初始化 owner 改为 Arc<OnceLock<_>>，所有 job clone
  共享同一生命周期，修复 prepare 放弃时最后 owner 等待编码线程退出的阻塞；两个既有 dropped-prepare/worker 回归恢复。
  SQL 340 项、ASOF 49 项、order 11 项及全 feature Clippy 通过。固定 N 分区、Window/v4 其余 CPU 编码迁移和性能门槛尚未完成。
- 2026-11-20 P6 Window 非空 checkpoint operations 编码已从 spawn_blocking 迁入 managed morsel，
  operations/spec/状态身份均由 owned work 携带，空操作保持直接返回，编码前后检查取消，池关闭后拒绝新编码。
  Window 55 项（1 ignored）、window_state 15 项、window_hopping 4 项及全 feature Clippy 通过。
  固定 N 分区、v4 store 与其余 CPU 编码迁移和性能门槛尚未完成。
- 2026-11-20 P6 runner opaque v4 编码已与持久化分离：checkpoint task 携带原 job context，
  managed morsel 只负责对象编码，V4OpaqueAdapter/V4Store 接收编码结果并保持原排序、统计和持久化语义；
  state 层不反向依赖 runtime，session 的 N=1 路径未改成跨线程执行。池关闭在写入前拒绝，开放池真实写入/读回通过。
  v4 focused 10 项、SQL recovery 24 项、checkpoint lifecycle 3 项及全 feature Clippy 通过。
  固定 N 分区、native/table/source-history 其余编码迁移和性能门槛尚未完成。
- 2026-11-20 P6 runner native v4 编码已与持久化分离：state 新增轻量 V4ObjectStore 接口，
  原 V4Store 行为不变，runner 的 ManagedNativeStore 仅在 write_objects 时切换到 managed morsel，
  previous manifest/index、page 分配、引用校验和 carried 计数继续复用原实现。池关闭在写入前拒绝，
  开放池完成 partition/page 持久化。native page 3 项、Join 581 项、SQL recovery 24 项及全 feature Clippy 通过。
  固定 N 分区、table/source-history 其余编码迁移和性能门槛尚未完成。
- 2026-11-20 P6 runner source-history 编码已接入 managed morsel：V4ObjectStore 同时覆盖 native/opaque，
  SourceHistoryContext 可注入共享对象存储且公共 archive_file 签名和默认行为不变；frozen source 安装时注入
  真实 job 的 ManagedHistoryStore，archive/seal/load 连续通过，池关闭在写入前拒绝。source-history 6 项、
  connectors 6 项及全 feature Clippy 通过。table v4 的 session N=1 coordinator 保持原路径，
  固定 N 分区及其余 CPU 编码迁移和性能门槛尚未完成。
- 2026-11-20 P6 grouped integer SUM 已建立真实固定 N 状态所有权：plan 时固定最多 4 个分区，
  全局 KeyStore 只路由一次并保留 group-id，group_id % N 的分区各自持有 SUM/undo，输出按全局 id 合并，
  不复制 key、重哈希或合并跨分区累加器。N 与 fixed-group-id-modulo-v1 路由身份纳入 layout fingerprint 和
  wire metadata，N=1/2/4 输出与 owner 跨批次一致，N 不匹配恢复拒绝。SQL 342 项及非 test/test Clippy 通过。
  本步不含并行 worker 或性能提升声明；并行执行、其他 keyed 算子和 P6 性能门槛尚未完成。
- 2026-11-20 P6 grouped integer SUM 固定分区已接入实际并行执行：N>1 且输入不少于 64,000 行时，
  各分区 SUM/marks 移入 scoped worker，KeyStore 路由与 lease 仍使用原共享锁；undo、reservation、输出和错误按分区序号恢复。
  N=1 或小批保持串行，不增加线程成本。逐行捕获 worker panic 后恢复原分区状态，回滚所有分区及 key births，
  清除故障后同一输入可重试；强制并行测试证明至少两个真实 worker 参与。SQL 344 项及全 feature Clippy 通过。
  本步不含性能提升声明；其他 keyed 算子、统一分片执行面、其余 I/O 分离和 P6 性能门槛尚未完成。
- 2026-11-20 P6 keyed SUM 首轮四例配对验收已完成，证据归档于 `benchmarks/sum-state/p6-fixed-partitions-20261120/`。
  基线为 `aefd17dc`（固定 N 串行 apply），候选为 `212ca55b`（scoped parallel apply），每例 2 轮×10 对 AB/BA，
  全部 snapshot/lifecycle oracle 通过，采集 120.299 秒，未增加样本或重试。stock-stream-1cpu 从 199.879 ms 增至
  249.977 ms（+25.06%，regression），stock-stream-8cpu 从 136.195 ms 增至 164.141 ms（+20.52%，regression），
  sparse-stream-1cpu +5.18%（inconclusive），finite 控制 +1.82%（inconclusive）。release 构建分别为 671.666/970.241 秒。
  因此当前并行实现不是性能升级，P6 4 核 ≥3 倍门槛明确未通过；后续先定位并消除回退，不扩张 keyed 功能面。
- 2026-11-20 P6 dense partition marks（`d80a334a`）已消除上述回退，证据归档于 `benchmarks/sum-state/p6-dense-marks-20261120/`。
  marks 改为按 `group_id / N` 索引的 dense u64 vector，SUM、undo 和 checkpoint layout identity 不变；
  SQL 344 项及全 feature Clippy 通过。相同 `aefd17dc` 基线下，stock-stream-1cpu 为 198.546 ms（−2.73%，no confirmed regression），
  stock-stream-8cpu 为 136.283 ms（+1.28%，no confirmed regression）；sparse 为 +1.70%（inconclusive），finite 为 +1.10%（inconclusive）。
  采集 118.698 秒、release 构建 628.631 秒，全部 oracle 通过。当前只恢复到基线附近，P6 4 核 ≥3 倍仍未达到；
  下一步处理共享 KeyStore 锁与状态所有权，不扩张 keyed 功能面。
- 2026-11-20 P6 批量旧组 lease 未证明收益并已撤回（`39ef601a`，revert `c75fe501`），证据归档于
  `benchmarks/sum-state/p6-batched-leases-20261120/`。stock-stream-1cpu 基本持平（−0.05%），stock-stream-8cpu
  为 148.248 ms（+8.24%，inconclusive），sparse/finite 分别 +3.80%/+2.61%（均 inconclusive）；
  该实现比 dense marks 更慢且增加分区扫描，不能留在默认路径。采集 120.230 秒、release 构建 621.022 秒，
  全部 oracle 通过。撤回后 `keyed_sum.rs` 与 `d80a334a` 逐字一致，不重复同一性能检查。
  后续先量化全局 key routing 与实际 partition update 成本，不再凭锁假设增加优化。
- 2026-11-20 P6 生产并行激活已修复并完成首轮真实测量（`6dcacab6`）：此前 `update_keyed_records`
  每个 4,096 行调用 `apply`，64,000 行并行门槛永不触发；现仅路由阶段按 morsel 切片并合并 group ids，整批可进入 worker。
  70,001 行测试证明至少两个真实 worker 激活、完整 oracle 与内存释放，SQL 345 项及 Clippy 通过。
  证据归档于 `benchmarks/sum-state/p6-production-parallel-20261120/`。兼容串行基线改为 `2ac1af5f`，
  避免旧 `aefd17dc` harness 身份不匹配。stock-stream-1cpu −0.66%（no confirmed regression），stock-stream-8cpu
  +15.63%（inconclusive），sparse +3.72%、finite −3.33%（均 inconclusive）；P6 4 核 ≥3 倍明确未达到。
  示例 route/apply 约 60/53 ms（串行）与 65/73 ms（并行），并行没有降低 apply 时间且 CPU 更高。
  此前 lease 实验没有真正执行并行，不能据此判定锁分摊无效；下一步仅在 worker 内预取旧组，避免串行新增扫描，仍无收益则关闭默认并行。
- 2026-11-20 P6 worker 预取 lease 仍未达到门槛，默认并行已撤回（`84570246`，revert `ebc64d9f`；
  激活 `6dcacab6`，revert `3c830fee`），证据归档于 `benchmarks/sum-state/p6-worker-prefetch-20261120/`。
  stock-stream-1cpu +0.45%（no confirmed regression），stock-stream-8cpu +9.02%（inconclusive），
  sparse/finite −0.52%/−1.23%（分别 no confirmed regression/inconclusive）。采集 120.652 秒、release 构建 632.512 秒，
  全部 oracle 通过。撤回后相关源码与 `2ac1af5f` 一致，不重复同一检查。thread-per-record 与 lease batching 路线已证伪；
  P6 仍需不同的路由/状态所有权设计，不能把当前状态称为性能升级。
- 2026-11-20 P6 SQL restore/capture prepare 已从独立 `spawn_blocking` 执行迁入 job morsel：
  保留 JobSqlRecoveryOwner 的串行准入、join loan、候选算子安装与 abandoned-work 结算，启动 gate 改为异步 oneshot，
  morsel lease 并入 prepare 的取消检查。池关闭后拒绝新 prepare，不再绕过统一准入；3 项 SQL worker 生命周期测试及全 feature Clippy 通过。
  迁移后独立 target 对照发现 `e824b3f4` 已有同一组 11 个 ledger 断言失败：字段和值正确，
  但候选 oracle 尚未纳入固定分区引入的 `key_layout` schema metadata。测试现已独立校验该 metadata 的唯一键与 64 位 hex 格式，
  并继续严格校验字段和值，不写死随机器分区数变化的哈希；完整 sql_recovery 27 项通过。其余 CPU 工作、
  其他 keyed 算子和 P6 性能门槛尚未完成。

## 1. 目标

1. 尽可能提高小 batch 场景下流式引擎的执行性能，与 Arroyo 和 TA-Lib 可比。
2. 重构 checkpoint 机制，使其清晰，并尽量不影响执行性能和并行计算。
3. 调整架构，提升对多线程执行和 SIMD 的支持。
4. 提升 join、group by 的性能。
5. 为 window sum 一类算子单独给出性能提升方案。

整个方案允许对架构和内部实现做较大调整。

## 2. 已批准的决定

| 编号 | 决定                                                                           | 说明                                                                                     |
|------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1    | 引入公开的同步 push API `StreamSession`                                        | 推翻此前对 `StreamingRunner::step` 的移除；保留 runner 只换入口的方案估计停在 200–300 µs |
| 2    | 删除 `stable_v1` 和 `stable_v2`，新算术是唯一数值契约                          | 见第 5 节                                                                                |
| 3    | 旧 checkpoint 不需要可恢复，状态格式和协议完全重做                             | 新格式自身的恢复仍保证逐位续算                                                           |
| 4    | 保持 `unsafe_code = "forbid"`，SIMD 依靠可自动向量化的循环和 `x86-64-v3` wheel | 不引入带 `unsafe` 的 kernels crate                                                       |
| 5    | 便捷流默认关闭 checkpoint，有序输入按批推进 watermark                          | 两者都是可见的行为变化                                                                   |
| 6    | 聚合的 changelog 输出为显式选项，完整快照仍是默认                              | 该项未单独答复，按建议执行；P4 规格前可再确认                                            |

## 3. 现状诊断

### 3.1 成本分层

负载为 warm 状态的 dual SMA(5)−SMA(20)。数据来自 PR #616 在 `b9a46e12` 上的本地测量产物
（`target/compiled-region-performance/`，未入库），主机为 WSL2。

| 层                                                              | 1 行 append | 99 行 append | 6,400 行，64 实体 |
|-----------------------------------------------------------------|-------------|--------------|-------------------|
| Python 流式全链路                                               | 599 µs      | 615 µs       | 908 µs            |
| 原生算子 dispatch（rolling 加 projection，不含 source 和 sink） | 6.1 µs      | 9.3 µs       | 329 µs            |
| TA-Lib 适配器（参照）                                           | 5.8 µs      | 5.9 µs       | 229 µs            |

由此得到三层成本：

- **异步外壳**：约 590 µs，与行数无关，位于算子之外。
- **算子 dispatch 固定成本**：约 6 µs。
- **每行内核**：33–51 ns。

P1a、P1b 融合和 #616 region 优化的是算子之间的那一段，所以没有收益：
#616 的配对比值为 0.96–1.11x，置信区间全部跨 1.0。

### 3.2 没有同步执行核心（目标 1）

- 算子全部是 `#[async_trait]` handler（`crates/calc-flow/src/operator/stream.rs:753`），只能由 Tokio 任务图驱动。
  同步 push 入口是被刻意移除的（`crates/calc-flow/src/lib.rs:19`）。
- 边是带互斥锁的 `VecDeque` 加 `Notify`（`crates/calc-flow/src/runtime/streaming/edge_queue.rs:12`）。
- 基准拓扑下，一次 append 在计时窗口内有 4 次 Tokio↔asyncio 跨线程唤醒、2 次 asyncio 事件循环轮转、
  约 5 次 Tokio task 跳转和 3 次 GIL attach。Data 和 Watermark 各是一次完整的 Python 回调往返。
- 1 行 append 的分阶段中位数：入队到 source 41 µs，data 到 watermark 250 µs，watermark 到 sink 215 µs，
  sink 到接收 81 µs，其中 `to_pyarrow` 占 48 µs。
- 单次回调的桥接成本（`benchmarks/rolling/callback-sparse-818d69c.json`，开启 profiling）：
  排队到分发 48–62 µs，完成到 Rust 恢复 56–80 µs。
- `to_pyarrow` 走 `RecordBatchReader` 再 `pa.table`（`crates/calc-flow-python/src/batch.rs:417`）。
- 任务拓扑不是主因：Tokio worker 从 32 减到 1 只有 1.02–1.11x
  （`benchmarks/rolling/scheduler-32-to-1-818d69c.json`）。
- 便捷 API 的默认 watermark 是 `max_seen − 1 µs`，由 100 ms 定时器发出
  （`python/calc_flow/_stream_inputs.py:124`）。这是一个与 CPU 无关的延迟下限。
- 与 Arroyo 的集成路径对比（2026-10-21 基线，WebSocket JSON 到计算到 stdout JSON，calc-flow 侧含 Python 适配；
  `.codex/artifacts/arroyo-comparison/REPORT.md`）：burst 吞吐 Arroyo 是 calc-flow 的 2.26 倍（投影与过滤）
  和 4.50 倍（四秒窗口）。

### 3.3 Checkpoint 不是可插拔的一层（目标 2）

- 开启 checkpoint 会选择另一条更慢的执行路径：非测试代码里有 24 处 `.checkpointing()` 分支，
  分布在 join、window、rolling、asof 和 sql。
- Rolling 开启后失去 in-place 更新和延迟历史（`crates/calc-flow/src/operator/rolling/ordered_stream.rs:337`），
  并为每个实体维护第二份状态 `RollingHistories`。
- Python 默认 `checkpointing=True`（`python/calc_flow/runtime.py:773`）。
- 触发时停流：barrier 要等 prepare、capture、encode 和落盘全部完成才转发
  （`crates/calc-flow/src/runtime/streaming/operator_task.rs:1722`）。
  落盘持有全 job 一把锁（`crates/calc-flow/src/state/transaction.rs:164`），
  每个新 segment 做三次 SHA-256、至少两次 fsync。每次 checkpoint 停顿 40–240 ms。
- capture 有四种机制：SQL 整算子出借、join 的 owned capture、ASOF 的 managed capture，
  其余算子走同步的 `checkpoint(&mut self)`。
- 复杂度：9 个算子族、约 20 个 layout 或版本常量、约 12 种二进制 magic、4 套增量方案和 3 个全量重写族。
  checkpoint 相关的非测试代码估算约 5 万行，其中 join 约 1.39 万、ASOF 约 1.06 万、rolling 约 0.97 万、
  `state/` 约 0.9 万。
- `benchmark-checkpoint-off` feature 没有从产品路径编译掉任何东西。

开启但未触发就已经有成本：

| 负载                        | 关闭     | 开启未触发          | 周期触发                                      |
|-----------------------------|----------|---------------------|-----------------------------------------------|
| `window_sum`，10M 行        | 1,010 ms | 1,616 ms            | 1 s 间隔 1,930 ms；200 ms 间隔 2,719–3,034 ms |
| Interval join，每侧 200k 行 | 263 ms   | 404 ms（24 h 间隔） | 100 ms 间隔 880 ms                            |

来源分别是 `.codex/artifacts/specs/window-operator-async-checkpoint-flush.md` 和
`.codex/artifacts/analysis/issue406-checkpoint-modes.md`。

### 3.4 并行是单 owner 加临时外包（目标 3）

- 只有一个共享 Tokio 池：Python 用 `tokio::runtime::Runtime::new()` 创建
  （`crates/calc-flow-python/src/config.rs:325`），worker 数等于逻辑核数。
  I/O 和计算没有分离，CPU 工作直接在 Tokio worker 上内联执行。
- 每个算子或融合链一个 task，并行度恒为 1。没有 keyed exchange、shuffle 或算子实例的概念。
- 有三套互不相干的外包机制：`gather_work` 自建线程池（join、ASOF、排序、全局 SQL）、
  rolling lanes 的 `spawn_blocking`、checkpoint 编码的 `spawn_blocking`。
- 准入门槛很高：rolling lanes 要求不少于 64,000 行且每 lane 不少于 16,000 行；
  join 探测要求不少于 8,192 行且不少于 320,000 次访问。
  空载分发的中位数是 45 µs（1 个单元）和 88 µs（4 个单元）
  （`crates/calc-flow/src/operator/join/data_work.rs:20`）。
- 实测不扩展：join 在 1 核是 87.1 ms，32 核是 89.5 ms；grouped SUM 在 4 核以上停在约 1.2 s；
  ASOF 1M 行在 32 核只改善 38.5%。
- 三条并行 grouped SUM 路线（行区间 lanes、完整键分片、委托 DataFusion 的 delta 聚合）都被撤回，
  原因是每批的路由、拷贝和合并是串行的（`benchmarks/sum-state/`）。
- DataFusion 在流路径上默认单分区（`crates/calc-flow/src/datafusion.rs:104`）。
- SIMD 无从谈起：
  - rolling 状态是 enum 后面的 AoS，每行都有实体查找、enum 匹配、`VecDeque<TimedSample>` 的 push 和 pop、
    checked 计数、`Result` 和 builder 的 `append_option`；
  - 聚合是每组一个 enum 或 `Box<dyn Accumulator>`；
  - 几乎没有可自动向量化的数值循环；
  - 仓库没有 `.cargo/config`、`RUSTFLAGS` 或 `target-cpu` 设置，x86-64 wheel 只有 SSE2。

### 3.5 Join 与 group by 是行对象式状态（目标 4）

| 负载（单核）                    | calc-flow 流式 | 参照                               | 来源                                                                                            |
|---------------------------------|----------------|------------------------------------|-------------------------------------------------------------------------------------------------|
| Join，1M × 64                   | 78.5 ms        | DataFusion 13.0 ms，Polars 18.0 ms | `.codex/artifacts/analysis/join-single-thread-performance.md`；参照值来自 2026-10-20 的单核矩阵 |
| Interval join，每侧 1M          | 672 ms         | DataFusion 346 ms                  | 同上                                                                                            |
| ASOF，1M                        | 84.0 ms        | Polars 63.8 ms                     | 同上                                                                                            |
| Grouped SUM，10M 行，100×100 组 | 1.67 s         | calc-flow 批 SQL 0.43 s            | `benchmarks/stock-positions/engine-core-sweep-20261107.md`；两边计时边界不同                    |

- **Join**
  - key 逐行经过三层身份：批内的 `LocalIntern`、持久的 `Dictionary`、每个 key 的 `NativeIndex`。
  - 最内层循环每 64 次访问或 4 KiB 做一次 `check_cancelled` 和 `consume_budget().await`
    （`crates/calc-flow/src/operator/join/columnar/owned_copy.rs:65`）。
  - 每个准入行都走事务式 prepare，包括不保留的行。
  - 输出物化和逐对字节计量占 15–29%。
  - 非原生 key 走一条对物化状态跑 DataFusion 等值 SQL 的旧探测层，成本与状态大小成正比。
- **Group by**
  - 状态是 `Vec<Group>` 加 `HashMap<Arc<[u8]>, usize>`。每组带 `Arc<[ScalarValue]>`、聚合 enum
    和独立的 `MemoryReservation`（`crates/calc-flow/src/operator/sql/incremental.rs:139`）。
  - 每批为每个被触及的组建一个写时复制的候选。
  - 每批输出全部分组的完整快照。
  - 不符合准入的查询形状保留全部输入，每批重跑整个查询。
  - inline 整数 SUM 内核本身约 10 ms 每 1M 行，差距在表达式节点、快照输出和进度协调。
- **Window 聚合**
  - 状态是 `BTreeMap<WindowKey, AccumulatorRow>`（`crates/calc-flow/src/operator/window.rs:335`），
    key 含窗口起止和 `Arc<[u8]>` 分组键。每批排序加两次建树。
  - hopping 把每行展开到所有开放窗口，最多 1,024 个。
- **ASOF**：已经最接近列式设计（每个 key 持有排序的列式 run），准入阶段占 69%（含子调用）。

### 3.6 Rolling 内核的固定成本与布局（目标 5）

- Data 回调只缓冲，Watermark 回调才最终化并输出。
- 约 6 µs 固定成本的来源（读代码推断，没有分阶段实测）：
  - 每次 dispatch 新建三个 `RowConverter`；
  - 约 12 个阶段计时 guard，每个两次互斥锁加两次时钟读取；
  - 端口校验里的 `format!`；
  - 每批深拷贝所有被触及的实体（`crates/calc-flow/src/operator/rolling/kernel.rs:1740`）；
  - 输出装配和两次 `EnvelopeCost`。
- SMA 和 dual SMA 永远走不到已有的 in-place 路径，因为它排除了 `MeanKernel`
  （`crates/calc-flow/src/operator/rolling/kernel/in_place.rs:420`）。
- 每行成本（2026-11-08 的阶段计数，旧源码）：West 递推每窗口每行 16–20 ns（见第 5 节），
  实体路由约 12 ns，顺序证明约 7 ns。
- 状态布局是 `HashMap<Vec<u8>, usize>` 到 `Vec<Arc<TypedEntityState>>` 到 `Vec<TypedWindowState>`
  再到 `VecDeque` 或 `Arc<RowsTape>`，每行三到四次指针追逐，每个样本 24 字节。
- 通用路径悬崖：只要有一个 `lag` 或 `delta` 输出（`crates/calc-flow/rolling-kernels.json` 中标为 general），
  整个算子就退到 `ScalarValue` 和 `BTreeMap` 的行路径。
- 有限（批）路径使用同一个逐行内核。
- 累加器有四套独立实现：rolling、`array/rolling_mean.rs`、window 和 SQL compact。

### 3.7 共性

核心 crate 约 46 万行 Rust，其中约 16 万行是测试。join 约 8.6 万行，rolling 约 7 万行，asof 约 5.4 万行，
流式运行时约 8.6 万行。每个算子自带状态模型、准入计费、checkpoint 编码和并行设施，
没有共享的 keyed-state 层或状态快照层，所以每项优化都要逐算子重做。

### 3.8 证据边界

- 小 batch 的数字是 2026-11-15 的本地测量；分阶段归因结合了一个较旧的带 instrumentation 的构建，阶段之间有重叠。
- join、group by 和 checkpoint 的数字来自 10 月到 11 月初的产物，早于若干修复，未在 `39cab2fb` 上重测。
- 6 µs 固定成本的拆分和所有预期收益都是估计。
- 所有测量都在 WSL2 主机上，唤醒延迟在裸机上可能更低。

## 4. 目标架构

```text
异步 I/O 面 (Tokio):   sources ── checkpoint 持久化 ── sinks
                          │            ▲                ▲
                          ▼            │ 冻结的状态页    │
同步计算核心:   router → shard 0..N-1（每个 shard 是一条 run-to-completion 的 Pipeline）
                共享: keyed-state 核心 · 内核库 · 状态表
驱动方式:   StreamSession.push()（调用线程）| StreamingRunner（连接器）| 分片执行器
```

### 4.1 同步计算核心

算子实现同步 trait。下面是示意，最终签名由 P1 规格确定。

```rust
trait Operator {   // 无 async、无 Box<dyn Future>
    fn on_data(&mut self, port: PortId, batch: &RecordBatch, out: &mut Emitter) -> Result<()>;
    fn on_watermark(&mut self, wm: EventTime, out: &mut Emitter) -> Result<()>;
    fn on_end(&mut self, out: &mut Emitter) -> Result<()>;
}
```

- Pipeline 内部的边是函数调用。发射的批进入该边的预分配出箱，按拓扑序处理。
- watermark、barrier 和 EOF 在同一调用栈内按序处理，保持“先发射已关闭的结果，再转发 watermark”和 FIFO 控制顺序。
- 内核里没有 `.await`。取消检查按 4–8K 行的 morsel 做。
- 指标是 Pipeline 私有的普通计数器，按需或周期性快照发布到 status。
- 预算在入口对整个 envelope 做一次准入（超大消息拒绝保持不变），在出口对输出做一次检查。
  内部边不排队，所以不逐边计费。

同一个核心有三种驱动方式。

**`StreamSession`（调用线程）。** 下面是示意。

```python
session = program.session(config=...)                 # 编译一次，持有原生状态
out = session.push({"ticks": batch}, watermark=ts)    # 调用线程内跑完，直接返回结果
session.checkpoint()
session.close()
```

- 没有线程跳转；Data 和 Watermark 合并成一次调用。
- `watermark=` 让调用方声明“到此为止已完整”，结果在同一次调用里返回。
  不传时沿用 `max_seen − 1 µs` 的生成规则，并按批推进。
- Python 侧缓存 schema 后直接导出 `RecordBatch`，另提供 NumPy 路径。
- checkpoint 是两次 push 之间的一个点，不需要 barrier 对齐。

**`StreamingRunner`（连接器）。** 保留，变成 source task、Pipeline、sink task 三段。
单源链路直接在 source task 内联执行；多源时入口有一个带预算的邮箱，一次唤醒排空全部可用消息。

**分片执行器。** 见 4.5。

迁移方式：现有 handler 在直调 harness 里已经能约 6 µs 跑完，先用一个“poll 到完成”的适配器包住它们，
`StreamSession` 就能在重写算子之前交付。真正会挂起的算子（走 DataFusion `execute_stream` 或 `gather_work` 的）
先留在旧路径。每个算子迁移到同步 trait 时删除其旧实现。

### 4.2 运行时拥有的状态表

- 算子不再实现 `checkpoint()` 和 `restore()`，而是在运行时拥有的状态表里声明状态：
  按 key 的列、定长环形缓冲、追加日志。
- 状态按 key group 分区，一个 shard 拥有若干 key group。
- 页带 epoch 标记，写时复制，配页级脏位图。快照就是冻结页引用，成本与脏页数成正比，算子不做编码。
  SQL 的 `DirtyGroups` 位图是现成范例。
- 批内原子性用 undo log（推广 `kernel/in_place.rs` 的稀疏 undo），或者“先预检，再不可失败地应用”。
  不再为每批克隆被触及的状态。
- 关闭 checkpoint 时脏位图是空实现。热路径只有一条，不再有 `.checkpointing()` 分支。
- 逻辑状态计费按列缓冲区的算术计算，不逐行，也不逐实体持有 `MemoryReservation`。

### 4.3 共享 keyed-state 核心

- 每批对 key 列做向量化哈希，经一张哈希表映射到稠密 group id。
- key 存在列式 append-only 存储里，常见类型有专用快路径。
- 规范 key 字节只在 key 首次出现和 checkpoint 时计算。
- group by、window、join、ASOF、rolling 实体路由和 distinct 共用这一套。

### 4.4 内核库

- 实体状态是 SoA slab：稠密 `EntityId`、定长 f64 环形缓冲、累加器列，全部是连续向量。
- 内核分两级，在计划期选定：
  - 标量级处理极小 append：id 到 slab 到累加器到输出槽；
  - run 级处理大批：按 `EntityId` 稳定分组，对每个实体的 run 把累加器放在寄存器里，
    一次预扫描证明没有 null、NaN 和 Inf 后走无分支循环；
  - 完整 tick（每实体一行）时跨实体并排更新。
- 按 (dtype, frame, 无空值证明) 的特化在计划期完成，不在每行做 enum 分发。
- 输出直接写入 `&mut [f64]` 加位图，不逐行调用 builder。
- rolling、window pane、array 和 SQL 共用一个累加器模块。
- SIMD 交付靠可自动向量化的循环加 `x86-64-v3` wheel。

### 4.5 按 key 分片

- N 条 Pipeline 各自独占状态分区。入口按 key 哈希路由一次，出口做确定性有序合并。
- barrier 和 watermark 按 shard 广播，在合并处对齐。
- 小批时 N=1，在调用线程内联；阈值由测量决定。
- `gather_work`、`entity_work` 和散落的 `spawn_blocking` 合并成一个 morsel 池，只服务无状态的大批工作。
- I/O 面和计算面分离，计算不再占用 Tokio worker。

### 4.6 Changelog 输出协议

聚合只输出变化的分组；需要完整快照时在边界处物化。默认仍是完整快照。

## 5. 新数值契约

### 5.1 `stable_v1` 的除法依赖

`stable_v1` 用 West/Welford 递推算 rolling mean：每进一个样本、每出一个样本，都直接更新 `mean` 本身
（`crates/calc-flow/src/array/rolling_mean_math.rs:63`）。

```rust
// 进窗                         // 出窗
count += 1;                     count -= 1;
delta = x - mean;               delta = y - mean;
mean += delta / count;          mean -= delta / count;
m2   += delta * (x - mean);     m2   -= delta * (y - mean);
```

- 新的 `mean` 要用上一步的 `mean` 经过一次除法才能得到，CPU 必须等这次除法完成才能开始下一行。
- 一次 f64 除法延迟约 13–20 个周期，每个窗口每行两次，dual SMA 是四次。
  这给出每窗口每行约 10 ns 的下限，与其他优化无关；实测是 16–20 ns。
- 同一实体内的行因此无法向量化。
- 这个累加器由 sum、mean 和 variance 共用，所以即使只要 window sum，也在为 `mean` 和 `m2` 的除法付费。
- `stable_v2` 预览版每行仍是这套递推，只是多了周期性重建。

TA-Lib 的写法是 `total += x_new; out = total / n; total -= x_old`。
依赖链上只有加减法，除法结果不回写状态，相邻行可以重叠执行，也能向量化。

### 5.2 新算术

- **sum 和 mean**：带补偿的滑动和，mean 在读出时才做除法。
  补偿采用“误差项单独累加、不回写主和”的形式，依赖链上每行只有一次加法。
  现有的 `CompensatedSum` 是 Kahan 形式，补偿项回写会把链拉长，只用在周期性重建里。
- **variance、stddev、covariance、correlation**：以每实体基准值平移后的补偿和与平方和，
  即 `stable_v2` 重建时已在用的方法。
- **周期性精确重建**：每实体每 max(W, 64) 个位置从环形缓冲重建一次，界定累计误差。
  节奏按实体位置计数，所以结果与批次切分无关。
- **不变的部分**：null、NaN 和 ±Inf 的语义分类（有效样本计数、`min_periods`、同号无穷得无穷、异号无穷得 NaN）；
  整数 sum 的精确检查；EWMA、decay、RSI 等递推。

### 5.3 可见后果

1. 所有 rolling mean、variance 和 covariance 的输出末位会变，批和流两条路径都变。
   现有逐位对照的测试和 fixture 要重建。
2. 窗口内的和超过 f64 上限时得到 ±inf，而 `stable_v1` 在这种输入下 mean 是有限值。
3. project-v3 schema 里 `numerical_profile` 的 `stable_v1` 和 `stable_v2` 两个枚举值被删除，
   显式写了它们的 project 文档会被拒绝；`schemas/project-v3.schema.json` 有预期的 diff。

### 5.4 验收方式

验收从“与旧输出逐位一致”改为三条：

- 相对精确参考值在声明的误差界内；
- 相同输入在任意批次切分下逐位一致；
- 从新格式 checkpoint 恢复后逐位续算。

精确参考 oracle 和误差界的定义在 P0 建立。

## 6. Checkpoint 协议

### 6.1 协议

1. 源切点和 barrier 注入不变。`StreamSession` 驱动下不需要 barrier。
2. 算子在 barrier 处冻结状态页引用，随即转发 barrier 并继续处理。
3. 持久化服务在后台写出冻结的页。每页哈希一次，每个 epoch 合并 fsync。
4. sink pre-commit 后等待 manifest。
5. 所有句柄持久化完成后发布 manifest v4。它仍是最后发布的唯一持久真相，然后 sink commit、source ack。
6. 最多一个在途持久化 epoch；下一个 barrier 到来时若上一个未完成则背压。

### 6.2 兼容性

- manifest 升到 v4，含按分区的状态句柄。旧 lineage 启动时给出明确错误，需要新建 lineage。
- 每个算子迁移时删除其旧 layout 和旧编码，不保留只读恢复。
- `tests/fixtures/v1/` 是历史 v1 对照证据，不受影响。

### 6.3 P3 规格必须定义的内容

- 失败模型的变化。现在的规则是 stage 失败不转发 barrier（`docs/runtime-envelope.md:571`）；
  新协议下 barrier 已经转发，在途 epoch 失败时该 epoch 作废，不发布 manifest。
- 取消、磁盘满、写失败、校验失败、worker panic 和 manifest 发布不确定时如何收敛。
- 在途 epoch 钉住的内存上限。
- 恢复时的完整性校验不得后移到对外确认之后。

### 6.4 未迁移的算子

尚未迁移的算子，其旧快照先当作不透明状态块接入新协议，落盘移到后台。
它们在各自重建的阶段落到状态表上。

## 7. 阶段与验收门槛

| 阶段 | 内容                                                                                                                | 验收门槛（估计值）                                                                                                                      |
|------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| P0   | 固定基线和验收用例；建立新算术的精确参考 oracle                                                                     | 每个后续阶段有基线、目标用例和停止规则                                                                                                  |
| P1   | 同步核心和 `StreamSession`，先覆盖 rolling 后接 projection；Python push 与直接导出；便捷流默认值；并入 #616         | Python 1 行 append 从 599 µs 降到 ≤20 µs（Arrow，已移到 P2）、≤5 µs（NumPy）；大批对照无回退                                            |
| P2   | Rolling 内核重写：新算术、SoA slab、undo log、所有输出进 typed plan、两级内核；rolling 和 projection 迁到同步 trait | 原生 1 行 ≤1 µs；每窗口每行 2–4 ns；6,400 行 64 实体从 329 µs 降到 ≤100 µs；Python 1 行 append ≤20 µs（自 P1 移入）和每次调用的分配预算 |
| P3   | Checkpoint 重做：写时复制页和脏位图、后台持久化、manifest v4                                                        | 已迁移算子开启未触发开销 ≤3%；1 s 间隔 ≥90%、100 ms 间隔 ≥75% 的关闭态吞吐                                                              |
| P4   | keyed-state 核心、列式分组状态、changelog 输出                                                                      | Grouped SUM 10M 行单核从 1.67 s 降到 ≤0.5 s                                                                                             |
| P5   | Join、interval join 和 ASOF 在 keyed 核心上重建                                                                     | Join 1M 单核从 78.5 ms 降到 ≤30 ms；interval 从 672 ms 降到 ≤350 ms                                                                     |
| P6   | 分片执行器、统一 morsel 池、`x86-64-v3` wheel                                                                       | Keyed 大批负载 4 核 ≥3 倍；小批路径不变                                                                                                 |
| P7   | Window 改 pane；其余算子迁移；删除旧路径                                                                            | 高重叠 hopping 用例有改善；核心只剩一条执行路径和一种状态格式                                                                           |

Arroyo 对齐作为独立门槛：在同一集成 harness 上 burst 吞吐不低于 Arroyo 的 0.8 倍，
在 P1 加原生连接器链路之后验收。

### 7.1 P0：基线与验收用例

- 在当前 main 重新测量并固定以下用例的基线，复用现有 harness：
  - 小批 warm rolling：1 行、99 行单实体，6,400 行 64 实体；分别经 Python 全链路和原生 dispatch；
  - TA-Lib 适配器参照；
  - grouped SUM：10M 行，以及 640 行小批；
  - join、interval join 和 ASOF：1M 行，单核；
  - checkpoint 模式：关闭、开启未触发、1 s 间隔；
  - Arroyo 集成路径。
- 建立新算术的精确参考 oracle，写下误差界的定义。
- 产出每个用例的命令、构建身份、基线数值、目标值和停止规则。每个后续阶段从中选不超过五个用例。

### 7.2 P1：同步核心与 `StreamSession`

- 同步驱动器和 `StreamSession`，Rust 与 Python 两侧。
- 首批准入的图：单源、rolling 后接精确投影、无 checkpoint。现有 handler 通过适配器运行。
- Python `push`、缓存 schema 的直接 `RecordBatch` 导出、NumPy 路径。
- 便捷流默认关闭 checkpoint；有序输入按批推进 watermark。
- #616 的 region 准入逻辑和测试并入，不单独合并。
- 停止规则：适配器方式达不到 ≤20 µs 时先停下重新定位成本，不继续后续阶段。
- 执行结果：门槛未达到；定位成本后，用户决定把 ≤20 µs 门槛移到 P2（见执行记录）。

### 7.3 P2：Rolling 内核重写

- 第 5 节的新算术；删除 `RollingNumericalProfile` 的两个旧值。
- 实体状态改成 SoA slab，直接建在状态表接口上，先用最简单的整表快照实现。
- 事务改成 undo log，对所有 group 种类和开启 checkpoint 的 job 生效。
- 去固定成本：计划期缓存 `RowConverter`、持久的 key 到 `EntityId` 路由、类型化的 `(ts, seq)` 末位身份、
  预计算的端点字符串、无锁的阶段计数。
- 所有输出种类进入 typed plan，包括 `lag` 和 `delta`。
- 两级内核，加完整 tick 的跨实体更新。
- 2026-11-08 计划里的 decay 增量内核和多窗口共享历史在这里落地。
- 有限（批）路径和流路径共用同一套内核。
- 删除：每批克隆、`RollingHistories` 双份状态、`ScalarValue` 通用路径、layout 1–6 及其编码。
- rolling 和 projection 迁到同步算子 trait（4.1），`StreamSession` 不再经适配器驱动它们。
- 验收另含自 P1 移入的 Python 1 行 append ≤20 µs（Arrow）和每次调用的分配与内存占用预算，
  同时报告进程内热循环和 harness 两种延迟。

### 7.4 P3：Checkpoint 重做

- 状态表的写时复制页和脏位图实现；第 6 节的协议；持久化服务；manifest v4。
- rolling 已在状态表接口上，切到新实现，内核不用改。
- 其余算子按 6.4 接入。
- 删除全 job 的 stage 锁和重复哈希。
- 门槛针对已迁移的算子（此时是 rolling）。group by、join 和 window 在各自迁移的阶段继承同一组门槛。

### 7.5 P4：keyed-state 核心与 group by

- keyed-state 核心（4.3）。
- SQL compact 聚合改成按 group id 索引的类型化状态列；undo log 取代每组候选；
  取消检查按 morsel 做，去掉每个 chunk 的 `yield_now`。
- 完整快照输出改为切片状态列，只重建脏块。
- changelog 输出作为显式选项。规格需定义初始基线、group 身份、更新顺序、flush 时机和恢复后如何重建基线。
- 扩大 compact 准入，减少“保留全部输入并重跑”的查询形状。
- 状态落在状态表上；删除 SQL compact 的旧 layout。

### 7.6 P5：Join、interval join 和 ASOF

- build 侧改成列式 run：批切片加 key-id、time 和 row-id 列。
- 非保留行通过 retain 位图跳过；定宽 schema 按批计费。
- 探测产生 selection vector，输出用 `take` 和 `interleave` 一次物化。
- 淘汰按 run 的前缀进行。
- 删除 SQL 探测层、三层 key 身份和内层循环的 async 计量。
- interval join 和 ASOF 迁到同一核心。
- 删除 join v1、v2 和 ASOF 的旧编码。

### 7.7 P6：分片与并行

- 分片执行器（4.5）。
- 统一 morsel 池，取代 `gather_work`、`entity_work` 和散落的 `spawn_blocking`。
- I/O 面和计算面分离。
- `x86-64-v3` wheel。

### 7.8 P7：Window 与收尾

- Window 聚合改成按 slide 的 pane 列式状态，建在 keyed 核心上；关闭窗口时合并 pane。
  先接入 COUNT、MIN 和 MAX；SUM 和 AVG 的 pane 合并次序由规格定义后接入。
- cross-section、distinct、order 和 snapshot SQL 迁移到同步 trait 和状态表。
- 删除旧的 async `StreamOperator`、已被替代的 runner 任务图部分、`gather_work`、`entity_work` 和所有剩余旧编码。

### 7.9 顺序与衔接

- P1 最先：它对应最大的那一层成本，其余阶段都建在它之上。它也最便宜证伪。
- P2 和 P3 的协议部分可以并行。
- P2 的 slab 建在状态表接口上，P3 只替换接口后面的实现。
- P4 先于 P5，因为 keyed 核心是共享的。
- P6 必须在状态可分区之后：此前的分片尝试败在逐算子的路由拷贝上。

## 8. 交付与测量规则

- 每个阶段是独立的规格、逐任务计划和实现周期。
  公共契约变化（P1 的 API、P2 的数值、P3 的协议、P4 的输出协议）先经过规格、API 和 critique。
- 分支、提交和 PR 遵守 [AGENTS.md](../../AGENTS.md)。每个阶段可以拆成多个可独立撤回的 PR。
- 每个行为变化先写失败测试。
- 测量遵守 [AGENTS.md 的 Verification](../../AGENTS.md#verification) 和
  [benchmark suite](../benchmark-suite.md) 的配对规则：
  每个实现项最多五个用例、合计 600 秒；执行前登记基线、候选、样本数和预算；
  证据不足即停止，不追加样本，不更换用例。
- 迁移即删除：一个算子迁移到新核心的同一批 PR 里删除它的旧实现、旧 layout 和旧编码，不长期维护两条路径。
- 没有证明真实收益的阶段不进入默认路径。

## 9. 可见契约变化

| 变化                                                                   | 阶段  | 影响                                  |
|------------------------------------------------------------------------|-------|---------------------------------------|
| 新增公开的 `StreamSession`                                             | P1    | 新 API                                |
| 便捷流默认关闭 checkpoint，按批推进 watermark                          | P1    | 结果更早可见；不再创建临时 checkpoint |
| rolling mean、variance 和 covariance 的输出末位变化；溢出窗口得到 ±inf | P2    | 批和流都变；fixture 重建              |
| project-v3 的 `numerical_profile` 删除两个旧值                         | P2    | 显式写旧值的 project 被拒绝           |
| manifest v4，旧 checkpoint 不可恢复                                    | P3 起 | 升级后需要新建 lineage                |
| 聚合可选 changelog 输出                                                | P4    | 新选项，默认不变                      |
| `x86-64-v3` wheel                                                      | P6    | 见第 10 节                            |

## 10. 未决项与风险

1. P1 是证伪点。适配器方式的 `StreamSession` 达不到门槛时，后续阶段的前提需要重新评估。
   P1 未达到 ≤20 µs，处理方式见执行记录。
2. `StreamSession` 的准入范围从单源链路逐步扩大。多输入算子的 watermark 对齐语义在扩大时定义。
3. 新算术的误差界数值，以及是否保留与 DataFusion 滑动 AVG 的 NaN 粘性兼容，由 P2 规格确定。
4. 异步持久化的失败模型由 P3 规格确定（6.3）。
5. `x86-64-v3` wheel 的打包方式由 P6 规格确定。单一 v3 wheel 会放弃不支持 AVX2 的 x86 CPU；
   一个 wheel 内带两份 native 模块并在 import 时选择，则与“核心 wheel 只含一个 native 模块”的发布不变量冲突。
6. 分片后的规范输出顺序需要确定性有序合并，其成本在 P6 实测。
7. 决定 6（changelog 默认值）未单独答复。
8. 所有收益数字是估计；多数现状数字来自 10 月到 11 月初的产物（3.8）。

## 11. 与既有计划和 PR 的关系

- [2026-11-08 计划](2026-11-08-ta-lib-arroyo-performance.md)：
  - 它的 P2（decay 增量内核）和 P3（多窗口共享历史）落在本文的 P2；
  - 它的 D1（持久化与处理重叠）是本文 P3 的协议；
  - 它的 D2（changelog 输出）是本文 P4 的一部分；
  - 它的 C1（快照分块复用）并入本文 P4，C2（hopping pane）并入本文 P7；
  - 它的 C3（批量 MIN/MAX）只在长 run 有热点证据时进入内核库；
  - 它要求保持 `stable_v1`、manifest v3 和既有状态格式，这些约束被本文的决定 2 和 3 取代。
- PR #616（compiled stream regions）：region 的准入逻辑和测试可复用，但 region 必须同时拥有入口和出口。
  它自己的验收门槛没有通过，并入本文 P1，不单独合并。

## 12. 证据索引

- 小 batch 分层与 TA-Lib 参照：PR #616 的本地产物 `target/compiled-region-performance/`（未入库）。
- 回调桥接成本：`benchmarks/rolling/callback-sparse-818d69c.json`、`.codex/artifacts/stream-dispatch/README.md`。
- Tokio worker 数对照：`benchmarks/rolling/scheduler-32-to-1-818d69c.json`。
- Arroyo 对比：`.codex/artifacts/arroyo-comparison/REPORT.md`、`.codex/artifacts/arroyo-attribution/REPORT.md`。
- Checkpoint 成本：`.codex/artifacts/specs/window-operator-async-checkpoint-flush.md`、
  `.codex/artifacts/analysis/issue406-checkpoint-modes.md`。
- Join：`.codex/artifacts/analysis/join-single-thread-performance.md`、
  `.codex/artifacts/analysis/join-remaining-gaps-2026-10-17.md`、
  `.codex/artifacts/analysis/issue406-eight-workload-diagnosis.md`。
- Grouped SUM 与并行尝试：`benchmarks/stock-positions/engine-core-sweep-20261107.md`、`benchmarks/sum-state/`。
- Rolling 阶段计数：`benchmarks/runtime-fusion/p3-20261108/native-counters.json`。
