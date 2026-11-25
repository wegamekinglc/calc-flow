# 水位回调的实体路由与数组读写

本次历史对照是 11.14 的 `7012b5b4` 与定时主线 `72072c1d`。后续主线 `e87b138f` 只作源码复核，未进入本实验。每轮十对的水位回调配对中位差为 +184.73 和 +185.31 ms；以下操作与计数对应到相同工作负载。现有生产指标能确定回调及阶段范围，细分操作均未单独计时。

| 水位处理步骤 | 11.14 生产路径 | 定时主线生产路径 |
|--------------|----------------|------------------|
| 分派 | 作业启用 managed checkpoint。in-place 准入退出，每批记录一次 CPU limit fallback，调用串行 `prepare_ordered_stream_with_history`。 | `prepare_ordered_update` 先检查 slab 准入，先于普通 serial 和并行分派。有资格且没有标量历史的状态可延迟独立历史维护。 |
| 识别本批的实体 | `prepare_touched_entities` 发现非空单 Utf8 键，按原字符串分配稠密的本地 ID，将首次出现的 64 个不同键一次性送入 RowConverter。该过程处于 `EntityResolution` 作用域。 | `route_slab` 还要求 resident 实体数恰为 1 才会走同一直接路由。首批 resident 为 0，后九批均为 64，于是每次 `encoded_entity_routing` 先对一百万行执行 RowConverter，再由相同的稠密路由算法按编码后的键分组。它测量内部 `entity_encode_ns` 和登记行数，却没有给公开 callback 建立路由阶段。 |
| 为数值内核准备 ID 和状态 | `prepare_stream_entities` 按每个 distinct 键生成本批局部状态数组，继续使用同一份逐行 `Vec<usize>`。后九批分别复制 64 个既有实体，合计 576 个，状态准备有嵌套作用域。 | `prepare_slab_entities` 由 distinct 键查找或新增持久实体。然后遍历逐行本地 ID，分配第二份含一百万个持久 `EntityId` 的向量。两份逐行数组同时存活，后者在 `NumericUpdate` 时钟开始前已构造。采样中公开 `copied_entities` 为 0。 |
| 数值及输出 | `finish_typed_fill` 可为本例编译专用 `MeanKernel`，使用本地 ID 定位本批状态。 | `fill_slab_rows` 尝试 full-tick 更新。准入按 distinct 顺序对逐行持久 ID 进行整批模式校验，成功后再遍历逐行 ID 和价格更新 ring 与结果。有序完整 tick 与逐行回退共用 `NumericUpdate`。两端各十次串行，全部一千万行在水位处理。 |
| 历史与交付 | 样本登记 12,800 个准备追加的历史行，11,520 个准备淘汰的行。 | slab 已承担这部分状态，以上两个工作计数为零。两端都准备十个输出块、一千万行，实际 sink 也收到一千万行。 |

两个版本都遍历每行判定实体。差别在 RowConverter 的输入行数，从每作业 **640 行变成 10,000,000 行**，这是对固定非空 Utf8/64 键 fixture 按可达源码分支推导的数，未在生产插入编码计数器。另一个独立成本是定时主线的第二份逐行持久 ID 数组；64 位平台上一百万个本地 usize 已占约 8 MB，还需一百万个由三个 32 位字段组成的 EntityId。这里按元素计数，峰值驻留字节仍须单测 allocator 或按真实 `capacity() * size_of::<T>()` 取证。释放外层容器、事务、builder 创建等未标注工作也可进入 Other；±56 ms 路由阶段的分类变化不能直接当作算法节省。主线 e87b138f 的紧凑 EWMA 准入已按实体保存持久 ID，按行 SMA 仍选 `SlabEntities::PerRow`，单 Utf8 的 resident==1 条件亦存在。

[11.14 分派](https://github.com/wegamekinglc/calc-flow/blob/7012b5b409553df04e29e6d7f9a11abf919cbf93/crates/calc-flow/src/operator/rolling/ordered_stream.rs#L422) · [11.14 路由](https://github.com/wegamekinglc/calc-flow/blob/7012b5b409553df04e29e6d7f9a11abf919cbf93/crates/calc-flow/src/operator/rolling/kernel.rs#L1265) · [11.14 状态准备](https://github.com/wegamekinglc/calc-flow/blob/7012b5b409553df04e29e6d7f9a11abf919cbf93/crates/calc-flow/src/operator/rolling/kernel.rs#L1711) · [11.14 MeanKernel](https://github.com/wegamekinglc/calc-flow/blob/7012b5b409553df04e29e6d7f9a11abf919cbf93/crates/calc-flow/src/operator/rolling/kernel.rs#L1427)

[定时主线路由](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/crates/calc-flow/src/operator/rolling/kernel/slab.rs#L523) · [两种键编码](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/crates/calc-flow/src/operator/rolling/kernel.rs#L3112) · [持久与本地 ID](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/crates/calc-flow/src/operator/rolling/kernel/slab.rs#L547) · [full-tick 准入和更新](https://github.com/wegamekinglc/calc-flow/blob/72072c1dc6c0874b79ccd8ce78103c97caec623d/crates/calc-flow/src/state/table/core.rs#L2035) · [后续 main 的 SMA / EWMA 选择](https://github.com/wegamekinglc/calc-flow/blob/e87b138fc37aae4d21d8e498833dca5b9e3b99e1/crates/calc-flow/src/operator/rolling/kernel/slab.rs#L77)

## 可独立验证的改动

1. 先复用仓库已有的非空单 Utf8 直接路由。在真正的有序 slab 水位入口加入测试计数器，输入固定四个实体轮转多轮，首批为空状态、第二批为既有多实体、第三批引入新键并改变首次出现顺序。分别约束 RowConverter 的输入数为本批 distinct 数；与通用编码比较键字节、每行本地 ID、每实体首行和计数，覆盖空字符串与非 ASCII。恢复后校验首批等价。对复合键与不可准入类型沿用通用编码，对不允许的 null 保留既有首错语义。已有的两个 direct routing 单元测试可提供 oracle，新断言应覆盖 slab 的实际分派。
2. 独立移除 numeric slab 的逐行持久 ID 展开。保留本地 ID 向量和按 distinct 保存的持久 ID 表；证明 tick 时核对逐行本地 ID 的模式，对持久 ID 保留分区、身份和存在性校验。回退通过 local→persistent 查表处理行。用 64 键重复轮次、末尾不足一轮、重复和漏键、跨批轮次换序、按不同顺序恢复的 ID 及检查点验证数值结果和顺序。工作计数或分配观测应在相同行数、不同实体数下证明额外持久 ID 的容量随 distinct 增长，数值 scratch 预算需要在等界与刚超界两侧验证；失败不改变已提交状态。
3. 精定位剩余时间时，可在两个版本对齐路由作用域，并在相同测试构建中分别记录编码、ID 映射、full-tick 校验和 ring 更新，核对独占区间守恒，同时用生产 wheel 保留无附加计时的公开对照。这能区分修复带来的真实工作量变化和把时间从 Other 移到已命名阶段的分类变化。
