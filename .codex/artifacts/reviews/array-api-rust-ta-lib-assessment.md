# calc-flow array 接口、Rust 下沉与 TA-Lib 核心评估

评估日期：2026-11-08（Asia/Shanghai）。本报告是源码与现有测试覆盖的调查，不是实现方案验收或性能测量报告。

后续实施约束：用户已明确不引入 TA-Lib 代码依赖。本报告保留调查时的候选比较，其中直接复用 TA-Lib 库/provider 的建议不再属于实施路线；当前方案以 [原生 Array 实现计划](../specs/native-array-runtime.md) 为准，采用 calc-flow 自研内核。

- calc-flow：执行 `git fetch origin main` 后固定在 [`7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`](https://github.com/wegamekinglc/calc-flow/commit/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75)，提交时间 2026-11-08 07:41:34 +0800。独立 worktree 位于 `target/worktrees/array-api-assessment`。
- TA-Lib：通过 GitHub API 确认其 `main` 为 [`58160778ecaab21e90eafd6e5462f6e833100af9`](https://github.com/TA-Lib/ta-lib/commit/58160778ecaab21e90eafd6e5462f6e833100af9)。所读 Rust manifest 版本为 `0.8.4`；此处不把主线 manifest 版本视为已验证的发布包版本。
- 当前用户工作分支和未提交变更不作为能力基线；未修改产品代码、未构建 native、未运行全量测试或性能测试。测试名称仅表示已有证据覆盖，不表示本次重新运行通过。

## 结论

当前 array 是一个**受限的外部数值计算扩展层**：独立数组表达式覆盖基础运算，table 与静态矩阵的组合已有类型分析、行对应关系校验、执行与恢复接入。但它还不是可独立使用的完整数组计算 API，也没有与 Python array 功能对等的内置 Rust 数值接口。

值得优先推进两条线：

1. 将已有 Rust rolling 数值核心整理成可复用的类型化接口，让直接数组调用、Arrow table 和 streaming 共用算法与语义。
2. 补齐 table/array 的 native CPU 桥接，并用可选的 TA-Lib Rust 适配器扩展技术指标。NumPy/JAX 继续承担成熟矩阵与设备计算。

TA-Lib 当前主线已有官方纯 Rust 库及真正保留状态的 streaming API，不能再按“只有 C 库、Rust 只能 FFI、每次重算历史”评估。与此同时，其 handle 明确不序列化，部分窗口 stream 仍是 O(window)，指标语义也不等同于 calc-flow；直接整体替换并不合适。

## 1. 计算能力覆盖

需要分开看三个层级，避免把“可以声明”“可以执行”“底层 NumPy 本身具备”混为一谈。

| 层级                                         | 已有能力                                                                               | 当前边界                                                                                                |
|----------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `Batch.from_array` + external `expression@1` | NumPy/JAX；单输入 `x`；`+ - * / @ **`、正负号；`sum/mean/max/min/transpose/reshape`    | 受限 AST；归约无 `axis/keepdims` 参数；无任意函数、索引切片、多命名数组或自定义 array UDF 引用          |
| 高级 `ArrayExpr`                             | 算术、比较、布尔表达式；backend/dtype/shape/lineage 分析；`linalg.from_columns/matmul` | standalone array output 仅能声明和分析，不能经 `Program.collect/compute` 返回；可执行矩阵片段有固定形状 |
| table 上的 `row/ts/cs`                       | 丰富逐行数学、rolling/cumulative 统计、EWMA、横截面和 SQL                              | 输入输出仍是列/table；这些不是已开放的 standalone array 方法，不能计入 array API 完整度                 |

来源：[array provider](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/calc_flow/array.py)、[ArrayExpr](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/calc_flow/symbolic/expr.py)、[能力矩阵](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/docs/symbolic-api.md)。

### 基础数组计算：够用，但覆盖窄

`x - mean(x)`、缩放、简单广播、reshape、转置、matmul 已可执行。批式表达式只识别输入名 `x`，矩阵乘法可组合这个输入及其变换，不等于提供多输入通用张量图。

尚未在该 array API 开放的常见能力包括：比较/条件选择的字符串表达式、slice/gather/scatter、concat/stack、显式 axis 归约、排序/分位数、丰富逐元素数学函数、FFT、分解/求解器、自动微分、通用 array rolling，以及任意多个命名 array 输入。高级 `ArrayExpr` 已支持比较和布尔声明，不能与字符串表达式的 allowlist 混写。

NumPy/JAX 支持更多操作，不表示 calc-flow 已把这些操作暴露出来。当前代码逐节点调用 namespace，没有对整个表达式显式 `jax.jit` 的图级编译，也没有提供梯度接口。JAX 后端能力应描述为保留 JAX 数组、设备和受支持运算，不能宣传成完整 JAX 图集成。

流式 `expression@1` 进一步限制为依赖 `x` 的逐元素表达式：拒绝所有函数调用和 `@`，因此 `x - mean(x)` 只适用于 batch。静态权重矩阵走另一个已证明逐行独立的 `symbolic_matrix@1` 路径。

### 类型、安全和资源边界较扎实

- 输入所有权、防御性复制、不可写 NumPy 结果、metadata 保留、backend 校验已有明确代码和测试。
- NumPy 接受 Array API 数值/布尔 dtype；对象、结构化等类型会拒绝。JAX 的 64 位类型受 x64 配置约束，matrix 路径拒绝静默窄化。
- AST 限制为 4,096 字符、128 节点、24 层；幂指数限制绝对值不超过 64；reshape 有 rank/维度/元素数约束，运算输出上限为 1,000 万元素。它们是特定路径的防护，不是整体峰值内存保证。
- `_parse_valid_expression` 具有 256 项缓存；NumPy 结果已有内部 owned-buffer 接管路径，不能笼统说“每个输出都再复制一次”。

对应现有测试：[test_array.py](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/tests/test_array.py)，覆盖所有权、dtype、非法 AST、输出大小、backend 不匹配、copy ceiling 和 JAX residency。

### 时间序列与技术指标：核心覆盖显著强于 array 表面

Rust rolling manifest 已列出 22 个 kernel 条目，包括 lag/delta、count/sum/mean、variance/stddev、min/max、covariance/correlation、EWMA、累计统计、difference、argmin/argmax、unique_count 和 decay。公开 `ts` 还提供组合函数；例如 `ema` 是 `ewma` 别名，`macd` 为两条 EWMA 的差。

这说明“高性能数值核心从零开始”不是正确方向。已有 typed transitions、直接 mean 快路径、实体并行和受条件限制的 decay 优化，应先检查可复用边界。manifest 的复杂度标签不代表所有 dtype、window、输入顺序都命中相同快路径；实现仍有 admission 和 fallback。

但这也不等于 TA-Lib 指标覆盖：当前 `ts.macd` 没有 TA-Lib MACD 的 signal/histogram 三输出契约；RSI、ATR、ADX、KDJ、众多 candlestick 等没有对应的内置完整指标 API。可用基础算子组合，不等于已有经过兼容验证的指标实现。

来源：[rolling manifest](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow/rolling-kernels.json)、[ts/cs/row API](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/calc_flow/symbolic/ops.py)、[Rust kernel](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow/src/operator/rolling/kernel.rs)。

覆盖度判断：基础数值变换中等；通用 ndarray 操作偏低；table 因子与状态计算较强；TA-Lib 兼容技术指标覆盖有限。由于目标函数集合没有约定，不给没有分母的百分比。

## 2. 与 table 的结合紧密性

### 声明、校验与生命周期结合较紧

标准路径是：

```text
Arrow table / SQL
  → linalg.from_columns
  → 元素级表达式 + 一次 matmul(static weights)
  → table.attach_columns
  → 后续 SQL / table 输出
```

编译器追踪 backend、dtype、shape 和 row lineage，校验矩阵结果回贴的是同一张 table 的同一组行；所选列顺序有语义。静态权重在打开 source 之前冻结，每个 stream job 只放置一次；声明进入 plan fingerprint，权重值 digest 进入 checkpoint，恢复时拒绝权重变化。

实际组合强于“只在末端外挂一次矩阵”：`test_sql_composes_with_registered_matrix_and_static_weights` 覆盖 SQL → matrix attachment → SQL → collect；`test_window_and_independent_matrix_preserve_static_bindings` 覆盖 event window 和独立 matrix 的多输出编译及静态绑定。不能把局部 lowering helper 的单输出限制误当成整个 Program 只能有一个输出。

来源：[SQL 组合测试](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/tests/test_sql_pipeline.py)、[独立多输出测试](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/tests/test_symbolic_event_window_lowering.py)、[静态输入设计](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/docs/symbolic-design.md#static-values-and-matrix-placement)。

### 可组合形状与物理执行仍有限制

- 单个可执行 symbolic matrix 片段要求一个名为 `weights` 的静态参数，且它只出现一次，直接作为唯一一次 matmul 的右操作数。不能自然表达多权重、多层矩阵链或动态右矩阵。
- `from_columns` 要求所选列为同一 dtype，实际 matrix 执行要求非空 table、所选列无 null、primitive numeric。低级 `table_matmul` 的多个数值列可以做 safe common dtype promotion；不要把两层的类型边界混为一谈。
- 低级 `table_matmul@1` 返回 array `Batch`，且 batch-only；高级 `symbolic_matrix@1` 返回 table，支持静态 RHS 的 batch/stream。
- SQL 边界已支持后续 table 处理；无 SQL/其他分段触发条件时，直接 `attach_columns → filter/project` 的 lowering 路径仍有缺口。这是对 lowering 分支的静态推断，未在本次动态复现，宜表述为“组合路径不完全统一”。
- event-window、ASOF 等组合有各自边界，不能从独立多输出可行推导任意串联都可行。

执行上，Rust operator 通过 `spawn_blocking` 进入 `Python::attach` 调用 provider；NumPy/BLAS 或 JAX 完成数值工作。不能把整个计算都归因于 GIL，但每批仍存在 host callback、校验和调度成本。

`_numpy_table_matrix` 逐 Arrow chunk 取只读视图，再填入一块 dense matrix：取列视图可零复制，不等于整体 table→matrix 零复制。symbolic JAX 路径还会把 table 上传设备，再用 `np.asarray(result)` 把结果带回 host 构造 Arrow 列。低级 JAX `table_matmul` 返回 device array，没有这个结果回传步骤。

matrix segment 已合并为每批一次 provider 调用，但不是 DataFusion 内可与前后 SQL 一起优化的数值算子，也不意味着内部逐元素中间结果被完全消除。现有 `copy_bytes` 是逻辑传输统计，尤其不能代替峰值内存/RSS。

来源：[matrix lowering](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/calc_flow/symbolic/lower/strategies.py)、[analyzer](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/calc_flow/symbolic/analyzer.py)、[provider 执行](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow-python/src/provider.rs)、[matrix 测试](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/python/tests/test_symbolic_matrix.py)。

判断：语义与生命周期集成较好；组合规则中等；物理布局、执行优化和跨语言成本仍有明显改进空间。

## 3. 下放到 Rust 库的可行性

### 已有基础，不是空白

Rust 已公开 `BatchKind::Array`、`ExternalPayload`、`Batch::external`，以及 `static_array_bool/int/uint/float`、`StaticArraySnapshot`、`StaticArrayValues`。provider factory 可以注册 batch 和 stream 两种生命周期；无需 Python 即可实现数值 operator。

Rust 示例 06 已实现中心化，示例 07 和 `support/matrix.rs` 已实现 table×静态矩阵，以及 stream 静态权重的复用。这证明宿主、端口、图执行与静态输入契约已经具备。**示例自定义 provider 不等于库已经内置通用数组算子**：其形状、列名和数值处理是示例特定的。

`static_array_snapshot()` 每次复制 values/shape/nulls 等内容，且浮点 carrier 用 `Vec<f64>` 表示 float32/64，整数使用宽位 carrier；没有通用 strides/device/complex 数组视图。它适合冻结、摘要与 provider 放置，不宜直接变成每批计算的核心 buffer API。

来源：[Batch 与 snapshot](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow/src/batch.rs)、[Rust 中心化示例](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow/examples/06_numpy_array.rs)、[Rust matrix provider 示例](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/crates/calc-flow/examples/support/matrix.rs)。

### 建议分层

```text
Python 表达式 / Rust 直接调用
                ↓
版本化、类型化数值计划 + 只读输入视图
                ↓
纯数值 kernel / 有状态 recurrence
                ↑
Arrow table adapter / stream operator / NumPy-JAX provider
```

这只是建议架构，名称不表示现有 API。DataFusion 继续作为唯一 table 引擎；纯数值 kernel 为 Arrow/DataFusion/operator 服务，不另建竞争的 table runtime。

| 下沉项                       | 可行性                          | 必要工作                                                                            |
|------------------------------|---------------------------------|-------------------------------------------------------------------------------------|
| 逐元素、基础归约             | 高                              | 类型化计划；dtype promotion、overflow、null/NaN 规则；只读 view 和拥有输出的 buffer |
| table × 静态矩阵             | 高                              | 生产级 native provider；权重缓存；小矩阵列式计算和大矩阵 dense/BLAS 路径的明确选择  |
| 已有 rolling/cumulative 核心 | 高，但需重构                    | 从实体分组、Arrow、预算、checkpoint 协调中提取数值层；保留现有转移顺序和恢复格式    |
| TA-Lib 技术指标              | batch 高；durable stream 有条件 | 输出行对齐、warm-up、语义适配、状态 codec 或可证明的恢复方式                        |
| 完整 NumPy/JAX 通用张量 API  | 范围大，暂不建议整体搬迁        | GPU、复杂 dtype、广播、布局、线代、自动微分不由 TA-Lib 解决                         |

优先把数据搬运和 Python callback 移出小型 CPU 数值热路径，比单纯把 AST parser 改成 Rust 更可能有收益。后者单独实施仍可能保留 callback 与 pack/unpack 成本。大矩阵不能假设手写 Rust 循环胜过 NumPy 所调用的优化 BLAS；需要保留后端选择。

公开接口应提供稳定的类型、形状、参数、错误路径和输出契约，不直接暴露当前 `pub(super)` 状态结构。CPU-local kernel 保持同步；stream/operator 继续拥有异步生命周期、取消和资源预算。`calc_flow` 公共 fallible API 使用 `calc_flow::Result<T>`。

如新增持久化 recurrence，必须进入版本化状态编码、参数 fingerprint、内存预算、恢复一致性与 batch/stream 等价测试。不能仅把私有 kernel 改成 `pub` 就认为下沉完成。

## 4. TA-Lib 高性能核心调查

### 当前源码形态

该固定提交的 `ta_func_list.txt` 共 240 个非空函数条目，其中 61 个 `CDL*`。这是技术分析函数清单，不是完整 Array API，也不是 240 个相互独立的数学原语。C 和 Rust 实现由同一套 canonical 定义生成；调查同时读取了 C SMA/EMA 与 Rust SMA/EMA/RSI/MAX/VAR。

官方 Rust API 接受借用 slice、调用者提供的独立输出 slice，并返回 `OutRange { beg_idx, count }`；每个函数有 lookback。immutable `Core` 保存设置，可跨线程共享。stream 提供 `open/open_and_fill/update/peek/value/advance`。

来源：[函数清单](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_func_list.txt)、[Rust 库源码](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/lib.rs)、[官方 Rust API](https://ta-lib.org/api/rust/)。

### 值得借鉴的设计与算法

| 观察                                        | 性能意义                                              | calc-flow 的落点                                                 |
|---------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------|
| typed contiguous slices + 调用者提供输出    | 验证和分配移到循环外；避免逐元素动态 dispatch         | 直接 Rust 数值接口；Arrow contiguous non-null primitive 快路径   |
| SMA 滚动和 + ring buffer                    | 初始化后每次增减一个元素，O(1) update、O(window) 状态 | 已有 mean/rolling 核心，优先比较布局与调用成本，不重复实现       |
| EMA/RSI 递推                                | 每条序列保留少量状态，避免历史重算                    | 将 recurrence 同时用于数组 batch 与实体 stream；保留语义和持久化 |
| `open_and_fill`                             | 一次热身既产生历史输出又形成持续计算状态              | 可借鉴 batch 初始化与后续增量统一的接口形状                      |
| MAX batch 使用 Van Herk/Gil-Werman 分块扫描 | 固定比较次数、较规则循环，适合编译器优化              | 可作为有限、排序完成、无特殊 null 语义场景的候选 fast path       |
| VAR 使用 shifted sums 与条件重建            | 减少大均值小方差下的消减误差，并控制长期漂移          | 比较数值误差与长期状态稳定性，不能只比较吞吐                     |
| EMA 等采用 FMA；部分入口运行时选择硬件 FMA  | 减少运算/舍入成本，硬件路径受控                       | 兼顾结果契约；不代表所有递推可沿时间轴 SIMD                      |
| 一套定义生成多语言及 batch/stream 实现      | 降低语义漂移，支持参考实现验证                        | 扩展已有 rolling manifest 的契约与差分测试更现实                 |

算法证据：[SMA](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/ta_func/sma.rs)、[EMA](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/ta_func/ema.rs)、[RSI](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/ta_func/rsi.rs)、[MAX](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/ta_func/max.rs)、[VAR](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/src/ta_func/var.rs)。

### 不应直接照搬的部分

1. **状态持久化。** TA-Lib 明确把 stream handle 设计为不可序列化，恢复方式是保留历史后重开。对于 EMA/RSI 等递推，仅保存 lookback 长度的尾部通常不能恢复原状态。calc-flow 需要自己的版本化 codec、上游支持或可承受完整历史回放的明确方案；`Clone` 不能作为 durable checkpoint。
2. **最坏复杂度。** TA-Lib MAX stream 缓存极值索引，过期时重新扫描；单调下降等输入可达到 O(window)/update。文档明确说这不是摊销 O(1)。LINEARREG、CORREL 等部分 stream 也仍需窗口重算。已有 calc-flow typed extrema/statistics 可能更适合流式路径，不能因 TA-Lib 标签就替换。
3. **缺失值与 warm-up。** TA-Lib stream 拒绝 NaN/±Inf，`advance` 只移动输出范围、保留状态和值，不能默认等同于 Arrow null 的 skip/reset/propagate 策略。lookback 前无输出也必须映射为 table 的前导 null 或另一个明确约定，不能错位回贴。
4. **EMA/MACD 语义。** calc-flow EWMA 首个有效值初始化，默认 `min_periods=1`；TA-Lib EMA 以一段历史平均数初始化并受 lookback/unstable period 影响。calc-flow `macd` 仅两条 EWMA 之差；TA-Lib MACD 提供三输出及不同初始化安排。建议单独标明 TA 兼容语义，不静默替换现有 `ts`。
5. **流运行时范围。** TA-Lib handle 不承担实体分区、event time、水位线、乱序/迟到、replay cursor、背压、checkpoint barrier 和交付证明；这些仍由 calc-flow 管理。
6. **安全与依赖。** 所读 Rust manifest 的 MSRV 为 1.88，低于 calc-flow 的 1.94.0，版本上无明显门槛；主库禁止 unsafe，但伴随 dispatch crate 有受 CPU 特性检测保护的 unsafe 调用。不能宣称整个依赖树无 unsafe；也不需要因此放宽 calc-flow 的 lint。接入时仍须核验固定版本和实际跨平台构建。

来源：[stream 设计与复杂度](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/docs/streaming-api-design.md)、[官方 stream 契约](https://ta-lib.org/api/rust/stream/)、[Rust manifest](https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/output/rust/library/Cargo.toml)。

### 接入路线选择

- **优先候选：官方纯 Rust TA-Lib 的可选适配器。** 先补充缺失指标的 batch 能力，固定版本和兼容语义；不必绕 Python，也不必先写 C FFI。正式 durable stream 接入以恢复能力达标为前提。
- **长期核心：整理 calc-flow 自有共享 kernel。** 保留与 null、窗口、实体和 checkpoint 已对齐的 rolling 核心；从 TA-Lib 借鉴 tight loop、warm-up、输出分配和数值稳定性设计。
- **C FFI：备用路线。** 只有 Rust 版本缺口或实测优势值得付出绑定、生命周期与全局设置管理成本时再考虑；当前不构成首选理由。

## 5. 建议实施顺序与验证边界

1. **先统一能力契约。** 明确 array 与 column 的区别；补齐组合路径；锁定 dtype、null/NaN、overflow、lookback、EMA seed、多输出和不修改输入的行为。
2. **再做最小 native CPU 闭环。** 提取一组已存在 kernel，开放直接 Rust 调用并接回 Arrow/table；提供可选 native matrix provider。不要以“支持全部 NumPy”为首期目标。
3. **按需补技术指标。** 优先试接 RSI/ATR/完整 MACD 等现有覆盖缺口，batch 先行；持久化状态设计完成后接入 managed stream。
4. **用有限测量决定后端。** 分开测数值 kernel 与端到端时间，记录 pack/unpack、provider 调用、权重放置和输出转换；不能用 TA-Lib 单指标循环吞吐推导整个 calc-flow runtime 加速比。

若进入性能任务，建议最多五个独立 case：逐元素链、窄矩阵回贴、SMA、EMA、对抗性单调序列 MAX。前三类分别定位 callback/中间数组、跨边界复制和窗口更新成本；EMA 暴露递推状态与语义差异；MAX 检查最坏复杂度。基线是本报告固定 main，候选是完成实现后的固定 commit；各 case 先验证相同语义，再固定 10 个配对样本，总墙钟上限 10 分钟（包含 fixture、warmup、启动、正确性和清理），构建预算单列。当前只是后续建议，没有执行这些测量，也没有速度提升结论。

研究验证已检查固定 main 的相关源码、公开文档与现有测试；独立 specialist 复核特别纠正了 SQL 组合和多输出判断。工作树 schema 无漂移，报告仅新增本文件。跨平台发布兼容性、直接 attach→filter 的动态复现、以及任一数值性能优势，仍需后续专门验证。
