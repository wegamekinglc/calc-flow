# 原生 Array 与共享数值核心：实现计划

日期：2026-11-08。状态：设计交付，尚未实施。工作项 slug：`native-array-runtime`。

设计审查：`cf-api-designer` 已复核最小接口和接入边界，`cf-critic` 最终专项审核 Pass。精确签名与数值 oracle 在 A0 固定；此状态不表示实现或 CI 已完成。

## 1. 目标与基线

将 calc-flow 现有数值能力开放为无需 Python 的 Rust API，并让 Python array、Arrow table 和 managed stream 共用数值实现。补齐原生 CPU 数组执行、table/array 桥接和有限的序列指标，同时保留既有 NumPy/JAX、table、stream 和恢复语义。

本次重新 `git fetch origin main`，基线仍为 [`7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`](https://github.com/wegamekinglc/calc-flow/commit/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75)。前置调查见 [array 能力评估](../reviews/array-api-rust-ta-lib-assessment.md)。文中的新 API 名称为拟议表面，不表示当前可运行。

与 [既有性能计划](https://github.com/wegamekinglc/calc-flow/blob/7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75/docs/plans/2026-11-08-ta-lib-arroyo-performance.md) 的关系：

- 该计划仍管理 decay、共享历史、运行时融合、checkpoint 重叠和 changelog 工作。本计划管理 array 数据面、公开接口、适配和指标覆盖。
- 当前 main 已包含 PR #565 的 decay 实现；计划文件中的旧执行记录不覆盖提交事实，不重复实现 decay。
- 多窗口共享历史未作为本基线已交付能力。若其先合并，数值抽取以合并后的实现为基线；若未合并，保留当前历史 owner 接口，不复制另一套共享历史实现。
- 本计划不重启已因性能验收撤回的融合方案。消除 Python callback 与表格打包成本，不要求扩大 runtime task fusion。
- 现有工作区有其他未提交工作；实施从届时最新 main 建立独立 `feature/` 分支，不混入这些变更。

## 2. 交付范围与完成定义

完整主线分为三个可独立发布的里程碑：

| 里程碑          | 必须交付                                                                          | 完成判断                                                                              |
|-----------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| M1 原生数值闭环 | 最小数组存储、Rust 直接计算、共享 SMA/EWMA、Python eager API、原生 batch provider | 无 NumPy/JAX/Python 的 Rust 用例可运行；Python 与 table 的 mean/EWMA 复用同一数值实现 |
| M2 图与流闭环   | 原生二维矩阵、table 组合、独立 ArrayExpr batch 收集、原生有状态 stream 与恢复     | SQL/array/table 组合与分批不变性通过；managed restart 恢复状态、绑定和顺序            |
| M3 指标扩展     | native RSI、ATR、完整 MACD 分量接口                                               | 新指标的 warm-up、缺失值、输出对齐、batch/stream 与恢复合同通过                       |

通用大矩阵后端、更多既有 rolling 核心开放与自研指标扩展，是有进入条件的后续工作，不阻塞 M1–M3。完整 NumPy 兼容、任意 N 维 tensor、GPU runtime、自动微分、FFT/分解、稀疏矩阵、任意 Python callback 不在本计划承诺范围。

用户明确约束：不引入 TA-Lib 代码依赖。TA-Lib 仅作为算法、接口和数值边界的研究参考；不加入 TA-Lib Rust crate、C FFI、Python wrapper 或其间接封装，不 vendoring/移植其生成源码，也不将它作为必需的构建或测试依赖。指标使用 calc-flow 自有 Rust 实现，复用现有数值与状态基础设施。

兼容承诺：

- `compute/collect` 继续返回 table；不把它们改成 table/array 混合返回。
- 现有 NumPy/JAX 注册、表达式 allowlist、输出类型、数值语义和设备行为保持。
- 现有 `ts.ema` 与 `ts.macd` 保持其 EWMA/快慢线差语义；新能力另有明确入口。
- `Batch` 仍是唯一图数据 envelope，沿用 `BatchKind::Array`；DataFusion 仍是唯一 table 引擎。
- 保持 Rust 2024、当前 MSRV 和 `unsafe_code = "forbid"`。首期不新增独立 crate，避免先扩大包和发布矩阵。

## 3. 架构与所有权

```text
Rust array 函数                 Python eager / 现有 ArrayExpr
       │                                  │
       │                        PyO3 / 类型化数据计划
       └────────────────┬─────────────────┘
                        ▼
           共享数值 kernel 与 recurrence
                        ▲
                        │
   Arrow 列适配 / native provider / 现有 RollingOperator
                        │
             Batch、图执行、StreamingRunner
```

职责分离：

- 数值层：只读输入、数学状态转移、输出缓冲；不依赖 Python、DataFusion Session、event time 协调或 checkpoint I/O。
- 适配层：dtype/validity、Arrow chunk、静态参数、行对齐、实体分区与输入顺序。
- runtime：任务调度、取消、预算、barrier、source cursor、交付和持久化。

M1 中 array/table 共享实现的承诺具体指 rolling_mean/EWMA。普通 table 逐行表达式与 SQL 继续由 DataFusion 执行；本计划不为统一代码路径而替换全部 table 数学算子。

### 3.1 最小数据面

拟议类型：`calc_flow::array::FloatView<'a>` 与 `NativeArray`。前者是已验证的连续一维 f32/f64 借用视图，可附 validity；后者是私有字段、不可变、拥有 typed buffers 的输出。内部复用 Arrow 的安全 buffer 所有权设施，避免自行实现不安全内存管理。

M1 接受一维数值输入，输出可为向量或标量；比较/where 所需布尔值按实际功能引入。M2 才增加二维矩阵 descriptor。shape/dtype 枚举保留演进空间，但不预先实现任意 stride、device 或未使用的空模块。

标量 `shape=[]` 为一个元素、一行；向量 `[n]` 为 n 个元素、n 行；矩阵 `[n,m]` 为 n×m 个元素、n 行。validity 使用 true=有效的逐元素 bitmap，长度与元素数一致；没有 bitmap 表示全部有效。不能把 matrix 元素数作为流边 row count。

- Rust 借用视图只用于同步计算；跨异步任务的输入必须由 `NativeArray`/`Batch` 持有。
- Rust 输入 slice 不修改；Python 可变 NumPy 输入在释放 GIL 或移交 worker 前形成拥有的快照。
- Arrow primitive array 可共享被视为只读的 buffer；正确处理 slice offset 与 bitmap offset，不强制 `combine_chunks`。
- 输出 buffer 属于执行器，可内部复用；不要求公共调用者提供会被写入的输出数组。
- `NativeArray` 通过 `ExternalPayload` 接入 `Batch`，backend 为显式的 `native`；`.array` 对 native 返回该 owned 对象，其他 backend 保持原类型。
- `StaticArraySnapshot` 保持冻结、digest、交换职责，不扩展成计算缓冲。native 静态参数走 Rust 放置路径；冻结时必要的宽化复制与计费仍保留，不能绕过 canonical digest，也不能每批重新 snapshot。

### 3.2 明确数值语义

| 类别              | 本计划的规则                                                                                                                                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dtype             | M1 数值输入 f32/f64；布尔只用于比较、逻辑和选择；不隐式接纳 integer/complex                                                                                                                                   |
| promotion         | 同 dtype 逐元素保持 dtype，f32 与 f64 混合升为 f64；Python float literal 为 f64，整数 literal 首期仅接受 ±2^53 内并精确转 f64，bool 只参与布尔运算；保留 f32 需要显式 typed scalar，不通过 NumPy/JAX 探测类型 |
| narrowing         | 仅显式 cast 可缩窄，f64→f32 按 IEEE round-to-nearest ties-to-even；有限源值溢出为 Inf 则报错；非法常量和不支持的转换在执行前报错                                                                              |
| 普通逐元素        | null 传播；有效 NaN/Inf 保持 IEEE 运算语义；除零、sqrt/log 的特殊值按固定 native 数学合同处理，不转为隐式 null                                                                                                |
| 普通归约          | 默认 null 传播；`skip_nulls=True` 显式跳过 null，不把 NaN 当 null；无有效样本返回 invalid scalar；全体归约的 sum/mean 用 f64 累加并返回 f64，min/max 保留输入 dtype                                           |
| 归约稳定性        | sum/mean 使用精确二进制累加，最终按 ties-to-even 舍入；mean 在舍入前除以有效计数，具体 oracle 见 A0 API note；不承诺与 NumPy/BLAS 位级一致                                                                    |
| rolling_mean/EWMA | 直接沿用现有 ts/kernel 的缺失值、Inf、min_periods、窗口行计数、修复及 readout 规则；f32 使用 f64 状态和 f64 输出                                                                                              |
| 输出对齐          | rolling 每个输入位置对应一个输出位置；warm-up 用 validity 表示，不压缩输出、不移动行                                                                                                                          |
| 窗口              | 首期是行窗口；window > 0，1 <= min_periods <= window；EWMA span > 0、min_periods > 0；后者不套用 window 上界                                                                                                  |
| 空输入            | 逐元素/rolling 返回空结果；归约返回 invalid scalar；二维空行矩阵返回具有已知输出宽度的空结果                                                                                                                  |
| 输入验证          | shape 乘积、buffer 长度、bitmap、dtype、输出尺寸和预算均在分配/状态提交前验证，使用 checked arithmetic                                                                                                        |

普通归约与 ts rolling 是不同 API，不用一个全局 `ignore_missing` 开关覆盖它们。采用原生 Array API 运算子集的明确规范，并给 nullable 扩展单独说明；不宣称完整 Array API 标准实现。

上述 Python 整数字面量上限由声明前端在 JSON 编码前检查。A3 原生 factory
接收已解析的 `JsonMap`，只能检查 `serde_json::Number` 保留的表示：i64/u64
整数表示限于 ±2^53，有限 f64 表示保留完整范围。超出 i64/u64 的原始整数
token 可能已经被上游舍入为 f64；factory 无法恢复其原始写法或精度，也不声称
仍能验证原 token 上限。此处明确修订原先过强的 token 承诺，具体边界见
[A3 接口修订](../api-notes/native-array-runtime.md#a3-已解析数值字面量边界)。

### 3.3 预算、错误与可观察性

`ArrayExecutionOptions` 统一传递输出/临时内存限额。直接调用默认保留现有 1,000 万输出元素上限，并增加默认 256 MiB 的本次执行分配预算；它覆盖 owned 输出和临时缓冲，不包括借用输入与进程其他内存。允许显式调整，checked size/overflow 检查始终有效。有状态 handle 另有 `max_state_bytes`，默认 256 MiB；managed stream 使用 operator/job 的预算而非新增一套互不相干的内存池。

该执行 options 随 A2 的真实 kernel 交付。A1 只构造、验证及接管调用方已有的 Arrow/Vec 存储，实施 descriptor/visible/retained byte 的 checked 计数，不建立未使用的执行预算类型，也不把输入大小误当作输出元素上限。精确 A1 签名、字节公式及 A2 sum/mean 的精确二进制累加 oracle 见 [A0 API note](../api-notes/native-array-runtime.md)。

外部 payload 的逻辑可见字节、实际 retained allocation、scratch reservation 分开记录。共享 Batch 继续按每消费者进行边计费；不能用 `copy_bytes` 冒充峰值 RSS。bitmap、shape、输出、候选状态、checkpoint capture 和首次静态放置均进入各自预算，拒绝发生在不可逆提交之前。

错误至少定位到 `inputs.<name>.dtype/shape/validity`、`options.window`、`outputs.<name>.bytes` 或具体节点。Rust public fallible API 使用 `calc_flow::Result<T>`；PyO3 继续映射到现有 Config/Compile/Execution/Checkpoint 错误类别，并保留 source error。

观测区分编译、输入转换、kernel、输出转换、静态放置和状态/capture 成本。复用已有 metrics/counters；只有无法观察的实际成本才新增字段，不制造空占位指标。native array operator 不伪造 DataFusion metrics。

## 4. 拟议公开接口

### 4.1 Rust 独立调用

以下为后续计算接口草案；A1 存储与视图的精确签名已在 [A0 API note](../api-notes/native-array-runtime.md) 冻结。options 是验证过的数据值，不保存 callback。

```rust
use calc_flow::{Result, array};

fn calculate(values: &[f64]) -> Result<array::NativeArray> {
    let input = array::FloatView::from_f64(values);
    let options = array::RollingMeanOptions::new(20)?.with_min_periods(1)?;
    array::rolling_mean(input, options, &array::ArrayExecutionOptions::default())
}
```

相同风格提供逐元素运算、`sum/mean/min/max`、`ewma`。Rust 无 Python 运行时和 NumPy/JAX 依赖即可计算。

M2 增加拥有状态的 `RollingMeanState`、`EwmaState`，`process(&mut self, input)` 按块推进。所有可报告的输入/预算/计算错误不提交本批状态；batch convenience 从新状态调用同一条转移路径。public state 不暴露内部 serde 字段，不把 `Clone` 当 checkpoint。

### 4.2 Python eager 与结果转换

```python
import calc_flow.array as cfa

x = cfa.from_numpy(values)  # 显式创建 native 不可变快照
y = cfa.rolling_mean(x, window=20, min_periods=1)
result = y.to_arrow()       # 保留 validity
```

增加 `from_arrow`；NumPy 仍是可选依赖，Arrow→native→Arrow 路径不导入 NumPy/JAX。数值计算是同步 CPU-local 函数，可在计算期间释放 GIL；不为每个纯数值函数新增 async 同义 API。

`NativeArray.to_numpy()` 默认拒绝丢失 null bitmap；调用者可显式选择填充值策略，或使用 lossless Arrow 转换。带填充转换产生新存储，不能改变 owned 原值。无 null 的零复制只读导出必须证明其所有可达 base 无可写别名；证明不了则复制，不弱化不可变性。

图路径新增显式 `register_native(runtime)`，底层注册 Rust factory 并更新真实 capability/revision，不注册 Python callback。`Batch.from_array(..., backend="native")` 接受已拥有 native 对象；便利的 NumPy 输入也遵守先快照再移交规则。

### 4.3 高级表达式与 table

M2 扩展现有 `ArrayExpr` 和 `linalg`，不新增一套并行 DSL。为动态数组输入增加独立 `array_input` 声明；静态 `parameter` 继续表示静态值。

- 独立数组提供 `ArrayExpr.collect_array(...) -> NativeArray`，复用现有图编译与 BatchExecutionPlan；默认只接纳 native、batch、单个 ArrayExpr 结果。
- table 结果继续走 `collect`；混合 table/array Program 结果不在本轮开放。
- `linalg.from_columns(..., backend="native") → matmul → attach_columns` 接入同一原生执行层。
- SQL 前后组合、直接 attachment 后的 projection/filter、独立多输出应使用同一 segment 机制；保留每段的类型和行对应关系。
- M2 每个原生程序/表桥接片段最多一次 matmul；去掉 native 路径对参数必须叫 `weights` 的要求，并允许不同片段共享同一静态参数。独立数组收集允许每段一次的显式分段；Table 中多次 matmul 须经显式 Table 阶段边界。单个 attachment 的自动多 matmul 链与任意动态 stream RHS 留待后续准入扩展，不放宽 NumPy/JAX 原合同。
- Rust/Python 原生类型推导由同一 Rust 规则驱动，Python 可通过私有 planning binding 查询；不再对未知 backend 回落导入 JAX。

## 5. 分阶段实施与验收

### A0 契约、基线和测试样本（M1 前置）

交付：精确 public API、数值语义表、capability 表、错误表、项目/状态兼容策略和小型 fixture。

操作：

1. 从 main 固定现有 mean/EWMA/null/NaN/Inf/min_periods/readout 的测试结果与运算顺序。
2. 对普通归约选定可实现的抗溢出/补偿算法并固定误差规则；覆盖 `[1e308, 1e308]` 的 mean、大偏置小波动、消减及非有限值。
3. 确认 public `ts` 合法参数范围；新的直接 API 不反向改变既有节点的校验合同。
4. 记录接口/状态所有权，与共享历史工作确定 `rolling/kernel` 修改负责人。

验收：同名运算无未解释的跨层差异；每种预期变化有明确可写的失败测试。A0 只交付文档与必要 fixture，不建立不可执行空模块。

### A1 原生只读视图、owned payload 与资源核算（M1）

新增 `crates/calc-flow/src/array/` 的真实基础类型，在 `lib.rs` 暴露模块；为 `Batch` 增加 native payload 接入，保持既有 external 扩展方式。

测试先覆盖：f32/f64、空值/空输入、offset bitmap、非法 shape、长度不匹配、拥有结果生命周期、shared clone 的只读性、visible/retained 字节计数与 checked 溢出边界。确认错误在额外数值分配前报告，不通过 `StaticArraySnapshot` 执行逐批计算；A2 再验收执行分配预算的拒绝边界。

退出条件：最小 Rust 构造→计算占位以外的真实复制/读取→释放用例可运行；无需 Python；已有静态输入 digest fixture 不变。

### A2 共享数值 kernel 和 Rust 直接 API（M1）

分两个可独立审查的增量：

- A2a：逐元素 `+ - * /`、有限常量幂、负号、`abs/sqrt/log/exp`，比较、布尔和 `where`；形状先支持同长一维和 scalar 广播。归约 `sum/mean/min/max`。
- A2b：从现有 Rust rolling 提取 mean/EWMA 的数学状态；增加 `rolling_mean/ewma` 直接调用，让原 RollingOperator 也调用它。复用稳定性修复和 count/Inf 处理，不以简化 sum/count 替换既有算法。

`where` 明确：condition 为 null 时输出 null；只由所选分支决定结果有效性；不作为任意 Python 延迟求值入口。布尔逻辑采用固定的三值逻辑表，测试覆盖其全部组合。

提取时保留旧状态 codec 的字段与恢复解释；实体 ID、历史 owner、排序、内存池、并行调度与 emission commit 留在 operator 层。CPU kernel 不调用 Tokio，不新增每行动态 dispatch。已有 mean 快路径和新直接 API 均使用同一数学转移。

验收：现有 table rolling golden/regression 通过；新 array 同语义 fixture 输出和值有效性一致；同一 profile 的重构路径保持原位级要求；默认预算拒绝、错误不修改输入。添加独立 Rust example。

### A3 Python eager、原生 batch provider 与真实能力记录（M1）

接通 PyO3 owned type、转换、直接函数和 native 注册。native batch provider 采用版本化数据-only options，例如 `native:array_expr@1`；Rust 使用类型化 IR，Python lowering 只生成数据，不携带 callable/import path。旧字符串 expression provider 不迁移语义、不调用 Python eval。

Rust 是 IR 验证的最终边界：单节点 options 编码上限 64 KiB，最多 128 个表达式节点、24 层、32 个命名输入和 16 个命名输出；常量只允许有界、有限的数值或 bool，不接受内联大数组，幂指数绝对值不超过 64。逐步开放的 rank 上限、checked shape 乘积和运算输出上限必须在 Rust 重验。引用缺失、重复 ID、环、未知 op/version、非法 scalar/dtype、输出类型不一致均在执行前拒绝；不能依赖旧 Python AST 的限制保护 native IR。对手写 project 和绕过 Python 的 Rust 调用都增加拒绝测试。

纯 Rust kernel 同步运行；从 async batch/stream operator 调用时使用现有有界 CPU worker 设施，先获取工作和内存准入，不在 Tokio executor 上执行长循环。长批次按固定块检查取消；取消后等待拥有 worker 收尾、释放 reservation，再结束 job，不能遗留后台计算。直接 Python eager 调用在冻结输入后释放 GIL 运行同步 kernel。

必要接点逐项交付：

| 接点                                    | 要求                                                                                                |
|-----------------------------------------|-----------------------------------------------------------------------------------------------------|
| native/Python Runtime 注册              | registry、registration snapshot、revision、duplicate 与 compile cache 同步；失败注册不推进 revision |
| `capabilities.py` 与 Rust capability 源 | 支持的 dtype/shape/mode/partition/checkpoint 如实暴露；batch-only 不宣称 stream                     |
| `array.py`、analyzer                    | 增加 Rust native 类型推导分支，移除“非 NumPy 即 JAX”的隐含分派                                      |
| `calc-flow-python/src/batch.rs`         | `.array` 支持 native payload；旧 `python_payload()` 路径保持；有效性和生命周期完整                  |
| native provider/static input            | 不进入 Python placement callback；保留 canonical 静态摘要和恢复校验                                 |
| stubs/exports/examples                  | `_native.pyi`、Python 模块公开表面和真实运行例子同步                                                |
| project/schema                          | 优先沿用 external operator options；新 vocab 必须严格验证，schema 生成器与 fixture 同步             |

验收：不安装 NumPy/JAX 时 Arrow→native→Arrow 可执行；Python 原生运算不进入 Python provider callback；可变输入在 worker/GIL 释放前冻结；已编译计划复用与能力变更失效行为正确。M1 到此可交付。

### A4 二维矩阵与 table 桥接（M2）

增加二维 contiguous descriptor，明确行主序/列主序的内部读写接口；不开放任意 strides。通用二维存储保留 bool 与 null；matmul 操作数只接受 f32/f64、非 null，内维一致、输出宽度为正，允许零行输出。混合精度按 native promotion 执行。

原生 `table_array@1` 直接读 Arrow 列，优先按输出列写 Arrow buffer。只有所选矩阵 kernel 要求时才打包 dense storage；每份 packing、静态放置和输出分配都受预算控制。首期实现确定顺序的正确 CPU 基础 kernel，不默认引入新的 BLAS 依赖，不宣称胜过 NumPy。

静态矩阵在每个有限 run 或 managed job 内按可信 owner 冻结、规范编码和原生放置一次；多个节点共享不可变放置值，跨执行不复用。共享键包含执行身份、保活的可信 owner、已验证 backend/dtype/shape 与成功的规范准备身份。恢复 digest-v1 包含物理绑定名，因此每个别名仍独立校验声明、计算原协议 SHA 并核对恢复值；物理摘要不能用作共享 placement 键。按消费者预算流式编码有界 chunk，不保无界 body；准备、放置和 scratch 重叠受真实预算/取消约束，完整校验前不发布候选缓存。

复用 segment 边界接通 SQL→matrix→SQL、attachment→filter/project、独立多输出和不同参数名。保持行对应关系检查：跨表、过滤前后行集不同、输出宽度与字段冲突均编译失败。ASOF/event-window 限制继续按现有合同，不借此自动放宽。

验收：Arrow chunk/slice/空行、f32/f64、多个静态消费者、权重变化恢复拒绝、无 Python callback、无重复静态放置；原 NumPy/JAX 场景回归通过。二维 `transpose/reshape` 只在布局可证明时共享，否则显式分配并计费。

### A5 独立 ArrayExpr batch 收集和有限 shape/axis 能力（M2）

新增 dynamic `array_input`，复用现有不可变 Node、analyzer、capability 与 project lowering。公开 `collect_array`，将 logical binding 降低为现有 array ports，执行结果保持 native owned 类型。

支持多个显式命名的 batch array 输入；同长向量/scalar 广播、二维已证明的 shape 操作，以及归约 `axis=None/0/1`（仅适用合法 rank），`keepdims` 必须有确定输出 shape。不支持的广播组合明确拒绝，不交由执行时猜测。

`ArrayExpr` 从 table 取值后只有保持行轴的运算可以 attach；沿行轴归约、transpose 改变行轴后不能原样回贴。独立结果没有隐含 table lineage。计划导出只包含声明和 provider 引用，不嵌入输入数组数据。

验收：声明→分析→编译→实际执行全部打通；多输入长度不匹配、rank/axis 非法、失去 lineage 后 attachment 都有编译错误；`compute/collect` 原返回合同不变。高层 async array stream 收集与 mixed Program 输出暂不开放，低级 runner 仍是 array stream 入口。

### A6 有状态 array stream、table 共用与持久化恢复（M2）

先提供直接 Rust owned handle，再接入原生 `StreamOperator`。array-only stream 定义为每条输入边一条有序序列，按到达顺序应用行窗口；多实体、event time、乱序与 late output 继续走已有 table adapter，不从矩阵行号猜测实体。

stateful native capability 必须来自 Rust 真实注册记录：stateful、支持的 checkpoint/layout、输入顺序与分区约束、确定性和 finality 分开描述。禁止复用 `_StatelessProviderLifecycle` 把 rolling 标成 `row_axis_independent`。需要新增词汇时，严格扩展验证器和 public schema。

执行采用 owned candidate/prepare→现有输出接受点→commit。预算失败、计算错误、取消或输出拒绝不能提前提交状态；已被 sink 接受的输出不能回滚，不新增 exactly-once 承诺。控制消息的 FIFO、watermark/idle/barrier/EOF 与 runtime 现有 contract 保持。

状态 codec：

- 现有 table rolling 尽量保持原 codec 和 fingerprint；改了数学布局不能继续冒用旧版本。
- 新 array operator 使用独立版本化状态 layout，记录参数/语义版本、计数、ring、数值修复状态和递推 seed。
- 状态保存有效性、必要历史与已经累计的 recurrence，不能仅保留 window 长度尾部后重开 EMA。
- codec 验证结构、上限、计数一致性及 checksum；错误版本、参数、dtype、静态 digest 不匹配在 source 打开前拒绝。
- 新 state adapter 通过现有 backend/session/manifest 保存；不添加一个与 managed checkpoint 并行的 public 文件格式。

验收：同序列不同分批、空批、跨 Arrow chunk、序列前缀恢复与不中断执行一致；失败后重试不双算；慢 sink/取消/EOF/barrier；损坏或超限状态 fail closed；restart 必须用新 job/进程和可回放输入验证。M2 到此完整交付。

### A7 RSI、ATR 与完整 MACD 分量（M3）

指标由 calc-flow 自行实现，不引入 TA-Lib 代码依赖；新增算子使用共享 recurrence 基础和自己的明确合同。Rust、Python eager、table 和 stream 一起验收，避免先发布不能恢复的同名 stream 能力。

首版规则：

- RSI：period 个有效价格差先初始化平均 gain/loss，随后 Wilder 递推；无缺失序列首个输出位于 index=period。loss 为 0 且 gain > 0 时 100；gain 为 0 且 loss > 0 时 0；两者都为 0 时 50。该 flat 值是新 native 合同，不声称 TA-Lib 精确兼容。
- ATR：第一行仅保存 previous close；随后计算 true range；period 个 true range 的均值初始化，再 Wilder 递推；首个输出位于 index=period。要求 high >= low。
- RSI/ATR：默认 null/NaN 断开有效段并重置 warm-up，所在行输出 null；任一 Inf 拒绝该批并不提交状态。这个新指标合同不改变已有 mean/EWMA 的 missing 处理。
- RSI/ATR 的有限输入也必须检查差分、true range、初始化与 recurrence 的中间溢出；首版不能以规定算法有限表示时返回包含输入字段/位置的计算错误，不发布该批输出或提交候选状态。测试包含 RSI 的 `[1e308, -1e308, 1e308]` 和 ATR 的 `high=1e308, low=-1e308`，以及失败后的续算；不把溢出静默变成 RSI 极值或无限 ATR。
- 完整 MACD：新增 `macd_components`/`MacdOutputs(line, signal, histogram)`；line 与现有 `ts.macd` 同义，signal 对 line 使用同一 native EWMA，histogram=line-signal。初值与 min_periods 明确来自 calc-flow EWMA，不冒充 TA-Lib MACD。
- table 新入口为 `ts.rsi`、`ts.atr` 和 `ts.macd_components`；三个 MACD 输出共享一个 state owner。参数、输出 dtype、null 规则进入 schema/指纹/codec 身份。

验收：独立标量公式 oracle、小型固定向量、零波动/单边上涨下跌/跳空/断段、缺失值与 Inf；批流分割、实体交错、恢复、共享输出与只消费部分 MACD 分量的状态正确性。可采用记录来源、版本和参数的外部参考数值作为补充 fixture，但测试不导入或链接 TA-Lib；只比较语义相同的范围，并记录允许误差，不能用宽容差掩盖初始化差异。

## 6. 后续有条件扩展

### B1 自研指标覆盖与语义扩展

进入条件：存在具体指标或数值兼容需求。逐项扩展自有 Rust kernel 和指标注册表，明确公式、初始化、lookback、缺失值、输出 dtype、复杂度和状态上限；Rust、Python、table 与 stream 复用同一实现。

需要与外部指标一致时，以明确版本的公开行为和参考数值定义兼容范围，独立实现并做差分验证。不新增 TA-Lib provider 或依赖，不用同名接口静默覆盖既有 `ts` 语义，也不默认承诺全库逐位一致。

新递推状态由 calc-flow 自己拥有、计费和编码；batch/stream 共用数值合同，durable stream 交付必须包含版本化 codec 与恢复测试，不依赖外部 opaque handle 或无界历史重放。

### B2 大矩阵后端、批量特化和更多 kernel

进入条件：已有剖析/测量证明 native 基础 kernel 或布局转换是具体 workload 的瓶颈。先做一个 CPU 线代后端的局部验证，再决定是否引入依赖；明确线程预算，避免 BLAS、Tokio 与 DataFusion 并行过度叠加。维持后端/数值 profile 显式选择，不静默改 NumPy/JAX backend。

更多现有 rolling kernel 按调用需要逐个开放：sum/min/max/variance/stddev、covariance/correlation、cumulative、decay；同输入共享历史复用已有计划成果。批量 vHGW、SIMD/FMA、多窗口融合各为单独性能变更，不夹在行为保持的 kernel 抽取中。

## 7. 文件责任与 PR 拆分

| PR | 所属阶段   | 主要责任范围                                                              | 依赖                         |
|----|------------|---------------------------------------------------------------------------|------------------------------|
| 01 | A0         | 契约、API、fixture 与必要文档                                             | 本计划                       |
| 02 | A1         | `crates/calc-flow/src/array/`、`batch.rs`、`lib.rs`                       | 01                           |
| 03 | A2a        | 纯数值与归约 kernel、直接 Rust API/tests                                  | 02                           |
| 04 | A2b        | 共享 mean/EWMA、现有 rolling 适配与恢复回归                               | 02、01；与共享历史负责人协调 |
| 05 | A3         | PyO3、native batch factory、capabilities、Python eager、stubs/examples    | 03、04                       |
| 06 | A4         | 2D/storage adapter、static placement、matrix provider、table segmentation | 05                           |
| 07 | A5         | `array_input`、analyzer、ArrayExpr 收集、axis/shape lowering              | 05、06                       |
| 08 | A6a        | 直接 owned state、共享候选转移、状态与峰值预算                            | 04、05                       |
| 09 | A6b        | 真实 stateful factory、codec、managed restart、能力迁移与控制/预算验收    | 08；与 06 静态输入对齐       |
| 10 | A7 RSI/ATR | recurrence、Rust/Python/table 表面与恢复                                  | 09                           |
| 11 | A7 MACD    | 共享多输出与原语组合、跨表面契约                                          | 09                           |

PR 数量是审查单元建议，不强制以一个大 PR 装完复杂状态功能。每个 PR 必须包含能执行的行为、测试与对应文档；涉及公开语义的子阶段先完成 spec/API/critic，最后有 specialist review。

可并行：03 与 04（在数据合同冻结后）；07 与 08/09（避开共同 analyzer/capabilities 文件并指定 owner）；10 与 11（在 state/codec 基础冻结后）。不让多个实现者同时改 `rolling/kernel.rs` 或静态输入放置逻辑。

重要现有接点：

- `crates/calc-flow/src/operator/rolling/kernel.rs`、`kernel/mean.rs`、`state_codec.rs`、`state_v3.rs`、`ordered_stream.rs`。
- `crates/calc-flow/src/static_input.rs`、`operator/mod.rs`、project/capability inventory 与数据-only options 校验。
- `crates/calc-flow-python/src/batch.rs`、`provider.rs`、`runtime.rs`；新增纯 Rust 数值 PyO3 adapter 可拆为 focused module。
- `python/calc_flow/array.py`、`pipeline.py`、`capabilities.py`、`_native.pyi`；symbolic 的 `expr/types/analyzer/ops/lower`。
- `python/tests/test_array.py`、`test_symbolic_matrix.py`、`test_static_stream_inputs.py`、`test_sql_pipeline.py`；Rust 同模块测试和必要 integration tests。
- `docs/array-guide.md`、`python-api.md`、`rust-api.md`、`symbolic-api.md`、`streaming-guide.md`、真实 examples 及 qualifying CHANGELOG。

## 8. 验证与性能协议

### 8.1 每阶段的正确性门槛

每个行为变化先有 focused failing test，记录预期失败，再实现 green。重构使用已有 oracle/golden 加缺失反例，不能写只镜像实现的测试。只运行受影响模块、必要格式/lint/编译检查；完整 workspace、多平台和 90% Rust coverage 属于 CI，不为本计划构建 native。

至少覆盖：

- 所有权：调用方 mutation、释放原对象、结果导出、异步移交和重复运行。
- 数值：精度提升、普通归约与 ts 区别、缺失值/非有限值、极端有限值、warm-up、输出 dtype/validity。
- 图：多输入、静态/动态区分、SQL 前后、直接 filter/project、行对应关系、输出字段冲突、能力缓存。
- 状态：分块不变性、实体隔离、恢复、error/cancel/retry、不重复提交、控制消息顺序。
- 资源：边预算、输出/scratch/state/capture 同时存在的峰值计费、尺寸溢出、超限恢复、失败放置不入缓存。
- 协议：旧 NumPy/JAX 项目与静态 digest fixture、现有 rolling checkpoint；新版本不支持时明确 fail closed。

schema 若无计划变更必须零漂移；若有新增词汇/算子形状，审查期明确列出预期生成差异并更新 fixture，稳定后重新执行无漂移检查。每个 PR 运行 `git diff --check`。提交/推送后仅取一次非阻塞 CI 状态，不等待或轮询；pending 只允许交接，不代表可合并。

### 8.2 有界性能验证

性能工作是对应实现阶段的独立小任务，不在本次计划交付时执行。每次先登记基线/候选 SHA、锁文件、构建参数、CPU affinity、线程数、dtype、输入、timer 范围、正确性标准、样本数及停止规则。

复用 `benchmarks/test_array_kernel.py`、`test_array_plan.py`、`test_array_provider.py`、`test_rolling_kernel.py` 和 `scripts/profile_warm_stream.py`。只补必要 case，不新建通用 benchmark framework，不运行完整 six-case array acceptance set 来包装一个局部特性。

| 候选 case            | 固定最小负载                                    | 对应问题                          | 基线与候选                                           |
|----------------------|-------------------------------------------------|-----------------------------------|------------------------------------------------------|
| E1 逐元素链          | f64，4,096 元素，同长数组，单输出               | callback/临时数组/编译与执行边界  | 已发布 NumPy provider vs native；kernel 时间另列诊断 |
| E2 归约 mean         | f64，4,096 元素，无 null、有限值                | 分配、归约精度与 native 调用成本  | 现有 mean provider vs native，预先冻结等价/误差规则  |
| M1 matrix attachment | f64，256×8 table、8×2 静态权重                  | table pack/unpack 与首次/稳态放置 | 现有 NumPy symbolic matrix vs native 同形状          |
| R1 shared SMA        | 1 实体，2,048 历史行、256 append、window=20     | kernel 抽取是否拖慢现有路径       | 实施前 main 的 table rolling vs 抽取后同一表面       |
| R2 EWMA stream       | 8 实体、总计 2,048 历史行和 256 append，span=20 | 递推状态、实体分区与提交开销      | 实施前 main vs 候选同图、同批次                      |

这是候选池，不是每个 PR 都跑五项。M1 常选 E1/E2/R1，矩阵阶段选 M1，stream 阶段选 R1/R2。新增 dtype、尺寸或 checkpoint 配置都算新 case，不暗中乘成矩阵。

每次默认固定 10 个配对样本，总墙钟上限 10 分钟，包含 fixture、preflight、warmup、正确性、启动和清理；构建估计单列，优先复用兼容 artifact。若维护 harness 对所选 verdict 要求更高样本数，执行前采用其固定要求并缩减 case；预算装不下则报告未测，不降低工具标准。重型 native 构建只在目标明确的性能任务进行，输出保留在 `target/`。

基线无对应功能的新接口只报告成本或与明确参考实现比较，不伪造“旧实现速度”。NumPy/native 浮点语义不同则不能计入同义加速 verdict；先选可比输入并固定误差约束。kernel、provider、plan 和包含转换的端到端分别报告，JAX 未作为本池目标，不借 CPU 结果推导 GPU 收益。

保持现有 harness 的配对统计、provenance、进程退出和回归规则。新 API 功能交付不要求虚构加速；宣称优化、引入更复杂 fast path 或更换默认执行策略，必须有所选目标的实际收益证据且无已确认控制回退。inconclusive 不代表提速或等价，不增加样本/负载重试以追求有利结果。

## 9. 发布、回滚与结束条件

- 每个里程碑是可独立使用的增量，前一里程碑无需等待所有指标或后端。
- native 始终显式选择；回退 NumPy/JAX 需要用户改变 backend，不在失败时静默切换语义或设备。
- 内部快路径可回退到同语义参考路径；provider capability、版本和指纹保持可解释。新 kernel 已写出的状态必须有声明的读取支持，撤回实现不能假装旧版本能读取新状态。
- 保持项目 v3 与 manifest v3 的外层合同；必要的新 kernel/state layout 独立版本化。若外层必须改变，另列迁移 PR，不夹在性能重构中。
- Rust/Python 版本按仓库 calendar version 同步，依赖 exact pin、wheel/sdist 内容和已安装 wheel 测试进入正常发布流程。依赖清单、lockfile 和发布产物不得因本计划加入 TA-Lib 或其包装依赖。
- 全部 M1–M3 完成需满足：Rust 独立示例、Python eager、native table/array 图、batch/stream/recovery、现有行为回归、资源/错误合同、文档和最终 specialist review。required CI/coverage/cross-platform 绿灯是合并条件，发布另需 artifact verification 与 post-build unit tests；本计划不授予合并或发布操作。

本次交付仅为计划和设计审查。没有产品实现、测试执行或性能结果；所有新 API 需按 A0 固定合同并逐阶段实现。
