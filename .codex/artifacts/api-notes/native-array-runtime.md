# Native Array：A0 冻结接口

日期：2026-11-08。工作项：`native-array-runtime`。本文冻结 A1 接口及后续阶段必须保持的语义；不表示这些接口已经实现。

依据：[主计划](../specs/native-array-runtime.md)、[源码评估](../reviews/array-api-rust-ta-lib-assessment.md)、[领域术语](../../../docs/introduction.md)、[代码规范](../../../.agents/skills/code-style/SKILL.md)。源码检查基线为 `7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`。遵守用户约束：不引入 TA-Lib crate、C FFI、Python wrapper、间接封装或 vendored/generated TA-Lib 源码；算法由 calc-flow 自行实现。

## 1. A1 公开表面

在 `calc_flow::array` 模块公开以下表面。`ArrayRef`、`NullBuffer` 使用当前 DataFusion re-export 的 Arrow 类型；不增加第二套 bitmap 或不安全 buffer。以下 Rust 为签名清单，不是预先添加到源码的空实现。

```rust
use datafusion::arrow::{array::ArrayRef, buffer::NullBuffer};
use calc_flow::Result;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrayDType {
    Float32,
    Float64,
}

#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum FloatValues<'a> {
    Float32(&'a [f32]),
    Float64(&'a [f64]),
}

pub struct FloatView<'a> { /* private fields */ }
impl<'a> FloatView<'a> {
    pub fn try_new(
        values: FloatValues<'a>,
        validity: Option<&'a NullBuffer>,
    ) -> Result<Self>;
    pub fn from_f32(values: &'a [f32]) -> Self;
    pub fn from_f64(values: &'a [f64]) -> Self;
    pub fn dtype(&self) -> ArrayDType;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn values(&self) -> FloatValues<'a>;
    pub fn validity(&self) -> Option<&'a NullBuffer>;
}

pub struct NativeArray { /* private fields */ }
impl NativeArray {
    pub fn from_arrow(values: ArrayRef, shape: Vec<usize>) -> Result<Self>;
    pub fn from_f32(
        values: Vec<f32>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<Self>;
    pub fn from_f64(
        values: Vec<f64>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<Self>;
    pub fn dtype(&self) -> ArrayDType;
    pub fn shape(&self) -> &[usize];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn as_arrow(&self) -> &ArrayRef;
    pub fn float_view(&self) -> Result<FloatView<'_>>;
    pub fn estimated_bytes(&self) -> usize;
    pub fn retained_buffer_bytes(&self) -> usize;
}
```

`FloatView` 实现 `Clone + Copy`；`NativeArray` 实现共享存储的 `Clone`。两者可实现只显示 dtype、长度、shape、null count 的 `Debug`，不得打印元素值。`NativeArray` 实现既有 `ExternalPayload`，因此同时满足 `Any + Debug + Send + Sync`。不提供 `DerefMut`、可变 buffer getter 或公开字段。

### 1.1 形状、validity 与所有权

- A1 只支持 rank 0/1 和 float32/float64，不接受 bool/integer/complex、matrix、strides 或 device。`shape=[]` 恰有一个元素；`shape=[n]` 恰有 n 个元素，允许 `[0]`。多维输入先报告 rank 不支持，不能把 `[usize::MAX, 2]` 当合法 shape 再分配。
- `FloatView` 是扁平一维借用序列，不携带 scalar/vector 身份。`NativeArray::float_view()` 对 scalar 成功返回长度 1 的视图；图中的 scalar 广播身份仍由 `NativeArray::shape()` 决定。
- `validity=None` 表示全部有效；bitmap 的 `true` 表示有效。bitmap 长度必须与 typed values 长度相同；必须尊重非零 bitmap offset，保留显式 all-valid 和 zero-length bitmap。null 位置仍有一个 typed value 槽位，但计算不得读取它来决定有效输出。
- `from_arrow` 仅接受真正的 `Float32Array`/`Float64Array`；仅声明 `DataType::Float64` 的其他自定义 `Array` 不足以通过验证。它接管一个 `ArrayRef`，共享已有只读 Arrow buffer，并保留原 array 的 offset/validity；不复制 values。
- `from_f32/from_f64` 消费 `Vec` 和 bitmap，在验证完成后通过安全 `ScalarBuffer`/primitive array 构造接管存储；不借用可变 caller 内存，也不把 float32 宽化为 float64。失败前不分配新的数值 buffer。
- `as_arrow()` 借出同一个 `ArrayRef`；调用者可 clone 它延长只读存储生命周期。`float_view()` 从已经验证的 Arrow primitive storage 借用，不分配、不调用 `StaticArraySnapshot`；A1 对所有合法 NativeArray 恒成功，后续非浮点 dtype 返回 `input.dtype` 错误。此 fallible 签名保证 A2 增加真实 Boolean 输出时不必破坏接口。
- shape 使用恰好 rank 个 `usize` 的紧凑拥有存储，例如 `Box<[usize]>`。不保留 caller `Vec` 的无关预留容量。
- 构造函数允许 NaN、Inf 和所有 IEEE payload bits；A1 不做数值规范化。NaN canonicalization 只适用于后续指定的运算结果或既有 static digest 协议，不改变输入存储。

### 1.2 字节成本与 Batch 接入

设 `n=values.len()`，`w` 为 dtype 字节宽度，`r=shape.len()`，`s=size_of::<usize>()`。构造时用 checked arithmetic 计算并缓存：

```text
bitmap_visible = 0                                  if no bitmap or n == 0
bitmap_visible = ceil(((bitmap.offset() % 8)+n)/8)   otherwise
estimated_bytes = n*w + bitmap_visible + r*s
retained_buffer_bytes = values_buffer.capacity()
                      + bitmap_buffer.capacity()   (0 if no bitmap)
                      + r*s
```

`estimated_bytes` 是当前 typed 槽位、可见 bitmap byte span 和 shape 的逻辑边计费，不包含 allocator/Arc/对象固定开销，不代表 RSS。`retained_buffer_bytes` 是 Arrow 暴露的 backing-buffer capacity 加 shape 存储；slice 可能保留远大于其可见值的父分配。它同样不是 RSS，也不能测得外部 Arrow owner 另外保留的内存；跨 buffer/consumer 不去重，允许保守重复计费。不得通过 unchecked 的容量求和隐藏溢出。

`ExternalPayload::backend()` 返回 `"native"`；`NativeArray::len()` 与 `ExternalPayload::len()` 均定义为逻辑行数，在 A1 与元素数相同，scalar 为 1，空 vector 为 0；`estimated_bytes()` 返回上述缓存；`as_any()` 返回自身。A4 增加矩阵时，两者均返回第一维行数，不能以全部元素数代替边行数；若需要独立元素数访问器，在 A4 随真实用途新增，不改变 `len()` 合同。

不新增另一条 `Batch` 构造路径。A1 提供真实文档示例：

```rust
use std::sync::Arc;
use calc_flow::{Batch, BatchMetadata, Result, array::NativeArray};

fn input() -> Result<Batch> {
    let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2], None)?;
    Batch::external(Arc::new(array), BatchMetadata::default())
}
```

A1 没有执行 kernel，因此不创建空的 `ArrayExecutionOptions` 或预算框架。构造时核实描述符、bitmap、visible/retained byte 加法；A2 才为新增输出与 scratch 分配执行主计划的元素/字节限额。传入已拥有 Arrow/Vec 的存储不是 A2 新执行分配，不能把输入大小误当输出元素上限。

### 1.3 验证与错误

全部可报告错误返回 `CalcFlowError::InvalidArgument`；不得 panic。公开稳定的是字段路径及约束，消息包含实际/预期 descriptor，不能包含数组内容。

| 违反约束                                             | field            | 消息要求                                         |
|------------------------------------------------------|------------------|--------------------------------------------------|
| 不支持的 Arrow dtype 或非对应 primitive array        | `input.dtype`    | 要求 Float32/Float64 primitive，并标明收到的类型 |
| rank 超过 1                                          | `input.shape`    | A1 只允许 scalar/vector，标明 rank               |
| shape 元素数与 values 不同                           | `input.shape`    | 标明 expected/received 元素数                    |
| bitmap 长度不匹配                                    | `input.validity` | 标明 expected/received 长度                      |
| typed bytes、bitmap span、shape 或 retained 总量溢出 | `input.bytes`    | 标明 `usize` 字节计数溢出                        |

验证顺序固定为 dtype、rank/shape、validity、byte counts，再构造拥有对象。`FloatView::try_new` 只验证其适用的 dtype、validity、visible byte counts，不复制数据；无 bitmap 的 slice 便利构造是可证明有效的 infallible 包装。禁止用 `StaticArraySnapshot` 实现这些构造/访问器。

## 2. A2 普通归约的数值合同

这里冻结后续实现的 oracle，A1 不增加归约代码。普通归约与既有 rolling 是不同数学合同；不改 RollingOperator 的累加顺序、状态编码或 readout。

`sum/mean` 的 input 是 `FloatView`，返回 rank-0 float64 `NativeArray`；`ReductionOptions` 私有字段，默认 `skip_nulls=false`。不跳 null 时任一 null 使结果 invalid；显式跳过后无有效值也 invalid。invalid scalar 用 positive-zero 槽位及 false validity 表示。有效 NaN 不当 null：任一 NaN 或同时出现正负 Inf 返回 canonical quiet NaN；只有一种符号的 Inf 返回该 Inf。

有限值使用 calc-flow 自研的**精确二进制累加器**，不采用可能先溢出的浮点 sum，也不采用可能把极小残差缩放为零的 max-scaled 求和：

1. float32 精确转换为 float64。将有限 float64 拆成符号、整数 significand 和 exponent；以 `2^-1074` 为单位累加正负整数绝对值。
2. 每侧使用 `ceil((2098 + usize::BITS)/64)` 个 `u64` limbs，足够容纳至多 `usize::MAX` 个有限 float64 的总和。使用安全整数操作和显式 carry；不新增 bigint/TA-Lib 依赖，不使用 unsafe。
3. 正负总量相减得到精确带符号和。`sum` 最终只做一次 IEEE round-to-nearest ties-to-even；真正溢出时返回相应 Inf，不把临时求和溢出当最终结果。
4. `mean` 对精确总和除以有效值数量，用商/余数决定最终 ties-to-even 舍入，不能先把总和转换为可能溢出的 f64 再除。有限输入的数学均值有界，不能因内部累计而产生 Inf。
5. 精确零的 sum/mean 统一输出 `+0.0`；canonical f64 NaN bits 为 `0x7ff8_0000_0000_0000`。内核顺序确定，不做 float tree reassociation；未来优化必须满足同一个精确舍入 oracle。

这是 O(n)、固定且有界 scratch 的算法选择，不是已测得的性能优势。A2 可以优化 limbs 更新，但不能降低本合同。独立测试 oracle 使用 Python 标准库 `Fraction.from_float`/精确整数及小型固定 IEEE 边界 fixture；不让测试重复生产的 limb 算法，也不引入 TA-Lib 测试依赖。

必须覆盖：`[1e308,1e308]` 的 mean 有限且为 `1e308`；`[1e308,1.0,-1e308]` 的 sum 为 `1.0`；`[f64::MAX,最小正subnormal,-f64::MAX]` 保留 subnormal；normal/subnormal 的 halfway ties、真实 sum overflow、正负 Inf、NaN、null、空输入、全 null 和 signed zero。min/max 保留输入 dtype，任一有效 NaN 传播该 dtype 的 canonical NaN（f32 为 `0x7fc0_0000`）；零值候选同时含正负符号时 min 取 -0、max 取 +0，只有一种零符号时保留该符号，无有效值 invalid。

## 3. 后续接口与兼容边界

- **A2a** 才加入布尔结果、比较/where、scalar 广播与归约。用私有字段的 `ArrayExecutionOptions` 实现主计划默认 1,000 万输出元素和 256 MiB 本次执行分配上限。不得为了它们在 A1 增加未使用 variants 或 stub。新增 dtype enum variants 保持 non-exhaustive；不能把 `FloatValues` 变成可写输入。
- **A2b** 的 `rolling_mean(FloatView, RollingMeanOptions, &ArrayExecutionOptions) -> Result<NativeArray>` 与 `ewma(FloatView, EwmaOptions, &ArrayExecutionOptions) -> Result<NativeArray>` 接到共享原生状态转移。f32 使用 f64 状态和 f64 输出；逐位置输出、前导 invalid、null/NaN/Inf、min_periods 和窗口计数沿用现有 table 契约。现有 `ts.ema/macd` 不变。
- **A3** Python `NativeArray` 是不可变 buffer owner，`from_numpy` 在 GIL 释放/worker 移交前复制可变输入；`from_arrow` 共享安全只读存储。`.to_arrow()` 保留 validity；`.to_numpy()` 默认拒绝丢失 null，显式填充转换产生新存储。`compute/collect` 继续返回 table。原生注册记录真实 capability/revision，不通过 Python provider callback 执行。
- **A4/A5** 才引入 rank 2、布局、矩阵和 `ArrayExpr.collect_array`。native dtype/shape 推导由 Rust 规则决定，不对未知 backend 回落 JAX；不改变既有 NumPy/JAX 的运算、输出类型和设备行为。静态参数冻结/摘要是独立职责，不能因 native carrier 可共享而绕过 canonical digest。
- **A6** `RollingMeanState/EwmaState` 拥有状态，`process` 对可报告的输入/预算/计算错误不提交本批；其状态 codec、版本和恢复验收由 managed operator 提供。A1 无 public snapshot/restore/serde surface，`Clone` 不等于 durable checkpoint，直接 handle 不承担 watermarks/实体调度。
- **A7** RSI/ATR/完整 MACD 使用自有实现及明确新入口。TA-Lib 仅作研究参考，不加入直接、间接、vendored 或测试代码依赖。

## 4. A1 交付验收

针对真实接口先记录 focused RED，再实现。至少检查 f32/f64 的原始 bits、scalar/empty/vector、显式与缺省 validity、非零 array/bitmap offsets、错误路径、共享 clone/export 的生命周期、caller 数据不变、shape 的复制、visible 与 retained 字节差异、checked byte 边界及 `Batch::external` 往返。验证 `NativeArray` 为 Send+Sync，借用 view 的生命周期由 Rust 保证，不把 borrowed view 搬入 async 任务。

仅运行相关 Rust unit/integration、必要 fmt/clippy/rustdoc；完整矩阵与 90% coverage 留给 CI。本文是文档变更，检查 Markdown、链接、spec references、schema 无意外漂移和 whitespace，不为计划构建 native。

## A3 已解析数值字面量边界

2026-11-08 接口及专项设计审查通过本修订。它明确替换先前过强的原始整数
token 承诺，不表示该承诺曾被实现。原生 factory 的边界是已解析 `JsonMap`，
没有原始 JSON 拼写信息。

- `serde_json::Number` 保留为 i64/u64 的整数表示必须位于 ±2^53，再转 f64。
- 已保留为 f64 的表示必须有限，可以使用完整有限范围；不能用数学上的整数性
  或 `fract() == 0` 将它重新分类为原始整数。
- 超出 i64/u64 的整数 token 可能在上游被舍入为有限 f64。该值与相同 bits 的
  显式浮点表示无法区分，按相同规则处理。例如保留的 `u64::MAX` 整数会被拒绝，
  更大的 `18446744073709551616` 却可能已被解析为可接受的 f64。此非单调边界
  必须公开，不声称恢复原始写法或丢失的精度。
- 显式 f32 字面量仍在上述检查后舍入且不得溢出为 Inf；pow 仍检查解析后的
  指数绝对值不超过 64，然后才转 f32。null、数组及数值到 bool 的隐式转换仍拒绝。
- Python 声明前端必须在还持有 Python `int` 时、JSON 编码前执行 ±2^53 检查。
  普通 external options 和手写 project JSON 不声称经过该前端检查。

JSON Schema 的 `integer` 描述数学值，不是 Rust Number 的保留表示。因此不能
用 `type=integer` 条件模拟此规则而错拒大整值 f64。schema 描述该语义，原生
factory 执行它；单独 schema 校验不足以证明通过全部原生语义检查。数组和
scalar 输入的数据值合同不变。本修订不改变数值算法、解析器 feature、依赖或
IR/capability 版本。

回归覆盖真实 serde_json 解析、原生 literal 校验及执行结果 bits：正负 2^64
和它们相邻的原始大整数拼写会形成相同 f64；保留的整数表示仍按边界拒绝。
