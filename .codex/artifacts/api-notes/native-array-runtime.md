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

## 5. M2（A4/A5）已批准设计合同

以下归档已经 API 与 critic 审查通过的精确合同及澄清；这是分阶段实施的
设计依据，不是功能、测试或 CI 完成声明。原 A1/A2/A3 及已解析 literal 边界
保留；本节仅按阶段扩展对应表面。Table bridge、完整 symbolic 收集和 A6b
不能因本归档而被标记为已实现。参见[主计划](../specs/native-array-runtime.md)
及[专项审核](../critiques/native-array-runtime.md)。配置保持 data-only，Batch
仍是图数据 envelope，DataFusion 仍是唯一 table 引擎；不引入 TA-Lib 代码。

### 5.1 Packed matrix 与精确公开表面

NativeArray 扩展为 packed row-major rank 0/1/2，允许 nullable bool/f32/f64。
`len/is_empty` 按逻辑行计数：`[]` 为 1，`[n]` 为 n，`[r,c]` 为 r；
`element_count` 和 ArrayView 的 len/is_empty 按全部槽位计数，故 `[3,0]`
有 3 行而 view 为空。Arrow 导出保留扁平存储，shape 单独携带；NumPy 导出
保留 shape，输入先按原 immutable snapshot 合同捕获。rank-2 float_view
在 `input.shape` 拒绝，不隐式 flatten。保留 Arrow value/validity offsets。

```rust
impl NativeArray {
    pub fn element_count(&self) -> usize;
    // Existing view() -> ArrayView<'_>, constructors and Arrow accessors remain.
}
impl<'a> ArrayView<'a> {
    pub fn reshape(self, shape: &[usize]) -> Result<Self>;
}
pub fn matmul(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions)
    -> Result<NativeArray>;
pub fn reshape(input: &NativeArray, shape: &[usize], execution: &ArrayExecutionOptions)
    -> Result<NativeArray>;
pub fn transpose(input: &NativeArray, execution: &ArrayExecutionOptions)
    -> Result<NativeArray>;
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction { Sum, Mean, Min, Max }
impl ReductionOptions {
    pub const fn with_axis(self, axis: Option<usize>) -> Self;
    pub const fn with_keepdims(self, keepdims: bool) -> Self;
    pub const fn axis(&self) -> Option<usize>;
    pub const fn keepdims(&self) -> bool;
}
pub fn reduce(input: ArrayView<'_>, operation: Reduction, options: ReductionOptions,
              execution: &ArrayExecutionOptions) -> Result<NativeArray>;
```

旧 sum/mean/min/max(FloatView, ...) 保留，并将无 shape 的 view 作为 vector。
axis 默认 None、keepdims 默认 false；Python eager 对应归约增加同名 keyword。
新增 Python eager matmul/reshape/transpose，沿用 execution options 约定。

- 逐元素广播只接受 scalar 或完全相同的 shape；不把 `[n]` 当 `[n,1]`，
  不沿 singleton matrix 维广播。promotion/null/Kleene/where/pow 保持原合同。
- Matmul 仅 `[m,k] @ [k,n]`、n>0、f32/f64 且实际 null count 为零；即使输出
  为空也验证两侧。nullable schema 本身不是拒绝理由。m=0 返回 `[0,n]`；
  k=0 返回 positive-zero `[m,n]`；f32/f32 保持 f32，混合升 f64。每输出按
  递增 k 在提升后的 dtype 中先乘再从 +0 左折叠相加；无 FMA、重排、隐藏
  宽累加或 exact-reduction 替换，不承诺 NumPy/BLAS 位级一致。
- Reshape 要求 rank<=2、checked product 与槽位数相等；无负数、-1 推断、
  stride/order 参数。owned reshape 共享数据，仅新 descriptor；view 保留借用。
  Transpose 的 rank 0/1 为 identity；一般 matrix 复制 value/validity 为 packed
  存储，零元素或 size-one-axis 可共享。所有 owned 输出拥有独立生命周期。
- Reduce 只接纳浮点；axis=None 归约全部，否则 axis<rank，拒绝负 axis。
  keepdims 将被归约轴设为 1，axis=None 则每原轴为 1，scalar 仍为 scalar。
  每 lane 使用 A2 exact sum/mean 及既有 min/max/null/NaN/signed-zero 规则；
  空或全 null lane 为 invalid，sum 也不例外。例：`[0,3]` axis=0 得三个
  invalid，`[2,0]` axis=1 得两个 invalid。

按元素而非行检查 max_elements。新输出、bitmap、descriptor、packing 和峰值
scratch 分配前计费；reshape/trivial transpose 的共享数据只计一次，新 shape
仍计费，多个 named outputs 仍逐个计输出元素。借用输入不变成新 owned 分配。
取消检查必须进入长 dot product、单 lane 和 transpose 内层，且不改变累加顺序。

### 5.2 IR2、ArrayExpr 与 table 边界

保留 `native:array_expr@1` identity，新增 ir_version=2；IR1 的字段、rank<=1、
结果和拒绝条件不变。IR2 保留 64 KiB、128 nodes、depth 24、32 inputs、16 outputs
上限，描述符允许 rank<=2 的非负整数或 null extent。新增 matmul(left,right)、
reshape(input,shape)、transpose(input)，reduce 增加 axis=null|0|1 与 keepdims。
每 IR program 最多一次 matmul；rolling/EWMA 仍为 vector。抽象输出 assertions
必须与推导完全一致，实际执行先复核 concrete shape、尺寸溢出、dtype 和预算。

`NativeArrayFactory::infer_program(&JsonMap) -> Result<JsonMap>` 与私有绑定
`_native._infer_native_array_program(program_json: str) -> str` 共用 Rust
parser/inference。query outputs 只有 name/node，返回 name/node/dtype/shape；
执行 IR 仍要求 assertions。仅用 descriptor/literal，不读行、不创建探测数组，
不通过 NumPy/JAX 推导；查询缓存依赖 runtime registration revision。

```python
def array_input(
    name: str, /, *, dtype: str, shape: Sequence[int | None],
    backend: Literal["native"] = "native",
) -> ArrayExpr: ...

class ArrayExpr:
    def reshape(self, shape: Sequence[int], /) -> ArrayExpr: ...
    def transpose(self) -> ArrayExpr: ...
    # sum/min/max use the same keyword shape as mean.
    def mean(self, *, axis: int | None = None, keepdims: bool = False,
             skip_nulls: bool = False) -> ArrayExpr: ...
    def collect_array(
        self, inputs: Mapping[str, NativeArray | Batch | TableData], /, *,
        runtime: Runtime, options: ExecutionOptions | None = None,
        array_options: ArrayExecutionOptions | None = None,
    ) -> NativeArray: ...
```

array_input 是 dynamic 声明；parameter(kind="array", mutability="static", ...)
仍是 static。collect_array 只返回一个 native 结果，允许 table、dynamic array
及 static 参数共同作为根；要求已注册 Runtime、精确 named bindings、真实 native
payload、匹配的 dtype/shape，复制输入 mapping，不修改调用者。每次拥有 fresh
batch plan，复用既有 segment、logical/physical bindings 和 immutable template；
cache key 包含 normalized array_options，payload 不入声明缓存。该选项只限每个
native IR 执行，非全图 RSS。阻塞入口拒绝运行中的 event loop。compute/collect/
Program 继续返回 tables；不增加 async array stream 或 mixed Program results。
Program.with_input 接受 ArrayExpr 时只允许 array_input 根，不能接受计算结果。

新增 `NativeTableArrayFactory`，identity `native:table_array@1`：真正的 batch
与受限 stateless stream factory。wrapper options 为 ir_version=2、inputs、nodes、
outputs、execution；恰有一个 required exact-schema table 和一个 output，至多一个
`kind="array", binding="static"` 的矩阵输入，名字任意，仅作唯一 matmul 的直接
RHS，无 dynamic RHS。table input 的 columns 非空、唯一、同 native dtype；null
可入存储，matmul 再按实际 null 拒绝。outputs 有两种闭合形状：

```text
table: {name, kind:"table", node, attach_to:<table input>, names:[...]}
array: {name, kind:"array", node, dtype, shape}
```

数值结果必须保留原 table row origin，rank 1 或 2；table attachment 的 vector
需一个名字，matrix 需已知正宽度及同数目的名字，不能与原字段冲突。仅接纳同
origin 的逐元素、static-RHS matmul、matrix axis=1 reduce 和已证明保留行映射的
shape 操作；拒绝 rolling/EWMA、axis=0/full reduce、rank-2 transpose 或未证明的
reshape。stream 还必须对任意批行数保持准入及行映射；锁定 batch length 的
concrete reshape 不合格。编译及 source-open 前验证 RHS 是真实 latched static，
不能只信 `binding` 字符串。保留全部 Arrow chunks/slices 的逻辑行序、原 columns/
Arrow metadata/Batch metadata、精确空输出 schema；packing 必须计费。

table-rooted collect_array 复用 table/SQL segments，table_array 可先返回 array；
full reduction/transpose/dynamic RHS 等 batch-only 工作在后续 array_expr segment。
attach 后直接 filter/project、SQL 前后组合、独立多输出及共享不同名参数均需真实
执行。origin 是语义，不由长度恢复：transpose/axis=0/full reduce/未证明 reshape
丢失后，即便 shape 再次相等也不能回贴；独立 array_input 没有 table origin。
保留既有 ASOF/event-window 及 NumPy/JAX 准入边界。

独立 array collection 可以每段一次 matmul 地分解链；**单次 attachment 表达式
的两个 chained matmul 不在 M2 准入内**，报 CompileError，例如
"native attachment supports at most one matmul between table stages; found 2"。
显式路径为 attach 第一结果 → from_columns(新 TableExpr) → 第二 matmul → attach
到该新 TableExpr → project 最终列。无需自动隐藏列，也不能凭同长度恢复 origin。

### 5.3 Native static scope、digest-v1 与具体预算

冻结/规范编码/native placement 对一次执行中的每个 trusted owner 各做一次；
**SHA-256 仍按 physical alias 分别执行**。digest-v1 包含 version、kind tag、
length-prefixed physical name，再接 descriptor/cells；每 alias 必须与既有
`static_input.rs::digest_for_name(alias, batch)` 逐位相同，不改 digest/manifest。
初始化各 alias 的原前缀 SHA state，一遍规范编码按固定 4 KiB chunk 喂给全部
hashers，无 body-sized Vec；hash 工作仍随 alias 数×payload bytes 增长。
每 alias 独立验证自己的 StaticInputSpec 和恢复 digest，在 source open 前完成。
同 owner 某 alias 的 dtype/shape 错误，必须该 physical path 失败且不发布 placement。

共享 key 为 `(execution scope, retained trusted owner, validated backend/dtype/shape,
engine-minted canonical-preparation identity)`。retain owner 防地址复用；不同 owner
同值不合并，不凭相同底层 buffer 或 caller metadata 共享，不用不同 salted alias
hash 作共享 key，不引入可持久化的替代 digest。每 finite run/job/recovery job 都有
fresh scope，失败/取消不发布可复用 entry。

既有 prepare_static_inputs 没有 budget/cancellation 参数，不能假定已有该机制。
实施需 native-only 私有受控路径：compiled native consumers 提供真实 node/port→
physical static projection 和 `B_j = max_allocation_bytes`；finite plan 保留相关
StaticInputSpec，RunContext clones 共用本次 fresh scope，managed job 同理。
公开 static_input 的既有 global lookup 语义不改，不向 native 回落到无预算 helper。

对 owner g 以 checked arithmetic 计算：F_g 为 frozen carrier、宽化值、null flags、
shape/descriptor 及 owner 额外保活的不同 source buffers；P_g 为 packed native
values/validity/shape；D_g 为每 physical alias 的 digest text/新增 descriptor；
W_g 为 4 KiB chunk、全部 live SHA states、真实 conversion/snapshot peak scratch。
已有 frozen carrier 仍计保留成本；f32 的宽化 frozen 与 packed f32 是两份分配。
仅在 owner identity 证明同一 allocation 时去重，检查 actual capacities；此逻辑
存储计费不含 allocator/map-node overhead，不是 RSS。

```text
G_j = consumer j 的所有不同 native static owners
R_j = sum(F_g + P_g + D_g for g in G_j)
对 owner g 的每个 consumer j: R_j + W_g <= B_j
consumer j 的后续 numeric packing/output/scratch 预算 = B_j - R_j
```

owner groups 串行 prepare，分配前预测全部最终 R_j；同 consumer 的重复 alias
只计 owner 一次，但所有 owner 必须求和。每 group 的完整 shared workspace 对
其每个 consumer 校验；较宽预算不能补贴较窄预算，不将无关 consumer 的最小值
变成全局上限。缓存命中不取消 R_j。仅新 native static path 扩展 consumer 预算，
不改无 static 的 native 或 NumPy/JAX 行为；不另加覆盖显式配置的默认上限。

重型 prepare 使用 bounded owned worker，并传入实际 RunContext 或 job/admission
的取消/deadline 检查，不伪造 per-batch context。检查 admission 前、最多每 4,096
槽（包括 null）、每编码 chunk、最多每 64 alias-hash updates 和最终 publication 前。
所有 group、每 alias descriptor/recovery digest、预算/实际容量及最终取消检查通过后，
才一次发布局部候选 map；错误保持 scope 不变。错误定位 `static_inputs.<alias>.bytes`
并写明 consumer、required/limit；numeric 错误保留 output/scratch 路径并说明 R_j。

### 5.4 M2 capability 与验收边界

M2 保持 capability schema 4。旧 native_scalar_vector_broadcast@1 不改，新增真实
native_packed_rank2@1、native_matmul@1、native_axis_reduction@1、
native_table_static_rhs@1。完整 M2 中 array_expr 仍 batch-only/checkpoint unproven，
实际 finite static 接入后才声明 supports_static_inputs；table_array 声明 batch+stream、
per_row_final、stateful=false、microbatch_invariant/deterministic/replay_safe=true、
checkpoint_support=stateless、state_version=None、supports_static_inputs=true、
partition_contract=row_axis_independent。不得用 Python stateless marker 代替 Rust 实现。

register_native 公共签名不改，原子插入 array_expr batch + table_array batch/stream
三项，固定 batch→stream lock 顺序，先验证所有 key/descriptor/records，首次插入后
无 fallible 验证。成功为两个 native_provider records、revision len+2、两种 symbolic
cache 只 invalidate 一次；任何 collision/failure 保持 maps/records/revision/caches/
generation。保留 singular descriptor query，plural 返回 array_expr/table_array。

M2 验收必须执行 tiny 独立数学 oracle、空/null/offset/alias-budget/cancellation、
真实 SQL/table/array segments、显式两段 attachment 与错误 lineage，以及一次 scope
共享、fresh scope/不同 owner 隔离、每 alias digest golden、不同 consumer 限额、
多 owner 求和/f32 overlap/hash workspace/cached-output 预算和失败零 publication/
source-open。不以 schema、counter 或 document-only 测试代替端到端行为；不宣称提速。

## 6. A6a 已批准合同：Rust owned sequence state

A6a 可独立交付，尚不发布 sequence factory/checkpoint capability。只新增 Rust
生命周期 owner，Python eager API 保持；不增加 Python state classes、高层 async
ArrayExpr stream、mixed Program 输出、公开 serde/Clone/文件 snapshot 表面。

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ArrayStateOptions { /* private fields; Default max_state_bytes = 256 MiB */ }
impl ArrayStateOptions {
    pub const fn with_max_state_bytes(self, max_state_bytes: usize) -> Self;
    pub const fn max_state_bytes(&self) -> usize;
}
pub struct RollingMeanState { /* private owned state */ }
impl RollingMeanState {
    pub fn new(options: RollingMeanOptions, state: ArrayStateOptions) -> Result<Self>;
    pub fn process(&mut self, input: ArrayView<'_>, execution: &ArrayExecutionOptions)
        -> Result<NativeArray>;
    pub fn reset(&mut self);
    pub fn state_bytes(&self) -> usize;
}
pub struct EwmaState { /* private owned state */ }
impl EwmaState {
    pub fn new(options: EwmaOptions, state: ArrayStateOptions) -> Result<Self>;
    pub fn process(&mut self, input: ArrayView<'_>, execution: &ArrayExecutionOptions)
        -> Result<NativeArray>;
    pub fn reset(&mut self);
    pub fn state_bytes(&self) -> usize;
}
```

new 验证空状态 charge，不按 window 预分配；reset infallible、释放历史并清零
recurrence/counters，保留配置。Debug 只显示选项/count/charge，不打印输入。
process 只接 rank-1 f32/f64（含 `[0]`），拒绝 scalar/matrix/bool，不隐式 flatten。
直接 handle 可在不同 call 混用 f32/f64，合法 f32 精确宽化，均输出同位置 f64。
所有 fallible 工作在 candidate 完成后才替换 owned state；失败保持原 counters、
history、capacity、bits，允许安全重试；输入及已返回输出始终只读。纯 CPU API 同步。

mean 复用 M1 add-before-expire、FloatSummary 的 sum/mean/**m2**/非有限计数及
readout；null/NaN 占窗口位置、不计有效样本，Inf 有效。StableV1 default/nonfinite
refold/extreme 行为不改，非 exact 普通 mean 或 SQL AVG 的 NaN-value 路径。
StableV2Preview 对应 `stable_v2`，累积 row counter 保留既有周期 repair cadence，
跨 chunk/process 调用及恢复保持累计位置；仅显式 reset 清零。EWMA 保留 first-valid seed、
`alpha=2.0/(span as f64+1.0)`、`previous+alpha*(sample-previous)`、span-one 分支；
null/NaN 不推进 valid count，warm-up 后 carry previous，Inf/生成的 NaN 仍有效。
mean: window>0、1<=min_periods<=window；EWMA: span/min_periods>0，可 min_periods>span。
参数为 u64；checked processed-position counter 跨 call 保留，EWMA 另存 valid count。
空 chunk 不推进/repair，返回既有 eager 空表示；u64::MAX 后新行失败，空 chunk 不溢出。

共享私有 transition/candidate，但 eager 旧签名、数值、空 validity 和原 execution
预算不被新增 handle state cap 收紧；table codec/layout/fingerprint 及数学合同不改。

```text
state_bytes(mean) = 64 + 16 * allocated_history_slots
state_bytes(ewma) = 24
S = old committed state; C = isolated candidate; O = output/validity/shape
T = copy/numeric/encoding/decoding scratch; K = distinct live captured segments
P = S + C + O + T + K <= execution.max_allocation_bytes
C <= state.max_state_bytes; output positions <= execution.max_elements
```

以上是逻辑 storage charge，非 ABI/RSS；不含 options/handle/allocator/container
overhead。slots 按 actual capacity 含 insertion spare，不只算 live rows；window
巨大时最多为已观察所需的 window+1 transient slot，checked bound，禁止预留整个
window。new 的 state cap 至少 64/24；reset 回到该 charge。新 stateful API 的
max_allocation_bytes 明确涵盖全部参与的 live ownership，非仅增量分配；相同 Arc
allocation 可去重，复制 candidate 必须另计。分配前预测，实际更大容量发布前复核。

checkpoint prepare 取 C=O=0，加实际 encoding/capture；restore 同时计 encoded
segments、old state、candidate、scratch。segment lease 跟随实际 StateSegment 的
最后 owner，包括后台 staging；缓存引用或 reset 不能提前解除 K。已知长度必须
在 managed preload 分配前做 native reservation，不能无界 load 后才拒绝。
仍按每 edge 计 Batch，非全图内存池，不改既有 eager/table/NumPy/JAX 预算。

## 7. A6b 已批准合同：真实 ordered stream 与 managed recovery

A6b 把 factory、Rust lifecycle、codec、managed restart、capability migration 一并
交付；不得先发布未实现的 checkpointed capability。本节是该后续阶段的合同。

### 7.1 Identity、options 与输出接受点

新增 `calc_flow::array::NativeArraySequenceFactory: Default + StreamOperatorFactory`，
并有 `pub fn capability_descriptor() -> JsonMap`。identity 为
`native:array_sequence@1`，stream-only，无 batch factory；固定 required array
ports input/output，无 Arrow schema，factory 自行验证 identity/ports，即使绕过 registry。
独立于 array_expr IR 的闭合 options 为：

```json
{
  "ir_version": 1,
  "input_dtype": "float64",
  "kernel": {"function":"rolling_mean","window":20,"min_periods":1,"numerical_profile":"stable_v1"},
  "execution": {"max_elements":10000000,"max_allocation_bytes":268435456},
  "state": {"max_state_bytes":268435456}
}
```

ir_version/input_dtype/kernel required；dtype 仅 f32/f64。rolling_mean 要求 window，
min_periods=1、profile=stable_v1 默认，可显式 stable_v2；ewma 要求 span、
min_periods=1 默认，拒绝 window/profile/alpha/adjust。execution/state 可省略，
各取上述默认。严格整数/count/profile/未知字段校验，64 KiB 和现有 JSON depth
边界；raw duplicate keys 由上游 strict parser 拒绝，JsonMap 不恢复丢失拼写。
保持前述 A3 parsed-value 边界，不新增 callbacks/source/import paths。

每 Batch 必须是真实 NativeArray、匹配 declared dtype 的 `[n]`，空批也检查；
一批输出一个同行数 f64 Batch 并保留 metadata。按唯一 dynamic edge arrival
位置推进；metadata.sequence 不负责排序/去重/reset。拒绝 static/第二输入/隐式 zip/
latest-value/实体推断/矩阵 flatten，source open 前验证真实 dynamic ingress。
多实体/event-time/late 仍走 table；独立节点/分支有独立 state owner。

process_data 完整 validate/preflight → bounded worker 生成隔离 candidate/output →
完成所有 fallible 包装/预算/取消检查 → await `emit("output", batch)` → 成功后
立即 nonfallible/non-awaiting commit 并返回。emit 成功与 commit 间不得再检查取消/
deadline、分配、编码或 fallible metrics；失败保留旧 state。partial fanout 沿既有
失败 job/恢复规则，不能承诺撤回已接受输出或新增 exactly-once 保证。

每 operator 除 factory-wide worker limit 外，必须有 pending-work owner/gate：
awaiter 丢失时旧 state/candidate/output/scratch/permit/credit 保留至 settlement。
后续 process/capture/restore 先 settle，或在同一 P 中计入全部仍 live 工作；
sync reset 不能抹 credit，sync restore 必须拒绝 busy 或先完整计费。worker 只返回
candidate，无共享 mutable state/stale commit。活跃 runtime 下 supervisor/reaper
拥有 join；正常 Tokio Runtime drop 等待 blocking work；调用方主动选择
shutdown_timeout/shutdown_background 不保证返回前 settlement，不新增清理线程。
Watermark/Idle/Barrier/EOF 仍由 runtime FIFO 转发；数值 watermark/progress handler
无额外输出，EOF 无额外行，barrier 只捕获此前成功 emit 已 commit 的状态。

### 7.2 Lifecycle、异步 restore 与 terminal 路径

扩展既有 non-exhaustive enum，保留原 variants；native sequence 返回 version=1
及三个 true，compiler 映射为同正版本 CheckpointedStateful，零/不一致版本拒绝。
microbatch_invariant 指保序重分批后拼接的数值位置一致，非 Batch metadata 相同。
stateful external 的 delivery 检查同样验证 determinism/replay safety；Python dict
不能伪造 Rust proof，不借用 `_StatelessProviderLifecycle`。

```rust
// Additional StreamOperatorLifecycle variant:
CheckpointedStateful {
    state_version: u32,
    microbatch_invariant: bool,
    deterministic: bool,
    replay_safe: bool,
}
// Defaulted additional method on the existing async_trait StreamOperator:
async fn restore_async(
    &mut self, snapshot: &OperatorStateSnapshot, context: &StreamOperatorContext<'_>,
) -> Result<()> {
    context.check_cancelled()?;
    self.restore(snapshot)
}
```

prepare_checkpoint_async 在 bounded worker 受控编码/copy/SHA，prepared segment 绑定
committed revision；checkpoint(epoch) 只 clone segment/小 metadata，revision 变更
后未 prepare 则 checkpoint error，无同步重编码 fallback。unchanged revision 可共享
segment。native restore_async 同样有界 decode/validate 后才换 state；sync restore
复用 decoder 并遵守 pending-credit 约束。正常 managed entry 在 operator ack 和
任何 Source::open 前 await 它；已知 segment size 在 preload 前 admission，加载/
hash/编码也须有取消接点，不能最后用 monolithic SHA 绕过控制。

terminal-complete fast path 会绕过普通 external restore，因此**每个 terminal manifest
中的 native sequence node**也必须在报告 completed 或打开任何 sink 前通过同样的
bounded preload、checksum、codec、identity、limits 检查；即使永不打开 source。
只增加最小 native validation dispatch，不改变 legacy external 或 table/SQL/join
恢复合同。reset 可丢本地 capture 引用，但 runtime 仍持有的 segment lease 不解除。

### 7.3 闭合 state layout 1

通过既有 StateBackend/session/StateHandle/OperatorStateSnapshot 持久化；manifest
仍为 3，sequence semantic state/numeric semantics/layout 各为 1。每 checkpoint
即使空数值状态也恰有 `state` segment，缺失即损坏；fresh job 用 reset。strict inline
metadata 记录 identity/layout/numeric semantics/state fingerprint；runtime 保留其
reserved version wrapper、progress/cursors/frontier/长度/checksum 职责。

state fingerprint 为 canonical data-only identity/input_dtype/kernel/normalized math
parameters/profile/numeric semantics/layout 的 SHA-256；全 pipeline fingerprint 仍校验
配置/bindings，不能因预算不改变数学就绕过它。以下整数固定 little-endian，顺序固定：

```text
76-byte header:
  magic[8]="CFASEQ\0\0", layout:u32=1, numeric_semantics:u32=1,
  kernel:u8 (1 mean,2 ewma), input_dtype:u8 (1 f32,2 f64),
  profile:u8 (mean 1 stable_v1 / 2 stable_v2; ewma 0), reserved:u8=0,
  state_fingerprint[32], window_or_span:u64, min_periods:u64, processed_positions:u64
mean payload (all listed counters/bits u64):
  valid_count, sum_bits, mean_bits, m2_bits, nan_count, positive_infinity_count,
  negative_infinity_count, history_length;
  oldest-to-newest history slots: tag:u8 (0 missing,1 present), value_bits:u64
ewma payload: valid_count:u64, previous_bits:u64
exact bytes: mean=140+9*history_length; ewma=92; no trailing bytes
```

所有必要浮点保存 raw IEEE bits（含 signed zero/Inf/summary NaN payload）；missing
tag 的 bits 必须 0，present history NaN 拒绝，因为 M1 已将输入 null/NaN 归为
missing，不能读取其无用 payload。不写非有限 JSON float，不套用 static digest 的
NaN canonicalization。恢复 ring 的逻辑顺序即可，不持久化 allocator capacity、
不按 window 预分配、不重算 summary/repair position 或从 history tail 重启 EWMA。

分配/decode 前检查 magic/versions/tags/length/fingerprint/params/profile/dtype、
checked size、预算及 inventory；history_length=min(processed_positions,window)，
valid_count 等于 present 数，Inf counts 与历史相符、nan_count=0；EWMA valid_count
不大于 processed_positions，零 valid 时 previous_bits 必须 +0。拒绝 unknown/
duplicate inline fields、错误计数、overflow、extra/missing segment/bytes。保留 summary
bits，正常下一行才按既有规则 repair。StateSegment/StateHandle 提供 checksum，非
认证；错误不改变旧 state，并在 source open 前或 terminal 完成前失败。

### 7.4 计划中的 A6b capability 4→5 与原子注册

**M2 保持 schema 4，A6b 才迁移到 5**：closed ProviderCapability 增加
`state_layouts: tuple[int, ...] = ()`。checkpointed_stateful 必须非空、严格递增、
positive exact integers；其他 checkpoint kinds 必须空。PartitionContract 新增
`single_ordered_sequence`，表示 arrival 顺序单 owner，无 partition independence/
multi-input alignment；无需额外 order flag。sequence descriptor 为：

```text
native:array_sequence@1; port_binding=fixed; required array input/output
modes=[stream]; stateful=true; microbatch_invariant=true; requires_watermark=false
finality=per_row_final; checkpoint_support=checkpointed_stateful
state_version=1; state_layouts=[1]; deterministic=true; replay_safe=true
supports_static_inputs=false; partition_contract=single_ordered_sequence
supported_dtypes=[float32,float64]; safe_dtype_rule=native_float_dtype@1
shape_rules=[native_sequence_vector@1]
```

fixed ports 可提供 strict options_json_schema，但 options_schema 必须 None，二者
互斥；原 fixed/no-JSON-schema 仍合法，ports 不从 schema prose 推断。旧 providers
layouts 默认为空，array_expr batch-only/unproven 与 table_array stateless/
row_axis_independent 语义不变；NumPy/JAX/plain providers 原默认不变。无兼容 shim，
消费者按 schema_version 迁移，不能将 v5 当 v4；project/manifest 仍 3、digest-v1 不改。

register_native 保持签名，一事务插入 array_expr batch、table_array batch/stream、
array_sequence stream 四项；Python 三个 native_provider records，plural descriptor
顺序 array_expr/array_sequence/table_array，singular 仍 array_expr。成功 revision
len+3、两种 cache invalidate 一次；所有 key/descriptor/records 先准备，失败保持
两 maps/records/revision/cache contents/generation。snapshot/worker replay 按 bundle
注册一次，不能逐 record 重放；不完整或矛盾 native evidence 拒绝，不能降级 lifecycle。

同批更新 Rust descriptors/lifecycle/admission，Python version literals/strict keys/
lists/integers/rules/partition/layout validators/defaults/canonical snapshots，PyO3
helpers 与 `_native.pyi`/公共 typing/schema，registration snapshot/replay/count/cache
消费者，以及 docs/examples/fixtures 中 version/字段/layout/两条旧 record 假设。
project-v3 schema 必须无漂移；能力迁移不提前混入 A6a 或当前 M2。

### 7.5 错误、实施拆分与验收

direct input/state/peak/count 错误沿 InvalidArgument，字段分别为 input.shape/
input.dtype、state.max_state_bytes、state.peak_bytes、input.rows；输出沿
outputs.output.*。stream wrong dtype/counterfeit 定位 inputs.input.dtype/backend，
kernel 参数定位 options.kernel.*，消息说明 expected/received 或 required/limit，
不打印数值。managed codec/identity/limits 错误使用带 operator/field 的
CheckpointMismatch→既有 Python CheckpointError；底层 I/O source 与实际取消身份保留。

A6a 交付 direct state/shared transitions/limits/rustdoc/20–50 行 Rust 示例；对比
one-shot 与 splits/empty/Arrow offset/f32 widening/null/NaN/Inf/zero/warm-up/巨大 window/
两 profile repair 边界，验证 failed call 后重试、输出生命周期、old/candidate peak
及 table/M1 oracle 不变。A6b 完整交付后才声明真实 stream/checkpoint capability，
验收包括：

- 真实 runner 单 ingress 结果/隔离、错误 static/多输入、emit 拒绝及成功后立即
  cancel 的 commit 边界、slow sink、control FIFO/barrier/EOF、partial fanout。
- aborted waiter 后 tight-budget reuse/reset/restore/capture 不能遗漏 pending
  allocations；worker/permit settlement、旧 capture 在新处理/reset 后继续计费。
- raw bits/计数 round trip、错误 checksum/长度/tag/version/dtype/params/巨大 state；
  新 managed job/runner + replayable source + stable backend 真正恢复 committed prefix，
  与不中断输出相符，检查 seek/open，不能只做 encode/decode 测试。
- 普通及 terminal recovery 均测试 corrupt/oversized/valid state；terminal 还需
  valid managed checksum 但内部 codec/layout 错误，证明 semantic validation，失败
  不开 source/sink、不报告完成；其他分支 static identity 改变仍 fail closed。
- 四 registry collision、descriptor failure、cleared Runtime、bundle/replay 的
  三 records/四 entries/一次 invalidate，以及旧 providers/table checkpoint 回归。

每行为先 focused RED，再实现与必要 lint/rustdoc/example 检查；完整矩阵/coverage
留给 CI，本设计无性能测量或提速承诺。A7 保持原计划另行设计，本归档不新增指标、
OHLC zipper、MACD 多输出 owner 或通用框架。

## M2 native preparation budget boundary clarification

Status: **Pass for the explicit accounting-boundary clarification**. This
appendix is the authoritative clarification of section 5.3's F/P/D/W and
per-consumer reservation contract. It preserves the earlier API/A6 archive as
history and does not change any public signature, digest-v1 byte, manifest-v3
protocol, NumPy/JAX budget interpretation or ordinary native execution without
static reservations. It is not an implementation, full-feature/source-review,
executed-test, performance or merge-readiness approval.

The reviewed source is section 3 of the design input titled
`m2-chain-attachment-api.md`, whole-document SHA-256
`675302e98bdf69f32ca12f0eefa0ae3b42449d52b9497673924fb348fed927d0`.
The focused review titled `m2-manifest-budget-critic.md`, SHA-256
`28e852b3fb1d3f47f0d1af70e0a5160e9b25c04a8d0294aef1e46e8650268a0e`,
passed these explicit boundaries; its
[canonical disposition](../critiques/native-array-runtime.md#m2-native-metadata-budget-boundary-review)
preserves the source-review obligations. This archive supplies the clarification
referenced by the [resource specification](../specs/native-array-runtime.md).

### Trusted scope and consumer limits

Existing `prepare_static_inputs(declared, supplied)` has neither a budget nor
cancellation argument; do not treat it as an existing controlled preparation
mechanism. The new native-only path receives trusted compiled records of node
identity, local-port-to-physical-static projection, StaticInputSpec and each
consumer's validated `ArrayExecutionOptions.max_allocation_bytes`, called B_j.
Caller metadata and ordinary dynamic input cannot fabricate these records.
Non-native providers may keep the legacy helper; native preparation cannot
silently fall back to it.

Finite compilation retains the required static declarations. Each execute owns
one fresh private preparation scope shared through that run's RunContext
clones; each managed job/recovery job owns one fresh scope in its job context.
A trusted owner is retained for the scope's lifetime. Sharing requires that
owner and successful validated backend/dtype/shape/canonical preparation
identity; equal contents or allocator addresses alone do not establish sharing.
Physical aliases still have separate declarations, salted digest-v1 prefixes
and independently checked recovery digests. Freeze/canonical encoding/native
placement may be shared once per trusted owner, but SHA work remains per alias.
No global pool or cross-run placement cache is introduced.

### Included F/P/D/W ownership

Use checked sizes and actual retained capacities where a buffer has capacity:

- **F_g:** frozen carrier and widened scalar storage, null flags, shape and
  backend/dtype descriptors retained for the execution. A supplied already
  latched carrier is still charged. Include distinct source buffers kept alive
  by retaining the trusted owner; they cannot become an uncharged extra copy.
  Do not substitute a saturating estimated_bytes value for checked capacity.
- **P_g:** packed native dtype-width values, validity and shape. An owner may
  share a buffer between representations only when allocation identity proves
  it. Widened frozen f32 and packed f32 are distinct allocations and both count.
- **D_g:** independently owned native alias names, digest-version/SHA text and
  newly owned descriptors for **every physical alias**, including the concrete
  metadata copies described below. Equal text in two allocations counts twice;
  an actually shared Arc allocation counts once.
- **W_g:** the peak simultaneously live preparation scratch: the fixed 4 KiB
  canonical encoding chunk, actual live SHA states, conversion/snapshot
  temporaries and necessary controlled worker scratch. Private read-only
  access should avoid full-copy public snapshots; if a full temporary is
  required, admit its complete charge before creating it.

Managed D includes native strings/descriptors independently owned by the scope,
prepared latched/digest maps, installed static map, Run/Job settings JSON,
required persistent checkpoint identity, the live recovered selected manifest,
the newly constructed current-epoch manifest, and temporary native JSON-map
strings used during checksum/serialization. The live selected manifest is not
backend history. A native field does not become excluded merely because its
owner is a manifest or JSON value. This includes newly materialized native-only
maps/strings even when a later whole-document serializer is excluded.

### Explicit exclusions and the actual guarantee

The guarantee is a per-native-consumer limit on **specified native-owned
payload, descriptor and buffer storage and preparation work**. It excludes:

- existing allocator/map-node overhead and the pre-existing outer
  `static_inputs` settings key/frame that exists with zero native aliases;
- the whole generic manifest canonical String/Vec allocation and global I/O
  envelope, as indivisible generic checkpoint machinery;
- backend historical manifest retention maintenance, including its separately
  loaded transient history Vec.

This exclusion is limited to the generic whole-manifest/I/O envelope; copying
native array storage or encoded native operator state does not become exempt
because it occurs in a backend or I/O path. Such copies must be eliminated or
charged under the applicable F/P/W/K contract; A6 state/capture accounting is
unchanged.

Do not assign an invented fraction of a generic serializer buffer's physical
capacity to native fields. Their encoded byte contribution is not that
allocation's capacity. Conversely, the separately owned native field/string
copies inside selected/current manifests or JSON maps remain in D. Mixed
legacy operator, sink and schema contents do not acquire a new generic
serialization budget through this feature. Borrowed dynamic input remains
excluded as before.

These are named exclusions, not consequences silently inferred from the old
allocator-overhead exclusion. Do not describe this limit as all live bytes,
all checkpoint memory, an exact runtime-memory bound, total process memory or
RSS. No existing NumPy/JAX behavior or public budget knob changes, and no new
implicit limit overrides an explicitly configured native consumer limit.

### Lifecycle overlap and descriptor reservations

Determine D from the maximum **actual simultaneous ownership** across the
bounded lifecycle, including each independently owned copy within a phase.
Preinstall, installed, capture/checksum and publication are useful names for
organizing a proof; phase names themselves do not establish non-overlap.

Before installation, local preparation maps, root settings, persistent identity
and selected-manifest validation may overlap. Treat consumed preparation maps
as released before task scopes only if the actual transfer/drop ordering proves
that release. Later selected/current manifests and native JSON-map temporaries
must be included wherever they coexist. A fresh managed run may reserve one
possible selected-manifest slot as an explicit conservative bound because
selection is not yet known; it is not spare credit for another component.

The engine-owned settings-copy inventory must be derived from actual clone
sites and dispatch: root job context, logical operator tasks, dynamic source
bindings, named output sink tasks (not each sink inside one task), and the
optional benchmark context. Verify the managed SQL capture/restore dispatch's
thin SqlRecoveryContext rather than counting a direct adapter's full-context
clone as another managed task. Arbitrary user-created context clones inside
external callbacks are outside this bounded engine-owned inventory. This
appendix does not certify an evolving resident-context formula or multiplier.

Taking a maximum across phases is valid only after final source review proves
non-overlap for the relevant allocations, including cancellation/worker
settlement, asynchronous staging, SQL temporary release before publication,
and any epochs that can coexist. Otherwise charge their overlap or a proven
conservative bounded combination. No fixed copy count may be invented from
phase labels or certified by a test that repeats the implementation's own
unverified constants. Verify observed descriptor capacities before publishing
or installing the scope, and retain the full admitted later-phase D bound
while that lifecycle remains possible.

### Per-consumer reservation and independent capacity checks

For each consumer j, let G_j contain its distinct trusted native static owners:

```text
R_j = sum(F_g + P_g + D_g for g in G_j)
For every consumer j of the owner g being prepared: R_j + W_g <= B_j
After preparation: numeric packing/output/scratch <= B_j - R_j
```

Predict all owners' final retained charges before any freeze/allocation,
including owners placed later. Prepare owner groups serially. Repeated aliases
of one owner do not duplicate F/P in a consumer's sum, but D still covers every
alias and required independent metadata copy. Sum distinct owners; charge the
complete shared W_g against every consumer needing g. A looser consumer cannot
subsidize a tighter one, and unrelated consumers do not create a global minimum
for unrelated owner groups. A cached placement retains R_j; cache hits do not
restore the full numeric B_j.

Verify F and P independently against their admitted predicted charges and
verify instantiated alias-descriptor capacities. **Future D is reserved
capacity, not spare credit:** an oversized actual frozen/source or packed
allocation cannot pass merely because the intermediate aggregate still fits a
larger full reservation that includes not-yet-created metadata. Reject such an
over-allocation, or explicitly re-admit the actual F/P together with the
complete still-required D bound and W before publication for every affected
consumer. Never consume future D to conceal the discrepancy. All checked
predictions precede allocation and actual retained capacities are validated
before anything becomes reusable.

### Cancellation, digest preparation and publication

Keep digest-v1 exact. Initialize each physical alias's SHA state with its
unchanged version/kind/length-prefixed-name bytes. Encode the shared descriptor/
cell stream once into the fixed 4 KiB chunk and feed each chunk, including the
final partial chunk, in order to every alias state. W includes the chunk and
all live hash states; finalized digest strings enter D. Do not create a
body-sized Vec or introduce a replacement digest/private body hash.

The private fallible cancellation callback uses the actual RunContext's
cancellation/deadline in finite execution and actual job/admission cancellation
and deadline in managed preparation. Check before admission, at most every
4,096 scanned slots including null slots, every encoded chunk, at most every
64 alias-hash updates, and after final placement before publication. Do not
synthesize a per-batch context that loses the job's clock/deadline. Heavy
preparation runs in the existing bounded owned worker mechanism; waiter loss
or teardown keeps inputs, candidate storage and permits owned until settlement.

Validate every alias's StaticInputSpec before placement and compare all
physical recovery digests before opening sources or exposing a reusable
placement. Keep all candidate maps local until every owner, capacity check,
per-consumer reservation, alias digest and final cancellation check succeeds.
Publish into the fresh scope once. Failure/cancellation leaves it unchanged,
with no partially prepared reusable cache. Ownership of unsuccessful workers
must settle before their reservations are released.

Choose the first offending physical alias/consumer in deterministic name order.
The preparation error path is `static_inputs.<alias>.bytes`, naming consumer,
required bytes and limit. Numeric execution retains its existing output/scratch
error paths while explaining the reserved static bytes R_j.

### Final source-review and acceptance obligations

This boundary Pass leaves the following proof obligations with implementation:

- Account for the actual copy sites/capacities in fresh and recovered jobs,
  selected plus current manifests, native JSON-map clones, settings/task
  contexts, SQL dispatch and staging. Show genuine Arc sharing where claimed.
  Prove the bounded overlap of phases/epochs, including cancellation and late
  drops; do not certify guessed multipliers.
- Test tighter/looser consumers, several owners whose sum exceeds one limit,
  frozen-f32/packed-f32/source overlap, live hash-state/chunk scratch and a cached
  numeric output that fails only after its static reservation is included.
- Exercise actual F/P capacity above its own prediction while unused future D
  exists: aggregate fit alone must not accept it. If re-admission is used,
  include the full still-required D bound for every consumer.
- Test null scanning and many-alias hashing cancellation, independently owned
  metadata in current/selected/JSON representations, and asynchronous lifetime
  boundaries. Assert failure publishes no reusable placement and managed
  startup/recovery opens no sources. Keep per-alias digest golden bytes and
  non-native preparation behavior unchanged.

These are required source/behavior proofs, not reported results. The relevant
source boundaries include
[`CheckpointManifest::canonical_bytes` and `recompute_state_checksum`](../../../crates/calc-flow/src/state/manifest.rs),
which separate native JSON strings from whole-document serialization, and
[`retain_locked`/`load_retention_manifests`](../../../crates/calc-flow/src/state/transaction.rs),
which own backend retention-history maintenance. Existing examples, selected
checks and final implementation review must verify the complete feature under
normal repository policy; this clarification supplies no performance claim.


## Managed Python error projection clarification

The A6 error paragraph's `CheckpointMismatch -> CheckpointError` statement
describes the generic direct-core exception conversion. It does not replace
the existing managed streaming exception boundary.

For managed `StreamingRunner` startup/recovery, preserve the existing frozen
`StreamingRuntimeError`: native codec or state-identity failures retain
`category="checkpoint_mismatch"`, `component_kind="operator"`, the actual
`component_id`, and the safe state field (for example `state.layout`) in the
message. No new exception class, hierarchy or native-only conversion is added.
The direct-core conversion continues to use `CheckpointError` where it already
applies; cancellation identities and underlying managed I/O categories retain
their existing projections.

This qualification follows the actual `continuous.rs` managed mapper and
`error.rs` generic mapper. It corrects the earlier shorthand while preserving
the requirement to reject invalid native state before source/sink opening or
terminal success. Acceptance must exercise a real managed restart with valid
outer state/manifest integrity and corrupt inner native codec, asserting the
public managed exception category, operator and state field. Custom provider
errors retain their existing redaction boundary.
