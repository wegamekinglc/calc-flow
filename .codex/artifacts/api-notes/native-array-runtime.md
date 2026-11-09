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

## 8. A7 approved indicator contract

Status: **Design Pass after revision**, archived as the authoritative A7
surface and numerical contract for `native-array-runtime`. The reviewed input
was titled `a7-indicators-api.md`, 1,206 lines, SHA-256
`9016f0c67d219490a80a9750d316fa5256373f3355a32d3eb27e982913e84fee`.
The [focused final review](../critiques/native-array-runtime.md#a7-indicator-contract-focused-review)
closed two blockers and four Table clarifications. The [parent specification](../specs/native-array-runtime.md)
incorporates the Table callback, numerical-oracle and budget qualifications.

This archive changes heading scope, review metadata, historical source paths
and adoption wording only; the approved actionable contract below is preserved.
Earlier sections remain historical contracts; this section supplies their
previously deferred A7 design. It is not a claim of implementation, executed
acceptance tests, CI or merge readiness. A7 implementation still depends on
completed A6b factory, worker ownership and managed recovery work.

The original IEEE operation order is retained. The orchestrator accepted the
narrow use of existing Rolling StateBudget.max_bytes as an indicator-node peak
limit; its current default and lack of a Python/project override remain
explicit below. No TA-Lib code dependency is introduced.

### Audiences

- Rust users: synchronous calculations and owned sequence states with aligned,
  immutable float64 output, bounded storage and atomic errors.
- Python users: `calc_flow.array` eager functions and `ts` table declarations,
  with the same algorithms in finite execution and managed streaming.
- Integrators: strict project declarations, truthful native capabilities and
  versioned state that survives a new job and process.

This design uses calc-flow implementations only. It adds no TA-Lib dependency,
FFI, vendored implementation, test import, or universal indicator framework.
The three indicators are specific extensions of the existing native numerical
module and RollingOperator. Existing `ts.ema` and `ts.macd` remain unchanged.

### Surface today and source findings

The source baseline inspected is `c9e39cea33f7661062ba50640ddbfa30ba7a3f5b`.
Repository-relative source references below refer to that revision. Sources are
`docs/introduction.md`, the canonical native-array-runtime specification/API,
`crates/calc-flow/src/{lib.rs,array/,operator/rolling.rs,operator/rolling/}`,
`python/calc_flow/{array.py,_native_array.py,_native.pyi,symbolic/ops.py}`, the
symbolic analyzer/lowerer, error projection, and native array/state examples.
The approved A6 contracts in sections 6/7 above and their
[review archive](../critiques/native-array-runtime.md) were also consulted.
This is source research, not a new test or performance result.

Current Rust exposes `ewma(FloatView, EwmaOptions, &ArrayExecutionOptions)`,
`EwmaState`, `RollingMeanState`, immutable `ArrayView`/`NativeArray` and the
existing table RollingOperator. Python eager functions take `NativeArray`.
A6b's approved sequence identity is `native:array_sequence@1`, with fixed
required array ports `input` and `output`, one ordered dynamic ingress,
capability schema 5, semantic state version 1 and layout 1.

Facts that constrain MACD:

- `symbolic/ops.py:611` implements `ts.macd` as two EWMA expressions and a
  subtraction. It **already rejects `fast_span >= slow_span`**. Its defaults
  are 12, 26 and `min_periods=1`; both EWMA readouts get that same minimum.
- `ts.ema` delegates to `ts.ewma`. EWMA accepts positive span and positive
  min_periods; min_periods may exceed span. Its first valid sample seeds it.
- Native/table EWMA uses `previous + alpha * (sample - previous)`, with
  `alpha = 2.0 / (span as f64 + 1.0)`, and direct replacement at span one.
  Null and input NaN do not advance it. After warm-up they carry its output.
  Inf and arithmetic-generated NaN are valid state/output.
- Existing finance-reference tests explicitly show that a carried MACD line
  updates an outer EWMA on a missing original price. The new signal must do so.
- Table EWMA accepts Arrow signed/unsigned integers of widths 8/16/32/64 and
  Float32/Float64. Integer samples use the existing conversion to float64.
  Direct native arrays accept only their supported floating input types here.
- `RollingOutputSpec` is a public, currently exhaustive Rust enum. Adding
  variants is a Rust source-compatibility change for exhaustive matches; this
  is called out below rather than described as universally nonbreaking.

### Approved Rust surface

Export the following from `calc_flow::array`. Options have private fields,
getters and consuming setters. All fallible functions return `calc_flow::Result`.
These signatures are approved for implementation; approval does not make them
current callable APIs.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RsiOptions { /* period: u64 */ }
impl RsiOptions {
    pub fn new(period: u64) -> Result<Self>;
    pub const fn period(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtrOptions { /* period: u64 */ }
impl AtrOptions {
    pub fn new(period: u64) -> Result<Self>;
    pub const fn period(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacdOptions { /* five private u64 parameters */ }
impl Default for MacdOptions { /* 12, 26, 9, 1, 1 */ }
impl MacdOptions {
    pub fn new(fast_span: u64, slow_span: u64, signal_span: u64) -> Result<Self>;
    pub fn with_min_periods(self, min_periods: u64) -> Result<Self>;
    pub fn with_signal_min_periods(self, signal_min_periods: u64) -> Result<Self>;
    pub const fn fast_span(&self) -> u64;
    pub const fn slow_span(&self) -> u64;
    pub const fn signal_span(&self) -> u64;
    pub const fn min_periods(&self) -> u64;
    pub const fn signal_min_periods(&self) -> u64;
}

pub struct MacdOutputs { /* immutable NativeArray owners, private fields */ }
impl MacdOutputs {
    pub const fn line(&self) -> &NativeArray;
    pub const fn signal(&self) -> &NativeArray;
    pub const fn histogram(&self) -> &NativeArray;
    pub fn into_arrays(self) -> (NativeArray, NativeArray, NativeArray);
}

pub fn rsi(
    input: ArrayView<'_>, options: RsiOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray>;
pub fn atr(
    high: ArrayView<'_>, low: ArrayView<'_>, close: ArrayView<'_>,
    options: AtrOptions, execution: &ArrayExecutionOptions,
) -> Result<NativeArray>;
pub fn macd_components(
    input: ArrayView<'_>, options: MacdOptions,
    execution: &ArrayExecutionOptions,
) -> Result<MacdOutputs>;

pub struct RsiState { /* private owned state */ }
impl RsiState {
    pub fn new(options: RsiOptions, state: ArrayStateOptions) -> Result<Self>;
    pub fn process(
        &mut self, input: ArrayView<'_>, execution: &ArrayExecutionOptions,
    ) -> Result<NativeArray>;
    pub fn reset(&mut self);
    pub fn state_bytes(&self) -> usize;
}

pub struct AtrState { /* private owned state */ }
impl AtrState {
    pub fn new(options: AtrOptions, state: ArrayStateOptions) -> Result<Self>;
    pub fn process(
        &mut self, high: ArrayView<'_>, low: ArrayView<'_>, close: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
    ) -> Result<NativeArray>;
    pub fn reset(&mut self);
    pub fn state_bytes(&self) -> usize;
}

pub struct MacdState { /* one owner of all three recurrences */ }
impl MacdState {
    pub fn new(options: MacdOptions, state: ArrayStateOptions) -> Result<Self>;
    pub fn process(
        &mut self, input: ArrayView<'_>, execution: &ArrayExecutionOptions,
    ) -> Result<MacdOutputs>;
    pub fn reset(&mut self);
    pub fn state_bytes(&self) -> usize;
}
```

All parameters are positive exact u64 values. MACD also requires fast_span to
be less than slow_span, preserving the existing `ts.macd` constraint. Neither
minimum is capped by a span; signal_span has no relation to either other span.
RSI/ATR have no separate min_periods, adjust, seed or missing-value option:
their period fixes the stated warm-up. Requiring period follows existing
`rolling_mean(window=...)`/`ewma(span=...)`; there is no hidden period-14 default.

Every new eager/state input must be a rank-one Float32/Float64 view, including
`[0]`; scalar, Boolean and matrix views fail. ArrayView is deliberate: FloatView
alone loses the rank that the owned-state contract must validate. ATR requires
equal lengths, permits each operand to have either float dtype, and never
broadcasts. Each valid f32 sample widens exactly to f64 before arithmetic;
f64 bits remain unchanged. Direct states are not bound to a dtype across calls.

All results are rank-one float64 arrays of the same length as input. Invalid
slots contain +0.0 and false validity; valid signed zero is preserved unless
the RSI/ATR nonnegative normalization below specifies +0.0. Each new function
returns an explicit validity bitmap, including an empty bitmap for `[0]`.
Outputs outlive inputs/state; no caller-owned value or bitmap is modified.

The three owned states have the A6a lifecycle: no public Clone, serialization,
snapshot bytes, restore, transaction hooks or async CPU aliases. `reset` keeps
options/limits, clears numerical state and counters, and cannot fail. Eager
functions start fresh using the same private controlled transitions; they do
not inherit an independent state cap. Failed calls leave committed values,
counters and charges unchanged. Empty calls produce empty outputs without
advancing or resetting state, but still validate shape, dtype and limits.

### Approved Python surface

Public eager exports are under `calc_flow.array` (`cfa` below), matching the
current module. One generic immutable data container is exported as
`calc_flow.MacdOutputs` and re-exported by `calc_flow.array`; it holds either
three array owners or three ColumnExpr references. Implement with the existing
Python 3.9-compatible frozen/slot dataclass helper and `Generic[TypeVar]`.

```python
@dataclass(frozen=True, slots=True)
class MacdOutputs(Generic[T]):
    line: T
    signal: T
    histogram: T

# calc_flow.array
def rsi(
    array: NativeArray, *, period: int,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray: ...
def atr(
    high: NativeArray, low: NativeArray, close: NativeArray, *, period: int,
    execution: ArrayExecutionOptions | None = None,
) -> NativeArray: ...
def macd_components(
    array: NativeArray, *, fast_span: int = 12, slow_span: int = 26,
    signal_span: int = 9, min_periods: int = 1, signal_min_periods: int = 1,
    execution: ArrayExecutionOptions | None = None,
) -> MacdOutputs[NativeArray]: ...

# Methods on the existing calc_flow.ts declaration namespace.
def rsi(value: ColumnExpr, /, *, period: int) -> ColumnExpr: ...
def atr(
    high: ColumnExpr, low: ColumnExpr, close: ColumnExpr, /, *, period: int,
) -> ColumnExpr: ...
def macd_components(
    value: ColumnExpr, /, *, fast_span: int = 12, slow_span: int = 26,
    signal_span: int = 9, min_periods: int = 1, signal_min_periods: int = 1,
) -> MacdOutputs[ColumnExpr]: ...
```

Only native owners enter eager functions. Existing `from_arrow`/`from_numpy`
are the explicit import boundaries, so mutable arrays are frozen before GIL
release or worker transfer. Arrow-to-native-to-Arrow requires no NumPy/JAX.
Python bool, negative/zero, fractional, and >u64::MAX counts are rejected before
native conversion; Rust revalidates every parameter. There are no Python state
handle classes, high-level array streams, mixed Program outputs or indicators
inside array_expr IR in this delivery. Table `collect`/`stream` retain their
existing return types and lifecycle.

The private binding/stub contract is concrete:

```python
def _native_array_rsi(
    array: NativeArray, period: int, max_elements: int, max_allocation_bytes: int,
) -> NativeArray: ...
def _native_array_atr(
    high: NativeArray, low: NativeArray, close: NativeArray, period: int,
    max_elements: int, max_allocation_bytes: int,
) -> NativeArray: ...
def _native_array_macd_components(
    array: NativeArray,
    options: tuple[int, int, int, int, int],
    max_elements: int, max_allocation_bytes: int,
) -> tuple[NativeArray, NativeArray, NativeArray]: ...
```

The MACD tuple order is fast_span, slow_span, signal_span, min_periods,
signal_min_periods. The adapter exposes named arguments and wraps returned
owners in MacdOutputs. No mapping, source text, import path or callback enters
these options. PyO3 reuses existing exception translation and detaches only
after retaining all immutable inputs and validating admission.

### Numerical contract: RSI and ATR

Numeric semantics version is 1 for each new indicator. The following operation
order is part of the contract. Use individual float64 operations with IEEE
round-to-nearest ties-to-even, no reassociation, no FMA substitution, no
fast-math and no alternative seed hidden behind a numerical profile.
Finite underflow, including a rounded zero, is allowed. A nonfinite result at
any required finite arithmetic step is an error, not a null or saturation.
Underflow can materially change RSI's normalized ratio, even when the exact
real-number gain/loss average is balanced. In particular, for period 2 and
prices `[0.0, 2^-1074, 0.0]`, the required output is `[null, null, 100.0]`:
the second gain update retains the minimum subnormal, while the second loss
update rounds to zero. The exact rational seed would yield 50. This is an
accepted consequence of the specified IEEE algorithm, not an error or a
reason to change the algorithm or enlarge a global tolerance. The Fraction
accuracy check below deliberately excludes this domain.

#### Missing values, validation and counters

Read validity before reading a payload slot. A null payload must not be
converted, tested for finiteness, compared or otherwise inspected. A valid NaN
is missing. A valid positive or negative Inf is an error.

For ATR, at each row inspect all **valid** operands in high, low, close order
for Inf before deciding whether any operand is missing. A valid Inf must not
hide behind a null/NaN in another operand. Whenever high and low are both valid
and finite, require high >= low and finite `high - low`, even if close is
missing or this is the first row of a segment. This is an explicit input-domain
check; the first complete row still contributes no true-range sample. When
either high or low is missing, do not compare or subtract that pair.

A missing RSI price or any missing ATR operand emits null and resets the entire
segment: previous price/close absent, seed count zero, and all means +0.0. The
next complete finite row only establishes the previous price/close. Missing
rows advance total processed positions, but not any warm-up count.

Each state has a checked total processed-position u64. Segment sample count is
the number of differences/TRs used for seeding, capped at period once ready.
Never allocate period history or form `period + 1` in u64. With period=1,
the second complete row is the first result and each later sample replaces
the respective means directly. Period=u64::MAX is valid, constant-space and
cannot finish warm-up before the total-position counter is exhausted: the
first row uses a position without supplying a difference/TR. A further
nonempty call then fails atomically; an empty call still succeeds if admitted.
This unusual but precise boundary is documented, not silently clamped.

#### Mean seed and Wilder transition

Gains, losses and true ranges are nonnegative finite float64 samples. Normalize
their zero to +0.0. For the kth sample of the seed:

```text
k == 1: mean = sample
k > 1:  difference = sample - mean
        increment = difference / (k as f64)
        mean = mean + increment
```

At k=period the first output becomes valid. Later samples use the same three
operations with denominator `(period as f64)`; period=1 uses direct assignment.
The u64-to-f64 conversion is the ordinary correctly rounded conversion, including
counts above 2^53. Check every difference, increment and updated mean for finite
representation. No raw running sum, `(period - 1) * mean` product, or unbounded
history is used. This avoids sum/product overflow for representable nonnegative
means without changing the specified price-difference/TR overflow errors.

#### RSI

After the previous price exists, compute `delta = price - previous_price` and
reject a nonfinite delta. Set gain to delta if positive, otherwise +0; loss
to -delta if negative, otherwise +0. Update both means as specified above,
then replace previous price. Warm-up needs period differences after the first
complete price, so a no-missing result first appears at zero-based index period.

For ready average gain G and average loss L:

```text
G == 0 and L == 0: RSI = 50
L == 0 and G > 0:  RSI = 100
G == 0 and L > 0:  RSI = 0
G >= L > 0:       ratio = L / G; RSI = 100 / (1 + ratio)
0 < G < L:        ratio = G / L; RSI = 100 * (ratio / (1 + ratio))
```

Use that parenthesization. The ratio is at most one; do not compute an
overflowing G/L, form G+L, or clamp a nonfinite result. Every valid result is
finite and in [0,100]. The flat result 50 is this API's explicit contract.
`[1e308, -1e308, 1e308]` must fail at the first unrepresentable difference,
even during warm-up, and leave a previously existing state unchanged.

#### ATR

After a complete row has established previous close, compute, in order:

```text
range = high - low                         # already domain-validated
high_gap = high - previous_close
low_gap = low - previous_close
TR = max(range, abs(high_gap), abs(low_gap))
```

Check the two signed gap differences before abs. Use comparisons over these
finite nonnegative values and normalize zero to +0.0. Feed TR to the seed or
Wilder transition and then replace previous close. Close need not lie within
[low,high]. A no-missing first result is at zero-based index period.

`high=1e308, low=-1e308` fails the range domain check even on a first row or
with missing close. A finite current high/low and an opposite extreme previous
close can fail a gap subtraction separately. Neither case emits Inf ATR or
commits any partial candidate. These checks are not applied to MACD.

### Numerical contract: full MACD

For each position, literally compose these operations using the shared native
EWMA transition and the existing missing normalization:

```text
fast = EWMA(price, fast_span, min_periods)
slow = EWMA(price, slow_span, min_periods)
line = fast - slow                         # null if either is null
signal = EWMA(line, signal_span, signal_min_periods)
histogram = line - signal                  # null if either is null
```

An original null/NaN price skips fast/slow updates but may yield a carried valid
line. That line **is still offered to signal**. A valid NaN line is normalized
to missing by EWMA, so it does not increment signal's valid-sample count;
signal may carry its prior valid result. A valid Inf line updates signal.
Histogram subtraction preserves validity of operands, so NaN arithmetic is
valid NaN, not null. Finite-input overflow follows existing EWMA/subtraction
IEEE behavior rather than becoming an RSI/ATR-style error.

All three recurrences advance in one owned candidate even if only line or
histogram is consumed. Signal's valid count may exceed the original price's
valid count because carried lines count. For finite, nonmissing, nonoverflowing
inputs, line first becomes valid at min_periods-1 and signal/histogram at
min_periods+signal_min_periods-2. These are descriptive mathematical indices;
use checked counters, not unchecked parameter addition, in implementation.

`macd_components(...).line` must match existing `ts.macd` on their common input
domain, and Table MACD accepts exactly the current integer/float numeric types
listed above. Table integer conversion keeps the existing f64 rounding contract;
it is not a new integer-exact array API. New table RSI/ATR accept Float32/Float64
only, with mixed floating ATR columns allowed. All derived columns are nullable
Float64. No TA-Lib lookback, adjusted seed, RSI reset rule or finite-only filter
is imposed on old or new MACD.

### Table declarations, shared lowering and schema

The new ts functions require the existing declared entity_by, event_time and
sequence_by order and a common table row lineage for every operand. ATR cannot
combine columns from different tables or differently filtered row sets.
Existing materialization of column expressions occurs before the stateful
stage; exact input schemas and unsupported temporal compositions still fail
through the analyzer/compile boundary. No new table engine is introduced.

Extend the existing Rust `RollingOutputSpec` with these **new** variants:

```rust
Rsi {
    primitive_version: u32, input: String, output: String, period: u64,
},
Atr {
    primitive_version: u32, high: String, low: String, close: String,
    output: String, period: u64,
},
MacdComponent {
    primitive_version: u32, input: String, output: String,
    fast_span: u64, slow_span: u64, signal_span: u64,
    min_periods: u64, signal_min_periods: u64, component: MacdComponent,
},

// Export with the other rolling declaration types at the crate root.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MacdComponent { Line, Signal, Histogram }
```

Serialized kinds are `rsi`, `atr`, `macd_component`; primitive_version is
exactly 1. Period is required. MACD spans and minima use the defaults from the
Python surface when omitted in JSON and serialize canonically to their
normalized values. Unknown fields, invalid counts and unsupported versions
fail closed. Do not add these primitives to fused `RollingFloatPrimitiveSpec`
as another route; existing post-stage row arithmetic is sufficient.

Each Python MACD reference is an ordinary immutable component node with the
same value node and five parameters plus its component selector. Its owner key
excludes component/output name and includes input lineage, temporal order,
all five parameters and numeric semantics version. Lowering emits only the
consumed column declarations, but the compiled owner always contains fast,
slow and signal state. The Rust compiler groups matching declarations by that
same state identity, independently of the Python frontend. Duplicate aliases
may read one component without advancing it twice.

All components in one Program with the same owner key are evaluated in one
physical RollingOperator before output branches fork. Extend the existing
shared-state lowering for this specific group; do not produce three separately
owned stages or duplicate a state owner per logical Program output. The node's
single table output can contain any requested subset of columns. No consumer
is required for the other components; they remain state, not extra required
ports. A Program consuming none of the three has no live owner.

The existing table `input`/`output` ports, optional late diagnostic port,
event-time ordering, duplicate identity rules, lateness policy, watermarks and
delivery proofs remain the lifecycle contract. `with_late_output` treats one
MACD owner as one stateful stage and keeps its existing operand/upstream-stage
restrictions. Late rows dropped or routed to `late` do not update indicator
state; they follow the existing late policy, not an indicator retry path.

For admitted rows, Inf and ATR high/low domain checks happen before a new input
Batch is accepted into the table reorder buffer. Sequence-dependent delta/gap
checks happen when the ordered row is actually finalized. That may occur on a
later watermark after its original Batch was buffered. Such an error aborts
the current transition and job; it cannot retroactively unaccept an earlier
buffered Batch. An unordered input Batch cannot be completely numerically
validated before its neighbors arrive. Direct/array-sequence calls validate
and evaluate the whole ordered chunk before emitting any output.

**A7 adds a required preparation transaction for indicator-containing Table
nodes on both ordered and reordered paths.** Do not assume every existing
rolling branch already supplies it: baseline on_watermark/on_end can take
reordered rows out of the buffer before numerical preparation fails. For one
triggering process/watermark/EOF callback, retain the original numerical state,
pending rows/order proof, charge, output sequence, metrics and finality frontier
until the whole preparation succeeds. Its candidate covers every row that
this callback would finalize, including all output chunks. Complete all
arithmetic, output construction and edge-size admission, sequence-count checks,
candidate/peak admission and the final pre-emission cancellation check before
**any** output from that callback escapes. Do not validate and emit a valid
prefix before calculating a later chunk that could fail.

A preparation error emits zero rows and leaves those committed fields and
pending rows unchanged. A detached/taken buffer may be used internally only
with ownership that restores it on every preparation failure, including an
aborted awaiter; its bytes remain charged while detached. Include the ordered
last-identity/order-proof state, last_input_watermark, ended flag and both main
and late output sequences in this invariant. For a zero-output callback, the
same preparation checks precede the frontier/EOF commit. Any off-task
preparation owns an isolated candidate and cannot commit after waiter loss.

After preparation, existing output-emission/delivery rules still apply. A
failure after one of several prepared chunks was accepted fails the job and
does not promise to retract that chunk. After the final successful emission,
commit prepared numerical/buffer/sequence/frontier state without another
fallible check or await. The zero-output case uses the same nonfallible commit.

Correcting and retrying an input chunk is promised only when that chunk was
rejected before acceptance. A bad row already accepted into a Table buffer
cannot be replaced by retrying its watermark/EOF callback: it stays pending
when preparation fails, and the managed job follows its failure/recovery and
source replay contract. Tests may retry a cancellation/admission-only callback
after its cause is removed, but must not pretend a numerical buffered row was
never accepted. The parent A7 specification qualifies “whole batch” with this
Table callback boundary while retaining full direct and single-ordered-chunk
atomicity.

Keep `RollingValuePolicy::StatefulNumericV1` for existing declarations. Its
existing per-primitive rules remain unchanged; the new primitive_version=1
definitions specify RSI/ATR reset and MACD composition separately. Old
StableV1/StableV2 behavior and counters are unchanged in mixed nodes; neither
profile selects an alternative RSI/ATR/MACD algorithm.

#### Versioned table state and intentional generated-schema changes

- New constant `ROLLING_INDICATOR_STATE_LAYOUT_VERSION = 5` admits RSI/ATR and
  every prior primitive. Use it only when an indicator needs it, or when the
  caller explicitly selects it; old graphs retain their current layout and
  canonical bytes. It retains layout-4 columnar dictionary, pending rows,
  old primitive summaries and per-entity transition counts.
- New `ROLLING_MACD_STATE_LAYOUT_VERSION = 6` additionally admits MACD groups.
  RSI/ATR-only default to 5 after MACD ships. MACD-containing graphs use 6.
  Versions 1-4 keep their exact readers/writers and fingerprint interpretation;
  a new indicator in an old layout is rejected. Layout 5 rejects MACD.
- Layouts 5/6 add nullable Binary `_indicator_state` after layout 4's fields,
  and row-kind 5 for a unique `(entity_id, compiled_indicator_group)` record.
  That row carries only entity_id, group and the indicator payload; unrelated
  history/scalar/count fields are null. Legacy row kinds keep their meaning
  and have null `_indicator_state`. Entity rows always include transition
  counts. The exact zero/positive-transition inventory rule below determines
  when indicator records are present; absence must not erase a finalized prefix.
- Reuse the table state schema metadata, pipeline/operator identity, ordering,
  key dictionary, pending-buffer and segment inventory checks. No indicator
  reconstructs its recurrence from retained table history or a warm-up tail.
- Project format and managed manifest remain 3; Rolling configuration version
  remains 1. `schemas/project-v3.schema.json` intentionally gains the three
  output variants, MacdComponent enum/defaults, and the documented new layout
  range where generated. Matching schema generator fixtures and strict project
  validation change in the same PR. Unrelated schema changes are a failure.
- Existing project bytes/fingerprints do not acquire new default fields. New
  options, component selection and layout enter the ordinary project hash;
  grouped owner identity ignores output aliases only for state sharing. The
  full pipeline fingerprint still distinguishes output routing/projections.

Adding RollingOutputSpec variants breaks downstream exhaustive Rust matches.
Release notes must explicitly name the new variants and require callers to add
branches (or a deliberate fallback) when upgrading/recompiling. Do not also add
`#[non_exhaustive]` to old types in this feature. Existing serialized graphs
remain readable; older binaries reject new kinds/layouts. New state cannot be
downgraded by changing a version tag; recover it with a supporting binary or
replay inputs into a fresh compatible job.

### Native ordered array stream

Extend the existing sequence factory; do not add provider identities or extra
asynchronous ingress ports. `input` and `output` remain required array ports
without Arrow schemas, with Batch metadata preserved for every emitted Batch.
The actual payload must be immutable NativeArray with the declared float dtype.

| Kernel            | Input shape and order                      | Output shape and order                                    |
|-------------------|--------------------------------------------|-----------------------------------------------------------|
| rolling_mean/ewma | `[n]`, unchanged                           | float64 `[n]`, unchanged                                  |
| rsi               | `[n]` price                                | float64 `[n]` RSI                                         |
| atr               | packed row-major `[n,3]`: high, low, close | float64 `[n]` ATR                                         |
| macd_components   | `[n]` price                                | packed row-major float64 `[n,3]`: line, signal, histogram |

The ATR input is one synchronized row record per matrix row. It is not called
OHLC because no open value is used. Reject width other than three, rank-one
`[3*n]`, transpose/reinterpretation, mixed dtypes inside a matrix, static input,
second ingress and any implicit zip. `[0,3]` is valid. The existing Matrix
NativeArray constructor/import with explicit shape suffices; the adapter reads
the packed rows directly without allocating three deinterleaved copies.

MACD emits one packed Batch so a row's three components have one acceptance
point and cannot be split across partial port emissions. It always computes
and emits all three columns on this low-level surface. Table users select
columns by name; direct eager users get MacdOutputs. A low-level sink may choose
to use only one component after accepting the Batch, with no effect on state.
`Batch.len()` remains n, while packed output element admission counts 3*n.
Do not advertise `[n,3]` MACD output as acceptable vector input to another
sequence operator without an explicit supported transformation.

RSI/ATR PR adds **sequence IR2**, preserving IR1 unchanged. IR2 accepts all IR1
kernels plus `{"function":"rsi","period":14}` and
`{"function":"atr","period":14}`. MACD PR adds **IR3**, preserving IR1/2
unchanged; IR3 additionally accepts:

```json
{
  "ir_version": 3,
  "input_dtype": "float64",
  "kernel": {
    "function": "macd_components",
    "fast_span": 12,
    "slow_span": 26,
    "signal_span": 9,
    "min_periods": 1,
    "signal_min_periods": 1
  }
}
```

Required outer fields stay ir_version/input_dtype/kernel. Function is required;
RSI/ATR period is required; omitted MACD parameters use the public defaults.
Existing optional execution/state objects and defaults are unchanged. All
grammar branches reject unknown keys and retain the 64 KiB/depth/duplicate-key
and exact-integer guards from A6b. Old kernels retain numeric/layout identity 1
even in IR2/3; new kernels choose their specified layouts below. The declared
IR version is still part of the full project fingerprint.

#### Capability and registration projection

Capability schema stays **5**: its structural fields already express these
operators. RSI/ATR extend sequence `state_layouts` to `[1,2]`; MACD extends
it to `[1,2,3]`. Semantic `state_version=1` remains. No extra provider record
or registration count is added. Registration snapshots/replay, descriptors,
native admission and Python strict validators change together and continue
to insert/rollback the native bundle atomically.

Keep `native_sequence_vector@1` as the IR1 rule. Add
`native_sequence_indicators@2` for the exact IR2 per-kernel shapes and
`native_sequence_indicators@3` for IR3 shapes. Descriptor shape_rules lists
the supported versioned alternatives; consumers must dispatch by ir_version
and kernel, not infer that every kernel accepts every listed shape. The full
options_json_schema has separate version branches and exact kernel unions.
Unknown rule/version combinations fail closed. Do not weaken the old vector
rule to accept arbitrary matrices or mark the sequence row-axis independent.

All new sequence variants truthfully retain: stream-only, stateful,
checkpointed_stateful, microbatch_invariant, deterministic/replay_safe,
per_row_final, no watermark requirement, no static inputs, and
single_ordered_sequence partition contract. Microbatch invariance concerns
concatenated ordered positions, not identical Batch envelopes. Direct eager
and table entry points do not imply array_expr/ArrayExpr indicator support.
The table_array stateless bridge and all NumPy/JAX records remain unchanged.

The **built-in rolling capability** is a separate required update in
`python/calc_flow/capabilities.py`, its Rust/source projection where present,
and `test_capability_catalog_reports_every_durable_state_layout`. Preserve
existing supported layouts 1-4; advertise 5 only with the complete RSI/ATR PR,
then 6 only with the complete MACD PR. On the inspected baseline the catalog
still lists `(1,2)` although the runtime implements 3/4, so integration must
reconcile the actual supported set: `(1,2,3,4,5)` after the first PR and
`(1,2,3,4,5,6)` after the second. This is a rolling@1 inventory correction and
extension, not a provider registration, capability structural-version change,
or reason to alter sequence provider record counts. Keep the capability schema
at the A6b version 5 and preserve the existing built-in semantic state version.

### Exact state, codec and budget contract

#### Numerical state and fixed charges

Each direct/sequence state has these logical fields and deterministic charge:

| Owner | Fields excluding immutable parameters                                     | Bytes |
|-------|---------------------------------------------------------------------------|-------|
| RSI   | positions, has_previous, seed_count, previous_price, mean_gain, mean_loss | 48    |
| ATR   | positions, has_previous, seed_count, previous_close, mean_tr              | 40    |
| MACD  | positions, fast_count/value, slow_count/value, signal_count/value         | 56    |

Charge each flag as eight bytes. These charges exclude allocator/options/object
overhead and are neither ABI sizes nor RSS. Each group is constant space;
table entity/key/reorder-buffer storage remains separately budgeted. Constructor
and reset retain exactly the stated fixed charge. No period-proportional reserve,
three price histories, cached result vectors or alternative EMA seed is needed.

For an owned call use A6's participating-peak rule:

```text
P = committed S + isolated candidate C + outputs O + scratch T
  + distinct retained checkpoint/capture allocations K
P <= execution.max_allocation_bytes
C <= state.max_state_bytes
```

Count actual overlapping allocations, with checked arithmetic and reservation
before allocation. Shared immutable storage is charged once per allocation
owner; copied candidates and separate buffers count independently. These are
the native-array execution/state limits. Table has no ArrayExecutionOptions;
its exact independent mapping to the existing StateBudget is defined next.

For direct MACD, max_elements counts the sum of all three returned lengths, 3*n;
O includes three value buffers, three bitmaps and three rank-one shapes.
Packed stream MACD also counts 3*n, with one bitmap of 3*n bits and rank-two
shape. RSI/ATR count n. Actual allocated capacities, conversion/packing scratch
and any staging buffers must fit too. Table may materialize only selected
MACD columns while maintaining the full 56-byte recurrence. Eager calls admit
their fresh candidate plus outputs/scratch without an independent state limit.
Neither a retained capture nor a worker survives for free after waiter loss.

#### Exact Table limit and compatibility boundary

The sole Table limit used here is the `StateBudget` already owned by
RollingOperator and set by `RollingOperator::set_state_budget`. Its
`max_rows()` and `max_bytes()` are positive JSON-safe u64 values. The inspected
`operator/state_budget.rs` defaults **both to 9,007,199,254,740,991**. The
`config.rs::rolling_node` path constructs the operator with that default;
there is currently no project field, Python ts option or StreamRuntimeConfig
rolling-budget override. Keep those facts: this design adds no Python/config
knob and no implicit 256 MiB Table cap. The very large default is a counter
bound, not a promise of a useful operational memory ceiling. Rust callers can
set a finite budget before compiling either execution mode.

Define an **indicator node** precisely as a RollingOperator containing at
least one Rsi, Atr or MacdComponent output, necessarily using layout 5 or 6.
For this predicate only, change max_bytes to bound the operator's participating
peak. A mixed node containing any old rolling/cumulative outputs plus a new
indicator uses the new rule for the **whole node**, including its old groups,
row histories, buffers and all emitted columns. Nodes containing only old
primitives retain the exact old retained-only semantics, including checkpoint
copy exclusion, even if a caller explicitly selects layout 5/6 for such a node.
Layouts 1-4 and existing graph defaults/fingerprints do not change. Adding enum
variants must not accidentally change old-node state charges through a changed
`size_of::<WindowState>()`; preserve the old accounting result for those nodes.

For an indicator node, distinguish retained state from overlapping work:

```text
R_rows = existing retained-row charge for entities + history + pending rows
R_bytes = existing deterministic retained-state charge + indicator group state
P_table = S + C + O + T + K

At every committed/candidate state: R_rows <= state_budget.max_rows()
At every participating peak:       P_table <= state_budget.max_bytes()
```

Keep existing row-count meanings; do not multiply max_rows by the number of
MACD components or charge output/capture rows against it. Each initialized
indicator group adds its fixed 48/40/56-byte numerical charge once, replacing
any otherwise duplicated generic window-slot charge for that group. Existing
entity/key/history/buffer charges remain. A buffer-only entity with no
finalized transitions has no indicator group allocation or charge. It is still
charged for its pending rows and keys. The hypothetical candidate must meet the
same retained-row check before replacing the old state.

In P_table, S is all currently committed retained storage; C is the complete
additional isolated candidate/transition storage while S remains live; O is
the complete prepared output backing storage and chunk/metadata structures;
T is additional sorting, row materialization, copy/packing, encode/decode and
worker scratch; K is distinct encoded snapshot/restore-segment storage still
owned on behalf of this operator. Use deterministic retained component charges
for S/C, and allocated capacities plus deterministic descriptors for new
output/scratch/segment allocations. Existing conservative structural charges
are not claims about RSS. Include typed numerical state and its materialized
checkpoint/history copy if both exist; do not regard two representations as
one state allocation.

The Table ledger uses observable buffer storage and specified descriptors.
For every Arrow buffer, its payload charge is the maximum of reported capacity
and checked `ptr_offset + len`; aliases with the same backing base take the
maximum charge. Array Arc allocations and identifiable container descriptors
remain charged, including owned String, Box and Vec storage and their defined
capacity charges. Immutable Arrow Buffer/Bytes private owner control blocks are
excluded from retained S/C/O/K: the public backing pointer does not identify
that private owner, and independent wrappers can share data while owning
different control blocks. The convention applies to imported and newly created
buffers alike; it is not a claim to account for all heap allocations.
Construction/workspace reservations may conservatively include new private
control blocks and converge to the retained logical charge after preparation.
Allocation-counter construction peaks must therefore be distinguished from
retained logical-footprint checks. This exclusion does not remove payload,
Array/container descriptors, segment reservations, capture leases or unsettled
work from the ledger. No private-owner provenance registry is introduced.

P_table does not recursively charge the container allocations of owner-key
indexes used solely by Footprint/Charges accounting or the numeric reservation
registry. These indexes may only reference already funded owners and numerical
charges; they must not retain another unfunded data representation. This narrow
boundary does not apply to computation, sorting, deduplication, schema,
row/key/history, output, segment, or encode/decode/workspace containers. Their
data and positive fallback/residual charges remain included. Moving data or
work ownership into an accounting owner or lease does not change its charge
or funded lifetime.

Shared immutable state/output/segment components are charged once at their
underlying ownership boundary. Their largest applicable component charge is
used while several references overlap; an Arc clone does not create another
allocation. A deep copy or newly allocated candidate representation is distinct
and is charged separately, even if its values match. Shared MACD state is charged
once per entity/group, while distinct output value/validity buffers are charged
individually. Implement this as narrow Rolling-owned reservations/leases on
existing buffers and StateSegment owners, not a global allocator or a new public
memory framework. Admission includes the sum of all simultaneously live work
reservations; moving work to another task cannot reset that sum.

Borrowed caller input is excluded until the operator retains it in a buffer,
candidate or output. A shared Arrow output column that is also retained state
does not get counted twice, but its backing storage remains charged while the
operator still owns either reference. Admission uses checked u64/usize sizes
and occurs before copying/reserving; an unexpectedly larger allocation capacity
must be checked before further work or publication. Output transferred wholly
to downstream/application ownership is outside this operator after its last
reference drops; edge budgets continue to charge every delivered Batch.

The same selected StateBudget applies at these lifecycle boundaries:

| Boundary                       | Retained-row admission                                                                                                              | Peak-byte admission and ownership                                                                                                                                                                                                                                                    |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Finite BatchOperator::process  | Check ephemeral rolling entities/history at each prefix; temporary sort/input materialization and output rows are not retained rows | Start a fresh calculation; C is its ephemeral recurrence/history, T is complete sort/materialization scratch, O all returned columns. Include any actually live S/K on the operator. No streaming state is committed; release reservations as results transfer and temporaries drop. |
| Stream input admission         | Check resulting pending rows/entities/history, including mixed old/new groups                                                       | S remains; candidate pending rows/copies and any late output enter C/O/T. Inf/domain/late checks finish before accepting the envelope.                                                                                                                                               |
| Watermark/EOF preparation      | Check the whole prepared next state                                                                                                 | Hold S, all selected pending rows, C and every prepared output chunk under one peak. Numerical preparation errors release C/O/T after work settles and preserve S/pending rows/frontier.                                                                                             |
| Checkpoint preparation/capture | Existing retained state still meets max_rows; encoded rows add no row charge                                                        | S plus any live work/captures, encoding/materialization T and newly allocated K must fit together. Attach a lease to each immutable StateSegment owner; retained snapshot bytes stay charged after the prepare/checkpoint call returns.                                              |
| Normal or terminal restore     | Validate complete decoded retained rows before replacement/completion                                                               | Reserve known encoded lengths as K before reading, then incrementally admit decoded candidate C and scratch T alongside old S and other captures. Never preload/decode an unbounded snapshot and check only afterward.                                                               |
| Reset or budget change         | Reset clears committed rows; a lower budget must admit the current live ledger                                                      | Reset cannot drop K or pending-work credits while their owners exist. set_state_budget rejects a limit below current rows or total reserved peak without changing the old limits.                                                                                                    |

Finite execution uses this same max_bytes rule for typed fast paths and scalar
fallbacks. The current finite implementation does not already enforce this
peak contract; adding it for indicator nodes is part of A7. There is no hidden
ArrayExecutionOptions default on Table. The default makes no smaller memory
guarantee, but explicitly selected finite StateBudgets must be honored.

Checkpoint preparation for indicator nodes is bounded/cancellable and runs
through the existing asynchronous preparation/owned-worker mechanism; the
following checkpoint capture only attaches/clones the prepared immutable
segment and metadata for that exact revision. For direct synchronous trait
callers, an unprepared revision returns a clear checkpoint-preparation error,
following A6b's new-node policy, rather than falling back to an unadmitted
scan. A stale preparation cannot become a checkpoint of newer state. Every
encoding/copy/hash allocation and actual overlap is admitted before use.

Normal managed restore and the terminal dispatch described below reserve
known segment lengths before read, then use bounded decode under this same
operator ledger. A direct synchronous restore validates already-supplied
segment ownership against the limit before decoding and accounts for those
encoded bytes while live; it cannot undo the caller's prior allocation. These
new-node paths use A6b's async restore hook/private admission seam where needed.

The lease ledger belongs to operator resource ownership, outside resettable
RollingStreamState. Pending workers retain input/candidate/scratch reservations
until they settle, even if their awaiter or the operator disappears; existing
supervisor/reaper ownership joins them. A reset cancels old work and changes
its generation, clears committed numerical state and prevents a stale result
from committing, but preserves the pending reservation. Subsequent calls must settle
it or include it in P_table. Captures retain their accounting owner until the
last StateSegment/allocation reference drops, including asynchronous staging
and storage failure; resetting/replacing/dropping live numerical state cannot
return that credit early. A future lower set_state_budget must include these
live reservations. No public reset/prepare/commit or capture-handle API is added.

Document this versioned exception in StateBudget/RollingOperator rustdoc and
the streaming/batch guides: the old statement “checkpoint copies are outside
this limit” stays true for old-only nodes, while indicator nodes use the new
peak meaning. DataFusion memory pools, other operators, downstream queues and
application-retained results remain independent; P_table is not total-process
memory. Explicit limit changes retain the existing fingerprint/configuration
mechanism; do not invent a project budget field or claim the current Rust-only
setter has become serialized.

Array sequence continues A6's prepare -> successful `emit("output", batch)` ->
nonfallible immediate commit ordering. All numerical/size/cancellation checks
finish before emit; no new check or allocation can interrupt successful emit
and commit. Error before acceptance discards the candidate. Existing fanout
partial-delivery failure still fails the job; there is no new exactly-once or
all-consumer rollback promise. Pending worker ownership, cancellation/reaper
settlement and per-operator reservation gates carry over unchanged.

#### Sequence layouts 2 and 3

Retain A6's layout-1 reader/writer byte-for-byte for mean/EWMA. New RSI/ATR use
layout **2**; MACD uses layout **3**. New kernel tags are 3=RSI, 4=ATR,
5=MACD. Every new kernel has numeric_semantics=1, and is persisted in exactly
one `state` segment, including empty/reset numerical state. The runtime keeps
manifest/version wrapper/progress/checksum ownership. Inline metadata is strict
and repeats the operator/kernel/layout/numerical identity and state fingerprint.

New segment encoding is fixed little-endian, without variable collections:

```text
68-byte header:
  magic[8] = "CFAIND\0\0"
  layout:u32                         # 2 RSI/ATR, 3 MACD
  numeric_semantics:u32 = 1
  kernel:u8                          # 3, 4, 5
  input_dtype:u8                     # 1 f32, 2 f64
  reserved:u16 = 0
  state_fingerprint[32]
  processed_positions:u64
  parameter_bytes:u32
  payload_bytes:u32

parameters, all u64:
  RSI/ATR: period                    # 8 bytes
  MACD: fast_span, slow_span, signal_span, min_periods, signal_min_periods
                                     # 40 bytes

RSI payload, 40 bytes:
  has_previous:u8, reserved[7]=0, seed_count:u64,
  previous_price_bits:u64, mean_gain_bits:u64, mean_loss_bits:u64
ATR payload, 32 bytes:
  has_previous:u8, reserved[7]=0, seed_count:u64,
  previous_close_bits:u64, mean_tr_bits:u64
MACD payload, 48 bytes:
  fast_count:u64, fast_value_bits:u64, slow_count:u64, slow_value_bits:u64,
  signal_count:u64, signal_value_bits:u64
```

Exact total lengths are RSI=116, ATR=108 and MACD=156. Reject other lengths,
trailing bytes, reserved values, unsupported combinations and extra segments.
The state fingerprint hashes canonical provider identity, input dtype/shape
rule, kernel and normalized parameters, numeric semantics and state layout.
The full pipeline fingerprint separately verifies budgets, outputs and binding
identity. Budget changes do not bypass full managed recovery validation.

Save every f64 as raw bits, including MACD Inf, signed zero and generated NaN
payloads. Never use nonfinite JSON numbers or the static-input canonical-NaN
digest policy. RSI/ATR state values must be finite, means nonnegative, and
inactive/empty means +0.0; previous_price/close may be negative. Reset state has
has_previous=0, seed_count=0 and all floating bits +0. A present previous with
seed_count=0 also has all means +0. has_previous is only 0 or 1. Validate
seed_count <= period and <= positions.saturating_sub(1); presence requires
positions>0. Once ready, seed_count remains period, not an unbounded total.

MACD requires fast_count=slow_count<=positions. Zero-count values must have
+0 bits; otherwise raw finite/nonfinite bits are preserved. Signal count is
bounded by positions, **not** by fast_count. If fast_count<min_periods it must
be zero; if min_periods<=positions, it cannot exceed
positions-(min_periods-1). This permits valid carried-line updates. Parameter
and count validation happens before replacing any committed state.

Checksums detect accidental corruption, not authenticity or proof that every
arbitrary numerical bit pattern arose from a real history. Validate structural
and numerical invariants without replaying guessed history or normalizing
valid restored bits. Restore/capture admission covers encoded bytes, old state,
decoded candidate, immutable captures and scratch before loading/allocating.
Normal and terminal managed restore both perform these validations before
source open or successful terminal completion. Genuine restart uses the
existing StateBackend/session/manifest path; direct state handles acquire no
public file format.

#### Table indicator payload

The table `_indicator_state` payload uses these exact fields:

```text
numeric_semantics:u32=1, kernel:u8, reserved[3]=0, processed_positions:u64
normalized kernel parameters in the sequence order above
kernel payload in the sequence order above
```

Thus RSI is 64 bytes, ATR 56 bytes and MACD 104 bytes. The enclosing rolling
layout/kernel tag determines the length; no trailing data is accepted. Table
parameters and derived group identity must match the compiled declaration.
The numerical processed_positions equals that entity's transition count for
the group, including missing finalized rows; pending rows do not advance it.
For each entity/group validate the same numerical invariants and the outer
dictionary/group uniqueness/order. Mixed Float32/Float64 ATR types are
represented in the exact input schema and full rolling fingerprint; the
payload stores normalized f64 recurrence only. No MACD field is pruned because
a component lacks a consumer.

The inventory rule is deterministic and based on the explicit entity row's
transition count, not inferred from the presence of a summary or pending row:

- **Count zero:** this is a buffer-only entity. Require at least one pending
  row; there is no finalized history or indicator record. Encode no indicator
  records and reject even zero-valued indicator records at count zero. Restore
  the pending rows without allocating indicator states. Their first finalized
  row creates all declared indicator groups and advances each to position one.
- **Count positive:** require exactly one record for every compiled indicator
  group, even when all finalized inputs were missing, the most recent row
  reset warm-up, there is no retained history, or every MACD component except
  one is unconsumed. Each payload position must equal that positive entity
  count. Missing/duplicate/unknown groups fail. Such an entity remains retained
  with its group records even when there are no pending rows or legacy history;
  do not evict it merely because its numerical warm-up state is empty.

Legacy group/history checks still apply in mixed nodes. A zero-count dictionary
entry without any pending row is unused and rejected. Deleting the last
indicator record of a positive-count entity must fail even when pending rows
keep its dictionary entry alive. No decoder is allowed to turn a missing
positive-count group into fresh state. Empty global state may retain the
existing no-segment representation; these rules apply to every entity that
appears in a layout-5/6 segment, with no arbitrary empty entity inserted.

#### Explicit Table terminal recovery dispatch

A6b's native sequence terminal validation does not cover RollingOperator.
Add a targeted Rolling layout-5/6 validation dispatch in the existing
`runtime/streaming/runner.rs` terminal-manifest branch, before
`recover_terminal_manifest`, successful completion, or any sink opening.
Select the actual compiled Rolling operator/layout, not a provider name or a
user metadata claim. Validate each selected operator's segment inventory,
exact schema/metadata and kernel identity, entity dictionary, transition
counts, complete indicator-group inventory, raw payloads and selected budget
using the same checked decoder/admission as normal Rolling restore. Include
the explicit zero-count rule above and mixed legacy groups. Release decoded
candidates/reservations after validation if terminal completion needs no live
state; that does not permit skipping the decode or peak checks.

Both normal managed restore and this terminal path must admit encoded bytes
before load and decoded rows/storage before allocation. Indicator nodes use
the new P_table limit; old-only nodes explicitly using layouts 5/6 use their
unchanged selected retained-budget semantics, while still validating the new
format. Unrelated terminal behavior for layouts 1-4 remains unchanged. A
valid outer checksum does not make an invalid indicator payload trustworthy.
This is a narrow additional terminal dispatch, not a new universal recovery
framework or changes to unrelated operators.

### Error cases and cross-layer projection

Direct native invalid input/limits/numerics use the existing
`CalcFlowError::InvalidArgument { field, message }`; PyO3 maps it to ConfigError.
This preserves the established array exception taxonomy, even for a numerical
domain failure. Python declaration argument checks retain TypeError/ValueError
and analyzer compile errors in the existing style. Operator/runner wrapping
retains the field/position text and adds existing node/job context; it does not
turn a numerical error into a missing result. Corrupt checkpoint state remains
a checkpoint error. Do not add a new broad exception/reason-code hierarchy here.

Positions below are zero-based within the submitted direct/sequence chunk.
Table errors identify the input column and the coordinate actually available
at the validation boundary. Admission rejects a bad row before it enters the
accepted sequence, after late-row classification. Report `input row N; ordered
position not assigned` there, using the original submitted Batch row. Do not
label that envelope index as an ordered index or defer rejection merely to
obtain one: later stream arrivals can still change event-time order.

During finalized numerical processing, report the entity-local cumulative
`ordered position M`, adding the original Batch row when available. Rejected
rows have no assigned ordered position. These coordinate labels clarify the
same existing diagnostics; they add no public exception class, reason-code
hierarchy or serialized state field. Do not expose entity keys or sample values.

| Input violation                  | Field/path                                                 | Required message content                                                                            |
|----------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Zero RSI/ATR period              | `options.period`                                           | `must be a positive u64`                                                                            |
| Reversed/equal MACD spans        | `options.fast_span`                                        | `must be less than slow_span`                                                                       |
| Invalid warm-up                  | `options.signal_min_periods` or `options.min_periods`      | `must be a positive u64`                                                                            |
| ATR scalar/matrix direct input   | `inputs.high.shape` (likewise low/close)                   | `expected a rank-one floating vector`                                                               |
| Unequal ATR length               | `inputs.low.shape` or `inputs.close.shape`                 | `expected length <n> to match inputs.high, received <m>`                                            |
| Invalid floating type            | `input.dtype` or `inputs.<operand>.dtype`                  | `expected float32 or float64, received <dtype>`                                                     |
| Valid Inf                        | `input[<i>]` or `inputs.<operand>[<i>]`                    | `RSI/ATR requires finite non-missing samples`                                                       |
| ATR high below low               | `inputs.high[<i>]`                                         | `must be greater than or equal to inputs.low[<i>]`                                                  |
| RSI delta overflow               | `input[<i>]`                                               | `RSI price difference from previous price is not representable as finite float64`                   |
| ATR range/gap overflow           | `inputs.high[<i>]` or `inputs.low[<i>]`                    | name `high - low`, `high - previous_close`, or `low - previous_close` and finite-float64 constraint |
| Seed/Wilder arithmetic overflow  | offending row field                                        | name indicator, seed/recurrence and finite-float64 constraint                                       |
| Position overflow                | `input.rows`                                               | `processed position count would exceed u64::MAX`                                                    |
| MACD output element limit        | `outputs.elements`                                         | `three MACD components require <3*n> elements; limit is <limit>`                                    |
| State/peak limit                 | `state.max_state_bytes` / `execution.max_allocation_bytes` | required participating charge and configured limit                                                  |
| Packed ATR wrong width           | `input.shape`                                              | `ATR sequence input requires [n, 3] in high, low, close order`                                      |
| New kernel under old IR/layout   | precise kernel/version field                               | supported kernel/version pairing                                                                    |
| Corrupt or incompatible snapshot | existing checkpoint context                                | indicator/layout plus failed field, length, count, identity or limit                                |

Validate configuration and descriptors before numerical iteration. ATR validates
operand descriptors in high/low/close order, lengths next, then checked sizes,
admission and row values. Scan rows in input order and operand validity/Inf in
the defined order; report the first violation encountered by that deterministic
procedure. A later bad value cannot publish an already-computed prefix.

### Example

This proposed eager example needs no NumPy or provider registration. It is the
seed for a 20-50-line shipped example after the complete implementation lands;
it has not been run against today's package.

```python
from __future__ import annotations

import pyarrow as pa
import calc_flow.array as cfa


def main() -> None:
    close = cfa.from_arrow(pa.array([10.0, 11.0, 12.0, 11.0, 11.0]))
    high = cfa.from_arrow(pa.array([11.0, 12.0, 13.0, 12.0, 12.0]))
    low = cfa.from_arrow(pa.array([9.0, 10.0, 11.0, 10.0, 10.0]))
    rsi = cfa.rsi(close, period=2)
    atr = cfa.atr(high, low, close, period=2)
    macd = cfa.macd_components(close, fast_span=2, slow_span=4, signal_span=3)
    result = pa.table({
        "rsi": rsi.to_arrow(), "atr": atr.to_arrow(),
        "line": macd.line.to_arrow(), "signal": macd.signal.to_arrow(),
        "histogram": macd.histogram.to_arrow(),
    })
    assert result["rsi"].to_pylist() == [None, None, 100.0, 50.0, 50.0]
    assert result["atr"].to_pylist() == [None, None, 2.0, 2.0, 2.0]
    print(result)


if __name__ == "__main__":
    main()
```

Table use after declaring `quotes` with its schema and entity/time/sequence:

```python
parts = cf.ts.macd_components(quotes["close"], signal_span=9)
signals = quotes.with_columns(
    rsi=cf.ts.rsi(quotes["close"], period=14),
    atr=cf.ts.atr(quotes["high"], quotes["low"], quotes["close"], period=14),
    histogram=parts.histogram,
)
```

Selecting only histogram in this example still retains all three MACD states.
Finite execution uses `collect`; continuous execution uses the existing owned
`async with ...stream(...)` result. Durable restart examples must use explicit
replayable SourceBinding/SinkBinding and a new StreamingRunner/managed state
root, not claim that an ordinary async iterable is restartable.

### Independent numerical oracles and acceptance

Use two complementary references without importing implementation transitions:

1. Small hand-derived vectors pin warm-up positions, zero/flat rules, reset
   boundaries and exact outputs. The independent scalar Python float64
   reference expresses the specified individual operation order and serves as
   the **primary oracle for every admitted IEEE case**, including normal,
   subnormal, underflow, signed zero, NaN/Inf, finite extremes, required
   overflow errors and synthetic large counters. It imports no production
   transition. Compare finite results/signed zero bit-for-bit across Rust
   eager, owned state, Python eager, table and sequence on the same platform;
   compare validity and nonfinite classification as specified for MACD. The
   named `RSI(period=2, [0, 2^-1074, 0])` regression requires exactly
   `[null, null, 100.0]`, never 50. Codec round-trip separately preserves raw
   NaN payload bits, rather than requiring cross-platform NaN payload equality.
2. A supplementary **well-scaled fixture set** uses standard-library Fraction
   arithmetic. Its explicit domain predicate is: period is 1..32; each finite
   segment has at most 64 difference/TR updates; normalized prices/high/low/
   close are multiples of 1/16 in [-4096,4096]; high>=low; and every positive
   exact seed/recurrence mean for G, L or TR lies in [2^-20,2^13] at every
   update (zero is also allowed). Validate that predicate using the exact
   reference before applying this accuracy assertion; missing-value segment
   boundaries are checked separately. These bounded dyadic inputs have exact
   representable finite differences/TRs and keep all positive mean/ratio
   quantities far from underflow. The named subnormal example and extreme or
   long-decay fixtures do not satisfy this predicate and use oracle 1 only.

   For this concrete fixed fixture set, form the exact seed as a sum divided
   by period, then exact `(previous*(period-1)+sample)/period` and exact
   `100*G/(G+L)` with explicit zero cases. This is a different formulation from
   the production incremental mean. Require ATR relative error <=256*2^-52
   and RSI absolute error <=100*256*2^-52; an exact-zero ATR must be +0.0.
   These are accuracy acceptance bounds for this set, not a universal finite-
   input error promise or permission to approximate the primary IEEE oracle.
   The set and predicate must be fixed in the fixture provenance, not filtered
   after observing test failures. Denominator/count rounding above 2^53 is
   outside this set and uses the primary operation-order oracle.

No tolerance applies to validity, first output, segment resets, count behavior,
the named underflow result or required errors. Do not change the arithmetic or
enlarge any tolerance to mask a seed/operation-order disagreement. In particular,
the exact rational ratio need not be close to the chosen IEEE result near
underflow; the public numerical documentation must say so explicitly.

MACD's primary oracle is the independent scalar composition of three existing
EWMA recurrences with intermediate line validity, plus parity against actual
old `ts.macd` and `ts.ewma(old_line, ...)`. Reusing the production EWMA result
alone is insufficient. Keep the existing nonfinite behavior rather than using
the rational RSI/ATR oracle to redefine it.

Required focused behaviors:

- Parameter/dtype/rank/length checks; mixed f32/f64 ATR; null payload slots with
  sentinel Inf/NaN bits that must not be read; ownership, offsets and Arrow
  chunk boundaries; no NumPy/JAX import on the Arrow path.
- Flat/up/down sequences, gaps, period one, period u64::MAX, reset before/at/
  after warm-up, zero rows, finite extremes, +/-Inf, NaN, signed zeros,
  underflow and the specified +/-1e308 errors. Inf in one ATR operand must
  fail even when another is missing; high/low domain checks with missing close
  follow this explicit policy.
- Prefix then failed mixed-validity/overflow chunk then corrected retry equals
  uninterrupted output. Check counters and state bits as well as final values.
  Constructor/state/output/peak limits fail without partial output/commit.
- For Table ordered and reordered buffers, accept finite prices whose later
  ordered delta/gap overflows, then trigger watermark and EOF separately.
  Ensure an earlier valid output chunk could have been produced. Each full
  callback's preparation must emit zero chunks and preserve numerical bits,
  all pending rows/order proof, retained/peak reservations after work settles,
  output sequences, last watermark and ended/finality state. Do not replace an
  accepted bad row by replaying a corrected chunk into that same job. Include
  cancellation/budget preparation failures whose cause can actually be removed
  for a legitimate callback retry.
- MACD minima greater than spans, fast>=slow rejection, span-one signal, the
  existing missing-price carried-line example, signal_count>price_count,
  generated NaN and Inf. Single-component consumers still update all states.
- Same ordered rows under different chunk boundaries, interleaved entities,
  event-time reorder and late policy; group dedup across component aliases and
  separate Program outputs; compare both values and one-owner state charge.
- Managed capture then **new job and new process** with replayable sources:
  warm-up, ready state, reset segment, mixed entities, MACD subset and retained
  nonfinite bits. Prefix-plus-restored-suffix must match uninterrupted output.
  A restore onto the same live object is not this acceptance test.
- Checkpoint corrupt/truncated/extra bytes, wrong kernel/layout/parameters/
  dtype/fingerprint, nonzero reserved bytes, wrong flag/count, missing/duplicate
  table group, oversized encoded state and terminal recovery. Fail before
  source open/terminal success; keep old state if direct restore fails.
- Table layout-5/6 inventory: count-zero buffer-only entity has no indicator
  records and restores without synthesizing them; positive-count all-missing
  or reset state retains every group; deleting the last group while pending
  rows preserve the dictionary entry must fail. Check mixed legacy groups too.
- Targeted Table terminal dispatch: a checksum-valid segment with an invalid
  internal payload, missing positive-count group, oversized encoded/decoded
  state, and a valid completed checkpoint. The invalid cases fail before sink
  open/completion; the valid case completes without rerunning outputs. Generic
  checksum-corruption tests alone do not prove this path.
- Budget overlap with old state, candidate, output, scratch and live capture;
  shared captures charged once; cancellation during worker, output backpressure,
  capture and restore; waiter loss/reaper settlement; no post-emit cancellation
  check before commit. MACD packed output has one acceptance point.
- Explicit finite and stream Table StateBudgets on pure indicator and mixed
  nodes: retained rows fit but S+C+O+T+K exceeds max_bytes; output and capture
  have independent test cases. Hold a capture across reset and attempt a new
  call/lower budget; hold an aborted pending worker until settlement. Check
  exact release on the final owner drop, plus an old-only graph showing its
  prior retained-only budget and checkpoint-copy exclusion remain unchanged.
  Built-in rolling layout inventory and sequence provider inventories must
  reflect only the corresponding fully implemented PR's layouts.

### Delivery split and review status

Two implementation PRs, each complete across its promised surfaces:

1. **RSI/ATR**: native options/eager/owned state and focused numerical tests;
   Python adapters/stubs/exports; table primitives and layout 5; packed ATR and
   vector RSI sequence IR2/layout2/capability rules; real managed/terminal
   recovery; cancellation/budgets; project schema/fixtures; runnable examples,
   docs and release notes. Require A6b foundation first. It must not advertise
   a stream kernel whose codec or restart path is absent.
2. **MACD components**: one shared recurrence group and immutable output
   containers; direct/Python/table surfaces; shared/subset Program lowering;
   packed sequence IR3/layout3 and table layout6; capability/schema/codec
   updates; missing-line and legacy parity tests; subset/new-job recovery;
   budgets/cancellation; examples/docs and release notes. Build on the first PR
   without rewriting its already-released IR2/layout2 meaning.

Use focused test-first red/green and affected module formatting/lint/type/schema
checks; record the expected failure before implementation. Full coverage and
cross-platform regression gates remain CI responsibilities. No benchmark
matrix or speedup claim belongs to this API task. Any later performance task
must select a small bounded case set under the repository's existing protocol.

The signatures and choices in this section are approved: packed ATR, packed
low-level MACD, mandatory period, five MACD parameters, stable seeds, float-only
RSI/ATR table input, the preserved old MACD constraint, shared owner, new layouts
and Rust enum migration. The specialist critic passed the revised numerical,
Table transaction/codec, budget and capability contracts. After completed A6b,
hand off the two stated units to cf-implementer under the normal test-first
policy and final implementation review. This design archive supplies no product
implementation, test/CI result, merge authority or performance claim.


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


## A7 indicator source bindings

The common-row-set requirement also preserves the meaning of every consumed
column at its actual materialization context. For expressions containing a new
RSI/ATR primitive, compare referenced source and target column definitions with
canonical digest and full bytes. Recursively validate each consumed derived
definition on both sides at its own evaluation context, including definitions
before a later filter. Equal alias names, row origins or immediate definition
tags alone are insufficient.

Row-preserving project/with_columns/attach wrappers remain admissible when their
referenced definitions agree. Missing aliases, conflicting definitions and
references to overwritten ancestor columns fail with schema_mismatch; use
references from the actual target or explicitly materialize the desired input
there. This delivery does not add general foreign-derived-definition inlining.
The same proof covers references in outer arithmetic around the indicator.
Analysis and lowering independently enforce it before any name-based inlining.
Legacy-only EMA/MACD and row-local declarations retain their existing admission.
