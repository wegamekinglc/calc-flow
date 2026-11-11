# Rust runtime and extension reference

[Documentation](README.md) / 3.4 Rust runtime reference

The `calc-flow` crate implements Calc Flow's internal runtime. This reference
is for runtime contributors and extension authors working with native batches,
operators, graph plans, runners, stores, and checkpoints. The application API is
[Python expressions](python-api.md); Rust exports, traits, and package names remain
available for native integration. Native execution needs no Python process.

On this page:

- [Typed DataFrame and SQL relations](relational-frontends.md#rust)
- [Build and document](#build-and-document)
- [Native array storage](#native-array-storage)
- [Native array reductions](#native-array-reductions)
- [Native exponential averages](#native-exponential-averages)
- [Native rolling mean](#native-rolling-mean)
- [Owned native sequence state](#owned-native-sequence-state)
- [Checkpointed native array sequences](#checkpointed-native-array-sequences)
- [Native elementwise arrays](#native-elementwise-arrays)
- [Packed matrices and axes](#packed-matrices-and-axes)
- [Native RSI and ATR](#native-rsi-and-atr)
- [Native MACD components](#native-macd-components)
- [Native array batch provider](#native-array-batch-provider)
- [Native table/array bridge](#native-tablearray-bridge)
- [Expression pipeline](#expression-pipeline)
- [SQL operators](#sql-operators)
- [Snapshot SQL](#snapshot-sql)
- [Rolling windows](#rolling-windows)
- [Cross-section statistics](#cross-section-statistics)
- [Late-row policy contract](#late-row-policy-contract)
- [Bounded backward ASOF Join](#bounded-backward-asof-join)
- [Stream compilation and continuous runtime](#stream-compilation-and-continuous-runtime)
- [Batches](#batches)
- [UDFs and external providers](#udfs-and-external-providers)
- [Continuous execution and recovery](#continuous-execution-and-recovery)
- [Projects and stores](#projects-and-stores)
- [Errors and cancellation](#errors-and-cancellation)

## Build and document

The [Rust package quickstart](../crates/calc-flow/README.md) covers Cargo
dependencies, the MSRV, and a native calculation. The first crates.io release
is prepared through the checks in the [release guide](rust-release.md).

Work from the repository root for the current native runtime and its examples.
Choose focused local checks under [Verification](../AGENTS.md#verification);
the full crate commands are:

```bash
cargo test -p calc-flow --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
```

## Native array storage

`calc_flow::array` exposes immutable native float32/float64 and packed Boolean storage for Rust
extensions. `FloatView` borrows contiguous values and an optional Arrow
`NullBuffer` for synchronous work. `NativeArray` owns Arrow buffers and supports
scalar shape `[]`, vector shape `[length]`, or packed row-major matrix shape
`[rows, columns]`; a scalar has exactly one value. `len()` counts logical rows;
`element_count()` and `ArrayView::len()` count flat positions. `[3,0]` has three
rows and no positions. Arrow export is flat and retains shape separately.
`float_view()` rejects matrices; explicitly reshape a matrix to a vector before
sequence/rolling work.

```rust
use std::sync::Arc;
use calc_flow::{Batch, BatchMetadata, Result, array::NativeArray};

fn native_input() -> Result<Batch> {
    let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2], None)?;
    Batch::external(Arc::new(array), BatchMetadata::default())
}
```

`from_f32` and `from_f64` take ownership of their vectors; `from_boolean`
takes ownership of a `BooleanBuffer`. `from_arrow` shares existing concrete
Float32/Float64/Boolean Arrow storage and preserves slice and validity
offsets. `as_arrow` exports a read-only Arrow reference, while `float_view`
borrows the typed values. Clones share Arrow buffers, and debug output contains
descriptors without element values. These objects need no Python process.
For non-matrix Boolean storage, `float_view` returns an `input.dtype` error;
all matrix `float_view` calls return `input.shape`. `view` borrows either dtype
and preserves the stored scalar, vector or matrix shape.

`estimated_bytes` charges visible typed slots, the bitmap byte span and shape
entries. `retained_buffer_bytes` reports exposed backing-buffer capacities plus
shape entries, which can be larger for slices. Neither value measures process
RSS. The native payload uses the existing `BatchKind::Array` and reports its
logical row count through the batch. Static parameter snapshots retain their
separate freezing and digest contract.

## Native array reductions

`array::sum`, `mean`, `min`, and `max` synchronously reduce a `FloatView`
without modifying its values or validity. Default options return an owned scalar
`NativeArray` with shape `[]`; sum and mean produce float64, while min and max
preserve the input dtype.

```rust
use calc_flow::{Result, array::{self, ArrayExecutionOptions, FloatView, ReductionOptions}};

fn total() -> Result<array::NativeArray> {
    array::sum(
        FloatView::from_f64(&[1e308, 1.0, -1e308]),
        ReductionOptions::default(),
        &ArrayExecutionOptions::default(),
    )
}
```

Sum and mean accumulate finite values exactly and apply one final float64
round-to-nearest ties-to-even conversion. Mean divides the exact total before
conversion, preserving a finite average such as `mean([1e308, 1e308])`.
This numerical contract is separate from existing table rolling contracts.

By default any null input yields a null scalar. Use
`ReductionOptions::default().with_skip_nulls(true)` to skip null positions;
empty input and input without accepted values still yield null. Null result
storage contains positive zero and an explicit invalid bitmap. Valid NaNs
propagate as canonical NaNs. Sum/mean return NaN for mixed-sign infinities and
preserve a single infinity sign. Min/max preserve a lone zero sign; when both
zero signs occur, min returns negative zero and max returns positive zero.

`ArrayExecutionOptions` defaults to 10,000,000 output elements and 256 MiB of
combined output-buffer and kernel-scratch accounting. The limits can be
changed with `with_max_elements` and `with_max_allocation_bytes`; zero is
allowed to forbid a nonempty result or allocated output. A scalar reduction
requires one output element regardless of input length. Borrowed inputs,
allocator/object overhead, and process RSS are outside this execution charge.
Limits are checked before output allocation or scanning input values. Exact
sum/mean also reserve their bounded accumulator/finalization scratch.

## Native exponential averages

`array::ewma` computes a Float64 result from a borrowed float32 or float64
view. `EwmaOptions::new(span)` requires a positive span and defaults to
`min_periods=1`; `with_min_periods` accepts any positive warm-up count, including
one larger than the span. The smoothing factor is `2 / (span + 1)`.

The first valid sample seeds the state. Null and NaN inputs leave it unchanged;
once warm-up completes, those rows carry the last value. Infinity remains a
valid sample, so subsequent IEEE arithmetic may produce a valid NaN result.
Finite-input overflow follows the same recurrence as the table operator.
Output rows align with input positions, including leading nulls before warm-up.
This is the existing table EWMA contract, shared through one numeric transition.

```rust
use calc_flow::{Result, array::{ArrayExecutionOptions, EwmaOptions, FloatView, ewma}};

fn smooth(values: &[f64]) -> Result<calc_flow::array::NativeArray> {
    ewma(
        FloatView::from_f64(values),
        EwmaOptions::new(20)?.with_min_periods(3)?,
        &ArrayExecutionOptions::default(),
    )
}
```

Execution limits include the Float64 output, compact validity bitmap, shape,
and fixed recurrence state. Inputs stay read-only. This finite call starts a
fresh recurrence; managed streaming state continues to belong to its operator.

## Native rolling mean

`array::rolling_mean` computes one ordered sequence directly, without a graph,
DataFusion session or Python process. `RollingMeanOptions::new(window)` selects
one minimum valid sample and the existing `RollingNumericalProfile::StableV1`.
Use `with_min_periods` for a count between one and the window, or explicitly
select `StableV2Preview` with `with_numerical_profile`.

```rust
use calc_flow::{Result, array::{ArrayExecutionOptions, FloatView, RollingMeanOptions, rolling_mean}};

fn rolling_prices(values: &[f64]) -> Result<calc_flow::array::NativeArray> {
    rolling_mean(
        FloatView::from_f64(values),
        RollingMeanOptions::new(20)?.with_min_periods(5)?,
        &ArrayExecutionOptions::default(),
    )
}
```

The Float64 output has one position per input, including for Float32 input.
Nulls and NaNs consume window positions but do not count as valid samples;
infinities remain valid. A missing current input still reads the surviving
window, and insufficient valid samples produce null. The default matches the
native `StatefulNumericV1` table path, including its released extreme-value
limits. It does not adopt ordinary array mean's exact summation or the SQL AVG
rewrite's sticky NaN-value policy. Array and table execution share add/remove,
mean/M2 repair, preview rebase and readout arithmetic; checkpoint fields and
the table's shared-history owner remain unchanged.

Execution limits cover output buffers, validity, shape, and bounded numeric
scratch/history. History has fixed storage for at most
`min(input_length, window + 1)` positions, evaluated without integer overflow;
`u64::MAX` windows remain usable with small inputs. Inputs are read-only, and
oversized results or scratch are rejected before allocating their buffers.

## Owned native sequence state

`RollingMeanState` and `EwmaState` preserve an ordered sequence across synchronous
`process` calls. Construct them with the existing kernel options and
`ArrayStateOptions`, then pass rank-one `ArrayView` chunks and per-call
`ArrayExecutionOptions`. Float32 and Float64 chunks may alternate; both use the
same Float64 recurrence. Scalars, matrices and Boolean vectors are rejected,
including empty inputs with those descriptors.

```rust
use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayStateOptions,
    ArrayView, FloatView, RollingMeanOptions, RollingMeanState}};

fn across_chunks() -> Result<calc_flow::array::NativeArray> {
    let mut state = RollingMeanState::new(
        RollingMeanOptions::new(2)?, ArrayStateOptions::default(),
    )?;
    let execution = ArrayExecutionOptions::default();
    state.process(ArrayView::from_float(FloatView::from_f64(&[1.0, 3.0])), &execution)?;
    state.process(ArrayView::from_float(FloatView::from_f32(&[5.0])), &execution)
}
```

Each result owns a Float64 vector with one output position per input position.
The existing eager mean and EWMA numerical contracts apply, including missing
positions, warm-up, infinities, operation order and mean/M2 repair. StableV2's
rebase cadence counts all sequence positions across chunks. An empty vector
does not advance or repair state. Input buffers remain read-only; output buffers
outlive the input borrow, reset and state drop.

The default retained-state limit is 256 MiB. `state_bytes()` charges 64 bytes
plus 16 per allocated mean history slot, including spare capacity, or 24 bytes
for EWMA. Construction allocates no window-sized history, even for `u64::MAX`.
`reset()` releases mean history, resets counters and retains options and limits.
These logical charges exclude allocator and object overhead and do not measure
RSS. Debug reports descriptors and charges without sample values.

For these state handles, `max_allocation_bytes` limits the total participating
old state, isolated candidate, owned output/mask/shape and numerical scratch.
Candidate capacity is also subject to `max_state_bytes`; `max_elements` counts
output positions. Empty calls still check output descriptors and limits.
Validation, allocation, count or numerical errors leave history, capacity and
recurrence bits intact so a rejected chunk can be retried. Retained and peak
violations use `state.max_state_bytes` and `state.peak_bytes`; output errors
retain their `outputs.output.*` fields. Existing eager functions keep their
output-and-scratch budget semantics and independent ephemeral states.

These types have no public cloning, serialization or checkpoint API. See the
standalone [owned-state example](../crates/calc-flow/examples/native_array_state.rs).

## Checkpointed native array sequences

`NativeArraySequenceFactory` creates stream-only `native:array_sequence@1`
operators. Its closed data-only options lock `input_dtype` to `float32` or
`float64`. IR1 selects `rolling_mean` or `ewma` on rank-one inputs. IR2 also
accepts `{"function":"rsi","period":...}` on rank-one prices and
`{"function":"atr","period":...}` on packed row-major `[n,3]` inputs with
high, low and close in that order. ATR reads those rows directly without three
intermediate vectors; `[0,3]` is valid. Every kernel returns an aligned float64
vector. IR2 retains the old kernels' layout and numeric identity.

Declare exactly one required Array port `input` and one required Array port
`output`, without Arrow schemas. Scalars, Boolean arrays, foreign payloads,
changed dtypes and per-kernel unsupported shapes are rejected even when empty.
ATR has no second ingress, implicit zip, flattening or matrix reinterpretation.

Each operator processes one sequence in accepted arrival order. Batch metadata
is preserved but does not reorder samples. It does not infer entities, event
time or multiple-input alignment. Its ingress must be dynamic; every static
input declaration is rejected regardless of the declared backend. A dynamic
source may emit equal or repeated immutable batches normally.

Accepted output commits an isolated numerical candidate. A rejected emission,
cancelled computation, count overflow or budget error leaves the committed
recurrence unchanged. Workers own pending candidates through cancellation and
abandoned waiters; a later operation must settle that work before reuse. Reset
cannot release credit still held by a running candidate.

The owned-state charges for EWMA, rolling mean and RSI/ATR also apply here.
Output-element limits count ATR rows, rather than its three input operands.
The allocation limit includes
old state, candidate, output, numerical scratch and retained encoded captures.
Distinct immutable capture allocations are charged once until their final
owner drops. EWMA/rolling mean retain byte-identical layout 1. RSI/ATR use
layout 2 with fixed 116/108-byte state segments, raw floating bits, complete
previous/seed/mean/counter state and an identity fingerprint including the shape
rule and normalized parameters. Even empty or missing-reset sequences retain
one state segment. Restoration validates reserved bytes, exact lengths,
parameters, dtype, flags, finite/nonnegative means, inactive positive-zero bits
and count invariants without reconstructing history. Capability schema 5 keeps
the same identity and adds layouts `[1,2]`, with separate versioned shape rules
`native_sequence_vector@1` and `native_sequence_indicators@2`.

Use `ManagedCheckpointRuntime` with stable graph/binding identities and an
exactly replayable source for durable restart. The current public managed
runtime uses local storage. Native recovery requires that certified bounded
reader; unsupported internal/custom reader routes fail closed for native
sequence nodes. Byte lengths and allocation limits are checked before reading
state into owned buffers. Ordinary and terminal-complete recovery perform the
same controlled SHA, codec and identity validation before any source or sink
opens. Invalid latest state fails closed; there is no numerical-state fallback.
Legacy external providers keep their existing recovery contract.

The executable [native sequence example](../crates/calc-flow/examples/native_sequence_stream.rs)
processes chunks through a managed EWMA operator, then starts a fresh job from
its terminal checkpoint without reopening the source. This Rust factory does
not add a Python expression-stream declaration API.

## Native elementwise arrays

`ArrayView` is a read-only, copyable operand descriptor. `from_float(FloatView)`
and `from_boolean(&BooleanBuffer, Option<&NullBuffer>)` borrow vectors;
`scalar_f32(Option<f32>)`, `scalar_f64(Option<f64>)`, and
`scalar_boolean(Option<bool>)` create nullable inline scalars without allocating.
`NativeArray::view()` borrows existing storage without copying. Boolean values
and validity can have independent bit offsets; both visible spans and retained
buffer capacities participate in native payload accounting.

Every operation takes borrowed `ArrayView` operands and
`&ArrayExecutionOptions`, returning `Result<NativeArray>` with fresh owned
storage. All-scalar inputs produce shape `[]`. Otherwise every operand must
have exactly the same shape, and scalars broadcast to that shape, including zero
extents. Vectors and matrices with the same flat count remain distinct.
A vector of length one remains a vector and cannot broadcast to another
vector length. These synchronous APIs support rank zero, one and two.

| Functions                                                 | Accepted operands and results                                      |
|-----------------------------------------------------------|--------------------------------------------------------------------|
| `neg`, `abs`, `sqrt`, `log`, `exp`                        | Floating input; same floating dtype                                |
| `add`, `subtract`, `multiply`, `divide`, `pow`            | Floating operands; float32 together, otherwise float64             |
| `equal`, `not_equal`                                      | Floating pairs or Boolean pairs; packed Boolean output             |
| `less`, `less_equal`, `greater`, `greater_equal`          | Floating operands; packed Boolean output                           |
| `logical_not`, `logical_and`, `logical_or`, `logical_xor` | Boolean operands; packed Boolean output                            |
| `where_(condition, if_true, if_false, execution)`         | Boolean condition; floating branches promoted, or Boolean branches |

`pow` requires a non-null finite scalar exponent whose absolute value is at
most 64; a one-element exponent vector is rejected. Boolean values never
implicitly convert to numbers. `where_` checks all three shapes and both branch
dtypes, including an unselected branch. It reads only the selected branch's
valid value; a null condition produces null.

Arithmetic and comparisons propagate nulls without reading null payload slots.
NaNs remain valid values: IEEE equality with NaN is false, inequality is true,
and ordered comparisons are false. Division by zero and floating domain errors
produce valid IEEE infinities or NaNs. Signed zeros follow the individual
floating operation; transcendental functions have no cross-platform bitwise
guarantee. Boolean logic uses Kleene semantics: false AND null is false, true OR
null is true, XOR propagates either null, and NOT null is null. Null output slots
contain zero or false.

Output elements and the peak values/validity/shape allocation are checked before
output allocation. Packed Boolean outputs use exactly `ceil(length / 8)` value
bytes; a candidate validity bitmap is charged when any operand contains nulls,
even if selection or Kleene logic ultimately produces an all-valid result and
discards that bitmap. Limits exclude borrowed inputs, fixed scalar locals,
allocator/object overhead, and process RSS. Errors are `InvalidArgument` with
operand dtype/shape paths, `right.value` for an invalid scalar exponent, or
`outputs.output.elements`/`outputs.output.bytes` for exceeded limits. Messages
do not include element payloads.

These APIs do not alter Python expression lowering, Table execution,
NumPy/JAX providers, or graph registration.

## Packed matrices and axes

`matmul(left.view(), right.view(), &execution)` requires floating rank-2 shapes
`[m,k] @ [k,n]` with `n > 0` and no actual null values. All-valid validity buffers
remain legal. Zero rows and zero inner width are supported; zero inner width
produces positive zeros. Mixed float32/float64 promotes to float64, while two
float32 inputs retain float32 accumulation. Each dot visits increasing inner
indices with a separate multiply and addition, beginning at positive zero.
There is no BLAS/FMA/reassociation or NumPy parity claim.

`reshape(&input, shape, &execution)` owns a new rank-0/1/2 descriptor while sharing
read-only Arrow storage. It requires an unchanged checked element count and no
inferred dimensions. Its budget charges descriptor bytes and final elements.
`ArrayView::reshape(shape)` changes only the borrowed stack descriptor.
`transpose(&input, &execution)` swaps two matrix axes; scalars/vectors and
empty/size-one axes alias storage. Other matrices copy packed values and validity,
including sliced Boolean buffers and raw floating bits.

`reduce(view, Reduction::{Sum,Mean,Min,Max}, options, &execution)` accepts
`ReductionOptions::with_axis(None)` or `with_axis(Some(axis))`, plus
`with_keepdims(bool)`.
Axis must be less than rank. Each lane uses the existing exact sum/mean and
min/max null, NaN and signed-zero rules. Empty/all-null accepted lanes are null:
`[0,3]` reduced on axis 0 gives three nulls, and `[2,0]` on axis 1 gives two.
Keepdims retains reduced dimensions at extent one; scalar full reductions remain
scalar. The original FloatView functions remain available and interpret their
input as a vector. Their default results and budget contracts are unchanged.

Execution limits count flat output elements, new descriptors/buffers and peak
scratch. Controlled paths check cancellation before allocation, at least every
4096 positions (including dot-product inner loops), and before returning.
The executable [native matrix example](../crates/calc-flow/examples/native_matrices.rs)
combines matrix multiplication, axis means and owned shape transforms.

## Native RSI and ATR

`rsi(view, RsiOptions::new(period)?, &execution)` and
`atr(high, low, close, AtrOptions::new(period)?, &execution)` require rank-one
float32/float64 inputs and a positive `u64` period. ATR operands have equal
lengths and may use mixed floating types. Each result is aligned float64 with an
explicit validity bitmap. Null slots contain positive zero; input buffers remain
read-only. The first complete row establishes the previous price/close, then
`period` differences/true ranges seed the first valid result.

Null and valid NaN reset the complete segment. Valid infinity is an error. ATR
checks every valid operand for infinity before missingness, and validates
`high >= low` plus finite `high - low` even when close is missing. Nonfinite
price differences, signed gaps or recurrence arithmetic are errors. Flat RSI is
50; all-gain and all-loss RSI are 100 and positive zero, respectively.

The seed uses `mean + (sample - mean) / count` in that operation order. Later
updates use the same operations with denominator `period`; period one replaces
the mean directly. No reassociation or fused multiply-add is used. Finite
underflow is allowed and can affect the ratio materially: RSI with period two
on `[0, 2^-1074, 0]` returns `[null, null, 100]`, rather than the exact-real
ratio's 50. `u64::MAX` is accepted without period-sized storage, but cannot
complete warm-up before the checked total-position counter is exhausted.

`RsiState` and `AtrState` accept the same options plus `ArrayStateOptions`.
Their fixed numerical charges are 48 and 40 bytes, including empty/reset
states. Chunk errors and cancellation preserve committed values and counters;
`reset` retains options/limits and clears recurrence and counters. These owners
have no public cloning or snapshot format. Calls charge committed state,
isolated candidate, output buffers/shape and any retained captures against the
execution limit; candidates must also fit the state limit. Eager calls charge
one fresh candidate plus output and have no independent state cap.
The standalone [indicator example](../crates/calc-flow/examples/native_indicators.rs)
shows chunked RSI and aligned ATR output.

## Native MACD components

`macd_components(view, options, &execution)` computes one aligned float64 line,
signal and histogram from a rank-one float32/float64 input. `MacdOptions::default()`
uses spans 12/26/9 and both warm-up minima one. `MacdOptions::new(fast, slow, signal)`
requires positive `u64` spans and fast<slow; consuming `with_min_periods` and
`with_signal_min_periods` setters accept independent positive minima, including
values greater than their spans. Direct states may change float dtype between
calls. Scalars, matrices and Boolean inputs are rejected without reading values.

The computation literally composes the existing unadjusted EWMA transition:
fast minus slow is line; EWMA(line) is signal; line minus signal is histogram.
Null/NaN prices skip fast/slow updates and carry a ready line. That carried line
still updates signal, so its valid-sample count can exceed the price count.
A valid NaN line is missing only for the signal update; signal can carry an
already valid result. Infinity updates signal. Arithmetic-generated NaN remains
valid for line/signal/histogram. Finite overflow keeps the old EWMA/subtraction
IEEE behavior. This is Calc Flow's first-valid seed, rather than a TA-Lib lookback
or an adjusted/alternative profile.

`MacdOutputs::{line,signal,histogram}` borrow immutable NativeArray owners;
`into_arrays()` moves all three without copying their buffers. Outputs outlive
inputs/state and retain explicit validity, including empty arrays. Null slots
contain positive zero and valid floating bits follow the individual operation.

`MacdState::new(options, state_limits)` owns all three recurrences with a fixed
56-byte numerical charge, including empty/reset states. Every call prepares one
isolated candidate and all three outputs before committing; errors/cancellation
preserve committed counters and values. `reset` retains options/limits and clears
the complete recurrence. There is no public cloning, serialization or async CPU
alias. No history proportional to a span or minimum is retained.

The output element limit counts `3*n`, and allocation limits include three
independent value buffers, three bitmaps and three rank-one shape descriptors.
Eager calls charge one fresh candidate plus outputs; owned calls charge committed
state, candidate (zero for empty input), all output/scratch and retained captures.
Limits are checked before allocation and rechecked against actual capacities.
All inputs stay read-only. The three outputs always advance one shared state,
even when only line or histogram is consumed.

The ordered `native:array_sequence@1` provider accepts `ir_version: 3` and
`kernel: {"function":"macd_components"}` with the same five optional parameters.
Its floating input is `[n]`; its single output is row-major `[n,3]` with line,
signal and histogram in that order. It writes one packed value buffer, one
validity bitmap and one rank-two shape directly. Its element budget is `3*n`,
and its peak includes committed state, candidate, packed output and captures.
Empty inputs emit `[0,3]` without advancing counts. A declared input dtype stays
fixed across this provider's stream.

The fixed 156-byte layout 3 checkpoint preserves all three recurrence counts
and raw floating bits, including active infinities, NaNs and signed zero.
Ordinary and terminal managed recovery validate parameters, identity and state
invariants before any source opens. Existing IR1/layout1 and IR2/layout2 kernels
retain their versions and byte layouts; IR3 also accepts those kernels.
The standalone [MACD example](../crates/calc-flow/examples/native_macd.rs)
shows carried-line updates and immutable component ownership.

## Native array batch provider

`array::register_native(&ProviderRegistry)` atomically registers
`native:array_expr@1` for batch processing, `native:table_array@1` for batch
and admitted stateless stream processing, and `native:array_sequence@1` for
checkpointed sequence streams. Any of the four lifecycle-key collisions leaves
all existing entries unchanged. The three factories share two worker slots.
`NativeArrayFactory::default()` also implements `BatchOperatorFactory` for
direct Rust integration; every factory validates its exact identity.

The factory's `capability_descriptor()` returns data describing its actual
batch behavior and a canonical `options_json_schema` string. Querying this
descriptor does not register a Runtime. Named, required Array ports come from
the options declarations; their sets must exactly match factory ports and each
invocation's inputs. Payloads must downcast to genuine `NativeArray` owners.
Other providers' payloads are rejected at `inputs.<name>.backend`.

The versioned data-only IR has `ir_version`, `inputs`, `nodes`, `outputs`, and
optional `execution`. Every object rejects unknown fields. Each input declares
`name`, `dtype` (`bool`, `float32`, or `float64`), and `shape`; each output also
declares the referenced `node`. Nodes have a unique `id` and one operation:

```json
{
  "ir_version": 1,
  "inputs": [{"name": "x", "dtype": "float64", "shape": [null]}],
  "nodes": [
    {"id": "x", "op": "input", "name": "x"},
    {"id": "average", "op": "reduce", "function": "mean", "input": "x"},
    {"id": "result", "op": "binary", "function": "subtract", "left": "x", "right": "average"}
  ],
  "outputs": [{"name": "output", "node": "result", "dtype": "float64", "shape": [null]}]
}
```

IR1 keeps rank zero/one and its original closed node fields. `input`, typed
`literal`, `unary`, `binary`, `where`, `reduce`, `rolling_mean`, and `ewma`
nodes reuse the eager kernels. Full reductions accept numeric scalars/vectors;
rolling nodes require numeric vectors, return float64, and
start fresh for each batch. Rolling means admit explicit `stable_v1` and
`stable_v2` profiles. Node order and port list order need not be topological or
identical. Missing references, cycles and unreachable nodes are rejected.

Shape `[]` is scalar; `[n]` is a vector, and `[null]` has an unknown vector
extent. `[1]` remains a vector. Known and unknown extents infer the known extent;
runtime rechecks actual equality. Output assertions must exactly match abstract
inference, including known versus unknown extents. Concrete descriptors are
re-inferred before numerical work. Literals contain finite numbers or bools.
Integer `serde_json::Number` representations retained as i64/u64 must lie within
±2^53; floating representations retain the full finite f64 range. Null literals
and inline arrays are rejected. Explicit float32 literals round once and must
remain finite. `pow` requires its right
reference to be a numeric literal of magnitude at most 64, checked before f32
conversion.

IR2 (`ir_version=2`) adds packed rank-two descriptors, `matmul` with `left`
and `right`, `reshape` with a concrete `shape`, and `transpose`. It admits at
most one matmul and reuses the same deterministic eager matrix kernel. All
non-scalar operand shapes must agree exactly; corresponding known dimensions
resolve unknown ones, with actual equality checked before allocation. A reshape
with an unknown input extent defers its element-count equality check until
execution. Inferred products use checked arithmetic even when combining two
independently valid declarations. Known zero dimensions remain ordinary values.

IR2 `reduce` adds optional `axis` (null, 0 or 1) and `keepdims` (false by default).
It uses the eager axis/null/dtype rules. Rolling/EWMA continue to require vectors.
IR1 rejects these new fields even when explicitly null or false, and rejects
rank-two declarations and layout nodes.

`NativeArrayFactory::infer_program(&JsonMap)` accepts descriptor-only IR1/IR2
queries: each requested output contains only `name` and `node`. It returns an
object containing `outputs` with canonical `name`, `node`, `dtype` and `shape`
descriptors. It does not require registration or read rows. Execution limits of
zero are valid for queries, since no numeric output or scratch is allocated.
Executable factory options still require exact dtype/shape assertions. The
capability's strict canonical schema exposes both versioned executable forms.

The factory receives an already parsed `JsonMap`, not original JSON tokens.
Upstream parsing can round integer tokens outside i64/u64 into f64, after which
their original spelling cannot be recovered. For example, a retained `u64::MAX`
integer is rejected, whereas `18446744073709551616` can become an accepted
finite f64, indistinguishable from an explicitly floating value with the same
bits. This boundary makes no raw-token precision guarantee. Frontends that
retain arbitrary-precision integer literals must validate them before JSON
conversion; native input array values are unaffected.

Options are bounded to 64 KiB of compact UTF-8 JSON before a typed copy, with
at most 128 nodes, 24 dependency levels, 32 inputs and 16 outputs. JSON nesting
also obeys the existing 32-level document bound. Identifiers use the existing
portable ASCII syntax with a 64-byte maximum. Duplicate names remain visible
in declaration lists; duplicate object keys discarded by an upstream JSON
parser cannot be reconstructed by this factory.

One execution budget covers all simultaneously retained computed buffers,
shape storage and each active kernel's scratch. Input aliases share existing
payload owners and add no array-buffer charge. Owned reshape and degenerate
transpose aliases charge each new shape descriptor while sharing one data-buffer
charge with their producer; general transpose also reserves an explicitly
present all-valid bitmap. Every computed intermediate
obeys the element ceiling, and all named output elements are summed separately,
including repeated aliases of one input. The full graph is admitted before
work, then each kernel receives the remaining byte allowance and actual
retained capacities are checked. A possible validity bitmap and reduction
accumulator are conservatively reserved during preflight even when actual
values later suppress a bitmap or trigger an early null reduction. Consequently
an ambiguous program can be refused even when a particular eager result fits.
Allocator/object overhead, borrowed inputs and process RSS remain outside the
buffer budget. No partial output map is published on failure.

A directly constructed factory shares two private worker slots across its operators. Computation
runs on Tokio's blocking pool; queued admission uses a 10 ms polling interval to observe cancellation and
deadlines while the runtime can schedule it. Kernels check
before allocation, every 4096 visited positions, inside rolling history refolds,
and before returning. Explicit cancellation waits for worker settlement. Dropping
or aborting a waiter requests local cancellation; an independent supervisor
keeps the blocking handle and the closure retains its permit and owned inputs
until it actually exits. During runtime teardown the guard requests local
cancellation and Tokio's ordinary `Runtime::drop` drains its blocking pool.
`shutdown_timeout` and `shutdown_background` explicitly stop waiting: their return
does not prove that a worker has ended, and closure-owned buffers and permits
remain retained until it does. The factory introduces no checkpoint or
cross-platform bitwise reproducibility promise.

## Native table/array bridge

`NativeTableArrayFactory` reuses IR2 kernels for one exact-schema table input,
zero or one static matrix RHS, and one array or attached-table output. Its
`capability_descriptor()` exposes a strict canonical wrapper schema, batch and
stream modes, stateless checkpoint support and row-axis-independent finality.
The array factory remains batch-only with unproven checkpoint support. Both
factories support declared finite native static inputs.

```json
{
  "ir_version": 2,
  "inputs": [
    {"name": "observations", "kind": "table", "columns": ["a", "b"]},
    {"name": "coefficients", "kind": "array", "binding": "static", "dtype": "float64", "shape": [2, 1]}
  ],
  "nodes": [
    {"id": "x", "op": "input", "name": "observations"},
    {"id": "w", "op": "input", "name": "coefficients"},
    {"id": "y", "op": "matmul", "left": "x", "right": "w"}
  ],
  "outputs": [{"name": "result", "kind": "table", "node": "y", "attach_to": "observations", "names": ["score"]}]
}
```

Selected columns are nonempty, unique and homogeneous bool/float32/float64.
Chunks and slices retain logical row order. Packing is charged and does not
require `combine_chunks`; invalid slots are never read. Table output preserves
original columns, Arrow metadata and Batch metadata, and appends conservatively
nullable inferred fields. Its declared schema, including schema metadata, must
match exactly. Array output instead declares `kind`, `name`, `node`, `dtype`
and `shape`, without `attach_to` or `names`; it retains rows and Batch metadata.

Admission proves the original table row lineage. Elementwise work, static-RHS
matmul, rank-two axis-1 reductions and a rank-one transpose preserve it. Full or
axis-0 reductions, rank-two transpose, rolling/EWMA and mismatched origins cannot
be attached merely because lengths match. Batch reshape requires unchanged row
mapping; stream rejects every concrete reshape because it can fix a microbatch
length. A matrix is permitted only as the direct RHS of the single matmul.
Every stream RHS must have an actual compiled static binding; direct Rust
builders receive the same startup check before any source opens.

Provide declared values with the existing finite input mapping or
`StreamingRunner::with_static_inputs`. Native static names exactly match exposed
graph bindings: bare when the local port is unique, `node.port` when repeated.
Finite `data_sources` omits native static arrays and rejects dummy dynamic
sources for them; legacy Table/NumPy/JAX static declarations retain their source
coverage rules. Native project validation applies the same static declarations
and trusted-consumer checks as compilation and returns its matching fingerprint.
Only genuine native or engine-latched
native owners enter this preparation path. A wrapper delegating `as_any` does
not acquire trusted owner identity. One run/job freezes, canonically encodes and
places each shared owner once across nodes, including array/table consumers.
Separate owners sharing values or backing buffers remain separate. SHA-256 is
still computed per physical alias with unchanged name-sensitive digest-v1;
encoding once does not eliminate per-alias hashing. Public global
`static_input(name)` lookup keeps its existing meaning; private execution uses
exact node-port-to-physical projections.

Each consumer reserves retained source/frozen storage, packed placement and
identifiable native descriptor copies. Preparation adds a fixed 4 KiB encoding
chunk and one SHA state per alias. Every consumer must admit its own retained
reservation plus shared preparation workspace before allocation; numerical
packing/outputs/scratch receive the remaining byte allowance. Actual data and
instantiated descriptor capacities are checked separately, so future descriptor
reservation cannot subsidize a larger observed array allocation.

Managed descriptor admission takes the larger of the pre-installation phase
and active checkpoint phase. It counts native strings in preparation maps,
installed static keys, resident settings copies, checkpoint identity, the live
selected/current manifest and temporary native JSON maps. Preparation maps are
consumed before task contexts are created. Resident settings copies comprise one
root, each logical operator, each dynamic source binding, each named output
sink task and the optional benchmark context. Built-in SQL capture/restore uses
thin controls. A fresh job conservatively reserves a possible selected manifest
slot before recovery is known; loaded selected native capacities are rechecked.
Existing global frames/map overhead, whole-manifest serialization/I/O buffers,
historical retention maintenance and user-created external context clones are
outside this identifiable-storage budget, which is not an RSS cap.

Preparation checks actual run/job cancellation and deadlines in bounded workers.
A dropped startup waiter requests cancellation; owned cleanup waits for actual
work, permits and inputs to settle, including queued admission. All physical
recovery digests and observed native descriptor capacities pass before scope
installation and source open. Failure/cancellation publishes no reusable scope.
The existing NumPy/JAX preparation and checkpoint protocols retain their behavior.

## Expression pipeline

This native implementation example is exercised by
[`crates/calc-flow/examples/expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs)
and computes the same `[3, 7]` result as the Python expression quickstart:

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = a + b",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![1, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0]
        .column_by_name("total")
        .expect("expression output contains total")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("total is an Int64 column");

    assert_eq!(totals.values(), &[3, 7]);
    println!("calculated totals: {totals:?}");
    Ok(())
}
```

`PipelineBuilder` consumes and returns itself, so partially built graphs cannot
be mutated through aliases. Compilation validates ports and topology and binds
an immutable `UdfRegistrySnapshot`. `BatchExecutionPlan::execute` is
asynchronous and accepts only `Batch` values.

### Engine boundary

`ExpressionOperator`, `SqlOperator`, `RollingOperator`, and
`CrossSectionOperator` and exact-schema table `UnionOperator` are built-in
operators that implement both `BatchOperator` and `StreamOperator`;
`WindowAggregateOperator`, `StreamJoinOperator`, and
`StreamAsofJoinOperator` are stream-only. Add them through `PipelineBuilder::add_node`, which accepts a
`NodeOperator` conversion from a boxed built-in, custom batch operator, or
custom stream operator.

Every operator implements `OperatorMetadata: Send + Sync + Any`, so concrete
operator types must satisfy `'static`; they cannot borrow non-static data.
The bound does not require an operator instance to live forever.
Custom finite operators implement
`BatchOperator`, whose `BatchOperatorContext` carries the run-scoped context;
custom continuous operators implement `StreamOperator`, whose
`StreamOperatorContext` carries the stream-job context, operator identity,
current input watermark, late-row counters, and read-only access to declared
job-static batches through `static_input(name)`. External operators resolve
through `ProviderRegistry` and lifecycle-specific factories, covered below.

`BatchExecutionPlan::datafusion_config()` returns `Option<DataFusionConfig>`:
`None` means the plan is external-only and owns no table resources, so its runs
create no DataFusion session or UDF snapshot.
`BatchExecutionPlan::requires_datafusion()` returns the classification alone.
`DataFusionConfig` keeps `parallelism_mode=Fixed` and
`target_partitions=1` as its compatible default. Opt-in `Auto` freezes a
whole-run external-row policy for batch graphs containing multi-input SQL, and
opt-in `AutoRows` derives the request from input rows without requiring trusted
entity statistics. `AutoPlan` additionally pins a batch graph to one partition
when every node is a proven simple row-local SQL projection, and otherwise uses
the external-row policy. Other `Auto` graphs and runtimes additionally require
the trusted `calc_flow.datafusion.active_entities` batch metadata value;
`AutoRows` and the standalone/streaming `AutoPlan` fallback use input rows
instead. No policy scans input rows. See
[SQL and DataFusion performance controls](sql-datafusion-performance.md) for
the complete configuration, telemetry, benchmark, and rollback contract.

## SQL operators

`SqlOperator::new` declares the input aliases that become operator ports. It
accepts one read-only `SELECT` or CTE. Connect upstream node outputs to those
aliases with `Edge`/`PortEndpoint`, or leave them unconnected as graph inputs.
DDL, DML, utility commands, and multiple statements are rejected before
execution. The canonical SQL example is
[`crates/calc-flow/examples/02_sql_join.rs`](../crates/calc-flow/examples/02_sql_join.rs),
which joins `orders` to `fees` for `net = amount - fee`, then doubles the net
amounts in an expression stage:

The run-scoped physical planner can rewrite bounded ascending `AVG(Float64)`
windows and compatible same-column, same-frame `COUNT` guards through the
crate-private `CalcFlowRollingExec`. AVG-only queries use the shared Native
numerical profile; queries admitted by COUNT use SQL sum/count transitions
throughout. The optimizer requires proven partition/order keys and sorted
physical input. Any logical eligibility fallback keeps the original DataFusion
plan for the whole query, including AVG-only queries. If physical rewriting
cannot cover every candidate window in
a query containing COUNT, the planner restores the original DataFusion plan
and reports zero rewritten windows. Only logically eligible AVG-only queries
can retain successful partial physical rewrites when other windows miss the
supported physical shape and remain on DataFusion. See the
[rolling rewrite boundary](sql-datafusion-performance.md#rolling-rewrite-boundary)
for exact eligibility and numerical semantics. `DataFusionQueryMetric`
reports the candidate/rewrite counts, stable fallback reasons, configured and
effective partition counts, and the physical plan. Fixed parallelism is
adapted downward below 65,536 input rows per useful partition and is never
increased above `target_partitions`. Auto mode is bounded by
`max_partitions`, work rows, and trusted active-entity statistics.

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = plan()?
        .execute(inputs()?, ExecutionOptions::default())
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let net = output.batches()[0]
        .column_by_name("doubled")
        .expect("sql output contains doubled")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("net is an Int64 column");

    assert_eq!(net.values(), &[140, 216, 72]);
    println!("doubled net amounts: {net:?}");
    Ok(())
}

fn nodes() -> calc_flow::Result<PipelineBuilder> {
    PipelineBuilder::new("orders-and-fees")?
        .add_node(
            "join",
            Box::new(SqlOperator::new(
                "join",
                "SELECT orders.order_id, orders.amount - fees.fee AS net \
                 FROM orders JOIN fees ON orders.order_id = fees.order_id \
                 ORDER BY orders.order_id",
                vec!["orders".to_string(), "fees".to_string()],
                Vec::new(),
            )?),
        )?
        .add_node(
            "double",
            Box::new(calc_flow::ExpressionOperator::new(
                "double",
                "",
                vec!["order_id".into(), "net * 2 AS doubled".into()],
                None,
                Vec::new(),
            )?),
        )
}

fn plan() -> calc_flow::Result<calc_flow::BatchExecutionPlan> {
    nodes()?
        .connect(calc_flow::Edge::new(
            calc_flow::PortEndpoint::new("join", "output")?,
            calc_flow::PortEndpoint::new("double", "input")?,
        ))?
        .compile_batch(&UdfRegistry::new().snapshot())
}

fn inputs() -> Result<BTreeMap<String, Batch>, Box<dyn std::error::Error>> {
    let orders = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("amount", Arc::new(Int64Array::from(vec![75, 120, 40])) as _),
    ])?;
    let fees = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("fee", Arc::new(Int64Array::from(vec![5, 12, 4])) as _),
    ])?;
    Ok(BTreeMap::from([
        (
            "orders".into(),
            Batch::table(vec![orders], BatchMetadata::default())?,
        ),
        (
            "fees".into(),
            Batch::table(vec![fees], BatchMetadata::default())?,
        ),
    ]))
}
```

## Snapshot SQL

`SnapshotSqlOperator::new(name, query, aliases, udfs)` declares a stream-only
read-only query over complete replacement relations. Set exact input and output
schemas with `with_ports`. Each incoming `Batch` replaces one alias's entire
relation; an empty table initializes or clears that alias. The operator waits
for every alias to initialize, then emits a complete result after each update.
Cross-source arrival order does not define an event-time atomic snapshot.

Input relations and results must fit their edge budgets as whole batches.
`StreamRuntimeConfig.sql_snapshot_state_budget` independently limits each
node's latest relations, replacement candidates, and held checkpoint captures.
Outputs currently connect directly to sinks and reject ExactlyOnce requests.
See the [native snapshot SQL contract](snapshot-sql-native.md) for accounting,
initialization, EOF, checkpoint, and recovery rules, and the
[Python streaming guide](streaming-guide.md#complete-replacement-snapshot-sql)
for application composition.

## Rolling windows

`RollingOperator` evaluates native lag, delta, EWMA, RSI, ATR and aggregate outputs over
entity-partitioned, event-time-ordered rows. Its data-only `RollingSpec`
declares ordered `partition_by` and `sequence_by` keys, a non-null UTC
`timestamp[us]` `event_time` column, and one `RollingOutputSpec` per rolling
column: kind `lag` or `delta` — `primitive_version` 1, input and output
column names, and a positive `periods` distance; kind `count`, `sum`, `mean`,
`min`, `max`, `variance`, `stddev`, `argmax`, `argmin`, `rank`, `quantile`,
`unique_count`, or `decay` — `primitive_version` 1, input and
output column names, a positive frame, and `min_periods`, with `variance` and
`stddev` adding `ddof`; or the pair kinds `covariance` and `correlation` —
`primitive_version` 1, left and right input column names, an output column
name, a positive frame, `min_periods`, and `ddof`. Kind `ewma` carries
input/output names, a positive `span`, and positive `min_periods`; it emits nullable `float64` from the
unadjusted `alpha = 2 / (span + 1)` recurrence and shares constant state by
`(input, span)`. Kind `difference` embeds two `mean`, `variance`, `stddev`, or
`ewma` readouts and one output name. The leaves create and share state but are
not output columns; the operator writes their nullable `float64` difference
directly.
Kinds `rsi` and `atr` use `primitive_version: 1`, a positive `period`,
and nullable float64 output. RSI names one floating input; ATR names floating
`high`, `low` and `close` inputs. They require
`ROLLING_INDICATOR_STATE_LAYOUT_VERSION` (5). Identical operand/period groups
share one recurrence per entity, independently of output aliases. They reuse
the [native indicator numerical contract](#native-rsi-and-atr), including
complete-segment warm-up and null/NaN resets. Indicator-containing mixed nodes
currently use the general row fallback.
Kind `cumulative_mean` carries an input, output, and positive `min_periods`;
it stores one count and arithmetic mean per entity. Kinds `cumulative_sum`,
`cumulative_min`, `cumulative_max`, and `cumulative_variance` carry numeric
input/output names and positive `min_periods`; variance also carries `ddof`
(`0` or `1`, default `1` in Python). They skip null/NaN and emit null below
`min_periods`; variance additionally requires `count > ddof`. Sum preserves
checked integer promotion, min/max preserve input types, and online mean/M2
variance emits `float64`, becoming NaN in the presence of infinity. These
summaries have constant size per entity. Supported primitive inputs use typed
Arrow transitions and ordered columnar stream buffering, including compatible
mixed stages with Mean or rolling outputs; other shapes retain the general
fallback.
`argmax`, `argmin`,
`rank`, and `unique_count` emit nullable `uint64`; `quantile` and `decay`
emit nullable `float64`. Extrema positions count rows back from the current
row and choose the oldest tie. Rank is zero-based ascending with the first
rank for ties; signed zero values tie and a singleton quantile is null.
Decay weights valid samples
linearly, newest highest.

MACD components use `RollingOutputSpec::MacdComponent` with
`MacdComponent::{Line, Signal, Histogram}` and require
`ROLLING_MACD_STATE_LAYOUT_VERSION` (6). The input accepts signed/unsigned 8–64-bit integers and
Float32/Float64, using the existing EWMA float64 conversion. The five positive u64 parameters default to fast/slow/signal
spans 12/26/9 and price/signal minima 1/1; fast must be less than slow. Minima
may exceed their spans. Components and aliases sharing the same input and all
five parameters advance one full recurrence per finalized row. A ready carried
line still advances the signal after a missing price. Selecting only histogram
keeps the complete 56-byte logical numerical group and 104-byte checkpoint
payload. Layout 6 also accepts RSI, ATR and the previous rolling outputs;
layout 5 rejects MACD and keeps its previous payloads. The same bounded worker,
state budget, checkpoint preparation and managed recovery apply.
[`native_table_macd.rs`](../crates/calc-flow/examples/native_table_macd.rs) shows
all three components over nullable integer prices.


Cumulative sum/min/max/variance require
`ROLLING_CUMULATIVE_STATE_LAYOUT_VERSION` (4) in both declarations and writer
state. Layout 4 extends the columnar layout with exact per-group summaries
(including binary64 bits), alongside entity keys, history and pending rows.
Other declarations keep their existing layout and checkpoint fingerprints.
Recovery validates the layout, complete summary inventory, configuration,
schema, kernel and numerical profile before installing state. Unknown versions
and incompatible configurations fail closed.

A frame is `rows(size)` —
the `size` rows through the current row of the entity total order — or
`duration(micros)`: the exact-width event-time interval `(t − micros, t]`,
open at the lower bound and closed at the upper bound, with equal-time rows
ordered by `sequence_by`. `configuration_version` and `state_layout_version`
use their validated declaration constants. `configuration_version` is
`ROLLING_CONFIGURATION_VERSION` (1). Declarations without cumulative summaries,
EWMA or cumulative mean use `ROLLING_STATE_LAYOUT_VERSION` (1), while either recurrence requires
`ROLLING_EWMA_STATE_LAYOUT_VERSION` (2). These values belong to project
documents and configuration fingerprints. Old-only operators without cumulative
summaries write `ROLLING_COLUMNAR_STATE_LAYOUT_VERSION` (3): one deterministic entity
dictionary, projected retained history, a columnar reorder buffer, recurrence
state, and kernel/numerical fingerprints. Restore continues to read the
declaration's v1/v2 layout, while newly emitted descriptors and pipeline
capabilities report writer layout v3. `allowed_lateness_micros` and a
`LatePolicySpec` of `Error` (envelope scope), `Drop` (metrics version 1), or
stream-only `SideOutput` (metrics and schema version 1) classify late rows
against the input watermark, and `RollingValuePolicy` is the frozen
`stateful_numeric_v1`, which preserves a null or NaN current or referenced
value.

Stream rolling state defaults to the JSON counter range,
9,007,199,254,740,991 rows and logical bytes, without a smaller implicit budget.
`RollingOperator::set_state_budget(StateBudget::new(...))` sets explicit row and
byte limits before graph compilation. For old-only operators, the charge covers accepted reorder rows,
retained per-entity history, keys, and state overhead; it excludes checkpoint
copies and process-wide resident memory. Input and watermark callbacks reject
state growth before committing it, and restore rejects an oversized snapshot.
Old-only batch evaluation does not use this budget. Project-created operators use the
default, and this runtime limit does not change the project fingerprint.
The preceding retained-only rule applies to old-only nodes, including old-only
declarations that explicitly select layout 5 or 6.

For nodes containing RSI, ATR or MACD, the same `StateBudget.max_bytes` instead
bounds the whole operator's participating peak: committed state, isolated
candidates, all prepared output chunks, scratch and live encoded
checkpoint/restore segments. Mixed old rolling outputs participate too.
Shared allocations count once by ownership, while distinct copies count
separately. `max_rows` keeps its retained-row meaning. The rule applies to
finite and streaming execution; it adds no serialized budget field or smaller
implicit default. Captures and unfinished workers retain their charges across
reset, and a lower budget must admit all live owners. These logical charges
exclude process RSS, independent pools/operators and transferred results.
The retained logical footprint counts observable Arrow buffer capacity/extent
and identifiable Array/container descriptors. It excludes private immutable
Buffer owner control blocks, whose identity cannot be inferred from the data
pointer; construction reservations can conservatively include that overhead.
Pure accounting owner-key/reservation indexes are not recursively charged;
they may only reference already funded owners and numeric charges. This
excludes no computational or payload containers and does not shorten their
funded lifetime when ownership moves into a lease.

Layout 5 preserves complete indicator groups and mixed old state with strict
schema/configuration/kernel/numerical fingerprints. Preparation runs through
the controlled asynchronous worker path. Capture only attaches prepared bytes
for the exact state revision; an unprepared synchronous capture returns a
preparation error. Unchanged payload storage can be reused across epochs;
inventory descriptors remain epoch-specific. Managed recovery validates both
ordinary and terminal layout-5 state before connectors open.

Indicator finalization prepares the whole candidate and output before emission.
A preparation failure preserves pending rows, recurrence state, frontiers,
sequences and metrics. This does not retract earlier accepted input or
downstream chunks if emission later fails. See
[streaming indicator semantics](streaming-guide.md#native-table-indicators) and
the executable
[`native_table_indicators.rs`](../crates/calc-flow/examples/native_table_indicators.rs).

The new `RollingOutputSpec::Rsi` and `RollingOutputSpec::Atr` variants require
Rust callers with exhaustive matches to update those matches.

`numerical_profile` is optional. Its default `stable_v1` is omitted from the
canonical configuration, preserving existing project and checkpoint hashes.
The explicit `stable_v2` value is a preview: floating numeric and pair windows
periodically rebuild from retained values with shifted compensated sums. The
per-entity transition count is stored in the existing nullable columnar-state
position field, so the cadence survives checkpoint recovery without changing
the layout-v3 field schema. The profile is part of both configuration and
kernel fingerprints, and restore rejects a profile mismatch.

Validation rejects unknown fields, empty or duplicate keys, duplicate
or input-colliding output names, nullable or floating sequence columns,
non-numeric `delta`, `ewma`, `sum`, `mean`, `variance`, `stddev`, `covariance`,
and `correlation` inputs, `min` and `max` inputs outside the total-order types
(booleans, integers, floats, strings, dates, and `timestamp[us]` with no
timezone or UTC), a `min_periods` below one or — for `rows` frames only — above
the frame size,
a `ddof` outside 0 or 1, and ambiguous column references; `count` accepts any
input column. The output schema is the input schema followed by the declared
outputs in declaration order, each nullable. Lag, delta, `min`, and `max`
keep the input type; the other aggregates follow the frozen output-type
table:

| Kind                           | Input            | Output type       |
|--------------------------------|------------------|-------------------|
| `ewma`                         | numeric          | `float64`         |
| `count`                        | any column       | `uint64`          |
| `sum`                          | signed integer   | `int64`, checked  |
| `sum`                          | unsigned integer | `uint64`, checked |
| `sum`                          | floating         | `float64`         |
| `mean` / `variance` / `stddev` | numeric          | `float64`         |
| `min` / `max`                  | total order      | input type        |
| `covariance` / `correlation`   | numeric pair     | `float64`         |
| `difference`                   | float readouts   | `float64`         |

Aggregates count valid samples — values that are neither null nor NaN;
infinities are numeric samples and count toward `min_periods`. A window with
fewer than `min_periods` valid samples reads null, and a computed NaN stays
observably distinct from that gate. `min` and `max` read a monotonic queue of
valid samples and preserve the input type. Floating sums follow IEEE
arithmetic: a window holding both signs of infinity sums to NaN, and a window
holding one sign sums to that infinity. Integer sums stay exact and checked:
the slide runs through a wide transient accumulator and narrows with a
checked conversion at readout, so a window whose true sum is representable
never reports a false overflow, while a genuine `int64` or `uint64` overflow
fails the run loudly. Mean, variance, and standard deviation read a
reversible West accumulator that also counts the window's positive and
negative infinities: a mean over both signs is NaN (the undefined ∞ − ∞), a
mean over one sign is that infinity, and a finite window keeps the
incremental West value. Variance and standard deviation over any window
holding an infinity are NaN — a pinned contract — and their divisor is
`valid_count − ddof`: a non-positive divisor reads null after the
`min_periods` gate and before the NaN classification. Negative `M2` drift
from reversible removal clamps to zero at readout.

`covariance` and `correlation` read a reversible West co-moment accumulator
over the pairwise-valid positions — rows where both operands are non-null
and non-NaN — so `min_periods` counts pairwise samples and their divisor is
that count minus `ddof`. Their classification order is frozen: the
`min_periods` and divisor null gates come first, then a window holding ±inf
on either side reads NaN, and only a finite window with a zero-variance side
reads null for `correlation` — an infinity window therefore never reads null.

Outputs over the same input column and frame share one accumulator group —
numeric, or one monotonic extrema group per direction for `min` and `max` —
and pair outputs over the same operand pair and frame share one pair group;
`min_periods` and `ddof` are readout parameters, not extra state. Each entity
retains the largest `periods` distance or row-frame size across every output
and, when duration frames are declared, every row inside the widest duration
bound, so mixed row and duration outputs retain by both bounds together. The
same kernel serves both lifecycles. Batch evaluation
orders the complete input canonically and classifies no late rows: every row
is final at end-of-input. Stream evaluation buffers rows until the input
watermark passes each row's closing coordinate — its event time plus the
allowed lateness — then emits the ordered final rows, rejecting the whole
envelope under `error`, dropping each late row and recording metrics under
`drop`, or routing it to `late` under `side_output`. See the
[late-row policy contract](#late-row-policy-contract) for diagnostic routing
and recovery. Duplicate row identities are rejected. Rolling
stream state checkpoints at the aligned epoch cut using columnar writer
layout `3`, or layout `4` when cumulative sum/min/max/variance are present.
It stores projected history, buffered rows, recurrence state, and any
cumulative summaries and processed-row counts;
restore validates the schema and fingerprints before rebuilding the kernel.
See [symbolic compiler design](symbolic-design.md#native-rolling-state) for
the distinction between declaration versions and checkpoint writer layout.
Sliding arithmetic that becomes non-finite re-folds the current window,
keeping the classification exact at an O(frame) cost per row for those
windows.

Three settled corner properties of this algorithm family are identical on
every path: the incremental mean can overflow opposite-sign extremes
({−1e308, +1e308} reads +inf though the true mean is 0); a floating sum can
overflow its finite partial sum before reaching an infinity ({−1e308,
−1e308, +inf} sums to NaN while the mean reads +inf); and `min`/`max` order
floating samples by the IEEE total order, so −0.0 and +0.0 are
distinguishable extrema — a self-consistent deterministic choice, not SQL
`MIN`/`MAX` semantics, which compare them equal.

## Cross-section statistics

`CrossSectionOperator` evaluates native complete-group rank, percentile,
z-score, demean, winsorize, top/bottom selection, and mean-fill outputs. Its
data-only `CrossSectionSpec` declares
`configuration_version` and `state_layout_version` (both 1 in this release),
a non-null UTC `timestamp[us]` `event_time` column, ordered non-empty
`entity_by` and `sequence_by` keys, an ordered `partition_by` group key —
optional, and empty means one global group per grouping coordinate — an
`ExactTime` or `FixedBucket { width_micros }` grouping (one group per exact
event time, or per UTC `[start, start + width)` bucket with the Unix-epoch
origin flooring toward negative infinity), one `CrossSectionOutputSpec` per
output, `allowed_lateness_micros`, a `LatePolicySpec` of `Error` (envelope
scope), `Drop` (metrics version 1), or stream-only `SideOutput` (metrics and
schema version 1), and the frozen
`CrossSectionValuePolicy` `NanExcludePreserveV1`. A row identity is the
`(event_time, entity_by, sequence_by)` value.

Each output declares `primitive_version` 1, input and output column names, and
`min_samples`; the per-variant rules below define which values are gated when
a group has fewer valid samples. `rank` and `percentile` add `direction`
(`ascending`/`descending`),
`tie_method` (`average`/`min`/`max`), and `null_placement`
(`exclude`/`first`/`last`); `zscore` adds `ddof` of 0 or 1. Rank, percentile,
demean, and z-score produce nullable `float64`: rank is the one-based position
after the tie method, percentile is
`(rank - 1) / (ordered_count - 1)` with a single ordered value reading
exactly `0.5`, demean subtracts the valid-sample mean, and z-score divides
by the standard deviation over the divisor `valid_count − ddof` — null when
the divisor is not positive or the deviation is zero.

`mean` broadcasts the valid-sample mean to every group row. `top_quantile`
and `bottom_quantile` use the average descending or ascending rank divided
by the valid sample size as an inclusive fraction threshold; both return
nullable boolean masks. `residual` uses pairwise-valid dependent and
independent columns, centers both before calculating a least-squares slope,
and scales finite extreme values when centered statistics overflow or
underflow. It returns null for missing pairs or a constant independent sample.
`winsorize` accepts only float32/float64, carries finite `lower` and `upper`
probabilities satisfying `0 <= lower <= upper <= 1`, and clamps to the
Hyndman-Fan type-7 quantiles while preserving the input type. `top` and
`bottom` carry a positive `count` and `include_ties`: they produce nullable
boolean masks over the exact scalar total order, include every boundary tie
when requested, and otherwise use canonical row identity to select exactly
`min(count, valid_count)` rows. `mean_fill` accepts float32/float64, replaces a
null with the complete valid sample's mean when `min_samples` is met, and
preserves valid values, NaN, and the input type. All compatible outputs share
one grouping, sample classification, and measured-value sort per input column.

Under `nan_exclude_preserve_v1`, NaN is excluded from every sample and stays
NaN at its own row, infinity is numeric, and null handling comes only from
`null_placement` or the variant's fixed non-ordering rule: excluded nulls read
null
and leave the ordering, included nulls form one tied class at the requested
end; a null `demean`, `zscore`, or `winsorize` input reads null; top/bottom
map both null and NaN to null; and mean-fill replaces only null. The mean and
deviation classify infinities exactly as the rolling aggregates do: a sample
over one sign of infinity has that infinity as its mean and over both signs
reads NaN, and any infinity in the group makes the variance — and with it
the z-score — NaN. Validation rejects
unknown fields, zero bucket widths, empty keys or output lists, duplicate key
or output names, input-colliding output names, non-numeric input columns,
nullable or floating sequence columns, unsupported key types, and ambiguous
column references. The output schema is the input schema followed by the
declared outputs in declaration order.

One micro-batch is never evidence of completeness. Batch evaluation treats
every group as complete at end-of-input, classifies no late rows, and emits
all groups once, ordered by finality coordinate then group key, with rows
inside a group ordered by `(event_time, entity, sequence)`. Stream
evaluation accumulates groups across envelopes: a group closes when the
input watermark reaches its finality coordinate — the exact event time or
the bucket end, plus the allowed lateness, with equality closing — and emits
once in the same canonical order, releasing its rows and identity index.
End of input flushes every open group once without synthesizing a sentinel
watermark; a repeated end is a no-op and data after end is rejected. A row
whose group has closed is late: the `error` policy rejects the whole
envelope before any state changes, `drop` discards the row and records the
late metrics, and `side_output` routes it to the diagnostic `late` port.
Duplicate row identities are rejected
transactionally — within one envelope across partition groups, and across
envelopes while the identity is still open. Open groups checkpoint at the
aligned epoch cut as an Arrow IPC base segment with state version 1 —
configuration-hash and schema-fingerprint metadata plus bounded inline
manifest fields — and a restored operator reproduces the same ordered
output, watermark frontier, output sequence, and metrics.

Stream cross-section state defaults to the JSON counter range,
9,007,199,254,740,991 rows and logical bytes, without a smaller implicit budget.
Direct Rust users can set positive row and byte
limits through `CrossSectionOperator::set_state_budget(StateBudget::new(...))`.
Project-created operators use the default; the budget is runtime tuning and
does not change project fingerprints. Charges count materialized values,
keys, and map overhead. An envelope that would exceed either limit fails
before any accepted row or late side output is installed, and restore rejects
oversized state. Batch evaluation does not apply this stream-state budget.

## Late-row policy contract

Rolling and cross-section specifications share `LatePolicySpec`.
`Error { scope: LateErrorScope::Envelope }` rejects the complete input envelope
atomically; its serialized form requires `kind: "error"` and `scope: "envelope"`.
`Drop { metrics_version: 1 }` discards late rows and records their metrics.

`SideOutput { metrics_version: 1, schema_version: 1 }` routes late rows to a
required `late` table port alongside `output` in native streaming execution.
The late schema retains input fields and metadata and appends non-null
`_cf_late_*` diagnostics for node, input port, event time, closing time,
watermark, reason, source, sequence, and row index. Input names starting with
`_cf_late_` are reserved for this policy. The generated
[project schema](../schemas/project-v3.schema.json) defines its version fields.

Batch compilation rejects side output with `unsupported_mode`. Stream graphs
can expose `late` as an output or pass it through built-in single-input
expression/SQL successors. Other successor kinds are rejected with
`temporal_output_unavailable`; diagnostic output has no event-time progress
contract. The runtime forwards neither watermark nor idle messages on `late`
or its derived outputs, while barrier and end-of-input retain FIFO order and
participate in the same checkpoint epochs as normal output.

Native operators prepare the whole input envelope and bounded diagnostic
chunks before emitting. Accepted state, metrics, and independent normal/late
output sequences advance only after successful emission. Failed or cancelled
emission forbids live callback retry; resume from a compatible durable cut.
Side-output checkpoints include a strict `late_output` inline object with
`version: 1`, `schema_version: 1`, and a `uint64` `next_sequence`, including
when no late row has been emitted. Restore validates this object before
installing state and restores the independent late sequence. Missing, null,
malformed, unknown-field, or incompatible objects are rejected with
`CalcFlowError::CheckpointMismatch`; error/drop checkpoints omit the object
and likewise reject one supplied for those policies.

Managed recovery covers both outputs, including empty late epochs and terminal
restart. Delivery remains a per-output contract: exactly-once requires the
normal source, operator, and transactional sink proofs; ordinary sinks can
observe replay duplicates. The [file connector](connectors/file.md) describes
transactional Parquet recovery. Project and manifest formats remain 3, with
rolling writer layout 3 or 4 according to the declared outputs, and
cross-section state layout 1.

Python declares paired outputs through
[`with_late_output`](python-api.md#lateoutputs-and-local-late-policy).
The global compilation argument still accepts only `late_policy="error"`
or `"drop"`; the paired declaration selects SideOutput locally.
ASOF and event-window lateness retain their separate operator contracts.
The [streaming guide](streaming-guide.md#route-late-rows) gives the exact
diagnostic schema, closing-coordinate rules, physical binding names, and
resource/recovery boundaries. See the [2026-09 changelog](../CHANGELOG.md#2026-09)
for Rust source migration requirements.

## Bounded backward ASOF Join

`StreamAsofJoinSpec::new(left, right, tolerance, limits)` takes two validated
`AsofJoinSide` values, an exact microsecond `std::time::Duration`, and positive
whole-operator `AsofStateLimits`. It defaults to `AsofLatePolicy::Error`;
`with_late_policy` selects `Drop`. `StreamAsofJoinOperator::new` takes the node
name, exact left/right `SchemaRef` values, and that spec. Add it through
`PipelineBuilder::add_node` and compile in stream mode.

The operator exposes `spec()` and payload-free `status()` and implements
`StreamOperator` with `checkpoint`, `restore`, and `reset`. Its independent
semantic state version is `3`; native snapshots require layout and accounting
version `10`. Restore rejects other native layouts even when their semantic
state version is `3`.
`StreamingJob::stream_asof_join_status()`
returns status by node ID. The [ASOF guide](asof-join-guide.md) defines exact
input types, inclusive predecessor selection, dual-watermark finality,
whole-operator budgets, and recovery; [architecture](design.md#bounded-backward-asof-join)
explains native state and DataFusion ownership.

## Stream compilation and continuous runtime

`PipelineBuilder::compile_stream` produces an immutable
`StreamExecutionPlan`. The plan records stable edges, source and sink binding
slots, stream requirements, and the semantic fingerprint. Pass it to the
crate-root `StreamingRunner` with `SourceBinding`, `SinkBinding`, and
`ManagedCheckpointRuntime` values.

A plan compiled from a project that declares static side inputs exposes them
through `StreamExecutionPlan::static_input_ids()` and `static_inputs()`;
`source_binding_ids()` excludes the static names. Supply the immutable
`Batch` values with the additive builder, leaving `StreamingRunner::new`
source-compatible:

```rust
let runner = StreamingRunner::new(plan, sources, sinks, checkpoints)?
    .with_static_inputs(BTreeMap::from([("weights".to_owned(), weights)]))?;
```

`with_static_inputs` copies the mapping and is `#[must_use]`; validation,
latching, and digest computation are deferred to `start`, where they happen
exactly once before any source or operator lifecycle runs. The complete frozen
static-input crate-root export set is `StaticInputSpec`, `StaticInputDigest`,
`StaticMutability`, `STATIC_INPUT_DIGEST_VERSION`, `StaticArraySnapshot`, and
`StaticArrayValues`. `Batch::static_array_snapshot()` provides an owned
snapshot with five read-only accessors. Declarations join the plan fingerprint;
`CheckpointManifest` records one `StaticInputDigest` per name under its
`static_inputs` field, omitted when empty. See
[static inputs](streaming-guide.md#static-inputs) for the full per-job,
digest, and recovery contract.

The source-driven runtime consumes that plan after a whole-job
preflight. It runs one task per source, compiled operator, and graph output;
preserves per-ingress FIFO; routes source control through a job-scoped progress
driver; and converges source, operator, sink, queue, reservation, driver, and
reaper state on every terminal path. The progress driver owns watermark
generation, idle/reactivation, deterministic timers, and multi-ingress
progress.

`WindowSpec` declares fixed UTC tumbling or hopping windows and ordered
aggregates. `WindowAggregateOperator` is stream-only: it updates incremental
state, classifies late row-window assignments against the current input
watermark, emits closed windows in deterministic order, and snapshots retained
Arrow IPC deltas. Retained accumulators default to the JSON counter range,
9,007,199,254,740,991 windows and logical bytes, without a smaller implicit budget.
A window stops counting once its close
is emitted, with checkpointing enabled or disabled, so the budget bounds open
windows regardless of the checkpoint interval. Rust callers can set a
different limit through
`WindowAggregateOperator::set_state_budget(StateBudget::new(...))` before
adding the operator to a graph. Admission and restore reject state that
exceeds the budget; checkpoint segment copies and process RSS are separate.
The complete
[`windowed_streaming.rs`](../crates/calc-flow/examples/windowed_streaming.rs)
example wires a source-provided watermark through this operator to a sink.
`StateBackend` opens an exclusive lineage session;
`LocalStateBackend` provides immutable checksum-verified segment publication.
`CheckpointManifest` is the strict bounded v3 state-manifest contract. Barrier
coordination cuts live sources, aligns operator ingresses, stages operator
state, and publishes the manifest last as the sole durable recovery truth.
Transactional sinks commit only after manifest durability. Recovery restores
mixed ended/live jobs, and terminal checkpoints capture post-`on_end` state
without forwarding a post-end barrier. Exactly-once compatibility is proved
for each requested graph output before lifecycle work; checkpointed
ordinary-sink outputs remain at least once.

Launch failure and cancellation settle source, operator, sink, queue,
progress, state, checkpoint transaction, and reaper ownership. Cleanup
failures and panics remain typed secondary diagnostics and do not replace the
primary outcome.

`StreamingRunner::start(self)` is one-shot and returns the sole
`StreamingJob` owner. The job exposes synchronous status plus async checkpoint,
shutdown, cancel, and wait operations. Public continuous lifecycle failures surface as
`CalcFlowError::Streaming(StreamingError)`. Contained task panics use
`StreamingErrorCategory::TaskPanicked`; internal task IDs and panic messages
never cross the facade.

`edge_channel` accepts an `EdgeBudget` with the fields
`max_rows` and `max_bytes`. For `EdgeBudget::new(R, B)`, envelope count and
charged rows are each independently capped at `R`, while charged bytes are
capped at `B`. Direct callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`.

## Batches

- `Batch::table(Vec<RecordBatch>, BatchMetadata)` creates an Arrow table batch.
- `Batch::external(Arc<dyn ExternalPayload>, BatchMetadata)` creates an
  explicitly registered external-provider batch.
- `Batch::static_array_snapshot()` returns `Some(StaticArraySnapshot)` only for
  an engine-latched static array. It makes one detached `O(rank + bitmap +
  non-null values)` copy with read-only backend, dtype, shape, null-bitmap, and
  compact-value accessors; tables and general external arrays return `None`.
  `StaticArraySnapshot` and `StaticArrayValues` are non-exhaustive, deliberately
  non-`Clone`/non-`Debug`, and preserve `Send + Sync + Unpin` plus unwind-safe
  auto traits.
- `Batch::kind`, `num_rows`, `metadata`, `table_payload`, and
  `external_payload` expose immutable observations.
- `Batch::with_metadata` returns a new envelope.

`BatchMetadata::new(source, sequence, attributes)` validates the source and stores
JSON-compatible attributes. Its sequence is descriptive batch metadata and is
independent of continuous source cursors and checkpoint epochs.

Python static placement creates this snapshot once per static input inside a
blocking worker and caches only the provider-owned batch after successful
placement and a post-worker cancellation check. The transient peak may include
the engine latch, snapshot carriers, Python host list, NumPy storage, and JAX
result simultaneously. `static_placement_bytes` counts only the logical
provider transfer, not that peak memory or the internal snapshot copy.

## UDFs and external providers

Create an application-owned `UdfRegistry`, register native DataFusion scalar
UDFs, then compile against `registry.snapshot()`. Operators hold serializable
`UdfReference` values; they never own source code or import paths.

`ProviderRegistry` binds `ExternalOperatorSpec` values to trusted
`BatchOperatorFactory` and `StreamOperatorFactory` implementations through
separate `register_batch`/`resolve_batch` and `register_stream`/`resolve_stream`
paths. External provider state and payloads must satisfy Rust's `Send + Sync`
boundaries.

An external `StreamOperator` reports its proof through `lifecycle()`. The
default is `StreamOperatorLifecycle::Unproven`. A trusted factory may return
`StreamOperatorLifecycle::Stateless` only when the operator owns no checkpoint
state; its `microbatch_invariant` flag must also be true so splitting or
combining input micro-batches cannot change the output rows.

Stream planning keeps lifecycle proof and checkpoint capability distinct.
`PipelineBuilder::add_node` classifies a proven external stateless operator as
`Stateless`, and `compile_stream` revalidates its lifecycle proof. The
stateless proof keeps `deterministic` and `replay_safe` separate: best-effort
and at-least-once plans may compile without those two claims, but a plan
requesting exactly-once delivery rejects the provider unless both are true.
Operators classified as `CheckpointedStateful` instead qualify through their
positive checkpoint state version and do not need a stateless lifecycle claim.

An `Unproven` external operator is not rejected merely by `compile_stream`, so
ordinary plan construction retains the existing two-stage contract. If a
`StreamingRunner` is configured with a checkpoint runtime, job admission
rejects that operator before any connector opens. Exactly-once delivery
requires a checkpoint runtime, so an unproven operator cannot be admitted on
that route either.

## Continuous execution and recovery

The complete checked example is
[`04_continuous_runtime.rs`](../crates/calc-flow/examples/04_continuous_runtime.rs).
Its central lifecycle is:

```rust
let runner = StreamingRunner::new(
    plan,
    BTreeMap::from([(source_id, SourceBinding::new(source))]),
    BTreeMap::from([(
        output_id,
        vec![SinkBinding::ordinary("archive", sink)?],
    )]),
    ManagedCheckpointRuntime::new(".calc-flow-continuous")?,
)?;
let job = runner.start().await?;
let outcome = job.wait().await;
```

A `StreamSource` declares replay, delivery, schema, watermark, and batch-bound
capabilities, then implements async `open`, `next`, and `close`. An ordinary
`StreamSink` implements async `open`, `write`, and `close`; transactional sinks
also expose epoch commit and recovery. Managed checkpoints bind source cursors,
operator state, and sink evidence to the plan fingerprint.

The managed local backend's crash-durability guarantee applies on Unix;
on Windows, parent directory updates are not synced, so recovery after a
power loss is not guaranteed.

Run the checked examples:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example 02_sql_join
cargo run -p calc-flow --example 04_continuous_runtime
cargo run -p calc-flow --example windowed_streaming
```

The [complete Rust inventory](../crates/calc-flow/examples/README.md) includes
counterparts for every runnable Python example, with connector features and
small stock/account commands.

The [continuous streaming guide](streaming-guide.md) explains cursor,
watermark, window, delivery, checkpoint, and recovery behavior across both
language surfaces.

## Projects and stores

`ProjectSpec` is the strict v3 data model. Use `validate_project` for a
`ValidationReport`, `compile_project` for a `BatchExecutionPlan`, and
`project_json_schema` for the generated schema.

`FileProjectStore` is the public async atomic local project store. Project JSON
is canonical; YAML is safe import/export only. Continuous recovery storage is
owned by `ManagedCheckpointRuntime`, while `LocalStateBackend` and
`CheckpointManifest` expose the data-only v3 state contract. Print the
canonical project schema with
[`crates/calc-flow/examples/export_schema.rs`](../crates/calc-flow/examples/export_schema.rs).

## Errors and cancellation

Public operations return `calc_flow::Result<T>`. `CalcFlowError` defines typed
variants for invalid arguments/documents, compilation errors,
execution/provider failures, checkpoint errors, I/O paths, cancellation,
closed stream edges, and private runtime task-panic capture. Private capture
keeps `TaskPanicked.message` valid UTF-8 and at most 1,024 bytes including its
ellipsis; non-string panic payloads use a fixed message. Fallible
`ManagedCheckpointRuntime`, `StreamingRunner`, and `StreamingJob` operations
expose only `CalcFlowError::Streaming(StreamingError)`; contained panics use
`StreamingErrorCategory::TaskPanicked` and retain neither the internal task ID
nor the panic message.

`ExecutionOptions` carries cancellation/deadline controls. Operators receive a
`RunContext` and must check cancellation at safe work boundaries.
`RunContext::settings()` returns the immutable run settings and
`RunContext::deadline()` returns the optional absolute UTC deadline by
reference, allowing providers to observe the authoritative run context without
duplicating it.
