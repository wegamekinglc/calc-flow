# 符号事件窗口聚合：实现方案与 API note

状态：APPROVED — 2026-09-08 用户授权按已审查方案实施；功能发布状态以 PR 和
CHANGELOG 为准。本文保留实施任务及验收合同。

工作项 slug：`symbolic-event-window-aggregation`。
核对日期：2026-09-08。
源码基线：`464a480a7d87d7e5933b54afa1d7a30af9d18971`。
范围：Python symbolic 到既有 Rust 原生事件窗口的编译、执行与恢复接入。

本文集中保存需求、一次 API 冻结、实施任务和验收矩阵，避免多份合同
相互漂移。实施前的 API 审查以第 3–7 节为同一个冻结单元。

## 1. 目标和设计基线事实

让用户在一个 `Program` 中声明按固定 UTC tumbling/hopping 窗口分组的
`count/sum/min/max/avg`，并编译为已有 `WindowAggregateOperator`。
聚合在 Rust 中执行，Python 只负责不可变声明、静态分析和生成 project-v3。
窗口前后无状态表变换的精确 schema 通过必要的内部 Rust/PyO3 规划适配确认，
不在 Python 重写 DataFusion 的 nullable 优化规则。

设计基线源码确认了以下缺口；下文任务用于填补这些缺口：

- `python/calc_flow/symbolic/ops.py::WindowNamespace` 只有几何和分组参数，
  没有聚合列入口。
- `nodes.py` 中 `window_tumbling@1`、`window_hopping@1` 已有稳定声明身份。
  给旧节点补一个默认 `aggregates=[]` 会改变规范化字节和 digest。
- `analyzer.py::_window_table` 只推导窗口边界和分组列，将 lineage 设为
  `None`，却继承输入的 event-time/entity/sequence 排序事实。
  它还把输入加入 `_temporal_lineages`，触发 rolling 的非空 UTC 时间、
  entity 和 sequence 检查。这不符合原生窗口允许 null 时间和无分组的语义。
- `python/calc_flow/capabilities.py` 尚无 `window` operator capability；
  capability 文件不在 `symbolic/` 目录内。
- `symbolic/lower/` 已拆为 `program.py`、`planners.py`、`segments.py`、
  `strategies.py`。已有 relational DAG 和状态共享实现可以复用其边界模式。
- `python/tests/test_symbolic_lowering_rejects.py::test_sql_window_is_rejected_in_stream_mode`
  记录旧声明的 `unknown_primitive_version` 编译拒绝。
- `crates/calc-flow/src/config.rs::OperatorSpec::Window` 已支持严格
  `{"kind": "window", "spec": ...}` 配置。其构造器要求一个名为 `input`
  的 required table port 和精确 schema，并校验派生输出。
- `crates/calc-flow/src/operator/window.rs` 已实现全部五种聚合、watermark
  final 输出、hopping assignment、状态快照及恢复。
- [PR #243](https://github.com/wegamekinglc/calc-flow/pull/243) 已合并；它增加
  checkpoint segment 的字节等价和损坏检测测试，不是尚待完成的 codec 前置项。
  本地 `window.rs` 已包含对应保护测试。

旧符号路线图的完成状态不因本提案改变；本功能作为独立扩展实施。

## 2. 首版范围

支持的窗口路径为：

```text
exact-schema table input
  -> 已支持的无状态 project/filter/with_columns 和 row 表达式
  -> 一个 tumbling 或 hopping window aggregate
  -> 窗口输出，或无状态 project/filter/with_columns 和 row 表达式
  -> 一个或多个命名输出
```

同一个 Program 可以包含多个独立窗口、相同窗口的分支，以及窗口之外的
既有合法输出。每一条包含窗口的依赖路径最多有一个事件窗口，且窗口前后
均限无状态表变换。窗口之外的输出继续遵循其原有合同。

首版拒绝窗口路径上的 rolling、cross-section、join、另一个 event window、
array/matrix attachment、外部 stateful provider 和跨行 reduction。
即使 Rust 支持某些组合，也不在本次 symbolic 开放范围内。

不新增 session/calendar/timezone-local window、offset、early trigger、
更新/撤回输出、窗口级 allowed-lateness 或自定义聚合。几何以 Unix epoch
为 UTC 原点，边界为半开区间 `[start, end)`。
不新增 BatchKind、状态布局、manifest 版本、Studio REST 端点和执行引擎。

窗口的时间列首版必须可追溯为输入时间列的原样传递或纯重命名。
允许筛选和派生普通聚合输入列；不允许对窗口使用的时间列做算术、截断、
有损 cast 或其他会改变 watermark 坐标的变换。对于未能证明时间坐标保持
不变的表达式，分析阶段拒绝，而不是自动重写 watermark。
源绑定仍必须为所选时间坐标提供符合既有 runner 合同的 watermark。

## 3. API note：不可变聚合声明和用法

### 3.1 新声明类型

在 `python/calc_flow/symbolic/windows.py` 增加导出的纯数据容器：

```python
@dataclass(frozen=True, slots=True)
class WindowAggregate:
    function: Literal["count", "sum", "min", "max", "avg"]
    column: str
    output: str
```

构造器也执行严格校验，不能仅依赖类型标注：function 必须为上述精确字符串；
column/output 必须为非空精确 `str`；不接受 ColumnExpr、callable、SQL、
任意映射或可执行对象。该类型是声明数据，不持有 Batch 或运行时资源。
从 `calc_flow.symbolic` 导出，不新增顶层 `calc_flow.WindowSpec` 包装体系。

在既有 `window` namespace 添加以下纯构造函数：

```python
window.count(column: str, /, *, output: str) -> WindowAggregate
window.sum(column: str, /, *, output: str) -> WindowAggregate
window.min(column: str, /, *, output: str) -> WindowAggregate
window.max(column: str, /, *, output: str) -> WindowAggregate
window.avg(column: str, /, *, output: str) -> WindowAggregate
```

`count(column)` 只统计该列的非 null 值，不能省略 column，首版没有
`count(*)`。要统计所有成交，可选择一个保证非 null 的成交 ID；如需先计算
聚合表达式，应在窗口前用 `with_columns` 命名，再按列名引用。
`avg` 表示普通算术平均，不能解释成成交量加权均价。

### 3.2 扩展既有窗口入口

保留所有现有参数，两个函数都新增 keyword-only 参数：

```python
aggregates: Sequence[WindowAggregate] | None = None
```

- `None`（包含省略参数）保留旧 `@1` 声明、原规范化字节和 digest，继续
  作为不可执行的 declaration-only 窗口。不要把旧空聚合声明自动升级成分组去重。
- 非空 sequence 生成对应 `@2` 可执行声明。显式空 sequence 以
  `invalid_literal` 拒绝；该限制只约束新 symbolic API，不收紧原生 WindowSpec。
- 拒绝 str/bytes、mapping、set、生成器及不含 `WindowAggregate` 的 sequence。
  立即复制为 tuple，后续修改调用者的 list 不得改变节点、fingerprint 或输出顺序。
- group_by 保持声明顺序并复制为 tuple；聚合列表保持调用者声明顺序，不排序、
  不按 function/column 去重。两个相同 function/column 使用不同 output 是合法声明。
- group_by 中的名称不可重复，且不能是 `window_start` 或 `window_end`。
  聚合 output 不得重复或与分组键、两个保留边界名称冲突。
  output 与不再出现在结果中的普通输入列同名是合法的。

上述新增的几何关系、名称和可执行性检查约束 `@2`。`@1` 的构造/分析兼容路径
保留既有行为，不把历史上能构造但无法执行的旧声明偷偷迁移到 `@2` 校验规则。
第 4–7 节的新增执行事实同样针对 `@2`；旧有效声明的黄金向量保持独立。

### 3.3 目标示例

以下是待实现 API 示例；不将它标记为当前已可运行：

```python
from calc_flow import Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, table_input, window

trades = table_input(
    "trades",
    schema=[
        Field("ts", "timestamp[us, UTC]"),
        Field("symbol", "string"),
        Field("trade_id", "uint64", nullable=False),
        Field("quantity", "int64"),
        Field("price", "float64"),
    ],
)

minute = window.tumbling(
    trades,
    event_time="ts",
    size_micros=60_000_000,
    group_by=["symbol"],
    aggregates=[
        window.count("trade_id", output="trade_count"),
        window.sum("quantity", output="volume"),
        window.min("price", output="low"),
        window.max("price", output="high"),
        window.avg("price", output="avg_price"),
    ],
)

summary = minute.with_columns(
    FeatureSet([("price_range", minute["high"] - minute["low"])])
)
program = Program(
    "minute-bars",
    inputs=[trades],
    outputs=[("minute", minute), ("summary", summary)],
)
plan = program.compile_stream(Runtime())
```

两个输出共享同一个窗口状态所有者。运行时再用既有 `SourceBinding`、
`SinkBinding`、`StreamingRunner` 和可选 `ManagedCheckpointRuntime` 绑定执行。
上述 `table_input` 不需要为窗口伪造 entity/sequence 或非 null 时间声明。

## 4. API note：类型、名称与原生语义

### 4.1 精确类型矩阵

首版输入范围为“既有 symbolic portable 字段类型”和“原生窗口支持矩阵”的
交集，不顺带扩展全局 Arrow 类型语法。原生支持的任意嵌套类型 count、
timestamp 秒/纳秒等更宽范围，不因此成为 symbolic 可声明类型。

| 用途       | 首版输入类型                                                         | 输出类型                     | Nullable                |
|------------|----------------------------------------------------------------------|------------------------------|-------------------------|
| event_time | timestamp[ms]、timestamp[us]、timestamp[us, UTC]                     | 两边界为 timestamp[us, UTC]  | 输入可 null；边界 false |
| group_by   | bool、整数、float32/64、string/large_string、date32/64、us timestamp | 保留所选输入字段的类型和顺序 | 保留输入字段声明        |
| count      | 任意现有可声明并可由 project-v3 表示的字段类型                       | uint64                       | false                   |
| sum        | int8/16/32/64                                                        | int64                        | true                    |
| sum        | uint8/16/32/64                                                       | uint64                       | true                    |
| sum        | float32/64                                                           | float64                      | true                    |
| avg        | 上述有符号/无符号整数及浮点类型                                      | float64                      | true                    |
| min/max    | 与 group_by 相同的受支持类型                                         | 保留输入类型                 | true                    |

上表 us timestamp 仅指 timezone-naive 或 UTC。`timestamp[ms]` 可做时间字段和
count 输入，不能做当前原生矩阵不支持的 group/min/max。time32/time64
可做 count 输入，不能做 event_time/group/sum/min/max/avg。
naive timestamp 按既有原生 UTC 坐标处理，不引入本地时区推断。

输出 schema 严格为：`window_start`、`window_end`、按声明顺序的 group_by、
按声明顺序的 aggregate outputs。没有隐式原始列、索引、sequence 或内部辅助列。
分组字段遵循原生字段克隆语义；目前 symbolic Field 只表达 name/type/nullable，
不额外承诺任意 Arrow metadata 的跨声明保留。
最终 schema 由 Rust 构造的窗口端口再校验一次；Python 推导矩阵必须由与
原生手工配置的定向对照测试保护，不建立另一套数据计算实现。

窗口前和窗口输出后的 project/filter/with_columns 片段先完成静态字段、类型、
来源和稳定路径检查，再通过 `symbolic/lower/schema.py` 按实际 row-only/CSE
图逐 stage 传播 Arrow schema。原生规划的字段名、顺序和 dtype 必须与冻结的
声明一致；nullable 以原生规划为准。窗口 group 字段复制这个精确输入 nullable，
窗口边界和 aggregate 本身的输出规则仍以本节类型矩阵为准。

`Runtime._infer_symbolic_expression_schema` 调用私有 PyO3
`_infer_expression_schema(select, filter, input_schema)` 取得 Arrow schema。
内部 Rust 适配对纯列投影复用 stream 快速路径；其他表达式在只有声明 schema、
没有用户行的空 MemTable 上运行 DataFusion physical planning，读取物理计划的
schema。不得 collect/execute 用户行、open source 或执行注册 UDF，不在 Python
复制 CASE、boolean、coalesce 等优化规则。规划失败或字段名/dtype 不符必须在
分析阶段拒绝，不能拖到 source open 后暴露端口不匹配。

### 4.2 几何与运行结果

- size_micros、slide_micros 必须是精确 Python int，范围为 `[1, 2^64-1]`；
  拒绝 bool、float、timedelta、字符串和超界值。采用整数算法，禁止浮点换算。
- hopping 要求 `size % slide == 0`，且 `1 <= size / slide <= 1024`。
  单次 assignment 的有符号 EventTime 越界仍由原生运行时检查，不能宣称
  所有依赖数据值的溢出都能在 source open 前判断。
- null 时间行整体丢弃并更新原生 null-event-time 指标，不进入任何聚合。
- lateness 按具体 assignment 的 `end <= input_watermark` 判断。hopping
  同一行可能部分 assignment 关闭、部分仍开放；只丢弃已经关闭的部分。
- 输入数据不产生提前聚合结果。watermark 等于窗口 end 时关闭该窗口；
  end-of-input 通过已有 `on_end` 刷新剩余未输出窗口，不伪造公共控制注入 API。
- 空窗口不补行；已有分组内聚合输入全 null 时 count 为 0，其他四种为 null。
- 每次关闭输出按原生 `WindowKey(start, end, stable_group_key)` 排序。
  group key 的 null、NaN、signed zero 和编码排序复用原生规则，不改为 Python
  或普通字符串排序；保留原生输出 chunking 和序列语义。
- 溢出、浮点/NaN、整型 avg 累加精度、64 KiB 分组键上限及统计行为均以原生
  结果为准。本工作不修订它们，不增加 Python 逐行累加器。

### 4.3 现有 compile_stream lateness 参数的归属

`Program.compile_stream` 已有 `allowed_lateness_micros` 和 `late_policy`，
当前由 rolling/cross-section 消费。新窗口不接收、序列化或解释这些参数，
其迟到 assignment 始终采用既有原生 drop 行为。

为了防止窗口用户设置参数后误以为已经生效：纯窗口/无状态 Program 若传入
非默认 `(0, "error")` 参数，编译以 `capability_mismatch` 明确拒绝，指出
该参数不适用于 event window。这里的默认 `"error"` 不表示窗口 late 行报错。
若同一 Program 有独立且合法的 rolling/cross-section 输出，则非默认参数
继续只传给这些算子；窗口配置保持不变，并在 explain 中标明两种政策的归属。
无窗口 Program 的既有行为不变。

不扩展 `Program.analyze/explain` 的方法签名：二者只接受既有 runtime/mode，
不检查某次 compile 调用的 lateness 实参。explain 明示使用默认编译参数，
并说明 rolling/cross-section 参数与窗口固定 drop 政策的归属，不声称展示
另一场非默认 compile 的实际配置。非默认参数是否有合法消费者、参数类型与范围
由 compile 检查。合法但没有消费者的非默认值使用
`<program.name>.compile_stream.allowed_lateness_micros` 或 `.late_policy`
路径与 `capability_mismatch`；先报 allowed_lateness_micros，再报 late_policy。

## 5. API note：身份、来源事实和能力

### 5.1 版本化身份

继续使用 `calc_flow.symbolic.declaration.v1` 编码，不修改 program fingerprint
算法。对窗口 primitive 的属性校验按 `(name, version)` 区分：

- `@1`：只接受旧 attrs，默认值和所有有效声明字节保持原样。
- `@2`：接受相同几何 attrs，加上必需的有序 `aggregates`。
  其编码为 `CSeq(CMap(function=CStr, column=CStr, output=CStr), ...)`；
  map 按既有 canonical key 规则编码，sequence 保留顺序。
- digest 覆盖 primitive/version、完整输入节点 digest、event_time、几何、
  有序 group_by 和有序聚合项；聚合 output 名称同样是语义身份的一部分。
- 不按几何合并不同聚合列表，不把互为子集的窗口自动扩为一个超集状态。
  相同声明的独立构造实例只要 digest 相同就共享；相同几何但不同输入筛选
  或聚合列顺序的声明不共享。
- 旧 `@1` compile 拒绝保留回归覆盖；新 `@2` 编译成功另设测试。
  未知版本仍 fail closed，绝不能仅按 op.name 接受未来版本。

冻结两种旧窗口的 byte/digest golden，以及各一份新 `@2` golden。
同时测试嵌入 Program 后旧 fingerprint 不变、序列变更会改变新 fingerprint。
旧构造域的 golden 还应覆盖非整倍数/overlap=1025 的 hopping，以及重复或
保留 group 名称：这些历史 @1 声明原本能够构造，不能因新增 @2 校验而拒绝。

### 5.2 新的行来源和 finality 边界

新窗口 TableFacts 的 lineage 使用独立的带类型身份，例如内部
`@dataclass(frozen=True, slots=True)` 的 `_WindowRowOrigin(digest)`；
窗口输出列以此为来源。TableFacts/ColumnFacts/ArrayFacts 的内部 lineage
类型一致扩展，已有 source lineage 字符串保持旧行为；新窗口身份与任何输入
名字字符串都不能相等。`window:<digest>` 仅为 explain 的展示文本，不能作为
实际相等性判断键。不得复用输入 lineage，也不得用 `None` 冒充“可以和任意列对齐”。

state 包含窗口标记；event_time、entity_by、sequence_by 清空为 `None/()/()`。
另由窗口计划/explain 表达其原生确定输出顺序，不把 `window_end` 伪装成可直接
用于 rolling 的逐行排序证明。无状态投影/派生列继承正确的窗口来源；filter
遵循既有筛选来源检查，禁止将筛选前后的数组依赖隐式对齐。

跨来源 ColumnExpr 混用也必须拒绝，不能只检查 attach_columns。例如
`minute.with_columns(FeatureSet([("raw", trades["price"])]))` 和
`minute["avg_price"] + trades["price"]` 都是无效来源关系。
另设对抗用例：先构造窗口，再创建名字恰为该窗口 `window:<digest>` 展示文本
的独立 table input。即使字段类型、行数或名称相同，来自该 input 的原始列
仍不能与窗口输出混合；不能通过用户可控名称伪造来源证明。
窗口前后数组 attachment 全部在首版拒绝，且必须从窗口依赖祖先识别，不能
因为 project/filter 包装后丢失边界。
在进入 `_from_columns_array` 等会把 lineage 用作 shape 维度的旧路径之前，
先拒绝窗口 array 依赖；内部 origin 对象不能被塞进只允许整数/符号字符串的
array shape。展示文本与来源相等性始终分离。

将“需要 rolling/cross-section 的输入排序证明”和“需要 event window 的
时间字段及 watermark 能力”分别跟踪。窗口不再把输入加入 rolling 的
`_temporal_lineages`；同一输入若也用于 rolling，其 rolling 分支仍必须满足
原有排序要求。无分组窗口、nullable 时间、乱序到达但窗口未闭合的行应被接受。

### 5.3 capability、analyze 和 explain

新增 operator capability `kind="window", version="1"`，与 project-v3
kind 一致；注意 Rust 运行配置中的诊断 kind 是 `window_aggregate`，不能
把这个字符串误写到 project operator.kind。

能力事实为 stream-only、required table input/output、
`group_final_append_only`、stateful、checkpointed_stateful、
requires_watermark、deterministic、replay_safe、microbatch_invariant。
window 本身 `requires_datafusion=False`；前后 expression 按既有能力要求
DataFusion。state_version/state_layouts 复用当前原生版本 1，不声明新布局。

schema version 3 的现有 capability 字段已经足够，添加条目不默认升级 schema。
更新严格解析/排序/快照测试；如果发现真实序列化形状变化，才同步对应消费者。

analyze 和 compile 都必须检查所需 window 能力及精确版本，不能出现 analyze
声称可执行而 compile 因能力缺失失败。batch 模式报告 `unsupported_mode`。
缺能力、能力字段为 unproven、错误 state layout 或不支持 stream 时，报告
`capability_mismatch`。禁止因表达式能力存在而绕过窗口能力检查。
上述一致性针对声明、mode 和 capability；仅 compile 接收的选项按第 4.3 节
检查，不要求 analyze 在未获得实参时判断这些调用参数。

explain 至少可见：声明版本、几何/overlap、分组/聚合顺序和 schema、独立
lineage、finality 边界、late-assignment/null 时间策略、唯一状态节点数、
共享输出数、window checkpoint layout。显示状态量依赖活跃窗口和分组数，
未知行数/字节量标为 unknown，不推断未经证明的 cardinality 或固定内存上限。

## 6. 编译和 lowering 实现

### 6.1 结构和文件职责

| 文件或模块                               | 改动职责                                                      |
|------------------------------------------|---------------------------------------------------------------|
| symbolic/windows.py、ops.py、__init__.py | WindowAggregate、namespace helper、签名、严格不可变声明和导出 |
| symbolic/nodes.py                        | 窗口 @1/@2 attrs 校验与 canonical identity                    |
| symbolic/analyzer.py                     | 类型矩阵、模式、来源、时间坐标、状态路径和稳定诊断            |
| capabilities.py                          | 原生 window capability 和快照                                 |
| symbolic/lower/planners.py               | 不可变 WindowPlan、digest 去重、精确输入/输出 schema          |
| symbolic/lower/program.py                | window 能力检查、模式分派、混合输出、compile cache 接入       |
| symbolic/lower/strategies.py             | 状态边界 DAG、窗口前后 fragment、边和输出绑定                 |
| symbolic/lower/segments.py               | 行变换边界识别、字段保留、必要的类型/schema 辅助              |
| symbolic/lower/schema.py                 | 实际 row-only/CSE 片段的逐 stage 原生 Arrow schema 传播       |
| symbolic/optimizer.py                    | 窗口 finality 优化屏障、共享和资源 explain                    |
| Python Runtime/builder/project 适配      | 复用 project-v3；内部 schema 规划与独立有界 schema cache      |
| Rust expression/DataFusion 及 PyO3       | 内部 schema 规划适配、stream 投影一致性及对应定向测试         |
| Rust config/window                       | 复用现有合同；仅对定向测试暴露的原生缺口作必要修复            |

如果现有 strategies.py 继续增加会混淆职责，可把新增窗口策略放入
`lower/event_windows.py`；只移动新增逻辑，不借机重构既有 join/rolling 路径。
避免为了本功能公开一个全新 Python builder.window API；手工对照图可直接用
现有 ProjectDocument 校验和内部 graph 编译路径构造；测试可用
`PipelineBuilder._from_json(...).compile_stream(...)`，不把这个私有测试入口
宣传为新增公共 API，也不混用需要完整 connector bindings 的 project 编译入口。

### 6.2 lowering 算法

1. 分析全部输出的依赖 DAG；先确定每一条窗口路径是否属于首版合法组合。
   按拓扑顺序收集 `@2` 窗口，生成 `digest -> WindowPlan` 表。
2. 为每个可达 table input 建立可共享入口；把窗口前 project/filter/with_columns
   降低为既有 expression fragment。保留时间、group 和所有 aggregate 引用列，
   对派生列物化，按实际 row-only/CSE stage 的原生规划结果固定精确 schema，
   正确处理替换列和列重命名，不手工推测 nullable。
3. 每个唯一完整窗口 digest 只创建一个 native window node。状态共享按完整
   digest 判断；物理节点 ID 遵循既有 project-v3 ASCII 标识符和 64 字符上限。
   使用 digest 前缀或哈希缩短名称，检测所有已占用名称（含用户 source/output）
   并确定性加后缀，不能因截断碰撞静默覆盖或合并不同状态。
4. 窗口 output 转成带独立来源事实的虚拟表输入，复用无状态 fragment lowerer
   编译后续变换，并对实际后置 stage 逐个确认原生输出 schema。
   窗口的所有分支连向同一个实际 `output` port。
5. 合并各窗口 fragment 和不依赖窗口的原有输出图；重新定位边、命名输出和
   source bindings，验证一个输入只连接一个 writer、无环、无遗漏、无悬空节点。
   共享只能复用相同语义源，不能因为节点内容相似就混接独立 input。
6. 将数据文档交回既有 Rust compile_stream 做最终 schema/port/config 校验。
   source/sink 的 I/O 生命周期此时尚未开始。

窗口节点的 operator 片段固定为：

```json
{
  "kind": "window",
  "spec": {
    "event_time_column": "ts",
    "group_by": ["symbol"],
    "geometry": {"kind": "tumbling", "size_micros": 60000000},
    "aggregates": [
      {"function": "count", "column": "trade_id", "output": "trade_count"},
      {"function": "sum", "column": "quantity", "output": "volume"}
    ]
  }
}
```

外层 node 使用一个 required table `input`，schema 为窗口前 fragment 的
精确输出。派生 output 可以沿用现有 `output_ports=[]` 由 Rust 推导的机制；
符号层推导的 schema 必须在定向测试中与派生端口逐项相等。

### 6.3 优化限制和 cache

窗口是不可跨越的 finality 和 cardinality 边界。首版不把窗口后过滤推到窗口前，
即使 predicate 只引用 group key；不把窗口前筛选推到聚合后；不跨边界合并
CSE、rolling state 或 array materialization。窗口前投影裁剪必须保留所有
隐式依赖列，窗口后输出筛选不得改变其他共享分支。

状态共享的判断依赖完整声明，而不是最终 output 名称或只看 SQL 文本。
compile cache 继续使用现有 runtime/session/revision、精确声明和 schema、
mode、选中 operator/provider/UDF 版本。窗口能力和 operator version 纳入
已有 key 的 operator 集合，验证不同 aggregate、schema、能力 revision 不能
命中旧缓存。缓存不可持有正在运行的窗口状态；每次启动 job 有自己的 state。
原生 StreamExecutionPlan 在 runner 启动时被消费，因此 symbolic stream cache
只存成功编译的不可变 project JSON，每次 compile 都创建独立 owning plan。
batch 继续复用其不可变执行计划。AC-17 必须执行两个 job，不能仅比较缓存对象身份。

Runtime 另持有独立 schema cache，最多 128 个不可变 Arrow schema；key 为有序
select、filter 和精确输入 Arrow schema 序列化。仅成功原生规划可写入缓存，
失败不可缓存为有效 schema。成功 provider/UDF/lifecycle 注册同时清空该缓存和
compile cache。schema cache 不持有 DataFusion session、用户行或运行中的状态。

## 7. 严格诊断与恢复

### 7.1 稳定错误边界

立即可判定的参数错误在声明构造时使用现有 TypeError/ValueError 边界并带
完整 namespace 参数路径；依赖 schema/capability 的错误在 analyze/compile
使用 `AnalysisIssue` 和 `{path}: {code}: {message}` 的 CompileError。
不解析 Rust 英文错误字符串来猜字段路径。

下表 `P` 表示到达该窗口节点的实际输出路径，例如 `outputs.minute`。
包装节点按现有路径 grammar 递归展开；共享节点诊断按稳定输出/声明遍历顺序
选择路径，同一无效节点不因 cache 命中而丢失诊断。

| 错误                            | 诊断位置                                       | 既有 code                        |
|---------------------------------|------------------------------------------------|----------------------------------|
| bool/非整数/零/负数/超 u64 几何 | 构造路径 .size_micros 或 .slide_micros         | 类型异常或 invalid_literal       |
| 非整倍数或 overlap > 1024       | 构造路径 .slide_micros                         | invalid_literal                  |
| 空聚合列表                      | 构造路径 .aggregates                           | invalid_literal                  |
| 无效函数/字段名称声明           | 构造路径 .aggregates[i].function/column/output | invalid_literal                  |
| 重复/保留名称/分组输出冲突      | 构造路径 .group_by[i] 或 .aggregates[i].output | duplicate_name                   |
| 缺失时间列                      | P.window_tumbling.event_time                   | unresolved_type                  |
| 不支持时间列类型                | P.window_tumbling.event_time.dtype             | unsupported_type                 |
| 缺失或不支持分组列              | P.window_tumbling.group_by[i][.dtype]          | unresolved_type/unsupported_type |
| 缺失或不支持聚合输入            | P.window_tumbling.aggregates[i].column[.dtype] | unresolved_type/unsupported_type |
| 时间坐标变换无法证明            | P.window_tumbling.event_time                   | capability_mismatch              |
| batch 执行新窗口                | P.window_tumbling                              | unsupported_mode                 |
| 不合法 stateful/array 组合      | 违反边界的 operand 路径                        | capability_mismatch              |
| 原始列/数组与窗口输出混合       | 违反边界的 operand .lineage                    | schema_mismatch                  |
| 缺少或不兼容 window capability  | P.window_tumbling                              | capability_mismatch              |

hopping 使用相同结构并替换 primitive 名称。公开 WindowAggregate 构造器的
单值错误以 `WindowAggregate.<field>` 定位；namespace helper 使用其调用路径；
组合名称碰撞由具体窗口调用给出聚合索引。

Project-v3 的 `additionalProperties: false` 与 Rust WindowSpec 校验继续生效。
额外 allowed_lateness/trigger/codec 字段必须被拒绝。直接手写 project-v3 的
诊断仍遵循现有 `graph.nodes[i]...` 合同；只有 symbolic API 要求上述输出路径。
首版不为统一错误文案而扩大 Rust 诊断重构。

### 7.2 checkpoint/restart

不新增 Python snapshot，不复制 native state，也不在 Python 缓存未闭合输入行。
使用既有 manifest v3、operator layout 1、配置 hash、state schema fingerprint、
输入 cursor 和 per-output delivery 协议。

同一 Program 的独立重建和重编译必须产生相同物理 state node ID 与 graph
fingerprint；输出分支不能增加窗口 state owner。改变 geometry、group/aggregate
顺序、输入字段类型、聚合 output 或图拓扑后，按既有 fingerprint/recovery
协议拒绝旧 checkpoint，不提供自动迁移或跨图恢复。

测试通过托管 runner 在窗口未闭合时提交 durable checkpoint，终止旧 job，
重新创建 Runtime、Program、plan、runner，再按 cursor 恢复并关闭窗口。
不得仅在内存中调用同一个 operator 的 restore 就声称验证了 durable restart。
损坏 segment、配置/布局/fingerprint 不匹配必须在 source 恢复 open 之前失败。

对照 symbolic 图与手工 WindowSpec 图各自的恢复结果，不要求两个不同 graph
fingerprint 的图互相加载 checkpoint，也不比较它们的 operator ID 或 manifest
原始字节。只有相同图重建的身份应一致。

最终结果的重复保证服从选定 sink delivery：普通 sink 不自动获得 exactly-once。
测试未闭合窗口恢复时可使用无前序聚合输出的收集 sink；涉及已输出后故障的
去重断言必须使用既有事务 sink/交付证明测试设施。

## 8. 实施任务与交付次序

建议一个独立 feature 工作项、一个最终实现 PR，按下面六个可审查阶段提交。
中间提交可以保留功能未开放状态，但最终 PR 不留 stub 或声明成功却无法执行的
`@2` 路径。分支名建议 `feature/symbolic-event-window-aggregation`。

### EW-0：冻结合同和建立 RED

- 审查本文第 3–7 节，固定 API、类型/nullability、@1/@2、路径、来源、首版边界。
- 捕获旧 @1 golden；新增最小 @2 编译/执行失败测试。当前应因 aggregates 参数
  缺失或窗口 lowering 未实现而失败，记录命令和对应预期原因。
- 先写无 entity/sequence、nullable 时间、跨原始 lineage、共享 state owner 的
  回归用例，防止仅补表面参数后遗漏核心语义。
- 输出：冻结合同和针对当前基线的 RED 记录，不执行性能测试。

### EW-1：声明与身份

- 实现 WindowAggregate、五个 helper、两个窗口的 aggregates 参数。
- 实现窗口 primitive 的版本化 attrs 校验；保留旧 golden。
- 验证严格数据输入、caller immutability、空/重复输出、几何边界和顺序身份。
- 完成条件：声明和 canonical identity 测试通过，不改编码版本或项目版本。

### EW-2：能力与分析

- 增加 capability，分离窗口时间证明与 rolling 排序证明。
- 推导完整输出 schema、新 lineage，拒绝越界组合和时间坐标变换。
- 静态检查通过后逐 stage 确认窗口前后无状态片段的原生 schema；字段名/dtype
  必须与冻结声明相符，nullable 采用原生规划结果，不复制表达式优化器规则。
- analyze/compile/explain 共用事实和诊断，覆盖缺失能力与模式拒绝。
- 完成条件：声明与原生 schema 分析测试通过；现有 rolling 的排序拒绝保持有效。

### EW-3：执行 lowering 和共享

- 增加 WindowPlan、native window 节点、窗口前后 fragments、独立输出拼图。
- 处理多 input/多 output、重复声明、节点名称冲突和精确端口 schema。
- 接入内部 Rust/PyO3 schema 规划适配及独立有界缓存，补充纯列投影和物理规划
  schema 的定向对照；规划全过程不打开源、不执行用户行或注册 UDF。
- 加入 finality 优化屏障和 compile cache，补充 explain 共享信息。
- 用手工 project-v3 图验证两种几何的端到端执行，保留 @1 拒绝测试。
- 完成条件：schema/值/null/顺序对照和一个状态所有者断言通过。

### EW-4：流边界与恢复

- 覆盖固定控制消息序列下不同 Batch 切分、局部迟到 hopping assignment、
  null 时间、watermark=end、EOI 和未闭合 checkpoint/restart。
- 检查 manifest 实际 state owner 数、重建节点身份和恢复前拒绝。
- 覆盖同一 Runtime 多 job 状态隔离、不同窗口声明不错误共享。
- 仅当测试发现原生缺口时补相应 Rust window/config 测试，再最小修改内核。
- 完成条件：symbolic 与原生图各自的不中断及恢复结果一致，未引入提前输出。

### EW-5：文档与最终审查

- 更新 `docs/symbolic-api.md` 的 event-window 能力表、API、版本兼容和限制；
  `docs/symbolic-design.md` 的编译/来源/状态边界；`docs/symbolic-workflows.md`
  的窗口流程。必要时更新 `docs/python-api.md` 和 `docs/streaming-guide.md`。
- 更新旧 `.codex/artifacts/specs/symbolic-relational-dag.md` 等规范性文件中的
  “event window declaration-only”边界说明，注明只开放本文首版路径；保留
  历史阶段的验收记录，不追溯改写旧功能状态。
- 增加独立 `examples/symbolic_event_window.py`，含确定的小样本输出、源/汇及
  必要 watermark，登记到 `examples/README.md` 和现有示例清单/检查入口；
  不占用当前工作区正在新增的编号示例。
- 实现完成后才写 CHANGELOG 的功能记录；本方案交付不声称功能已发布。
- 确认 project-v3/OpenAPI/TypeScript 没有无关漂移。仅真实合同变化才重新生成。
- 由最终 specialist reviewer 审查实现与本文验收映射；阻断发现修复后复审。

依赖顺序：`EW-0 -> EW-1 -> EW-2 -> EW-3 -> EW-4 -> EW-5`。
测试作者可在接口冻结后独立准备原生对照和恢复 fixture；不并行修改同一文件。
不以乐观工期代替阶段验收；主要工作量集中在 EW-2 的来源证明和 EW-3 的 DAG 接入。

## 9. 验收矩阵

新增集中测试模块建议为 `python/tests/test_symbolic_event_windows.py`、
`test_symbolic_event_window_lowering.py` 和 `test_symbolic_event_window_recovery.py`。
必要 fixture 各自放在聚焦模块内，不新建共享 conftest.py。

| 编号  | 测试内容                                                   | 通过证据                                                |
|-------|------------------------------------------------------------|---------------------------------------------------------|
| AC-01 | 旧 @1 含重复/保留 group、非整倍数/1025-overlap             | bytes/digest/Program fingerprint 不变；执行仍拒绝       |
| AC-02 | @2 声明、未知版本、不可变输入与有序聚合                    | 新 golden 稳定；list 改动不影响声明；顺序变化改变身份   |
| AC-03 | 两个 UTC 1 分钟窗口、分组、五种聚合                        | 与手工图的 Arrow schema、行值、null 和行顺序一致        |
| AC-04 | tumbling/hopping，各种预处理和后处理                       | 派生聚合输入与结果表达式正确；窗口前后 filter 未互换    |
| AC-05 | 类型矩阵、count null、全 null 聚合、null group             | 输出类型及 nullable 与原生一致；无空窗口补行            |
| AC-06 | 几何边界和非法配置                                         | 1024 overlap 通过，1025/非整倍数/零/负数/bool/超界拒绝  |
| AC-07 | 缺列、不支持类型、重复/保留名称、未知配置字段              | 稳定路径/错误码；source open 计数为 0                   |
| AC-08 | lineage 和组合边界                                         | 原始列/数组、伪造 input 名称、wrapped attachment 均拒绝 |
| AC-09 | 无分组、nullable 时间、无 entity/sequence、未闭合乱序输入  | 原生可接受的输入不被 rolling 排序要求误拒绝             |
| AC-10 | 同一声明两个输出、独立构造同 digest、不同声明              | 图和 manifest 中每个唯一窗口恰有一个 state owner        |
| AC-11 | 相同逻辑消息轨迹按单批/逐行/不规则/空批切分                | 每种切分与对应手工图一致；最终有序结果跨切分一致        |
| AC-12 | 窗口未闭合时 durable checkpoint/restart                    | 新 Runtime/runner 恢复后结果与不中断运行一致            |
| AC-13 | schema/spec/layout/fingerprint 不匹配及损坏 segment        | source 恢复 open 之前拒绝，不存在部分恢复后继续运行     |
| AC-14 | null 时间、watermark=end、EOI、hopping 局部迟到 assignment | 关闭前无输出；边界/指标/最终结果与原生图一致            |
| AC-15 | batch/缺能力/不兼容能力/时间列变换/无效 lateness 参数      | 声明/能力错误分析编译一致；lateness 实参仅 compile 校验 |
| AC-16 | 独立窗口、窗口外既有输出、ID 冲突和多 input                | 无误连/重复 source/多 writer；保留每个命名输出          |
| AC-17 | compile cache 与同图重建                                   | 键变化无旧结果；稳定图可恢复；多 job 不共享可变状态     |

### 9.1 最小确定性对照数据

使用 `2026-01-01T00:00:00Z` 为基准；按下列顺序到达，先不推进至窗口结束。
quantity 为 int64，price 为 float64；非 null 时间行的 trade_id 均非 null。

| 相对秒 | symbol | quantity | price |
|--------|--------|----------|-------|
| 5      | A      | 10       | 100   |
| 35     | A      | null     | 102   |
| 59     | A      | 30       | null  |
| 15     | B      | null     | null  |
| 60     | A      | 7        | 110   |
| 95     | A      | 13       | 90    |
| null   | A      | 999      | 999   |

推进 watermark 至 60 秒后，第一个窗口恰好输出 A、B 两组：

- A：trade_count=3、volume=40、low=100、high=102、avg_price=101。
- B：trade_count=1、volume=null、low=null、high=null、avg_price=null。

推进至 120 秒后第二窗口 A 为：2、20、90、110、100。
null 时间行不贡献聚合，null-event-time 计数为 1。
另外将 count 输入替换为 price，验证它统计非空价格而非所有成交。

hopping 另用 size=120 秒、slide=60 秒，含 epoch 前时间和半开边界点。
先推进 watermark=60 秒，再送 t=30 秒的行：`[-60,60)` assignment 应丢弃，
`[0,120)` assignment 仍应累计。由原生图验证 assignment 计数，不能将它按
“整行全丢弃”实现。

### 9.2 比较方法和恢复设施

手工图使用独立手写 `WindowSpec` project-v3 数据，不能调用新 symbolic
lowerer 或共用其聚合构造辅助函数生成预期配置。两边使用相同 source/sink
能力、进度策略、运行参数和逻辑数据顺序；不以 SQL rolling window 作为 oracle。

在 watermark、barrier 和 EOI 的同一逻辑位置切分数据，只改变相邻 Data 的
Batch 边界；不可让自动 watermark 因切分提前后还声称测试的是相同消息轨迹。
同时覆盖一个 Batch 内多个 Arrow record batches。

结果按每个命名输出的发出顺序拼接后比较，不先排序，不比较物理 Batch 大小或
计时。浮点测试包含普通有限值和 NaN/signed zero；使用 Arrow mask/必要的
位级检查，避免 `NaN != NaN` 导致错误结论。native 与 symbolic 不同节点 ID 的
Batch metadata 不要求字节相同，但各自序列必须符合原生交付合同。

指标在相同切分和相同消息轨迹下比较原生与 symbolic；跨切分只断言可保持的
统计量，例如 null 行和迟到 assignment 总数。affected_batches、任务耗时和
chunk 数可能依切分变化，不设伪等价门禁。

恢复 fixture 复用 `test_symbolic_stream_late_and_recovery.py` 的 scripted
source/cursor 和 managed-checkpoint 模式，使用事件/确认握手等待 checkpoint
已发布；不靠固定 sleep 猜测状态。断言 manifest 中真实的 window state entry，
不是只断言 explain 出现 `state_stages 1`。

## 10. 验证范围和完成条件

每个行为变更先记录 focused RED，再实现并跑对应 GREEN。纯方案/文档阶段
只检查结构、路径/链接、Markdown、换行和 diff，不构建 native 或运行上述未来测试。

实现期间按阶段选择测试文件/用例，不默认运行所有 symbolic 测试。全部新窗口
定向测试在最终本地验收时应有一次通过记录；只在修改或出现新问题后重跑相关项。
准备好的 Python/native 环境中，最终窗口测试命令为：

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" uv run pytest -q \
  python/tests/test_symbolic_event_windows.py \
  python/tests/test_symbolic_event_window_analysis.py \
  python/tests/test_symbolic_event_window_lowering.py \
  python/tests/test_symbolic_event_window_row_schema.py \
  python/tests/test_symbolic_event_window_schema_planning.py \
  python/tests/test_symbolic_event_window_recovery.py
```

格式/lint 只检查实际修改的 Python 文件。若需要重建 PyO3，按仓库规则把构建
和环境输出留在 target 下；不能遗留 source 中的 `_native*.so`。
内部 Rust/PyO3 schema 规划适配需要相应定向测试：复用原生 stream 纯列投影，
确认物理规划的字段名/dtype/nullable、实际 CSE stage 传播、缓存成功/失败边界和
注册失效。Rust 表达式规划用例可按 `infer_stream_schema` 名称过滤运行；PyO3
通过对应绑定测试及上述 schema-planning Python 用例覆盖，不为此运行整个 workspace。
若另外修改窗口内核，则按受影响范围追加对应测试，例如：

```bash
cargo test -p calc-flow --test window_compile
cargo test -p calc-flow --test window_tumbling
cargo test -p calc-flow --test window_hopping
cargo test -p calc-flow --test window_state
cargo test -p calc-flow --lib operator::window
```

上面是可选择的窗口测试清单，不要求每次全部重跑。未修改窗口内核时复用原生
结果进行 Python 对照；schema 规划适配仍运行对应定向检查，不为本功能重跑
codec、整个 Rust workspace 或本地性能矩阵。
必要时运行受影响旧用例，如 @1 拒绝、rolling 排序、capability 严格解析和
compile-cache 回归；记录具体用例名，不扩成无关全仓套件。

检查生成合同和 whitespace：

```bash
git diff --exit-code -- \
  schemas/project-v3.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

完整回归、Linux/Windows、Rust 90% 联合 coverage（含 connector services）、
Studio backend 85% coverage 和例行性能门禁交由 GitHub CI。未运行本地覆盖率
不能当作这些门禁已通过。没有新 REST 或项目合同变化时不重做 Studio 功能。

提交/PR 的 Test plan 附 AC 映射、RED/GREEN、基线/最终 SHA 和未运行范围。
常规交付只读一次非阻塞 CI 快照；只有后续明确要求最终 CI 或 merge 才等待。
合并仍须明确授权、最终 specialist review 无阻断，以及全部 required checks
通过。本工作当前已获实施和创建 PR、修复全部问题的授权；未获合并授权。

实现完成的判据是：全部 AC 有证据、文档与导出同步、旧身份未漂移、无 Python
数据执行、每唯一窗口一个 state owner、托管恢复与原生对照通过、最终审查完成。
仅能编译或仅有正常流输出不足以完成该工作项。

## 11. 资源目标、风险和依赖

每个唯一声明只有一份 native 状态；Python 的工作量和声明 DAG 大小相关，
不与输入行数相关。复用现有 bounded task/edge/backpressure、输出 chunking
和 checkpoint 管理；不添加 Python 无界缓存、额外输入收集或按行调用。

不能把“一份状态”解释为状态字节有固定上界：分组基数、活跃窗口数量、
watermark 停滞及原生状态资源政策仍会决定占用。保留原生限制和错误行为，
explain 不承诺未知工作负载的内存数值。本项不承诺未经测量的加速，也不以
新的本地 benchmark 作为默认交付任务。

主要风险及对应控制：

- 行数变化后错误继承来源或排序：带类型 lineage、名称伪造反例、清空逐行排序事实、AC-08/09。
- 误套 rolling finality/late policy：独立窗口能力和策略说明、AC-14/15。
- 看似共享而实际有两份状态：完整 digest 规划、图与 manifest 双重断言、AC-10。
- 筛选/CSE 越过 finality 边界：lowering 屏障和共享分支反例、AC-04/16。
- Python 类型推导与 Rust 漂移：实际 row-only/CSE stage 原生 schema 规划、
  精确输出及全支持矩阵对照；nullable 不复制 DataFusion 规则，AC-05/07。
- checkpoint 表面可恢复但重编译身份不同：重新构造 Runtime/图及负例、AC-12/13/17。
- 进度坐标被窗口前时间表达式改变：首版只接受可证明的传递/重命名，AC-15。

#243 已作为 codec 测试基线存在。本功能不依赖重做
`decode_accumulator_state`/`append_accumulator_state_array`，不修改 state
version、segment 编码或 Arrow IPC 元数据。若另一个后续 codec PR 同时修改
window.rs，实施者先对齐基线并保留其字节等价保护；只有实际冲突才协调文件修改。

新增特性不需要新第三方依赖，不触发版本联动发布；正式发布时另循既有 crate、
Python core、Studio 和 frontend 同步版本规则。
