# Calc Flow 借鉴 TA-Lib 与 Arroyo 的性能开发计划

日期：2026-11-08。状态：执行中；P1a 按性能停止规则撤回，P1b/P2 继续开发。

目标是降低已有计算和流式应用的实际执行成本，同时保持数值、输出、资源和恢复契约。
优先开发跨算子运行时融合、`ts.decay` 增量内核和多窗口共享历史。
检查点写入与处理重叠、增量结果输出分别作为协议设计项目；其余候选由具体负载和测量决定是否进入实施。

本计划整合两份 TA-Lib/Arroyo 源码研究及其复核结果，作为后续工作的统一入口。
[原跨库计划](cross-library-performance.md) 保留其历史含义；本计划不把其中的旧任务状态或历史测量转写为新成果。
当前实现契约仍以 [runtime envelope](../runtime-envelope.md)、
[streaming guide](../streaming-guide.md) 和源代码为准。

## 执行记录

- P0：研究基线已重新确认；实现基线为 `8d8be7d9`，记录随 P1a PR 交付。
- P1a：[PR #563](https://github.com/wegamekinglc/calc-flow/pull/563) 撤回，不合并运行时改动。
  融合模块36 passed、1 existing ignored，10项新测试、本地 Clippy/Codacy 通过；
  测量源码873的 Linux/Windows/macOS CI 最终全部通过。
  但唯一预登记测量的目标 speedup=0.966894，95% CI=[0.896492,1.080038]，未证明提速。
  对照 speedup=0.941725，95% CI=[0.890114,0.993488]，不构成已确认超过5%的回退，亦不证明等价。
  第三场景 descriptor 缺少行数字段，被身份校验拒绝，没有有效吞吐 verdict；
  已修元数据并补正确性测试，原失败证据保留，未补跑、增样本或调整负载。
  真实图静态追踪满足全部准入条件；没有物理 driver 直接 instrumentation，保留该限制。
  来源、原始样本与停止结论见 PR 分支的 `benchmarks/runtime-fusion/p1a-20261108.md` 及 raw JSON。
- P1b：独立规格/critique 已通过，基于最新 main 的 Expression→SQL 实现进行中；
  不合入 P1a 未证明受益的独立算术/过滤准入。
- P2：主规格与首轮 critique 已完成；具体误差认证与恢复设计已获 critic 放行；核心数值/owner 信用3项测试已通过，
  typed 接入、旧检查点与 per-Decay Reference 回退验证进行中。
- P3：共享历史规格与独立 critique 已完成，可进入 test-first 实现；P4 尚未开始。
- C1–C4：尚未达到进入条件，按各项条件评估。
- D1/D2：尚未开始协议设计，单独验收。
- 合并门槛：真实性能提升、受影响正确性与恢复检查、CI/Codacy、最终 specialist review。

## 1. 基线与证据

研究基线固定为三个提交，后续实施必须从当时最新 `main` 建立独立分支，重新检查受影响路径。

| 项目      | 固定源码提交                               | 用途                       |
|-----------|--------------------------------------------|----------------------------|
| Calc Flow | `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf` | 2026-11-08 核对的最新 main |
| TA-Lib    | `58160778ecaab21e90eafd6e5462f6e833100af9` | 指标递推、共享历史、批处理 |
| Arroyo    | `eac7541f061c467c729549caabd6658d00ce9b23` | 算子链、窗口、输出、检查点 |

这些 SHA 是复现入口；本地 `/tmp`、`target/` 快照和分支名不构成长期依赖。
TA-Lib 此提交的 VERSION 为 0.8.4，但不据此推断同一实现存在于所有旧版本。

### 1.1 已有测量决定的方向

- compact SUM 的三条并行路线已经测量并撤回：
  [行区间 lanes][lanes-study]、[完整键分片][shards-study]、
  [委托 DataFusion 的 delta 聚合][delegation-study]。本计划不重复建设它们。
- 最新 main 已复用物理计划模板和完整快照的 key 列。继续调查表达式处理、快照构造、
  跨任务协调和检查点等待，不把已交付优化列为新增项目。
- 历史持仓表格、checkpoint 占比和 CPU 采样只能定位调查对象。重新固定构建、计时边界、
  批次、输出量、CPU 配额和 checkpoint 配置后，才能判断当前收益。
  inclusive CPU 采样占比不得相加，也不得直接视为墙钟收益上限。
- 90/110 ms 的既有 SMA 预算属于规定的 SQL/DataFusion 比较：有效 p4，
  绝对时间门槛在至少 16 逻辑核的校准条件下适用；它不是 p16 配置，也不是所有新工作的统一目标。
  继续遵守 [SQL 性能合同](../sql-datafusion-performance.md#fair-benchmark-and-evidence-gates)。

### 1.2 借鉴结论与明确修正

| 技术或提议                      | 当前事实与开发决定                                                                 |
|---------------------------------|------------------------------------------------------------------------------------|
| WMA 双和递推                    | `decay` 仍逐窗扫描；新增专用内核，但先证明缺失值权重和数值契约                     |
| 同输入共享 tape                 | TA-Lib 当前用于 MAVP 多周期子流；先迁移到同列 Numeric 行窗口                       |
| 单调队列与批量极值              | Calc Flow 已有单调队列；vHGW 只作为批处理常数优化候选                              |
| mean-only 删除 M2、改 sum/count | 不采纳；M2 参与 refold，且 sum/count 会改变近溢出有限窗口结果                      |
| 每行除法替换为乘 1/window       | 不采纳直接替换；满窗 add 时 count 为 W+1，remove 后为 W，舍入亦不同                |
| 删除 NaN/Inf 与 checked 检查    | 仅研究已证明前提的特化；Inf 仍合法，溢出余量应批前证明，不能改成批末发现           |
| Arroyo 任务链                   | 借鉴跨原生算子边界的直接传递；Calc Flow 已有 lowering 融合和部分运行时融合         |
| future_to_poll 自动并行 CPU     | 不成立；主循环 await 批处理，真正重叠仍需有界工作执行与顺序控制                    |
| updating aggregate 全列更新     | Arroyo 无 retract 的 global 路径批量更新；keyed 路径仍逐行，不能作为分组 SIMD 证据 |
| 按 slide 聚合 pane              | 可减少高重叠窗口的重复行更新；Arroyo 当前构造只启用一层 pane                       |
| dirty-key 输出与定时 flush      | 属于可选结果协议设计，不能替换现有每批完整快照                                     |
| 字节背压与 barrier 局部暂停     | Calc Flow 已有；Arroyo 所引 channel 的 queued_bytes 是统计，不能作为更强准入机制   |
| 后台 checkpoint                 | 异步准备、增量段复用已存在；新增问题是安全地重叠持久化与后续处理                   |

[均值反例][mean-counterexample] 使用窗口 2 和
`[1e155, 1e155, 1e155, -1e155]`：保留 M2 的结果为 `0.0`，
移除其修复触发后残差超过 `1e130`。[近溢出测试][mean-overflow]
要求 `[1e308, 1e308]` 的均值仍为有限值。这两项是开发约束，不是可以用性能目标豁免的边界。
不保留未经验证的“mean-only 减少约 30% 操作”等收益承诺。

## 2. 所有工作共同保持的契约

1. **表与 API。** DataFusion 55 仍是唯一表引擎；不引入 TA-Lib FFI 或第二套表运行时。
   调用方 Arrow 数据不可变，schema、metadata、逻辑输出名称和结果行身份保持。
   现有 `Program`、`compute`、`stream` 入口和默认配置保持。
2. **数值。** 保留 `stable_v1` 默认值和 `stable_v2` 显式选择、现有容差、
   null/NaN/Inf/正负零、`min_periods`、预热、平局、refold/rebase 语义。
   只改变存储或执行组织的项目要求相同输入与切分下输出逐位一致。
   改变算法的项目必须满足已有逐项数值合同，不新增宽松容差。
3. **顺序。** 保留实体内递推、event time/sequence 顺序、规范输出顺序、
   late Drop/Error 的错误次序、watermark/Idle/Barrier/EOF 规则。
   慢活跃输入仍约束 watermark；Idle 不等于可以忽略所有慢输入。
4. **输出与提交。** 现有 SQL 聚合每批仍输出完整累计快照。
   状态仅在现有成功 emission 边界提交。失败、预算拒绝、取消和重试不得留下部分更新，
   不撤回已被普通 sink 接受的输出，不新增 exactly-once 承诺。
5. **恢复。** project-v3、managed manifest v3、既有 fingerprint 和支持的状态格式保持。
   内存布局变化优先继续导出既有持久格式。确需改变格式或语义时，先定义版本与兼容矩阵，
   不能用旧标识静默编码新含义。恢复失败不得替换已安装状态。
6. **资源。** 保留边的 envelope/row/byte 准入、超大消息拒绝、状态预算、工作内存预留、
   取消后回收和输入生命周期。共享缓冲按唯一 owner 核算；小切片不能无限保留大输入。
   后台任务的内存信用必须存活到任务实际结束。
7. **实现。** 保持 `unsafe_code = "forbid"`。不通过调大队列、关闭校验、改变输出量、
   跳过持久化或放宽测试来制造性能收益。

## 3. 开发顺序与依赖

| 阶段 | 工作                        | 进入条件                         | 交付物与退出条件                         |
|------|-----------------------------|----------------------------------|------------------------------------------|
| P0   | 固定证据与验收协议          | 本计划                           | 每个首选项目有基线、目标 case 和退出规则 |
| P1   | 扩大运行时算子融合          | P0；跨算子成本可观察             | 安全链执行、控制语义测试、配对证据       |
| P2   | decay 增量内核              | P0；数值准入规则通过审查         | 专用内核、参考路径、恢复与复杂度证据     |
| P3   | 多窗口共享历史              | P0；状态所有权方案通过审查       | 同列 Numeric Rows 共享队列与兼容恢复     |
| P4   | 批前证明与循环特化          | P1–P3 后仍有可定位的循环成本     | 单一受益路径的窄特化                     |
| C1   | 完整快照的聚合列分块复用    | 大基数且更新集中、存在干净分块   | 事务性缓存；密集更新对照不回退           |
| C2   | hopping pane 聚合           | overlap 高、每 pane 数据足够密集 | 先 COUNT 的等价实现与预算模型            |
| C3   | 批量 MIN/MAX 内核           | 连续长分区，极值内核为实际热点   | vHGW 专用路径和原流式回退                |
| C4   | source 行数、字节、等待阈值 | 指定 connector 存在缓冲等待问题  | connector 局部策略与延迟/吞吐证据        |
| D1   | checkpoint 持久化与处理重叠 | 当前关键等待仍在 staging         | 协议规格、失败矩阵，然后实现             |
| D2   | 可选 changelog 输出 API     | 应用接受增量输出与不同可见时机   | API/恢复/交付规格，然后实现              |

P1、P2、P3 是首轮主线，默认按表中顺序推进；它们不互相构成逻辑前置条件。
同一轮只验证一个主要机制，前一项收益不能计入后一项；后续基线包含已经合入的改动。
P4、C1–C4 仅在进入条件满足时执行，不是为了“做完清单”而自动开启的工作。
D1、D2 的规格设计可在主线期间进行，但协议未收敛前不修改运行时或公开结果行为。

## 4. 首轮实现

### 4.1 P0 固定可比较基线

逐项记录最新 main SHA、候选 SHA/工作树摘要、依赖锁文件、构建参数、CPU affinity、
Tokio workers、DataFusion partitions、输入批边界、checkpoint 状态和输出合同。
复用已有持仓、rolling 和 warm-stream 工具，不以工作区未提交的示例改动作为隐含基线。
缺少某个必要的定向 case 时，在现有 harness 中最小扩展，并明确标为新增覆盖。

先利用现有阶段计时、工作计数与既有报告定位成本，再决定是否需要独立采样或分配诊断。
时间占比、调用次数和分配量分别报告。P0 不触发全库 benchmark、核心数扫描或多引擎重跑。
每项性能执行必须先完成第 7 节的具体协议，不能把整份路线图视为一次测量授权。

### 4.2 P1 扩大跨算子运行时融合

依据：[Arroyo chaining][arroyo-chain]、[ChainedCollector][arroyo-collector]；
Calc Flow 的 [eligible_pair][cf-fusion] 当前主要接纳 Rolling/ASOF/纯投影后接 exact projection。

**实施。** 分为两个可独立撤回的增量：

- P1a：在已支持的链头后接确定性的无状态投影、算术和过滤。
  编译阶段证明单输入、单消费者、精确内部输入 schema、受支持且无副作用的表达式。
  Python lowering 的链尾可以不声明输出 schema；此时沿用原 emission/下游验证，
  不新增启动规划或提前触发空流错误。继续作为链头仍要求已知输出 schema。
  不跨 fan-out、late 分支、外部 provider 或 UDF 边界。
- P1b：在 P1a 的控制和计费模型成熟后，研究持仓链路的
  `Expression → compact SQL`。保留两个逻辑节点及独立状态身份，先仅合并任务执行边界，
  不重排浮点聚合、不合并数据批次，也不承诺消除表达式产生的 Arrow 数组。

由链执行器传递拥有的 `Batch`，保留虚拟内部边的类型校验、预算拒绝、观测和控制排序。
内部边成功接受输出后，上游按原 emission/commit 边界提交，再由下游处理；下游计算失败
按自身节点归属传播，不能把下游成功增加为上游提交条件。直接调用无法保持该边界时，
保留有界本地邮箱或调度交接，不在融合项目中暗改提交协议。
链尾未声明输出 schema 时仍使用原 collector 的声明端口校验、EnvelopeCost、sender 校验
及预算检查；这项准入调整须有真实 Python 导出形状的 driver 与失败对照测试。
保留每个节点的 checkpoint、错误归属和指标含义；重计算在既有有界 CPU 工作设施中执行，
不能阻塞 Tokio 或把 `select!` 当作 CPU 并行。
准入失败使用原任务路径；诊断可以关闭新融合，默认 API 不新增配置项。

**验收。** 针对空/全过滤批次、计算错误、窄边预算、背压、取消、EOF、barrier 和恢复做对照。
融合与未融合图的值、顺序、状态和逻辑节点观察保持；物理任务数允许变化。
既有纯投影融合继续工作。至少一个预注册目标按选定维护工具的统计合同提供真实改善证据，
所选对照无已确认回退，否则撤回该实现增量。工具提供 `improved` 判定时沿用其规则；
仅报告置信区间的工具必须在执行前固定交付标准，不能将两种统计口径混用。

### 4.3 P2 将 decay 从逐窗扫描改为增量内核

依据：[TA-Lib WMA][talib-wma]；Calc Flow 的 [evaluate_scan][cf-decay]
为每行创建有效样本列表并求加权和，typed plan 尚不接受 Decay。

**实施。** 首轮覆盖 Float64、Rows frame、已证明顺序的输入。
为每个实体保存有效值数、普通和、按有效样本顺序计算的加权和与离开窗口所需的队列。
令窗口宽度为 W，有效值为 `x1…xn`，则现有分子可写成
`Q + (W - n) * S`，其中 `S = Σxi`、`Q = Σi*xi`；分母为
`n * (2W - n + 1) / 2`。该恒等式只说明候选状态，不构成浮点等价证明。

例如 `[1, null, 4]`、W=3 必须得到 `(2*1 + 3*4)/5`。
原始行队列决定过期，有效样本序号决定权重，不能把缺失值当成保留权重位置的零。
预热、`min_periods`、W=1、W 大于当前历史和混合输出都沿用现有行为。

递推改变运算顺序，必须先定义现有容差下的准入和修复方案，覆盖强抵消、极值和长序列。
TA-Lib 的 `8*W` reseed 只能作为候选，不直接继承其数值结论。
有限输入也可能中间溢出。无法证明的形状保留参考扫描；运行时回退必须从未提交的旧状态
重新计算，不重复 emission 或污染指标。若无法满足既有数值合同，停止默认路径交付。

把 Decay 接入 typed plan，避免它迫使其他已支持输出共同回退。
首轮持久化优先继续使用现有历史契约；恢复须验证后续多批结果，而非只验证刚恢复的读数。
不能因新增 resident summary 改变原有声明的 kernel/checkpoint 身份。
若重建不能保持约定的续算结果，则先解决状态版本方案，不静默改变格式。

**验收。** 先写缺失值权重、批次切分、混合输出、错误/取消和恢复反例；与当前扫描及独立
窗口 oracle 比较。用测试工作计数证明正常有限值路径不再每输出扫描 W 个元素；
周期重建计入总工作量。端到端测量和大窗口对照证明实际受益，异常回退成本单独报告。

### 4.4 P3 同输入多窗口共享历史

依据：[TA-Lib MAVP 的 period bank][talib-tape] 和
Calc Flow 的 [SlidingSampleQueue][cf-queue]。同 input/frame 的 Numeric group 已共享，
首轮只处理不同 Rows frame 对同一输入列的重复保存。

**实施。** 在单实体的事务性状态内，为同列建立一个按行序号访问的历史 owner，
容量覆盖最大窗口和原有 add-before-expire 的临时位置。
每个窗口保留自己的累加器、有效计数和过期位置；每行只写入历史一次，各组仍按原顺序更新。
先支持两窗口 mean/difference，再考虑更多 Numeric 窗口；Extrema、Pair、Duration 不混入首轮。

保持现有 West/M2/refold，重算从各自窗口视图读取正确顺序。
事务准备按实体复制共享历史一次，失败时丢弃候选；持有旧快照的读者不可见新写入。
计费同时覆盖共享 payload、各组 summary、候选副本和暂存余量；不能仅降低估算值。
持久格式不变时，以旧基线 checkpoint 恢复到候选并继续计算验证兼容。

**验收。** 单窗口不额外付出共享管理开销；dual SMA 的值与状态迁移保持逐位一致。
测试证明共用一次历史存储与复制，样本 payload 从 `ΣW` 降至 `max(W)` 加必要余量，
不把它写成总内存或总算术工作等比例下降。配对观察包含 warm 小批、实体切换、持有旧快照
和单窗口对照；分配字节、峰值内存与端到端时间分别报告。

### 4.5 P4 基于批前证明的特化

P1–P3 后重新定位热点，只保留仍有成本的窄改动：

- 复用路由得到的每实体行数，在批前验证累计计数有足够余量；验证失败回原逐行路径，
  保留原第一错误。循环内仍推进每行 transition 值，保持 rebase 时点。
- 在已证明的有界 Rows 状态中，验证有效计数及 add-before-expire 的上界。
  不把窗口计数的证明泛化到累计计数、Duration 或不可信恢复状态。
- 对 `nan_as_value=false` 的私有输入路径研究 NaN 检查特化；Inf 处理、
  非有限中间结果检测和修复继续保留。SQL 的 NaN-as-value 路径继续使用原实现。
- 将确定不变的 group 类型、实体密度和引用准备前移。检查生成代码确认分支是否消失；
  不以源码行数、Result 数量或数学运算计数代替 CPU 测量。

本阶段不增加 unsafe，不替换 mean 算法，不直接引入 FMA 或乘倒数。
每个特化单独证明、测量与撤回；没有端到端收益且不能证明有价值的资源收益时不合入。

## 5. 按负载进入的候选

### 5.1 C1 完整快照聚合列分块复用

Calc Flow 的 [aggregate_output_chunk][cf-snapshot] 仍为每个分块重建聚合值列；
现有 key 缓存不能消除这部分成本。Arroyo 的 [updated_keys/flush][arroyo-flush]
提供变更跟踪思路，但其 retract/append 结果协议不适用于本项。

为当前事务的候选 group 建立 touched-chunk 集合，仅重建受影响的聚合列，其他块引用已提交的
不可变数组。checkpoint 的 dirty 生命周期不同，不直接复用它作为每批缓存失效集。
首轮只缓存聚合列，不缓存 volatile 投影；HAVING、ORDER BY、LIMIT 仍按原语义执行。
新组、尾块增长、restore、reset、输出失败和预算不足都必须正确失效或回退。

进入条件是实测存在显著干净块。当前 `CHUNK_ROWS=8192`，一万个分组只有两个块，
随机小批更新也很容易全部变脏。此项目不预设改善典型持仓 workload，也不以改变分组顺序
或切小输出块作为隐含前提。全量交付成本保持；稀疏局部更新和密集随机更新必须成对验证。

### 5.2 C2 高重叠 hopping window 的 pane 聚合

借鉴 [Arroyo sliding window][arroyo-window] 的 slide 桶 partial aggregation。
Calc Flow [当前路径][cf-window] 为每行展开所有开放窗口；新路径先更新所属 pane，
关闭窗口时合并 pane。降低的是原始行重复更新，最终合并和输出成本仍存在。

首轮仅 COUNT，且必须保留每个目标窗口的 late 判定和迟到指标；不能因为一个 pane 在
watermark 前就丢弃仍能贡献给其他开放窗口的整行。先明确逻辑窗口状态预算与物理 pane
预算的关系，避免通过改变预算单位默许更多状态。checkpoint、输出顺序和空窗口规则保持。

MIN/MAX 后续单独验证 null、IEEE total order 与键顺序。整数 SUM 的中间溢出和第一错误、
浮点 SUM/AVG 的合并次序没有等价证明前不接入。Arroyo 的多层容器不构成多级聚合已启用
或可获得对数复杂度的证据。用一个高 overlap、密集 pane 目标和一个低 overlap 对照验收。

### 5.3 C3 连续分区的批量 MIN/MAX

[TA-Lib MAX][talib-max] 的批处理采用 van Herk/Gil-Werman 前缀/后缀扫描，
流式版本采用另一算法。Calc Flow 已有摊还 O(1) 的单调队列，本项只争取常数收益。

只对已排序的长连续分区、固定 Rows frame 建立批量路径，控制 scratch 大小并保留流式回退。
warm-up、尾块、跨 RecordBatch 延续、null、NaN、Inf 和正负零均用当前比较规则校验。
不直接扩展到 argmin/argmax 的最早平局身份。若极值 CPU 占比很小，或短分区回退成本抵消收益，停止该项。

### 5.4 C4 指定 source 的缓冲等待策略

[Arroyo should_flush][arroyo-batching] 按行数、字节或等待时间触发发送。
先审计目标 connector 已有行/字节限制和真实缓冲行为，只给存在等待问题的 source 增加
有界 linger；不改统一 edge 准入，也不重新批处理已经由调用方给定的 Python Arrow 批次。

flush 策略不得把数据移过 watermark、barrier、EOF 或 replay cursor 切点。
说明最大等待时间、空缓冲处理、首行/末行延迟和小批开销；使用可控时钟测试，不依靠 sleep 猜测。
由于批边界影响累计快照的可见频率，本项必须显式定义 connector 行为与配置默认值，
先以 opt-in 交付，不能仅以最终表相同宣称行为完全不变。

## 6. 独立协议项目

### 6.1 D1 checkpoint 持久化与后续处理重叠

先按当前 main 分开测量 alignment、准备/编码、staging、manifest 发布和恢复读取。
已有异步准备、lazy delta、段哈希缓存和复用不重复实现；Join 的 lazy-delta bookkeeping
与整个运行时提前转发 barrier 是不同范围的改动。

[Arroyo TableManager][arroyo-checkpoint] 将 checkpoint 入队，flusher 完成后才发送
`CheckpointCompleted`。候选方案必须同样区分“捕获了 epoch 的不可变切片”和
“epoch 已持久化完成”，不能在 enqueue 时就推进 durable status、确认 source cursor 或提交 sink。

**先交付协议规格，至少解决以下问题：**

1. 在 barrier 切点冻结哪些状态、如何保证后续批次不修改该切片，以及 source/sink 如何参与。
2. 何时允许转发 barrier、恢复收批，何时才能发送 durable ack 与安装 manifest。
   现有 stage 失败不转发 barrier 的行为若变化，必须明确新失败模型及其对恢复的影响。
3. 初始方案最多一个在途持久化 epoch；次轮 barrier 到来时如何背压，工作队列如何有界。
4. 取消、磁盘满、写失败、校验失败、worker panic、manifest 发布不确定、进程退出和 then-stop
   如何收敛；失败后不使用未完成切片恢复，所有受托任务最终被等待和回收。
5. 校验不得移到对外确认完成之后。restore 的统计剪枝只能排除被证明无关的状态，
   不跳过恢复合同要求的完整性、schema、历史条目或状态预算验证。
6. 峰值内存、滞留输入、恢复时间和吞吐共同验收；不能以无限缓存换取 barrier 低延迟。

规格通过 API/运行时审查和失败矩阵审查后，再实施 capture/staging 解耦、集成 coordinator、
最后做故障注入与高频 checkpoint 的定向测量。收益不足或内存/恢复代价不可接受时，保留当前协议。

### 6.2 D2 可选 changelog 结果协议

当应用明确接受增量结果时，设计独立的 opt-in 模式；现有每批完整快照默认值保持。
不能直接把上游研究中的名字当成已经确定的公开 API。

规格必须定义初始 baseline、group 身份、retract/append 配对、删除/tombstone、
不变值判定、更新顺序、flush_interval、barrier/EOF 强制 flush、schema 和 metadata。
定义新消费者如何初始化、checkpoint 后如何重新建立 baseline、重放是否可能重复、
如何由消费端重建完整表；结果版本坐标不自动意味着 exactly-once。

实现顺序是 native 变更记录与提交 → 流消息/结果契约 → Python 所有权与迭代接口 →
可选完整快照重建适配器 → 文档和示例。项目序列化新增字段需有显式默认值与兼容验证，
不能直接修改现有 StreamOutput 的解释。

测量同时报告计算端、传输端、消费端重建成本、首条结果延迟和最大可见延迟。
与完整快照的比较必须标注输出协议和 flush 频率不同；每批强制重建的对照保留全部重建工作。
第一阶段成果是可审查的 API/协议规格，不承诺主线内部优化即可消除全量输出成本。

## 7. 验证与性能测量

### 7.1 正确性和资源验收

每个行为变化先增加最小失败测试，记录预期 red，再实现 green。
结构优化采用优化开/关对照；算法优化同时采用现有参考路径与独立 oracle。
受影响项目覆盖：预热/空批/多 RecordBatch、缺失与非有限值、实体交错、跨批切分、
旧 checkpoint 恢复后续算、预算拒绝、取消、输出失败与重试、共享缓冲存活期。
只选择与该项变化直接相关的组合，不将上述维度展开为全排列。

P1 重点运行链控制测试；P2/P3/P4 重点运行 rolling 内联测试和
`python/tests/test_symbolic_finance_reference.py` 的对应场景；C1 对应 snapshot cache/
output refusal；C2 对应 window 边界和恢复；D1 对应 checkpoint 故障与生命周期。
具体测试过滤器在实现时以实际测试名为准。本计划文档验收不构建 native。

### 7.2 每次测量协议

遵守 [AGENTS Verification](../../AGENTS.md#verification) 和
[维护中的配对合同](../benchmark-suite.md#revision-comparisons-and-regression-gate)。
每个实现项最多选五个实际 workload，**全部所选 case 合计 600 秒**，包括 fixture、
预热、恢复、正确性、进程启动、采集和清理；构建时间单独估算和报告。
以下是候选集合，不是一次运行所有行的矩阵。P1a/P1b 分别固定协议和预算。

| 项目 | 默认目标与必要对照                                           | 维护工具入口                                    |
|------|--------------------------------------------------------------|-------------------------------------------------|
| P1a  | Rolling/ASOF 后接算术或过滤链、一个不满足融合准入的对照      | stream operator/ASOF 工具的最小定向扩展         |
| P1b  | 小批持仓、较大批持仓、一个不满足融合准入的对照               | example 30 与 stock/sum 配对工具                |
| P2   | 单独 decay、大窗口 decay、decay 加已有均值输出               | test_stream_operators.py 中的 rolling scan 工具 |
| P3   | warm dual SMA、单 SMA 对照、一个稀疏实体 append              | profile_warm_stream.py、test_rolling_kernel.py  |
| P4   | 一个命中特化的既有目标、一个回退对照                         | 对应 P1/P2/P3 的现有工具                        |
| C1   | 多分块局部更新、同基数密集随机更新                           | compact SQL/stock 工具的最小定向扩展            |
| C2   | 高 overlap 密集 pane、低 overlap 对照                        | window/stream operator 的现有工具               |
| C3   | 长连续分区 MIN/MAX、短分区对照                               | rolling kernel 工具                             |
| C4   | 低流量首条可见延迟、连续输入吞吐                             | 目标 connector 的最小定向工具                   |
| D1   | 一项高频 checkpoint 负载、一项恢复、一项 checkpoint 关闭对照 | 已维护的 checkpoint/recovery 工具               |
| D2   | 增量消费、相同变更加完整重建、现有 snapshot 对照             | 输出协议冻结后扩展现有应用 harness              |

运行前补全精确 case 名、行数、批大小、状态规模、CPU 配置、baseline/candidate、
固定样本数与总墙钟估计。CPU 配置、GC 模式或恢复状态不同即为不同 case，不通过嵌套模式规避五例上限。
先选工具能支持的最小输入，不默认 10M 行或全核心数扫描。

使用 `release_pairs` 的版本比较固定为维护规则的 **两轮，每轮十个 AB/BA 配对**；
不得复用旧实验的五对样本来替代当前合同。独立 warm-stream 比较在本计划中固定 30 对，保留工具自身统计合同；
已有内部 60 对 kernel/reference 诊断仅用于其原目标，不能自动变成版本性能结论。
必须在协议中选择统计口径；不同工具的置信区间和 verdict 不拼接。

保留当前两轮 +5% 回归规则：两轮区间下界均超过 +5% 才判已确认回退；
任一上界超过 +5% 而不满足前述条件时为 inconclusive；两轮上界均低于 -5% 才判 improved；
其余按维护判定器处理。不额外要求每个区间都排除阈值，不把 inconclusive 解释为等价或提速。
目标 case 的收益必须达到其所用维护合同的要求，对照不得存在已确认回退。

主线性能改动以至少一个预注册目标的端到端收益为交付依据；P3 等资源优化若仅证明内存收益，
必须明确作为资源优化评估，完整记录时间结论，不能宣称 CPU 提速。
预算耗尽、环境不适合或证据不足时停止，等待并清理所有自有进程，报告已完成项与不确定性。
不追加 case、扩大样本或反复重试以取得有利结果。

`plan_end_to_end` 等原子 acceptance set 只有完整集才可给集合 verdict；定向子集只报告逐例。
不为获得集合结论自动扩成全套。本地不运行完整多引擎、核心数或数据类型矩阵；定时 broad
benchmarks 和完整回归留给 CI，除非用户另有明确范围要求。

## 8. 交付组织与结束条件

每个实现项使用从最新 main 建立的独立 `feature/<description>` 分支和隔离工作区。
P1a、P1b 分开交付；P2、P3 各一个可独立评估的改动。
同一项只保留一份主规格和必要的阻塞修正，避免为每个微优化建立独立框架。
团队规格/API/审查产物按既有约定进入 `.codex/artifacts/`，共享该项的 kebab-case slug。

清晰的内部改动按 implementer → reviewer → 按需 doc writer 推进。
P2 的数值状态、C2 的窗口语义、D1/D2 的协议由 spec/API/critic 先解决真实歧义，
再进入实现；performance specialist 负责已约定的定向证据，最终 specialist review 必需。
用户已授权逐项建立 PR、自主修复 CI/Codacy，并在真实性能提升、必要检查与最终审查通过后自主合并。
每个实现增量独立 PR；P0 的基线与验收记录随 P1a 交付。没有证明真实性能提升的候选不得合并。

单项 handoff 必须包含：最终行为、相对基线的实际改动、准入与回退范围、red/green 证据、
配对/资源测量及其限制、恢复兼容性、最终审查结果、必要文档更新与撤回办法。
未证明收益的原型不进入默认生产路径，可保留证据供后续判断；不以“测试通过”代替性能验收。

本地仅运行受影响的测试和必要格式/lint/类型检查。完整回归、服务覆盖率和跨平台门禁留给 CI；
Rust workspace 90% line floor 保持。推送或 handoff 后最多读取一次非阻塞 CI 状态，
pending/absent/cancelled/inconclusive 不等于绿色，合并仍需显式授权及 required checks 通过。

文档和实现收尾检查：

```bash
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

D2 若确有经规格定义的项目合同变化，应单独生成、审查预期 schema diff；其他工作不应产生漂移。
文档阶段只检查结构、表格、链接、事实和 diff，不为验证计划运行产品测试或 benchmark。

## 9. 保留研究项

向量化 repartition、Peek 影子状态、peak-trigger reseed、结构体布局和新的 UDF 执行方式
保留为后续研究，不建立空实现。只有出现明确 workload、兼容证明和测量预算才进入计划。
尤其不引入 Arroyo 的 unsafe 数据结构、允许超大批次绕过预算、通用更换 Parquet 状态后端，
也不因外部库某种算法更快就改变 Calc Flow 默认数值配置。

本轮完成标志是 P1–P3 各自得到可交付实现或有证据的停止结论；
条件项目没有满足进入条件时明确标记未启动，不要求为了完成路线图全部实现。
D1/D2 的规格与后续实现独立验收，不作为主线交付的隐含前置条件。

[lanes-study]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/benchmarks/sum-state/sum-batch-parallel-20261104.md
[shards-study]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/benchmarks/sum-state/sum-key-sharding-20261107.md
[delegation-study]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/benchmarks/sum-state/sum-delta-delegation-20261107.md
[mean-counterexample]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/rolling/kernel.rs#L4579
[mean-overflow]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/rolling/tests.rs#L4275
[cf-fusion]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/runtime/streaming/runner/operator_fusion.rs#L113
[cf-decay]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/rolling.rs#L4166
[cf-queue]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/rolling/kernel.rs#L2543
[cf-snapshot]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/sql/incremental.rs#L2070
[cf-window]: https://github.com/wegamekinglc/calc-flow/blob/8d8be7d907a095dbc0dc2b73dafcfdd842979cdf/crates/calc-flow/src/operator/window.rs#L1279
[talib-wma]: https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/input/wma/wma.c
[talib-tape]: https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/generator/src/backends/c_stream.rs#L4845
[talib-max]: https://github.com/TA-Lib/ta-lib/blob/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/input/max/max.c
[arroyo-chain]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-datastream/src/optimizers.rs#L43
[arroyo-collector]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-operator/src/operator.rs#L415
[arroyo-flush]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-worker/src/arrow/incremental_aggregator.rs#L638
[arroyo-window]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-worker/src/arrow/sliding_aggregating_window.rs
[arroyo-batching]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-formats/src/lib.rs#L12
[arroyo-checkpoint]: https://github.com/ArroyoSystems/arroyo/blob/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-state/src/tables/table_manager.rs#L95
