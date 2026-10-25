# ASOF / Interval join 多核优化性能报告

正式五配置测量完成：维护中的两轮配对规则确认 ASOF 32 CPU 改善，P50 耗时下降
38.53%；确认 interval 8 CPU 改善，下降 19.49%。ASOF 1 CPU、interval 1 CPU
和 interval 32 CPU 均为 `inconclusive`。未触发确认回退规则，但三个不确定结果
不能解释为无回退或等效。32 CPU 诊断确认局部并行已发生，但整段 CPU 利用
仍有限；ASOF admission 与 interval 输出/提交阶段仍是主要剩余耗时。

## 测量范围与计时边界

同一 WSL2 主机 `chengli-i9`，最多 32 个逻辑 CPU。WSL 内核报告 16 个不同
core ID 分组，这不足以确定宿主真实物理核数量或 SMT 拓扑。1 CPU 绑定 `[0]`；
8 CPU 绑定 `[0,2,4,6,8,10,12,14]`，各取不同报告分组的首个逻辑 CPU；32 CPU
绑定 `[0..31]`。CPU 配额不能替代实际任务并发证据。主机外部负载与电源模式
无法完全观测。
采集期间的本机竞争进程检查未触发；前后 1/5/15 分钟 load average 分别为
`[0.81298828125, 1.47998046875, 1.27783203125]`、`[1.13671875, 1.423828125, 1.27978515625]`，这不证明宿主机完全安静。

固定五个配置，每配置两轮、每轮十对交替 AB/BA；第二轮反转起始顺序。
共 200 个正式观察和 20 次排除预热，全部完成，没有失败、重采或增加样本。
ASOF 每侧 1,000,000 行；interval 每侧 262,144 行；每侧单批、单个 Arrow
RecordBatch，64 个实体。显式现有状态预算均保持 1 GiB。

计时覆盖已就绪 source 入队到预期输出交付和 Arrow concat。编译、启动、
完整 oracle、进度观察、EOF 和清理在该计时器之外，但计入采集总预算。
ASOF 使用现有 bounded-feed 适配器，interval 使用现有 exact-cursor 适配器。
完整 schema、身份、值和 interval 匹配次数检查均通过：ASOF 输出 1,000,000 行，
interval 输出 `11×262144−1920 = 2,881,664` 行。20 个 worker 完成并清理；每个
worker 留存一次预热与十个正式样本的独立状态目录及原始 stdout。

## 正式版本对照

| 配置                   | 基线 P50 ms | 候选 P50 ms | 时间变化 | 两轮维护规则判定       |
|------------------------|-------------|-------------|----------|------------------------|
| ASOF 1M / 1 CPU        | 86.44       | 91.84       | +6.25%   | inconclusive（不确定） |
| ASOF 1M / 32 CPU       | 96.99       | 59.62       | -38.53%  | improved（改善）       |
| Interval 262k / 1 CPU  | 205.86      | 210.00      | +2.01%   | inconclusive（不确定） |
| Interval 262k / 8 CPU  | 177.18      | 142.65      | -19.49%  | improved（改善）       |
| Interval 262k / 32 CPU | 184.63      | 168.77      | -8.59%   | inconclusive（不确定） |

| 配置                   | 基线 P95 ms | 候选 P95 ms | 第一轮配对区间     | 第二轮配对区间     |
|------------------------|-------------|-------------|--------------------|--------------------|
| ASOF 1M / 1 CPU        | 94.66       | 99.58       | [+7.71%, +18.18%]  | [-7.89%, +0.69%]   |
| ASOF 1M / 32 CPU       | 104.36      | 69.83       | [-45.02%, -28.30%] | [-43.39%, -25.54%] |
| Interval 262k / 1 CPU  | 228.59      | 235.71      | [+2.04%, +5.52%]   | [-4.39%, +10.61%]  |
| Interval 262k / 8 CPU  | 202.46      | 168.15      | [-25.74%, -11.40%] | [-24.00%, -15.84%] |
| Interval 262k / 32 CPU | 232.27      | 205.64      | [-25.18%, +9.01%]  | [-21.15%, +0.99%]  |

时间变化为候选相对基线，负数代表更快。P50 汇总每侧 20 个正式观察；主报告
P95 保持采集前计划的 nearest-rank 定义：排序后第 `ceil(0.95×20)=19` 个观察，
仅作描述。原始 `statistics.head_p95` 由维护函数按线性插值计算；原始
`descriptive.A/B.p95_seconds` 是本表使用的 nearest-rank 值。两种定义均保留在
原始报告中，不混用、不以线性插值替换预先声明的 P95。

判定直接采用 `scripts.benchmark_suite.report.comparison`，离线从所有原始观察
重算逐字段一致。两轮区间下界均高于 +5% 为 `regression`；若任一轮区间上界
高于 +5%，返回 `inconclusive`；否则两轮上界均低于 -5% 为 `improved`，其余为
`no-confirmed-regression`。本次使用该规则，没有额外增加等效或通过条件。

十对样本使用二项分布次序统计量区间，取配对变化排序后的第 2 与第 9 项，实际
覆盖率为 0.978515625。区间针对每对变化的中位数，不是汇总 P50 比值的区间。
该统计构造假设各对独立且来自共同变化分布；交替顺序不能证明该假设成立。

ASOF 1 CPU 的两轮配对中位数分别为 +9.80% 和 -3.16%，方向相反；汇总 +6.25%
不能单独判定回退。Interval 32 CPU 的汇总 -8.59% 也不能单独判定改善，第一轮
区间仍覆盖 +5% 以上。Interval 1 CPU 两轮上界均覆盖 +5% 以上，保持不确定。

## 扩展性

| 同一版本的比较      | 基线加速比 | 候选加速比 | 解释             |
|---------------------|------------|------------|------------------|
| ASOF 1 → 32 CPU     | 0.891×     | 1.540×     | 描述性跨配置比较 |
| Interval 1 → 8 CPU  | 1.162×     | 1.472×     | 描述性跨配置比较 |
| Interval 1 → 32 CPU | 1.115×     | 1.244×     | 描述性跨配置比较 |
| Interval 8 → 32 CPU | 0.960×     | 0.845×     | 描述性跨配置比较 |

加速比使用 `T较少CPU / T较多CPU`。跨 CPU 配置未构成时间配对，不能套用新旧
版本的配对显著性结论。ASOF 的描述性 1→32 CPU 比值由 0.891× 变为 1.540×；
interval 的 1→8 CPU 比值由 1.162× 变为 1.472×。候选 interval 32 CPU 的 P50
仍比其 8 CPU 高 18.31%，不能声称 8→32 CPU 继续获益。新旧版本在同一 32 CPU
配置上的 interval 比较也仍不确定。现有数据支持部分配置的吞吐改善，未证明
接近线性多核扩展，也未证明单核总工作量减少。

## 阶段诊断

以下均为 32 逻辑 CPU、每版每场景一次排除预热后的一次记录，使用单独的
`benchmark-internals` release。其时间不混入默认 release 的正式样本；单次阶段
变化用于定位，不提供阶段改善显著性，也不能解释未经诊断的 8→32 CPU 因果。

以唯一 ready 起止 marker 裁剪阶段。下表的父阶段墙钟是同名区间的并集；本次
同名父阶段彼此不重叠，所以等于它们的墙钟和。不同层级与 worker 存在包含或
重叠，不能相加。异步父阶段没有 thread CPU 测量；空缺不是 CPU 为零。

### ASOF：匹配/计划准备缩短，admission 仍占主段

| 阶段          | 基线墙钟 ms | 候选墙钟 ms | 阶段关系                          |
|---------------|-------------|-------------|-----------------------------------|
| admission     | 43.051      | 50.285      | 包含 state_install                |
| state_install | 12.326      | 15.367      | admission 的子阶段                |
| finalize      | 49.224      | 14.989      | 包含匹配、物化和提交              |
| match_plan    | 39.834      | 6.313       | 包含候选 match_worker 与 assembly |
| assembly      | 未记录      | 0.060       | 候选 fragment 计划组装            |
| materialize   | 0.746       | 0.025       | 本次未出现复制型 gather_worker    |
| state_commit  | 0.008       | 0.006       | finalize 的子阶段                 |

基线 `probe_selection` 记录 `snapshot-budget`，候选记录
`contiguous-fragments`、32 个 work units、100,121,133 字节的输入 workspace
charge。候选 32 个 `match_worker` 合计处理 1,000,000 行，证据指向连续片段的
匹配与输出计划准备。`admission` 仍约 50.285 ms，其中 `state_install` 15.367 ms
已包含在内；不能把这两者相加。此次 snapshot 中，新的并行匹配尚未改变
admission 主导整体 ready 区间的状况。

两版均没有 `gather_worker` 事件，结果均为一个 1,000,000 行 RecordBatch。
结合共享 Arrow 输出分支，本次 materialize 主要组装共享结果；这不构成新增
复制型 row gather 的性能证据，也不能把正式 38.53% 改善归因于该复制路径。

### Interval：state preparation 缩短，finish_data 仍占主段

| 阶段               | 基线墙钟 ms | 候选墙钟 ms | 阶段关系                             |
|--------------------|-------------|-------------|--------------------------------------|
| prepare_batch      | 62.784      | 25.989      | 包含 state_prepare                   |
| state_prepare      | 46.251      | 8.694       | 包含 retained_selection / chunks     |
| retained_selection | 20.199      | 3.617       | state_prepare 子阶段，含 worker 等待 |
| retained_chunks    | 26.040      | 5.066       | state_prepare 子阶段，含 worker 等待 |
| append_prepare     | 9.286       | 11.551      | 过期索引与追加准备                   |
| finish_data        | 114.856     | 120.240     | 含匹配收尾、输出、提交与等待         |
| state_commit       | 8.609       | 10.235      | finish_data 的子阶段                 |

两侧各一批，共两次 retained preparation；候选 `retained_selection` 和
`retained_chunks` 各记录 64 个 worker 任务，即每批 32 个。两次选择 note 都为
`prepaid`，没有出现预算回退；note 的 bytes 合计分别为 8,412,990 和
30,774,014。这些是各次 workspace charge 的记录合计，不是同时存活峰值、
实际分配总量或 RSS。显式状态预算仍为原定 1 GiB。

`finish_data` 约 120.240 ms，包含 count/gather、输出交付、状态提交和相关等待，
不能把这 120.240 ms 称为全串行 CPU。新的 state preparation 已明显缩短这次
观察中的相关墙钟，但剩余输出和提交阶段仍占本次观察的大部分墙钟；当前 trace 不足以
进一步拆分所有同步工作、调度或等待的因果贡献。

### 同步 worker 与全 ready 区间资源

| 同步 worker 阶段            | 基线任务数 | 候选任务数 | 基线墙钟和 ms | 候选墙钟和 ms | 基线 CPU 和 ms | 候选 CPU 和 ms |
|-----------------------------|------------|------------|---------------|---------------|----------------|----------------|
| ASOF match_worker           | 未记录     | 32         | —             | 46.816        | —              | 44.218         |
| Interval retained_selection | 未记录     | 64         | —             | 13.001        | —              | 12.501         |
| Interval retained_chunks    | 未记录     | 64         | —             | 25.070        | —              | 21.908         |
| Interval count_worker       | 32         | 32         | 6.684         | 6.302         | 6.641          | 6.262          |
| Interval gather_worker      | 64         | 64         | 34.049        | 46.004        | 33.660         | 40.933         |

worker 墙钟列是各任务时长的和，任务可并行重叠；它既不是端到端阶段时长，
也不能加到父阶段墙钟上。worker CPU 使用 `CLOCK_THREAD_CPUTIME_ID`，所有
本次 worker scope 都完整处于 ready marker 内，没有按比例分摊跨边界 CPU。
基线未记录某类 worker 不等于整个 job 没有线程或 CPU 工作。

| 32 CPU 诊断配置 | 版本 | ready 墙钟 ms | PhaseClock CPU ms | 平均 CPU 占用 | 活跃 worker 峰值 |
|-----------------|------|---------------|-------------------|---------------|------------------|
| ASOF 1M         | 基线 | 93.013        | 101.239           | 1.088         | 0                |
| ASOF 1M         | 候选 | 66.073        | 118.978           | 1.800         | 26               |
| Interval 262k   | 基线 | 188.045       | 265.814           | 1.413         | 21               |
| Interval 262k   | 候选 | 158.820       | 301.264           | 1.897         | 28               |

平均 CPU 占用为 PhaseClock 在同一包围 ready 的区间内测得的进程 CPU 时间
除以其自身墙钟时间；它包含该区间内的所有进程线程。该墙钟比原计时器多约
12–13 微秒，不用包含完整 oracle/EOF 的全调用 CPU 除以 ready 时间。

活跃 worker 峰值由 native 按 ThreadId 去重，只表示同时具有 instrumented
同步工作 scope 的线程数；不表示相同数量的物理核正在执行，也不表示整段
ready 持续使用 26 或 28 核。较高瞬时峰值与约 1.8–1.9 的平均 CPU 占用并存，
支持“局部并行已发生、整段利用仍有限”的判断；单次 CPU 总量不能证明算法
总工作量改善或回退。

ready 结束后，ASOF 两版各有三个 finalize 事件，其 scope 墙钟合计为基线
0.005562 ms、候选 0.004381 ms，均单独保留且未纳入表中。Interval 本次没有
ready 之后的 instrumented 阶段事件，这不代表 EOF/清理没有成本。
完整事件、裁剪索引、阶段聚合和限制见诊断离线分析（`target/performance/join-cpu-pipeline/diagnostic-analysis.json`）。

## 构建与来源

正式基线 source：`80d98ae089c8a317c2a884967b83728c94cb87ff`。
正式基线 tree：`e3d60f36f6a2ebd0ddd867a6c7a7242cd390e20e`。
正式基线 native SHA-256：
`7c78de89fa52d68a355d5a6c1e495a8c65f20eb99b87359d58324fa6548c0868`。
正式基线 wheel SHA-256：
`97394768fb641d18fe6310a1577e91a8b5845acc62fcf9577bd94d0a52fa48bb`。

候选 source：`61aab0e571800df8e8a18ad2244993d0929f5247`。
候选 tree：`15932415e462de46e0612ec26b724abf78ee82ae`。
候选默认 release native SHA-256：
`1c1ad9934bdbb229976ff4d0d283e751d5ce65dd80fdf2e6437014a54dfe2fca`。
候选默认 release wheel SHA-256：
`6dfca099e9fd0176baed387dd5c10b9e55eca7afa329ebefed61abede362cf3a`。
候选诊断 release 与正式候选使用同一 source/tree，启用 `benchmark-internals`。
其 native SHA-256：
`e8f42882a259a56fbda77848285aba5e03c02d2a843309aafc9a9fe743af3efd`；wheel SHA-256：
`8a6949ac9212e6e1341a3ed0973e336ff49c94c5e61c8a445ad6912d124f770e`。

以上正式候选和候选诊断 native 均明确绑定 `61aab0e5`。报告随后作为文档归档
不会改变被测 native 的源码来源；后续文档提交不应被当作重新构建或重新测量。

默认 release 的 Rust 1.94.0、Python 3.13.9、NumPy 2.5.1、PyArrow 24.0.0、
Polars 1.44.1、Cargo.lock、default features、release profile 和空 RUSTFLAGS
相容。Polars 版本仅属于依赖指纹，本次没有运行 Polars 基准或跨库对比。
离线核实源码 SHA/tree/clean、实际 native/wheel 字节、重新编译记录以及
每个 worker 实际导入的封存 native 均一致。

每配置 machine、dependency、workload 指纹在两版及两轮之间完全一致；跨 CPU
的 machine 指纹因 affinity 不同而不同，因此跨 CPU 只提供描述比值。
完整逐配置 SHA-256 见正式离线审计（`target/performance/join-cpu-pipeline/formal-offline-audit.json`）。正式报告 SHA-256：
`d99d28e559fcf6d6613830f7b9522463a846ace49bb2444fdb357f4f6b2c1879`；固定计划 SHA-256：
`294f1047f15bb11e1735f716fa5d6136f9e1b1020b32bb2226f9f6160a2065d4`。

诊断基线使用按 case 封存的 instrumentation-only 派生构建：ASOF 为 `c164c635`，
interval 为 `0094d7ca`；它们不能冒充正式默认基线。完整来源及修复记录见
诊断恢复离线审计（`target/performance/join-cpu-pipeline/diagnostic-recovery-offline-audit.json`）。候选与逐 case 基线的
诊断 machine/dependency/workload 及 compiled-plan 指纹一致；诊断与正式构建
仅在依赖指纹的 features 中不同，因此不混合它们的计时样本。
候选诊断报告 SHA-256：
`0afdf7708dfcc2d2b3d91dfdaf9e4e155002eeb6f91e3e19a8dd9d851920d99c`。
全部四个成功配置（共八次调用）来源、trace、oracle 与自然完成校验见
诊断来源离线审计（`target/performance/join-cpu-pipeline/diagnostic-offline-audit.json`）。

## 预算、恢复及证据

总采集预算 600 秒：诊断 90 秒、正式采集 480 秒、清理预留 30 秒。
正式 controller 从 preflight 前到全部清理完成共 116.399135 秒。
基线诊断累计 6.367333142014104 秒，包括原失败尝试 3.072655099007534 秒及
一次授权 interval 修复 3.29467804300657 秒。候选诊断耗时
3.606903419989976 秒，诊断总计 9.97423656200408 秒。最终采集累计
126.37337156600552 秒，低于预定 600 秒；没有用剩余预算增加样本或重试。

各 release 构建单独计时，不计入采集预算。预估约 600 秒/构建，实际用于本报告
证据的构建如下；工程与测试重建另计。

| 构建                   | 实际秒数 | 说明                           |
|------------------------|----------|--------------------------------|
| 正式基线 default       | 603.756  | 复用之前构建，本次未重建       |
| 正式候选 default       | 605.023  | 源码 61aab0e5                  |
| 基线诊断初版           | 867.917  | 超出原估计；保留 ASOF 证据     |
| 基线 interval 诊断修复 | 609.670  | 独立 instrumentation-only 来源 |
| 候选诊断               | 606.786  | 同一源码，单独 feature 构建    |

原 ASOF 的一次诊断预热和一次观察原样保留。interval 首次诊断预热因为事件
operator 名称为空而失败，随后只补一次预热和一次观察。原失败文件、SHA、
诊断源码修复和每 case 真实构建全部保留。因此实际共九次诊断调用：
四次成功预热、四次正式诊断观察及一回失败预热；成功样本与正式样本没有增加。该技术恢复不改变任何正式统计。
失败预热保留了错误报告与进程输出；校验器抛错前未写入其原始 trace，
因此没有该次失败调用的完整事件记录。

离线审核核实正式报告和全部原始 stdout 一致、独立状态目录库存、正确性及
完成证据、三类指纹与全部来源哈希、nearest-rank P95 和维护配对统计重算；
没有运行 native、构建或重新采集。以下原始证据保存在本机测量目录，
下列路径仅标识原测量机文件，不随仓库提供，也不是下载链接。
本报告与来源摘要随仓库归档；原始文件清单：

- 预声明采集口径（`target/performance/join-cpu-pipeline/acquisition-plan.md`）
- 固定计划（`target/performance/join-cpu-pipeline/plan.json`）
- 正式测量原始报告（`target/performance/join-cpu-pipeline/reports/measurement.json`）
- 正式测量逐配置摘要（`target/performance/join-cpu-pipeline/formal-measurement.log`）
- 正式来源与统计离线审计（`target/performance/join-cpu-pipeline/formal-offline-audit.json`）
- 基线诊断合并报告（`target/performance/join-cpu-pipeline/diagnostics/base/report.json`）
- 原失败报告原字节备份（`target/performance/join-cpu-pipeline/diagnostics/base-interval-repair/original-failed-base-report.json`）
- 一次 interval 修复报告（`target/performance/join-cpu-pipeline/diagnostics/base-interval-repair/report.json`）
- 诊断恢复离线审计（`target/performance/join-cpu-pipeline/diagnostic-recovery-offline-audit.json`）
- 候选诊断报告（`target/performance/join-cpu-pipeline/diagnostics/candidate/report.json`）
- 全部诊断来源与 trace 离线审计（`target/performance/join-cpu-pipeline/diagnostic-offline-audit.json`）
- ready 裁剪与阶段聚合（`target/performance/join-cpu-pipeline/diagnostic-analysis.json`）

## 适用范围与覆盖建议

本次五配置覆盖 ASOF 连续片段匹配/输出计划、interval retained-state 准备与
chunk 复制，以及它们共用的 gather executor admission；32 CPU trace 已确认
这些路径的实际 worker 调用。维护中的 EngineCase ASOF/interval 场景提供
端到端 oracle 与生命周期边界。ASOF 本次未触发复制型 row gather，不能声称
覆盖该路径的性能。

未测 ASOF 8 CPU、小输入、热点 key、宽列／变长／嵌套类型性能、长期稳态以及
checkpoint/recovery 耗时。正确性测试覆盖这些语义并不等于这些场景已有性能
证据。五配置子集不提供完整 benchmark suite 或全平台的整体通过结论。

后续覆盖建议按直接热路径排序，不在本次自动扩测：

1. 为未触发的 ASOF 复制型 row gather 在维护中的 ASOF 场景增加最小、明确
   不能共享输出 Arrow 列的 workload；使用既有 scale 和 oracle，并让 trace
   验证 gather worker 实际执行。现有连续片段场景保留作匹配/计划准备对照。
2. 为 interval retained-state admission/chunk 路径在维护中的 interval 场景补充
   明确的 coalesced 单批覆盖；现有标准 batch 几何不能替代新路径的触发证据。
3. 用维护中的小批场景检查共享 executor 的小工作回退，在现有 join
   materialization 目标补充需要时的宽列/变长 payload 覆盖；这些与多核阈值、
   内存复制成本有关，优先于继续扩展 CPU 或行数矩阵。
