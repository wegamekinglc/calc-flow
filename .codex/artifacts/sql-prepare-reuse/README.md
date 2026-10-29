# DAL-417 第一阶段：有限 SQL 准备复用

**固定 64×512 小批的完整消费中位延迟下降 46.9%，两轮成对区间均支持 improved。
整表的组中位数下降，但按同一规则仍为 inconclusive，不声称稳定提速。**

产品基线 `d9e2759fb31bfd0c915d31f0cc534665ece72ee9`，候选产品提交
`9db6d10b29bc6349b529d2807cfd23ba4dd6aace`。本次只优化有限 SQL 的重复准备，
不改绑定/Arrow 包装或 streaming。此前 Arroyo、窗口和 benchmark 审计证据保持原样，
没有重新测量或从历史样本拼成新的跨引擎倍数。

## 正式结果

同一份 32,768 行 Arrow 输入，过滤/投影后每次输出 16,384 行。每场景两轮、每轮十对
邻近 AB/BA，80 次正式观测全部逐列、逐值、按原顺序核对，共 **1,310,720 行输出**。
主要时钟为**已备输入放行 → 父进程完整消费 Arrow Table**，包含结果包装及输出 file IPC。
输入解码、声明编译和一次暖身位于 gate 前，冷准备单列；数据面始终 Arrow/Batch/IPC。

下表是各 side 的 20 次组中位数；吞吐单位为**输入行/秒**，不是成对比值。

| Input shape | Base rows/s | Head rows/s | Base p50 ms | Head p50 ms | Verdict      |
|-------------|-------------|-------------|-------------|-------------|--------------|
| 1x32768     | 9,258,608   | 12,680,119  | 3.539       | 2.584       | inconclusive |
| 64x512      | 614,386     | 1,157,049   | 53.335      | 28.323      | improved     |

小批 P95 为 62.007 → 32.740ms，整表 P95 为 5.465 → 3.755ms（20 次的 nearest-rank）。
仓库门禁使用每对**候选/基线耗时变化**的精确中位数区间，十对取第二、第九顺序统计量，
每轮覆盖率 97.8516%；阈值和两轮规则未改：

| Input shape | Round | Median change | Interval low | Interval high |
|-------------|-------|---------------|--------------|---------------|
| 1x32768     | 1     | -1.57%        | -28.78%      | +9.20%        |
| 1x32768     | 2     | -30.39%       | -35.60%      | +5.70%        |
| 64x512      | 1     | -48.45%       | -54.07%      | -42.44%       |
| 64x512      | 2     | -45.50%       | -51.35%      | -33.25%       |

整表区间上端仍超过 +5%，因此不确定；没有为了改变结论补样、删样或重跑。

## 成本定位与最小实现

预编译图此前仍在每次 `execute_async` 创建 run-scoped DataFusion runtime/session，
并重新逻辑/物理规划。独立诊断对两个既有 workload 分别调用公开同步/异步 API，
记录每次 DataFusion 阶段。基线 64×512 异步序列的观测为：

| Diagnostic stage                 | Observed ms |
|----------------------------------|-------------|
| RuntimeEnv acquisition, total    | 0.260       |
| Session initialization, total    | 5.425       |
| Logical planning, total          | 7.354       |
| Physical planning, total         | 18.126      |
| Execution, total                 | 2.331       |
| Async API plus packaging/getters | 58.565      |
| Sync API plus packaging/getters  | 46.337      |

诊断是独立单序列，包含逐次指标 getter，且构建阶段的主机负载未独占控制；不能用它
做稳定速度排名。DataFusion 的执行/collect 等有包含关系，不重复相加。跨层余项包含
图调度、校验、上下文创建/销毁、指标 getter、包装及 PyO3/Python↔Tokio 桥接；
同步/异步差值也包含线程执行位置和调度差异，**没有独立测得线程投递或 GIL 的独占份额**。
证据支持先处理重复规划，不支持把全部余项认定为可消除的调度成本。

实现只把现有受限 `QueryTemplates` 缓存提升到 **BatchExecutionPlan 私有边界**：

- 图、配置和 UDF snapshot 在编译计划内固定，缓存不跨计划共享，即使 fingerprint 相同。
- 复用要求查询文本、别名和精确 schema（含 metadata）一致；仅固定单分区策略启用
  跨调用复用，其他分区策略仍独立规划，避免输入规模影响分区布局。
- 沿用现有 immutable 函数和单表 row-local physical tree 的准入；聚合、stable/volatile
  查询等不扩大准入。每个计划最多八个模板，模板 scan 不保存输入行。
- 每次重新绑定为新物理执行树，并保留独立 session、内存池、指标、任务及取消 token。
  共享计划继续使用现有生命周期锁；没有把调用改成共享可变执行器。

独立诊断中，候选暖身后的小批物理规划次数为 0，基线为 64。正式采样不新增逐批指标
getter，复用已审 `sql_worker.api_calls`；详细末次指标在 IPC 后提取，明确标为末次查询。
正式 API 子阶段中位数如下（小批列是 64 次的连续序列/子阶段和）：

| Stage (ms)             | Base full | Head full | Base 64x512 | Head 64x512 |
|------------------------|-----------|-----------|-------------|-------------|
| API Arrow result ready | 1.165     | 0.696     | 49.840      | 25.056      |
| Native execute         | 1.071     | 0.600     | 45.919      | 21.307      |
| Input binding          | 0.034     | 0.038     | 1.663       | 1.618       |
| Result packaging       | 0.043     | 0.041     | 1.715       | 1.591       |
| Output IPC             | 0.441     | 0.456     | 0.737       | 0.779       |
| First warmup call      | 3.971     | 4.003     | 3.667       | 3.603       |
| Declaration/precompile | 3.774     | 3.740     | 3.690       | 3.664       |

Native execute 内含准备与执行，不与 API 总计重复相加。首次暖身调用包含首次模板准备，
未宣称降低冷调用成本；这里证明的是**复用同一个预编译计划**的收益。
便利 collect 每次创建新图，尚不获得此项跨调用收益。绑定/包装代码未改，其小幅波动
不能归功于 Arrow 包装优化。当前仍保留 session 初始化、物理树重绑定和异步桥接成本。

## 验证、预算与复现

- Python 定向回归 **9 passed**：重复/空/null/新输入的逐值结果和 IPC 输入字节不变，
  精确 schema/metadata，配置/图/同 fingerprint 不同 UDF snapshot，聚合/时变查询回退，
  暖计划并发、真实调用中取消、排队调用及后续恢复。
- Rust batch 范围 **11 passed**：模板不保留输入行/算子状态、既有分区策略及取消/deadline
  恢复。仅运行该范围；测试目标编译 23m28s，测试本体 0.07s。
- 80 次正式与四次预检的离线 Arrow/时钟审计通过，成对摘要精确复算；同步更新 hash 后的
  错值与遗漏正常 execute span 两个反例都被拒绝。验证器不依赖 Python assert。
- Rust fmt、相关 Python Ruff/format、实际 complexity ratchet、schema/whitespace 检查通过。
  全量回归、跨平台、Clippy 和覆盖率交现有 CI，门禁与基线未修改。

基线/候选 wheel 构建 908.746/572.901 秒单列。采集、环境、八个诊断序列、四次预检、
80 次正式及审计账本为 47.334 秒；另保守计入归档/校验预留 60 秒，共 **107.334/600 秒**。
正式观测 32.217 秒，0.1 秒轮询未见 cargo/rustc/maturin 构建进程。
CPU 4–7 跑 worker、0–1 跑 controller，Tokio 四线程，其余 helper 一线程；未锁频或独占 CPU。
没有用旧的不兼容 wheel。原始数据包含所有正式/预检输出、阶段时钟、命令、RSS、诊断、
构建日志、脚本和 native/wheel/source/dependency seals，不包含 wheel 或 native 二进制。

完整原始包 `data/DAL-417-sql-prepare-raw.tar.gz` 为 9,929,877 字节，SHA-256：
`0d84f4ef1117f46adeb57289680a8cb323abbbf162dca2c375085bc7cbceb04f`。
[预记方案](PLAN.md)、[实际命令与重现](REPRODUCE.md)、[成对摘要](data/summary.json)、
[身份](data/identity.json)、[验证](data/validation.json)、[预算](data/budget.json)。
新机器的新哈希和结果应另行记录，不回写这些已采样身份。

## 测量后的依赖策略修复

PR #516 的依赖策略检查发现 `libz-sys 1.1.29` 命中 `RUSTSEC-2026-0418`。
核对远端 `main@d9e2759f` 尚无修复后，仅将锁文件中的该包及 checksum 更新至修复版
1.1.31；其他包、依赖边、忽略策略和门禁保持原样。原 `cargo audit` 三项 ignore 参数
及 `cargo deny --locked check` 均退出 0，保留既有 duplicate/yanked warning。

`cargo tree --locked -p calc-flow-python` 确认本次默认 wheel/有限 SQL 的实际依赖图不含
`libz-sys`；全特性反向图显示它仅由可选 Kafka 的 `rdkafka-sys` 引入。上游修复移除了
构建脚本的环境命令日志；FFI `src/lib.rs` 与旧版相同，另有 WASI/aarch64 构建调整，
可选特性的跨平台构建兼容性由 CI 验证。本次没有重新构建或采样，复用前述 SQL 证据。

测量仍对应产品提交 `9db6d10b`、原 wheel/native/source 和锁文件身份。原始数据与
`data/identity.json` 不改写；依赖修复后的 head 并非新增已采样版本。新 `Cargo.lock`
SHA-256 为 `228b17f6384d1b1275755306dfe6c18d81d145f46de7994c1092080cbd38454e`。
