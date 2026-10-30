# DAL-417 第三阶段：Arrow 绑定与包装

本轮完成最小包装改动及定向验证，**整表、小批完整消费性能均为 inconclusive**。
尚不能证明稳定提速、等价或排除回归；保留全部样本，没有补样追求通过。
小批结果包装阶段中位数 1.461→1.299ms 是描述性观测，不单独作为稳定收益结论。

| 场景        | 完整消费中位延迟：基线→候选 | 绝对输入吞吐中位数：基线→候选 | 两轮结论     |
|-------------|-----------------------------|-------------------------------|--------------|
| 1×32768     | 2.574→2.432ms               | 1273.2 万→1347.3 万行/s       | inconclusive |
| 固定 64×512 | 25.104→25.052ms             | 130.5 万→130.8 万行/s         | inconclusive |

整表两轮成对耗时变化中位数 −9.99%、−1.78%，精确中位数区间分别
[−46.57%, −1.41%]、[−12.91%, +9.01%]；小批 −0.81%、−1.95%，区间
[−6.51%, +5.81%]、[−14.51%, +5.32%]。沿用仓库 `paired_round`/`_paired_verdict`
两轮 +5% 规则，十对样本的实际区间覆盖率 97.8515625%。延迟和吞吐分别取各次观测
中位数；不能由某一阶段或两个中位数的比值替代成对判定。

## 定位与最小改动

基线固定 `main@1041bb7b6d9bd3a16686b8753e856e9f12f2529f`，独立于 DAL-475 实施。
基线小批诊断：正常输入绑定 1.492ms、native execute 21.587ms、结果包装 1.471ms、
concat 0.056ms。cProfile 开启会扰动时间；它只用于定位，不参与正式统计。
concat 使用既有 PyArrow 默认共享 buffer 路径，未改调用，也没有将包装移到计时之外。

产品提交 `3c66dfefba618ace56cb3430f244f27d37453b96`：

- `Batch.from_pyarrow` 从三次协议属性查找改为一次当前 `__arrow_c_stream__` 查找/调用。
  缺少协议的原 TypeError、producer 异常、capsule 导入、逐批校验及 metadata 校验顺序保留。
  定向回归在旧 native 上实际观察到三次查找；候选一次，并在下次导入重新读取动态 exporter。
- `Batch.to_pyarrow` 使用既有 `PyRecordBatchReader` 安全 owned-reader API，将一个 owned
  RecordBatch 引用集合转交给当前 `pyarrow.table`。避免中间 PyTable 的重复 schema 扫描
  和再次克隆记录集合；核心不可变 TableBatch 已在唯一构造边界检查精确 schema。
  每次导出仍创建独立 reader、动态查找当前构造器，未缓存 Python Table/版本判断或输入行。
- `RunResult.outputs` 只复制输出映射，不再连带复制查询指标、计划字符串及 run metadata。
  锁在创建 Python 对象前释放；每次仍返回新字典/Batch 包装，原 payload rehome、GC 和
  cleared-state 错误保留。公共接口、依赖及 `_native.pyi` 无变化。

## 完整消费与阶段

复用第一阶段 SQL 预编译正常接口、Arrow fixture 和严格列/值/顺序 oracle。
两场景各两轮十对相邻 AB/BA，共 **80 次正式观测、1,310,720 行输出**，另四次预检、
基线/候选各两次诊断。每个新进程只暖身一次，在正式 gate 之前完成并单列冷成本。
正式输入逐次正常绑定，没有提前转 Batch；输出经过正常 RunResult.outputs/to_pyarrow、
concat/IPC，主计时结束于父进程 read_all。表行只经 Arrow/Batch/IPC，不使用 JSON 表输出。

| 阶段中位数（ms）      | 整表基线 | 整表候选 | 小批基线 | 小批候选 |
|-----------------------|----------|----------|----------|----------|
| 声明/预编译           | 3.146    | 3.186    | 3.195    | 3.279    |
| 冷准备/首批暖身       | 2.974    | 3.058    | 2.629    | 2.836    |
| 正常输入绑定          | 0.026    | 0.025    | 1.471    | 1.412    |
| native execute        | 0.546    | 0.576    | 19.532   | 18.973   |
| 结果映射及 Arrow 包装 | 0.031    | 0.030    | 1.461    | 1.299    |
| concat                | 0.007    | 0.007    | 0.060    | 0.059    |
| 正常 API 结果可用     | 0.617    | 0.647    | 23.112   | 22.174   |
| 输出 IPC              | 0.369    | 0.360    | 0.607    | 0.613    |

阶段时钟包含嵌套/跨层余项，不相加中位数或作独占因果归因。末次查询详细指标在输出
IPC 后读取，单独标明，不乘以批数。双方暖计划 physical_planning_count 均为 0。
所有正式输出的精确 schema/metadata 和 IPC 批形状与各场景首个基线输出一致。

controller CPU 0–1、worker CPU 4–7、Tokio 4，其余计算线程 1；硬件 i9-13900HX，
无独占/锁频保证。候选诊断/预检/正式的连续 28.394 秒按 100ms 检查构建活动，未观察到
cargo/rustc/maturin。含环境准备及保守归档审计预留的累计账本 **104.251/600 秒**。
构建单列：native wheel 222.576 秒，Rust 定向测试编译/执行 231.529 秒，Clippy 5.984 秒。

## 身份与验证证据

基线复用原第二阶段候选 wheel（源码 `cfa2a2a2`），该提交与指定 main 的 Rust/Python
产品源码、manifests、Cargo.lock 和 toolchain 完全一致；保留原身份，不冒充新构建。
基线 native SHA256 `d7b690a6c97c9bcc6def157907b5d321cb22f66c79cd1415ba5da932634c64f4`。
候选构建源与已提交产品逐一核对 813 个产品/配置文件及锁文件，native SHA256 为
`fa14e7484c59e172057017de998bf1df1405362ef14b6c90a673908f515d0841`。
实际 wheel/source/env 哈希、命令及资源记录在原始包和 `data/identity.json`、`data/seals.json`。

候选 Python **40 passed**（Batch/metadata 34 项，结果/并发取消/便利层/背压 6 项）；
Rust binding **18 passed**（Batch 模块 15 项，结果副本、外部 payload 循环及 cleared-state
三项）。覆盖精确 schema/field metadata、chunk/null、零行/零列、只读输入、重复导出及
原 Batch 释放后的 buffer 所有权，动态 exporter/当前构造器拒绝和 producer/metadata 错误顺序。
动态版本拒绝用构造器 guard 模拟，确认没有冻结或绕过当前检查；真实版本矩阵交 CI。
Rust fmt、相关 Clippy/Ruff/format、原复杂度门禁、schema/差异检查通过，未改门禁或基线。

原始输出逐值/顺序、输入形状、跨双方精确 schema/metadata/输出批形状、阶段库存/时钟、
seal/hash 和成对统计重算通过。干净固定源通过，staged/unstaged/untracked 产品源均拒绝；
更新 hash 后的错值、遗漏输入绑定时钟也被拒绝。原始包解包后再次复核全部正式观测。
封存自检额外覆盖“index 已改、工作树反向恢复基线”的边界，交付入口增加独立 cached diff
并拒绝此反例。采样时产品源码已提交且干净；采样版 `sampling/collect.py`、交付版
`harness/collect.py` 及各自哈希分别保留，产品/wheel 和观测数据不变，无重采样。
实际命令见 `REPRODUCE.md`、`data/validation.json`；包和记录校验见 `data/SHA256SUMS`。
原 SQL、streaming、Arroyo/窗口和历史 benchmark 审计证据复用，未重跑全量或追加性能场景。
本批交协调器聚焦审查及完整 CI；代码合并条件与稳定性能收益保持分别记录。
