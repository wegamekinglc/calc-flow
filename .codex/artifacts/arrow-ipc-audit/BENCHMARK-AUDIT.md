# 维护中 benchmark 的计时审计

固定 `b5ae227e48ac9bb8a81583a23485e2c6064cee52`，仅静态追踪及一个确定性时钟 probe；
没有运行完整集合，没有修改 benchmark、统计门禁或基线。`static-inventory.json` 保存
20 个维护 shard、12 个 Rust target、源文件 SHA 和逐行计时/转换锚点。
测试名含 benchmark 不等于性能入口：`test_engine_comparison.py`、`test_retained_stream.py`
等主要验证 harness 正确性，不是额外性能采样组。

## 从入口到 CI 的实际数据流

`.github/workflows/benchmark-suite.yml:55–58,295–304` 解析 ref/catalog，调用
`python -m scripts.benchmark_suite run --shard ...`。catalog 有四个 Python scale、
七个 engines scale、七个 warm scale、Rust 和 lifecycle，共 20 shard。

- engines/warm：`measure.py:111–152` 启动长期 worker，prepare/warmup 后按 AB/BA
  请求 sample；`worker.py:129–167` 调用 active.sample 后才 JSON 编码元数据。
  测量使用 sample 返回的 `seconds`，不是父进程 request RTT。
- Python/lifecycle：`legacy.py:22–80` 调 pytest 的 `benchmark(callable)` 时钟，保存
  原始样本；`normalize.py:26–43` 读取 stats.data。lifecycle 另调用 evidence verifier，
  不以一个全程秒数替代阶段/恢复/资源要求。
- Rust：`rust.py` 从 Cargo manifest 发现 target，构建和进程启动在各 target 的内部时钟
  之外。Criterion 的 ns/iteration 经 `normalize.py:46–67` 归一为秒；自定义 JSON report
  经对应 asof/join/sql/window loader 验证完整 inventory 和 correctness 后读取 seconds。
- `report.py:66–146` 汇总 P50/P95/min/max，N/P50 为吞吐；P95 是 invocation 时长分位数，
  不是逐行延迟。engine/warm 版本比较为两轮十对及成对中位数区间；legacy whole-suite
  blocks 的 minima 只给 informational。`aggregate.py:20–42,118` 只将明确 regression/
  证据无效等判失败。不能把 whole-suite minima 或本轮五次描述性结果替代正式门禁。

JSON 在这条主测量链中通常是**配置/统计元数据**，不传输被测结果表。
定向 probe 使用真实 1,024 行 CF SQL execution，替换的只是 harness 的时钟，事件顺序为
`fixture/compile → clock start → real execute → materialized Arrow → clock stop →
full validation → JSON metadata`；回复只有 seconds/correctness，未传 JSON 数据行。
`timer-probe.json` 的 1µs 是确定性测试值，不是速度实测。

## 逐组边界矩阵

G=数据生成，T=类型转换/序列化，S=进程/作业/事件循环启动，D=声明/图编译，
P=DataFusion session/物理规划，R=所声明阶段的结果形成/消费，V=该组现有正确性校验。
V 不保证每组都验证所有值；下面另列普通历史 pytest 的校验覆盖缺口。
外=在所列时钟之外；内=包含；混=按子场景区分；—=无该阶段。
T 包括正常 Batch/Arrow/API 包装，不表示必然复制数据，更不表示 JSON。

| 组                                                      | 计时入口/位置                                                                                | G / T / S / D / P / R / V                | 指标真正衡量什么                                                                                                                              |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Python DataFusion 表计算、runtime fan-out               | `test_datafusion.py:50–70`；`test_runtime.py:29–43`                                          | 外/内/外/外/内/内/外                     | 已编译图的完整 plan.execute，含 fresh-run/session、规划、结果 Batch/PyO3；不是只测 DF kernel                                                  |
| Symbolic 普通执行/rolling/cross-section/matmul          | `symbolic_support.py:349–360`；`test_symbolic_baseline.py:1103–1245`                         | 外/内/外/外/混/内/外                     | 已编译图执行；matmul provider/数组边界是正常 API 成本                                                                                         |
| Symbolic SCE 和 rolling-kernel paired outer benchmark   | `test_symbolic_baseline.py:911,966`；`test_rolling_kernel.py:163–169`                        | 外/内/外/外/混/内/外                     | 一个 outer sample 连续执行 reference 和 candidate 两方案；部分 reference 本身含两个 plan；独立成对时钟在 extra_info                           |
| Array backend_kernel                                    | `array_support.py:192–227`；`test_array_kernel.py`                                           | 外/外/外/外/—/内/外                      | 原 NumPy/JAX 运算并等待 JAX 完成，可称 backend kernel 边界                                                                                    |
| Array provider_boundary                                 | `test_array_provider.py:32`；`array_support.py:201`                                          | 外/内/外/外/—/内/外                      | 解析/验证/调用 provider、Python/native 与 ownership；不是裸数组 kernel                                                                        |
| Array plan_end_to_end                                   | `test_array_plan.py:40–100`                                                                  | 外/内/外/外/—/内/外                      | 已编译 plan.execute 加同步，provider registration/输入/warmup 不计时                                                                          |
| Array batch_ownership                                   | `test_array_ownership.py:23`                                                                 | 外/内/外/—/—/内/外                       | Batch.from_array 的所有权/复制边界；不是算术吞吐                                                                                              |
| Public Program.execute SQL                              | `test_program_engine.py:94–120`                                                              | 外/内/外/内/内/内/外                     | Program 对 Arrow 的正常 convenience 执行，输入 immutable，声明已建但 lowering/compile 可在调用内                                              |
| Public Program.execute streaming、九类 stream lifecycle | `test_program_engine.py:63,94`；`test_stream_operators.py:223–264`                           | 外/内/内/内/混/内/外                     | asyncio.run、native job、源/任务/通道、EOF/关闭和 Arrow 拼接，明确 lifecycle-to-Arrow                                                         |
| Engine matrix CF SQL                                    | `engine_comparison.py:311–330,655–684`                                                       | 外/内/外/外/内/内/外                     | PipelineBuilder 已编译；每次 plan.execute 仍可能新建 DF runtime/物理计划；输出 to_pyarrow 计时                                                |
| Engine matrix raw DataFusion                            | `engine_comparison.py:333–351,667`                                                           | 外/内/外/内/内/内/外                     | 新建 context、注册预备 batches、SQL 规划、collect、Table 包装                                                                                 |
| Engine matrix Polars/1T                                 | `engine_comparison.py:397–473,667`                                                           | 外/内/外/外/内/内/外                     | 预建 lazy plan 的 collect(engine=streaming) 到 Arrow，含引擎正常计划/转换                                                                     |
| Engine matrix TA-Lib、Finance-Python                    | `rolling_indicator_comparison.py:109–138`；`finance_python_rolling_runner.py:87–116,136–160` | 外/内/外/混/—/内/外                      | TA-Lib 切片/连续复制/重组；Finance transform 构造 expression、pandas→NumPy 与 NaN 规范化。不是相同裸 kernel 边界                              |
| Engine matrix CF ready stream                           | `engine_stream.py:465–484,566–614`                                                           | 外/内/外/外/混/内/外                     | ready 源放行到全部 Arrow 表拼接；包含 source/task/channel/WM/状态，启动和 EOF/关闭不计时；lazy 首批 DF 初始化仍可包含                         |
| Small-batch/replay 与 checkpoint-duration               | `engine_stream.py:835–877,949–1026`                                                          | 外/内/混/混/混/内/外                     | throughput ready 路径；checkpoint variant 刻意包含 pacing、非终止 checkpoint ack、取消/恢复与验证恢复输出，不能标为纯计算                     |
| Warm incremental                                        | `warm_stream.py:433–470`                                                                     | 外/内/外/外/混/内/外                     | 一个持续 runner 的增量 enqueue→Arrow receive；数据/WM 在开始前准备；首轮历史预热和最终关闭另计                                                |
| Lifecycle checkpoint/recovery 特殊组                    | `test_symbolic_baseline.py:1340–1439,1447–1525`                                              | 外/内/内/内/混/内/外                     | 编译/start/checkpoint/取消/恢复/shutdown、status 轮询和文件大小检查；20 次诊断阶段分位数与 pytest 外层样本是两类证据                          |
| Rust core Criterion                                     | `benches/core.rs:42–85,190–253,336–423`                                                      | 混/混/混/混/混/内/混                     | compile、runtime new、canonical JSON、channel/operator/fanout/backpressure 各自明确计时；输入 clone/map、锁/任务构造在迭代内；不是统一 kernel |
| Rust stream_union                                       | `benches/stream_union.rs:36–60`                                                              | 外/内/外/外/—/内/内                      | 两个 process_data、context/锁/Batch clone/collector drain，含小型输出数量断言，无数据 JSON                                                    |
| Rust m4_state_window                                    | `benches/m4_state_window.rs:59–137,185–233,294–324`                                          | 外/内/混/外/—/混/混                      | state I/O、hash/handle、window reset/admission/drain、checkpoint/restore 分组；process_data 不等于窗口最终关闭/全部聚合输出                   |
| Rust allocation_regression                              | `benches/allocation_regression.rs:1312–1370`                                                 | 外/内/外/外/混/内/内                     | 受控线程的 dispatch allocation count/bytes/peak，部分检查在测量 closure 内；不是墙钟性能数                                                    |
| Rust stream_join_perf                                   | `benches/stream_join_perf.rs:664–920`                                                        | 外/内/外/外/混/内/混                     | iter_custom 只累计标定的右输入处理/WM/checkpoint 阶段；左状态预载、operator 创建通常不计时，子场景需看各 start/end                            |
| Rust stream_join_materialization                        | `benches/stream_join_materialization.rs:221–270`                                             | 外/内/外/外/混/内/外                     | 完整输出物化、bounded edge、consumer drain 与故意的 slow-sink sleep；oracle=true 独立运行，正式 validate=false                                |
| Rust stream_asof_perf                                   | `benches/stream_asof_perf.rs:153–180,276–351`                                                | 外/内/外/外/—/内/外                      | 只计 watermark settlement 和 collector；admission、restore、checkpoint capture 与全行校验分列在外                                             |
| Rust stream_asof_e2e                                    | `benches/stream_asof_e2e.rs:246–315`                                                         | 外/内/外/外/—/内/混                      | admission 到 settlement 完整原生算子；eviction_ticks 子场景的逐 tick status 断言在 timer 内                                                   |
| Rust stream_asof_eviction                               | `benches/stream_asof_eviction.rs:374–408`                                                    | 外/内/外/外/—/内/外                      | 只累计若干 on_watermark 调用；status 验证、checkpoint/hash、recovery probe 在外，无输出吞吐分母                                               |
| Rust stream_sql_aggregate                               | `benches/stream_sql_aggregate.rs:480–515`                                                    | 外/内/外/外/内/内/外                     | 所有批增量聚合/快照，配置要求的 checkpoint prepare/capture 也包含；恢复和 snapshots oracle 在外                                               |
| Rust stream_window_groups                               | `benches/stream_window_groups.rs:400–445`                                                    | 外/内/外/外/—/内/外                      | process_data 与 on_end 两段之和；中间 checkpoint/hash 不包含，聚合最终 Arrow 结果才结束                                                       |
| Rust sql_datafusion_performance                         | `benches/sql_datafusion_performance.rs:632–659,727–910`                                      | 外/内/外/外或内/内/内/外                 | CF plan.execute envelope 对照 raw session/注册/规划/全流 collect/Batch envelope；另有 first-batch/phase 指标，不能只取 first batch 冒充全结果 |
| Connector decode throughput（CI Rust 附加步骤）         | `.github/workflows/benchmark-suite.yml:319–335`；`protobuf.rs:734–784`                       | 外/内/外/外/—/内/混                      | 预备 protobuf/JSON bytes→Arrow decode 丢弃结果；确实计 JSON 解析，是显式 codec benchmark，不是图间 JSON 或纯计算排名                          |
| Standalone rolling comparison                           | `rolling_indicator_comparison.py:532–562,657–686`                                            | 外/内/外/外或混/混/内/外                 | native complete RunResult、Finance transform、TA-Lib 多 symbol adapter 各自边界；worker 控制 JSON/输出 digest 在外，不能说对称 kernel 比较    |
| Standalone release / recovery / diagnostics             | `release_collectors.py:27–90`；`issue406_run.py`；`native_memory_profile.py:165–240`         | 按目标/按目标/分列/按目标/按目标/内/分列 | 继承单 case 原时钟/封存 wheel；额外 CPU、RSS、checkpoint 与准备预算是诊断证据，不是统一计算样本                                               |

矩阵中的 T=内需要分清：CF 的 Batch/PyArrow 包装、session 创建、运行结果物化、数组同步
属于公开 API 的正常工作；TA-Lib 的重组、Finance 的数据框适配、source/observer 记账是
测试/集成适配成本；JSON/Protobuf codec 和 state checkpoint 则是在测被命名的协议/存储工作。

## 已有数据对照与需要修正的解释

`existing-evidence.json` 从当前仓库原文件提取，未重算旧性能或回写数字：

- `benchmarks/sql-operators/results.json` 有 98 个 `execute-to-arrow` 单元，首个单元
  是 2×10 样本、P50=0.5157665ms、new-coverage，N/P50=19,388.62 行/秒。
  这证明当前汇总是包含规划/包装的 execute-to-Arrow 指标，不是独立 DF kernel 时间。
- `benchmarks/polars-operators/results.json` 有 84 单元，同样 2×10，首项
  P50=4.623768ms、external-reference。不能拿不同形状/库的 adapter 时间解释成内核差异。
- `benchmarks/symbolic/baseline-b9deff7-run1.json` 保留六个历史项；首项 extra_info 的
  datafusion_planning_ns=13,757,316、execution_ns=47,637,097 是独立 warmup 结果。
  源码 `test_datafusion.py:58–69` 和 symbolic 调用明确在 benchmark 开始前取这组 metrics。
  它不能拿来当各次 pytest 样本的分解，也不能与 pytest P50 直接相减。

需要澄清/后续修正的指标（本轮只报告，不修改原合同）：

1. **撤回把旧 JSON 出口热点当作核心产品优先级。** #494/#510 的结果只约束其外部集成
   适配；新核心测量仅 Arrow/Batch/IPC。历史原始证据保留，不重命名旧数据以冒充新口径。
2. **元数据名称/口径应更精确。** 普通 Python 的 datafusion_* 建议加 warmup 前缀，
   process_rss_bytes 是测前 RSS 快照，不能称峰值；需要实测峰值时另加有明确采样边界的新项。
3. **成对 outer sample 的吞吐含义需标注。** Symbolic/rolling-kernel outer callable 执行
   reference 和 candidate 两种方案，SCE14/multistage 的 reference 本身由两个 plan 组成；
   用一个 N 除其总时间只能称 pair 工作单位/秒，不能标为 candidate 吞吐。
   独立 paired_samples 仍是原门禁依据，不要从 outer pytest 时长推导候选单次加速。
4. **生命周期时钟不可混用。** pytest 包装了 asyncio.run、目录/graph 准备；内部 phase
   则在 lifecycle() 内开始。阶段分位数来自独立诊断，不能假设与 outer P50 同一次执行。
5. **少量真实测试适配在 timer 内。** union 数量断言、ASOF eviction_ticks status 检查、
   collector 计数、slow-sink sleep 是实际包含的成本。前三者适合在新合同中分列/移出，
   sleep 属于明确背压负载，不能当计时错误删除。任何改变都需要新 workload identity。
6. **结果完整性有契约边界。** WarmScenario 只 receive 一个非空 Arrow 表，再校验行数和
   全量值；当前 eligible 输入/输出按一个增量结果验证，若未来 operator 分块，现有路径
   会被行数校验拒绝，不能把首块时间当成成功样本。扩展时应按 expected_rows 聚合后结束。
7. **离线 JSON fixture/export 仍然存在。** Join/ASOF acceptance 的
   `join_asof_acceptance.py:368–472` 把参考/native 事件写到 fixture.json 以便独立 harness
   读取；它在处理时钟前完成，不污染处理时间，但不满足“整个测试准备路径也不写行 JSON”
   的更严格口径。建议后续迁移该准备/交接数据到 Arrow IPC，保留 JSON 配置和指标。
8. **正确性覆盖需按组陈述。** 普通历史 pytest 部分只断言行数、形状或 dtype，例如
   `test_datafusion.py:102` 的投影与 `test_runtime.py:45` 的输出数。其 pytest 成功不等于
   benchmark 内已做全行 oracle。现有 engine matrix 和本轮重测有逐值校验；建议为其他组
   增加明确的 validation_scope 元数据，必要时补 timer 外 oracle，不改动历史采样口径。

本次没有发现主矩阵在 `seconds` 停止后才执行计算、把 JSON 行发送时间偷偷计成核心时间，
或当前 normalize/成对统计公式单位错误的证据。上面的命名/解释问题、少量 harness 成本
和主动测 codec/存储的时间不能混为一种错误。未隔离 timer 内轻量观测函数的因果占比，
所以没有为移出断言承诺速度收益，也没有取消既有 gate。
