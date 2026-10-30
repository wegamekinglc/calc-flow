# DAL-417 第三阶段：Arrow 包装

基线固定 main `1041bb7b6d9bd3a16686b8753e856e9f12f2529f`，在本任务专用后续分支实施。
使用基线既有接口/依赖，不等待或复用 DAL-475 改造。先诊断 Batch.from_pyarrow、
RunResult/Arrow 导出及 concat；只消除可由现有安全 API 避免的重复绑定、检查或包装。
不得引入 unsafe、依赖 fork、缓存调用者可变输入或跳过动态协议/版本检查。

只测既有 SQL 预编译转换：整表 1×32768，固定 64×512。沿用第一阶段的 worker/API、
Arrow fixture、严格列/值/顺序 oracle、gate 到父进程 read_all 完整消费边界。
正常输入绑定、执行、RunResult 输出包装、to_pyarrow、concat 和输出 IPC 都留在对应时钟内。
首次冷调用暖身单列，不隐藏编译/冷准备成本；不预绑定正式输入或提前导出结果。

预先固定两轮、每场景每轮十对相邻 AB/BA，偶数 base/head、奇数 head/base，共 80 次。
另四次预检；基线/候选各两场景一次诊断，共四次。诊断开 cProfile，不进入正式统计。
固定样本数，不因 inconclusive 补样；沿用仓库 paired_round/_paired_verdict 的两轮 +5% 规则。
Arrow/Batch/IPC 承载表数据，JSON 仅记录时钟、配置、计数和身份。

累计采集硬预算 600 秒，含 fixture、身份/环境准备、诊断、暖身、预检、失败、进程启动、
清理和原始审计。每次观测 30 秒上限，至少保留 35 秒退出清理余量。构建单列：
基线若能证明产品代码、锁文件、编译配置及依赖兼容则复用原 wheel 身份；候选 native
重建预估 4–10 分钟，额外定向单测/Clippy 编译时间亦单列。
正式采集持续监测 cargo/rustc/maturin，发现本机构建争用即停止，保留已完成数据。
worker CPU 4–7、controller CPU 0–1、Tokio 4，其余计算线程 1，无独占/锁频保证。

直接回归覆盖只读输入、精确 schema/field metadata、动态协议/构造器查找、分块/空/null、
重复导出、所有权和 buffer 生命周期、错误先后/清理；沿用既有异步取消定向用例。
全量、跨平台和覆盖率交 CI；原 SQL、streaming、Arroyo/窗口及审计证据保留。
