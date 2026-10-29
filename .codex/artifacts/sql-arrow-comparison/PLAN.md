# DAL-417 SQL / 真实 Arroyo 补测预记

接续已合并 #511，专用 checkout 从 `816a9174373d7df659748d5775977eadb308779a`
建立 `agent/cf-performancer/dal-417-sql-arrow`。产品基线仍为 b5ae227e，sealed CF native
a0708fcf…；Arroyo v0.15.0/4ef15f6 的已验证 IPC binary 40588f3d…，不重建。
只新增本目录，不修改 doc-writer 的 README/REPORT 入口，不重跑 streaming/窗口。

固定原 32,768 行过滤/投影，两个场景：1×32,768、64×512。
A=CF SQL collect_async，B=真实 Arroyo，C=CF SQL 预编译 graph execute_async。
每场景五组，组顺序 ABC/CBA/ABC/CBA/ABC；B 总在中间，两个 CF 路径分别与 B 邻近且交替
前后，共 30 次正式观测。预检每场景每路径一次，共六次，单列且不排名；正式不因噪声补样。
预检、失败、正式采集累计硬预算 600 秒，每次调用上限 45 秒；采前必须有剩余配额。
机器、wheel、Arroyo、输入字节/批形状和资源复用 #511：worker 4–7，controller 0–1，
Tokio 4 threads，helper 单线程、Arroyo -p1。没有 CPU 独占/锁频保证。

主要可比较指标为**ready-input gate → 父进程完整 Arrow Table 可消费**，双方 file IPC，
输入解码/初始声明及预编译准备在 ready 前，输出 encode/rename/read_all 在共同区间内。
该指标是准备就绪后的执行+结果消费吞吐，包含输出 IPC 与门控/观察开销，不叫纯计算。
另外列出 CF 完整 API→结果 Table、Arroyo native core、公开列包装、IPC 与启动。
Arroyo core+结果 projection 是顺序阶段和，非额外的连续时钟，不与旧 streaming 样本混算。

CF public collect 的正常 input binding / prepare+native graph compile / output _tables
均保留在 API 时钟内；对三个同步函数仅加嵌套时钟，完整保留正常调用与返回。
预编译图的声明/降低/compile 在 ready 前，调用中 Batch.from_pyarrow、execute_async、
outputs/to_pyarrow、concat 均计入。图预编译不宣称跨调用 DataFusion 物理计划缓存。
结果包装成本单列为内含子阶段，不能再次从总计扣除以制造排名；跨线程 span 不相加冒充
独占 CPU。每次完整输出严格核对列、值、原顺序，不按顺序排序掩盖差异。

数据面保持 Arrow/Batch/IPC，JSON 仅图、配置、统计。保留预检/失败、全部 IPC、时钟、
RSS、命令/退出及脚本 hashes。正式结果与原 CF/Arroyo 历史表分开；最终单独结果页同时
列出已有 streaming 数据的独立边界，禁止跨时段拼成新的成对结果。
