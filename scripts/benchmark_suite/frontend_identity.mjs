import { cpus, hostname, release, totalmem } from "node:os";

export function machineIdentity() {
  const processors = cpus();
  return {
    platform: process.platform,
    architecture: process.arch,
    hostname: hostname(),
    kernel: release(),
    cpu_models: [...new Set(processors.map((cpu) => cpu.model))].sort(),
    logical_cpus: processors.length,
    memory_bytes: totalmem(),
    node_version: process.version,
    v8_version: process.versions.v8,
    node_options: process.env.NODE_OPTIONS ?? "",
    uv_threadpool_size: process.env.UV_THREADPOOL_SIZE ?? "",
    runner_name: process.env.RUNNER_NAME ?? "",
  };
}
