// Run each checkout's installed Vitest with raw samples retained on both sides.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { startVitest } from "vitest/node";

// Installed under each trusted checkout's node_modules/.cache by the adapter.
// No request-selected module or output path is imported or opened.
await mkdir("target/benchmark-suite", { recursive: true });
const context = await startVitest("benchmark", [], {
  root: process.cwd(),
  watch: false,
  benchmark: { includeSamples: true, outputJson: "target/benchmark-suite/vitest.json" },
});
try {
  await retainSamples(context);
} finally {
  await context.close();
}
// Vitest's JSON reporter deliberately removes samples, even with includeSamples.
// Recover the actual observations from its completed task results, never from
// median/mean estimates. Preserve the reporter's other fields unchanged.
function collect(task, entries) {
  if (task.meta?.benchmark) {
    const values = task.result?.benchmark?.samples;
    if (task.result?.state !== "pass" || !values?.length) {
      throw new Error(`Incomplete benchmark samples: ${task.name}`);
    }
    entries.push([task.id, [...values]]);
  }
  for (const child of task.tasks ?? []) collect(child, entries);
}

async function retainSamples(context) {
  const entries = [];
  for (const file of context.state.getFiles()) collect(file, entries);
  const samples = new Map(entries);
  const report = JSON.parse(await readFile("target/benchmark-suite/vitest.json", "utf8"));
  const count = attachSamples(report, samples);
  if (count === 0 || count !== samples.size) {
    throw new Error("Incomplete frontend benchmark inventory");
  }
  await writeFile("target/benchmark-suite/vitest.json", `${JSON.stringify(report, null, 2)}\n`);
}

function attachSamples(report, samples) {
  let count = 0;
  for (const file of report.files) {
    for (const group of file.groups) {
      for (const benchmark of group.benchmarks) {
        const values = samples.get(benchmark.id);
        if (!values || values.length !== benchmark.sampleCount) {
          throw new Error(`Benchmark inventory mismatch: ${benchmark.name}`);
        }
        benchmark.samples = values;
        count += 1;
      }
    }
  }
  return count;
}
