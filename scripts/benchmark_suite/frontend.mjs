// Run each checkout's installed Vitest with raw samples retained on both sides.
import { createRequire } from "node:module";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [sourceArgument, outputArgument] = process.argv.slice(2);
if (!sourceArgument || !outputArgument) {
  throw new Error("Expected frontend source directory and JSON output path");
}
const source = resolve(sourceArgument);
const require = createRequire(join(source, "package.json"));
const { startVitest } = await import(pathToFileURL(require.resolve("vitest/node")));
const context = await startVitest("benchmark", [], {
  root: source,
  watch: false,
  benchmark: { includeSamples: true, outputJson: resolve(outputArgument) },
});
// Vitest's JSON reporter deliberately removes samples, even with includeSamples.
// Recover the actual observations from its completed task results, never from
// median/mean estimates. Preserve the reporter's other fields unchanged.
const samples = new Map();
function collect(task) {
  if (task.meta?.benchmark) {
    const values = task.result?.benchmark?.samples;
    if (task.result?.state !== "pass" || !values?.length) {
      throw new Error(`Incomplete benchmark samples: ${task.name}`);
    }
    samples.set(task.id, [...values]);
  }
  for (const child of task.tasks ?? []) collect(child);
}
for (const file of context.state.getFiles()) collect(file);
const output = resolve(outputArgument);
const report = JSON.parse(await readFile(output, "utf8"));
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
if (count === 0 || count !== samples.size) {
  throw new Error("Incomplete frontend benchmark inventory");
}
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
await context.close();
