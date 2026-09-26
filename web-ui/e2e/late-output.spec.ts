import { expect, test } from '@playwright/test';

import { deleteWithLaunchToken } from './session';

const projectsUrl = 'http://127.0.0.1:8765/api/v3/projects';

test.use({
  launchOptions: {
    args: ['--disable-gpu', '--disable-software-rasterizer'],
  },
});

test('imports, binds and validates both late-output ports without losing delivery', async ({ page, request }) => {
  const id = `late_output_${process.pid}`;
  const fields = [
    { name: 'ts', data_type: 'timestamp[us, UTC]', nullable: false },
    { name: 'symbol', data_type: 'string', nullable: false },
    { name: 'seq', data_type: 'uint64', nullable: false },
    { name: 'x', data_type: 'float64', nullable: false },
  ];
  const connector = { provider: 'calc-flow-connectors', name: 'file', version: '2.0.0' };
  const document = {
    format_version: 3, id, name: 'Late output Studio workflow',
    runtime: { mode: 'stream', options: {} },
    graph: {
      name: 'late-output-studio', edges: [], nodes: [{
        id: 'roll', position: { x: 100, y: 100 },
        input_ports: [{ name: 'input', kind: 'table', required: true, schema: fields }],
        output_ports: [],
        operator: { kind: 'rolling', spec: {
          configuration_version: 1, state_layout_version: 1,
          partition_by: ['symbol'], event_time: 'ts', sequence_by: ['seq'],
          allowed_lateness_micros: 3, value_policy: 'stateful_numeric_v1',
          late_policy: { kind: 'side_output', metrics_version: 1, schema_version: 1 },
          outputs: [{ kind: 'lag', primitive_version: 1, input: 'x', output: 'previous', periods: 1 }],
        } },
      }],
    },
    sources: [{
      binding: 'input', connector, schema: fields,
      options: { path: 'late-workflow-input.parquet', format: 'parquet' },
      watermark: { policy: 'bounded_out_of_orderness', column: 'ts', delay_ms: 1, emit_interval_ms: 1 },
    }],
    sinks: ['output', 'late'].map((binding) => ({
      binding, connector, delivery: 'at_least_once',
      options: { path: `late-workflow-${binding}`, output: 'rows' },
    })),
  };
  await deleteWithLaunchToken(request, `${projectsUrl}/${id}`);
  try {
    await page.goto('/');
    await page.getByLabel('Import project').setInputFiles({
      name: 'late-output.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(document)),
    });
    await expect(page.getByLabel('Project', { exact: true })).toHaveValue(id);
    const node = page.locator('.react-flow__node').filter({ hasText: 'roll' }).first();
    await expect(node.locator('[data-handleid="output"]')).toBeVisible();
    await expect(node.locator('[data-handleid="late"]')).toBeVisible();
    await node.click();
    await expect(page.getByText(/no Watermark\/Idle/)).toBeVisible();
    await expect(page.getByLabel('Late row policy')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Batch', exact: true })).toBeDisabled();
    const outputs = page.getByLabel('Graph output', { exact: true });
    await expect(outputs).toHaveCount(2);
    await expect(outputs.nth(0)).toHaveValue('output');
    await expect(outputs.nth(1)).toHaveValue('late');
    const deliveries = page.getByRole('combobox', { name: 'Delivery', exact: true });
    await expect(deliveries).toHaveCount(2);
    await deliveries.nth(1).selectOption('best_effort');
    await expect(deliveries.nth(1)).toHaveValue('best_effort');
    await page.getByRole('button', { name: 'Stream', exact: true }).click();
    await expect(outputs.nth(1)).toHaveValue('late');
    await page.getByRole('button', { name: 'Validate', exact: true }).click();
    await expect(page.getByText('Graph is valid')).toBeVisible();
    const exported = await request.get(`${projectsUrl}/${id}/export?format=json`);
    expect(exported.ok()).toBeTruthy();
    const saved = await exported.json();
    expect(saved.graph.nodes[0].operator).toEqual(document.graph.nodes[0]?.operator);
    expect(saved.sinks.map((sink: { binding: string; delivery: string }) => [sink.binding, sink.delivery]))
      .toEqual([['output', 'at_least_once'], ['late', 'best_effort']]);
  } finally {
    await deleteWithLaunchToken(request, `${projectsUrl}/${id}`);
  }
});
