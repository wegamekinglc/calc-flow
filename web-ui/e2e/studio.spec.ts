import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const twoSourceProject = {
  format_version: 2,
  id: 'two_source_e2e',
  name: 'Two source E2E',
  description: 'Two independent saved sources join downstream.',
  pipeline: {
    name: 'Two source pipeline',
    nodes: [
      {
        id: 'left_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, value * 2 AS left_value FROM left_source',
          aliases: ['left_source'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 80 },
      },
      {
        id: 'right_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, adjustment AS right_value FROM right_source',
          aliases: ['right_source'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 280 },
      },
      {
        id: 'join_result',
        operator: {
          kind: 'sql',
          query: 'SELECT l.id, l.left_value, r.right_value, l.left_value + r.right_value AS total FROM left l JOIN right r ON l.id = r.id ORDER BY l.id',
          aliases: ['left', 'right'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 480, y: 180 },
      },
    ],
    edges: [
      {
        source_node: 'left_branch',
        source_port: 'output',
        target_node: 'join_result',
        target_port: 'left',
      },
      {
        source_node: 'right_branch',
        source_port: 'output',
        target_node: 'join_result',
        target_port: 'right',
      },
    ],
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [
    {
      id: 'left',
      input: 'left_source',
      format: 'inline_json',
      data: [{ id: 1, value: 3 }, { id: 2, value: 5 }],
    },
    {
      id: 'right',
      input: 'right_source',
      format: 'inline_json',
      data: [{ id: 1, adjustment: 10 }, { id: 2, adjustment: 20 }],
    },
  ],
  run_options: {
    max_input_bytes: 10485760,
    max_rows: 100000,
    timeout_seconds: 30,
    memory_limit_mb: 512,
    output_rows: 1000,
  },
};

test('builds and runs a persisted DataFusion UDF graph without browser code', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Build the flow')).toBeVisible();
  await expect(page.getByText('double_value', { exact: true })).toBeVisible();

  await page.getByLabel('DataFusion expression').fill('doubled = double_value(a)');
  await page.locator('.udf-option input[type="checkbox"]').check({ force: true });
  await page.getByRole('button', { name: 'Validate' }).click({ force: true });
  await expect(page.getByText('Graph is valid')).toBeVisible();

  await page.getByRole('button', { name: /Run preview/ }).click({ force: true });
  await expect(page.getByText('completed')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole('columnheader', { name: /doubled/ })).toBeVisible();
  await page.getByText('Physical plan').click({ force: true });
  await expect(page.getByText('ProjectionExec')).toBeVisible();
  await page.getByRole('button', { name: 'Inspect' }).click({ force: true });
  await expect(page.getByText('No stored checkpoint')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Benchmark comparison' })).toBeVisible();

  const projects = await page.request.get('http://127.0.0.1:8765/api/v2/projects');
  expect(projects.ok()).toBeTruthy();
  const [summary] = await projects.json();
  expect(summary.id).toMatch(/^project_[0-9a-f]{32}$/);
  const project = await page.request.get(
    `http://127.0.0.1:8765/api/v2/projects/${summary.id}`,
  );
  expect(project.ok()).toBeTruthy();
  const document = await project.json();
  expect(document.format_version).toBe(2);
  expect(document.pipeline.nodes[0].operator.udfs).toEqual([
    {
      provider: 'python',
      name: 'double_value',
      version: '1',
      kind: 'data_fusion_scalar',
    },
  ]);
  expect(JSON.stringify(document)).not.toContain('def double_value');

  const validation = await page.request.post(
    `http://127.0.0.1:8765/api/v2/projects/${summary.id}/validate`,
  );
  expect(validation.ok()).toBeTruthy();
  expect(await validation.json()).toMatchObject({ valid: true, issues: [] });

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON' }).click({ force: true });
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${summary.id}.json`);
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByLabel('Import project').setInputFiles({
    name: download.suggestedFilename(),
    mimeType: 'application/json',
    buffer: await readFile(downloadPath!),
  });
  await expect(page.getByText('Project imported')).toBeVisible();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Delete', exact: true }).click({ force: true });
  await expect(page.getByText('Project deleted')).toBeVisible();
  const remaining = await page.request.get('http://127.0.0.1:8765/api/v2/projects');
  expect(await remaining.json()).toEqual([]);
});

test('edits and runs a persisted two-source SQL join', async ({ page }) => {
  const projectUrl = 'http://127.0.0.1:8765/api/v2/projects/two_source_e2e';
  const created = await page.request.post('http://127.0.0.1:8765/api/v2/projects', {
    data: twoSourceProject,
  });
  expect(created.ok()).toBeTruthy();

  let deleted = false;
  try {
    await page.goto('/');

    const sources = page.getByRole('region', { name: 'Data sources' });
    await expect(sources.getByRole('article')).toHaveCount(2);
    await expect(page.getByLabel('Graph input 1')).toHaveValue('left_source');
    await expect(page.getByLabel('Graph input 2')).toHaveValue('right_source');

    await sources.getByRole('button', { name: 'Add data source' }).click({ force: true });
    await expect(sources.getByRole('article')).toHaveCount(3);
    await sources.getByRole('button', { name: 'Remove source 3' }).click({ force: true });
    await expect(sources.getByRole('article')).toHaveCount(2);

    await page.getByLabel('Data 1').fill('[{"id":1,"value":4},{"id":2,"value":5}]');
    await page.getByRole('button', { name: 'Save' }).click({ force: true });
    await expect(page.getByRole('status')).toHaveText('Project saved');

    await page.getByRole('button', { name: 'Validate' }).click({ force: true });
    await expect(page.getByText('Graph is valid')).toBeVisible();

    await page.getByRole('button', { name: /Run preview/ }).click({ force: true });
    await expect(page.getByText('completed', { exact: true })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole('columnheader', { name: /total/ })).toBeVisible();
    await expect(page.getByRole('cell', { name: '18', exact: true })).toBeVisible();

    const saved = await page.request.get(projectUrl);
    expect(saved.ok()).toBeTruthy();
    const document = await saved.json();
    expect(document.data_sources.map((source: { input: string }) => source.input)).toEqual([
      'left_source',
      'right_source',
    ]);

    const removed = await page.request.delete(projectUrl);
    expect(removed.ok()).toBeTruthy();
    deleted = true;
  } finally {
    if (!deleted) await page.request.delete(projectUrl);
  }
});
