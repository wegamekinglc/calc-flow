import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

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
