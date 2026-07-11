import { expect, test } from '@playwright/test';

test('builds and runs a persisted DataFusion UDF graph without browser code', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Build the flow')).toBeVisible();
  await expect(page.getByText('double_value')).toBeVisible();

  await page.getByLabel('DataFusion expression').fill('doubled = double_value(a)');
  await page.locator('.udf-option input[type="checkbox"]').check({ force: true });
  await page.getByRole('button', { name: 'Validate' }).click({ force: true });
  await expect(page.getByText('Graph is valid')).toBeVisible();

  await page.getByRole('button', { name: /Run preview/ }).click({ force: true });
  await expect(page.getByText('completed')).toBeVisible();
  await expect(page.getByRole('columnheader', { name: /doubled/ })).toBeVisible();
  await page.getByText('Physical plan').click({ force: true });
  await expect(page.getByText('ProjectionExec')).toBeVisible();
  await page.getByRole('button', { name: 'Inspect' }).click({ force: true });
  await expect(page.getByText('No stored checkpoint')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Benchmark comparison' })).toBeVisible();

  const project = await page.request.get('http://127.0.0.1:8765/api/v1/projects/untitled');
  expect(project.ok()).toBeTruthy();
  const document = await project.json();
  expect(document.pipeline.nodes[0].udfs).toEqual([
    { name: 'double_value', version: '1' },
  ]);
  expect(JSON.stringify(document)).not.toContain('def double_value');
});
