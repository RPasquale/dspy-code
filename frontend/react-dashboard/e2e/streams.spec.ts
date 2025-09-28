import { test, expect } from '@playwright/test'

test('streams explorer renders', async ({ page }) => {
  await page.goto('/streams')
  await expect(page.getByText('Streams Explorer')).toBeVisible()
  // UI elements present
  await expect(page.getByText('Topics')).toBeVisible()
  await expect(page.getByText('Time')).toBeVisible()
})

