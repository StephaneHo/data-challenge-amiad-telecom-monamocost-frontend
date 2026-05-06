import test, { expect } from "@playwright/test"

test('affiche la liste des papers sur la page d\'accueil', async ({ page }) => {
    await page.goto('/')

    const firstHeading = page.getByRole('heading', { level: 2 }).first()
    await expect(firstHeading).toBeVisible()
})