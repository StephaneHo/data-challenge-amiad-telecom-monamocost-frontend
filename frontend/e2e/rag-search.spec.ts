import test, { expect } from "@playwright/test"

test('recherche RAG end-to-end', async ({ page }) => {
    await page.goto('/search')

    const input = page.getByPlaceholder(/Votre question/)
    await expect(input).toBeVisible()

    await input.fill('What are transformers?')

    const button = page.getByRole('button', { name: /Rechercher/i })
    await expect(button).toBeVisible()

    await button.click()

    const response = page.getByText(/transformer|model|attention|neural/i)
    await expect(response).toBeVisible({ timeout: 30000 })

})