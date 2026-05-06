import { render, screen } from '@testing-library/react'
import { ErrorFallback } from './ErrorFallback'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'

describe('ErrorFallback', () => {
    it('affiche le message de l\'erreur', () => {
        const fakeError = new Error('Test error message')

        render(
            <ErrorFallback
                error={fakeError}
                resetErrorBoundary={() => { }}
            />
        )
        expect(screen.getByText('Test error message')).toBeInTheDocument()

    })

    it('appelle resetErrorBoundary au click sur Réessayer', async () => {
        const user = userEvent.setup()
        const onReset = vi.fn()

        render(
            <ErrorFallback
                error={new Error('whatever')}
                resetErrorBoundary={onReset}
            />
        )

        const button = screen.getByRole('button', { name: /Réessayer/i })
        await user.click(button)
        expect(onReset).toHaveBeenCalledTimes(1)
    })
})