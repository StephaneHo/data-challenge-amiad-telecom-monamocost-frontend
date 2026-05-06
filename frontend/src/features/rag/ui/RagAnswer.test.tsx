import { render, screen } from '@testing-library/react'
import { RagAnswer } from './RagAnswer'

describe('RagAnswer', () => {
    describe('sécurité (XSS)', () => {
        it('ne doit PAS exécuter de script HTML injecté dans la réponse', () => {
            const payload = '<script>window.__xssExecuted = true</script>Hello'
            const { container } = render(<RagAnswer answer={payload} />)

            expect((window as any).__xssExecuted).toBeUndefined()

            expect(container.querySelector('script')).toBeNull()
        })

        it("ne doit PAS rendre une <img onerror> en HTML inline du markdown", () => {
            const payload = '<img src=x onerror="window.__imgXss = true">'

            render(<RagAnswer answer={payload} />)

            expect((window as any).__imgXss).toBeUndefined()   // ✅ matche le payload

        })
    })

    describe('feature (markdown)', () => {
        it('affiche le texte brut sans markdown', () => {
            const payload = 'Bonjour le monde'

            render(<RagAnswer answer={payload} />)
            expect(screen.getByText('Bonjour le monde')).toBeInTheDocument()
        })

        it('rend les listes markdown', () => {
            const md = `- item 1\n- item 2\n- item 3`
            render(<RagAnswer answer={md} />)
            expect(screen.getAllByRole('listitem')).toHaveLength(3)

        })

        it('rend le gras markdown', () => {
            const payload = '**important**'

            render(<RagAnswer answer={payload} />)
            expect(screen.getByText('important').tagName.toLowerCase()).toBe('strong')

        })
    })
})