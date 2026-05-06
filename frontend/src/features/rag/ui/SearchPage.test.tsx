import { screen } from '@testing-library/react'
import SearchPage from './SearchPage'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { waitFor } from '@testing-library/react'
import { searchRag } from '../api/rag'
import { renderWithProviders } from '../../../shared/test/renderWithProviders'

import { axe } from 'vitest-axe'



vi.mock('../api/rag', () => ({
    searchRag: vi.fn().mockResolvedValue({
        answer: 'test answer',
        references: [],
        tokens_used: 0,
    }),
}))

const searchRagMock = vi.mocked(searchRag)

beforeEach(() => {
    searchRagMock.mockClear()
})

describe('SearchPage', () => {
    describe('rendu initial', () => {
        it('affiche un champ de recherche et un bouton', () => {
            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            expect(screen.queryByPlaceholderText(/Votre question/)).toBeInTheDocument()
            expect(screen.queryByRole('button', { name: /rechercher/i })).toBeInTheDocument()
        })

    })



    describe('validation', () => {
        it('affiche une erreur si moins de 3 caractères', async () => {
            const user = userEvent.setup()
            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const input = screen.getByPlaceholderText(/Votre question/i)
            const button = screen.getByRole('button', { name: /rechercher/i })


            await user.type(input, 'ab')
            await user.click(button)
            expect(await screen.findByText(/minimum 3 caractères/i)).toBeInTheDocument()
        })
    })

    describe('soumission', () => {
        it('appelle searchRag avec le bon payload quand le form est valide', async () => {
            const user = userEvent.setup()
            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const QUERY = "What is the state of ML today ?"
            const input = screen.getByPlaceholderText(/Votre question/i)
            await user.type(input, QUERY)

            const button = screen.getByRole('button', { name: /rechercher/i })
            await user.click(button);
            await waitFor(() => expect(searchRagMock).toHaveBeenCalledWith({ query: QUERY }, expect.anything()))
        })

        it('affiche la réponse quand la recherche réussit', async () => {
            const user = userEvent.setup()
            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const input = screen.getByPlaceholderText(/Votre question/i)
            const button = screen.getByRole('button', { name: /rechercher/i })

            await user.type(input, 'What are transformers?')
            await user.click(button)

            expect(await screen.findByText('test answer')).toBeInTheDocument()

        })


        it('désactive le bouton pendant le submit', async () => {
            const user = userEvent.setup()

            searchRagMock.mockImplementationOnce(() => new Promise(() => { }))

            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const input = screen.getByPlaceholderText(/Votre question/i)
            const button = screen.getByRole('button', { name: /rechercher/i })

            await user.type(input, 'What is the state of ML today ?')
            await user.click(button)

            expect(button).toBeDisabled()
            expect(await screen.findByText(/Recherche en cours/i)).toBeInTheDocument()

        })


        it('affiche un message d\'erreur si la recherche échoue', async () => {
            const user = userEvent.setup()
            searchRagMock.mockRejectedValueOnce(new Error('Backend indisponible'))

            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const input = screen.getByPlaceholderText(/Votre question/i)
            const button = screen.getByRole('button', { name: /rechercher/i })

            await user.type(input, 'What is the state of ML today ?')
            await user.click(button)

            expect(await screen.findByText(/Backend indisponible/i)).toBeInTheDocument()

        })


        it('affiche les références cliquables après recherche réussie', async () => {
            const user = userEvent.setup()

            searchRagMock.mockResolvedValueOnce({
                answer: 'test answer',
                references: [
                    {
                        arxiv_id: '1234', title: 'Paper One', published_at: '2023-12-12T00:00:00Z', venue: null,
                        url: 'https://arxiv.org/abs/1234', score: 0.9
                    },
                    {
                        arxiv_id: '5678', title: 'Paper Two', published_at: '2023-12-12T00:00:00Z', venue: null,
                        url: 'https://arxiv.org/abs/5678', score: 0.8
                    },
                ],
                tokens_used: 100,
            })

            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const input = screen.getByPlaceholderText(/Votre question/i)
            const button = screen.getByRole('button', { name: /rechercher/i })

            await user.type(input, 'What is the state of ML today ?')
            await user.click(button)

            await screen.findByText('Paper One')
            screen.getByText('Paper Two')

            const link = screen.getByText('Paper One').closest('a')
            expect(link).toHaveAttribute('href', '/papers/1234')
        })


    })

    describe('accessibilité (a11y)', () => {
        it("n'a pas de violation a11y au render initial", async () => {

            const { container } = renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            const results = await axe(container)

            expect(results).toHaveNoViolations()
        })

        it('le champ de recherche est accessible par son label', () => {
            renderWithProviders(<SearchPage />, { route: '/search', path: '/search' })

            expect(screen.getByLabelText(/question/i)).toBeInTheDocument()
        })
    })



})