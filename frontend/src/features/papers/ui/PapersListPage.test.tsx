import { screen } from '@testing-library/react'
import PapersListPage from './PapersListPage'
import { renderWithProviders } from '../../../shared/test/renderWithProviders'

describe('PapersListPage (intégration MSW)', () => {
    it('affiche la liste des papers depuis l\'API', async () => {
        renderWithProviders(<PapersListPage />, { route: '/', path: '/' })


        expect(await screen.findByText('ML on Graphs')).toBeInTheDocument()
        expect(await screen.findByText('Transformers')).toBeInTheDocument()
        expect(await screen.findByText('The future of AI')).toBeInTheDocument()
    })

    it('chaque carte est dans un Link vers le bon paper', async () => {
        renderWithProviders(<PapersListPage />, { route: '/', path: '/' })

        const card = await screen.findByText('ML on Graphs')
        const link = card.closest('a')

        expect(link).toHaveAttribute('href', '/papers/1234')
    })
})