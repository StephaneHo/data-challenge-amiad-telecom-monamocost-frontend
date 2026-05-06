import { screen } from '@testing-library/react'
import { vi } from 'vitest'
import PaperDetailPage from './PaperDetailPage';
import { mockPapers } from '../../../shared/mocks/mockData';
import { renderWithProviders } from '../../../shared/test/renderWithProviders';

vi.mock('../api/papers', () => ({
    fetchPaper: vi.fn().mockResolvedValue({
        arxiv_id: '1234',
        title: 'ML on Graphs',
        abstract: 'why GraphSAGE is very important',
        published_at: '2023-12-12T00:00:00Z',
        citation_count: 358,
        venue: 'Conference of Chicago',
        pdf_url: 'https://...',
    })
}))

describe('PaperDetailsPage', () => {
    it('affiche le titre du papier récupéré', async () => {
        renderWithProviders(<PaperDetailPage />, {
            route: '/papers/1234',
            path: '/papers/:arxiv_id'
        })
        expect(await screen.findByText('ML on Graphs')).toBeInTheDocument()
    })
})