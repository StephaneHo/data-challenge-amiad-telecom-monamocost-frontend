import { http, HttpResponse } from 'msw'
import { mockPapers } from './mockData'

export const handlers = [
    http.get('http://localhost:8000/papers', () => {
        return HttpResponse.json(mockPapers)
    }),

    http.get('http://localhost:8000/papers/:arxiv_id', ({ params }) => {
        const paper = mockPapers.find(p => p.arxiv_id === params.arxiv_id)
        if (!paper) {
            return new HttpResponse(null, { status: 404 })
        }
        return HttpResponse.json(paper)
    }),

    http.post('http://localhost:8000/rag/search', () => {
        return HttpResponse.json({
            answer: 'Réponse RAG simulée',
            references: [],
            tokens_used: 42,
        })
    }),
]