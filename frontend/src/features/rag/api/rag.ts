import type { Paper } from "../../papers/types"

export type RagSearchRequest = {
  query: string
  top_k?: number
  year_from?: number | null
  categories?: string[] | null
}


export type Reference = Pick<Paper, 'arxiv_id' | 'title' | 'published_at' | 'venue'> & {
  url: string
  score: number
}

export type RagSearchResponse = {
  answer: string
  references: Reference[]    // on ra
  tokens_used: number
}

export async function searchRag(payload: RagSearchRequest): Promise<RagSearchResponse> {
  const response = await fetch('http://localhost:8000/rag/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok) {
    throw new Error('La recherche RAG a échoué')
  }
  return response.json() as Promise<RagSearchResponse>
}