import type { Paper } from "../types"

export async function fetchPapers(): Promise<Paper[]> {
    const response = await fetch('http://localhost:8000/papers')
    if (!response.ok) {
        throw new Error('Aucun papier retrouvé')
    }
    return response.json() as Promise<Paper[]>
}

export async function fetchPaper(arxivId: string): Promise<Paper> {

    const response = await fetch(`http://localhost:8000/papers/${arxivId}`)
    if (!response.ok) {
        throw new Error('Aucun papier retrouvé')
    }
    return response.json() as Promise<Paper>
}