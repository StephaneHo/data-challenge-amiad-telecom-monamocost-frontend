import { fetchPaper } from "./api/papers";

export const paperQuery = (arxivId: string) => ({
    queryKey: ['paper', arxivId] as const,
    queryFn: () => fetchPaper(arxivId),
    staleTime: 5 * 60_000,
})