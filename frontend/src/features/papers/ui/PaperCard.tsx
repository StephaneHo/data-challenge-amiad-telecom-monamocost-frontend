import type { Paper } from "../types"

type PaperCardProps = {
    paper: Paper
}

export const PaperCard = ({ paper }: PaperCardProps) => (
    <article className="bg-white border rounded-lg shadow p-4 space-y-2">
        <h2 className="text-xl font-bold">{paper.title}</h2>
        <time className="text-sm text-gray-500">{paper.published_at}</time>
        <p>{paper.abstract}</p>
        {paper.venue && <p>Publié dans: {paper.venue}</p>}
        {paper.citation_count != null && <p>Nombre de citations: {paper.citation_count}</p>}
    </article>
)