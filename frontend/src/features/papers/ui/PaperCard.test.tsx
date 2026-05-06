import { mockPapers } from "../../../shared/mocks/mockData"
import type { Paper } from "../types"
import { PaperCard } from "./PaperCard"
import { render, screen } from '@testing-library/react'

describe('PaperCard', () => {
    it('affiche le titre du paper', () => {
        const fakePaper: Paper = mockPapers[0]

        render(<PaperCard paper={fakePaper} />)

        expect(screen.getByText('ML on Graphs')).toBeInTheDocument()
        expect(screen.getByText('why GraphSAGE is very important')).toBeInTheDocument()
        expect(screen.getByText(/358/)).toBeInTheDocument()
        expect(screen.getByText(/Conference of Chicago/)).toBeInTheDocument()

    })

    it('n affiche pas la conférence quand venue est null', () => {
        const fakePaper: Paper = mockPapers[1]
        render(<PaperCard paper={fakePaper} />)

        expect(screen.queryByText(/Publié dans/i)).not.toBeInTheDocument()
    })

    it('n affiche pas les citations quand le nombre est null', () => {
        const fakePaper: Paper = mockPapers[1]
        render(<PaperCard paper={fakePaper} />)
        expect(screen.queryByText(/Nombre de citations/i)).not.toBeInTheDocument()
    })
})