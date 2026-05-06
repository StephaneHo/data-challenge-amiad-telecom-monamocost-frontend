import { useParams } from "react-router"
import { PaperCard } from "./PaperCard";

import { useSuspenseQuery } from '@tanstack/react-query'
import { paperQuery } from '../queries';



export default function PaperDetailPage() {
    const { arxiv_id } = useParams<{ arxiv_id: string }>();

    const { data } = useSuspenseQuery(paperQuery(arxiv_id!))

    return (
        <PaperCard paper={data} />
    )
}