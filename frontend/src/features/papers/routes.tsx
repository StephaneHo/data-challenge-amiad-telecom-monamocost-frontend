import { lazy } from 'react'
import type { LoaderFunctionArgs, RouteObject } from 'react-router'
import type { QueryClient } from '@tanstack/react-query'
import { paperQuery } from './queries'

const PapersListPage = lazy(() => import('./ui/PapersListPage'))
const PaperDetailPage = lazy(() => import('./ui/PaperDetailPage'))

export const papersRoutes = (queryClient: QueryClient): RouteObject[] => [
    { index: true, element: <PapersListPage /> },
    {
        path: 'papers/:arxiv_id',
        element: <PaperDetailPage />,
        loader: async ({ params }: LoaderFunctionArgs) => {
            if (!params.arxiv_id) throw new Response('Missing', { status: 400 })
            return queryClient.ensureQueryData(paperQuery(params.arxiv_id))
        },
    },
]