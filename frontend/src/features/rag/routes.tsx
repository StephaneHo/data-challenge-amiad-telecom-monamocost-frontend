import { lazy } from 'react'
import type { RouteObject } from 'react-router'

const SearchPage = lazy(() => import('./ui/SearchPage'))

export const ragRoutes = (): RouteObject[] => [
    { path: 'search', element: <SearchPage /> },
]