
import { createBrowserRouter, Outlet } from 'react-router'
import { Suspense } from 'react'
import type { QueryClient } from '@tanstack/react-query'
import NavBar from './shared/ui/NavBar'
import { papersRoutes } from './features/papers/routes'
import { ragRoutes } from './features/rag/routes'

function RootLayout() {
    return (
        <>
            <NavBar />
            <Suspense fallback={<p className="p-4">Chargement...</p>}>
                <Outlet />
            </Suspense>
        </>
    )
}

export const createRouter = (queryClient: QueryClient) =>
    createBrowserRouter([
        {
            path: '/',
            element: <RootLayout />,
            children: [
                ...papersRoutes(queryClient),
                ...ragRoutes(),
            ],
        },
    ])