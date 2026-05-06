import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Routes, Route } from 'react-router'
import { render } from '@testing-library/react'
import { Suspense, type ReactElement } from 'react'

type Options = {
  route?: string
  path?: string
}

export function renderWithProviders(
  ui: ReactElement,
  { route = '/', path = '/' }: Options = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>
        <Suspense fallback={<p>Loading test...</p>}>
          <Routes>
            <Route path={path} element={ui} />
          </Routes>
        </Suspense>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}
