import type { FallbackProps } from 'react-error-boundary'

export function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
    return (
        <div role="alert" className="p-4 max-w-3xl mx-auto bg-red-50 border border-red-300 rounded">
            <h2 className="text-xl font-bold text-red-700">Quelque chose s'est mal passé</h2>
            <pre className="mt-2 text-sm text-red-800">
                {error instanceof Error ? error.message : 'Une erreur est survenue'}
            </pre>
            <button
                onClick={resetErrorBoundary}
                className="mt-4 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
            >
                Réessayer
            </button>
        </div>
    )
}