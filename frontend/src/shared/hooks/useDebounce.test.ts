


import { vi } from 'vitest'

import { act, renderHook } from '@testing-library/react'
import useDebounce from './useDebounce'

// TODO 2: beforeEach et afterEach pour les fake timers
beforeEach(() => {
    vi.useFakeTimers()
})

afterEach(() => {
    vi.useRealTimers()
})

describe('useDebounce', () => {
    it('retourne la valeur initiale immédiatement', () => {
        const { result } = renderHook(() => useDebounce('hello', 500))
        expect(result.current).toBe('hello')
    })

    it('met à jour après le delay', () => {
        const { result, rerender } = renderHook(
            ({ value }) => useDebounce(value, 500),
            { initialProps: { value: 'hello' } }
        )

        rerender({ value: 'world' })
        act(() => vi.advanceTimersByTime(500))
        expect(result.current).toBe('world')
    })

    it('annule le timer précédent si la valeur change rapidement', () => {
        // TODO 8: change la valeur 2 fois rapidement
        // TODO 9: avance le temps de 500ms
        // TODO 10: vérifie que c'est la DERNIÈRE valeur qui est retournée (pas l'intermédiaire)
        const { result, rerender } = renderHook(
            ({ value }) => useDebounce(value, 500),
            { initialProps: { value: 'hello' } }
        )
        rerender({ value: 'inter' })
        act(() => vi.advanceTimersByTime(200))
        rerender({ value: 'world' })
        act(() => vi.advanceTimersByTime(500))
        expect(result.current).toBe('world')


    })
})