import '@testing-library/jest-dom'
import { beforeAll, afterAll, afterEach } from 'vitest'
import { server } from '../mocks/server'
import 'vitest-axe/extend-expect'
import * as matchers from 'vitest-axe/matchers'
import { expect } from 'vitest'
import type { AxeMatchers } from 'vitest-axe/matchers'


declare module 'vitest' {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface Assertion extends AxeMatchers { }

    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface AsymmetricMatchersContaining extends AxeMatchers { }
}

expect.extend(matchers)

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())