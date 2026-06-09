import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

// Match tsup's build-time injection so `VERSION` (src/index.ts) resolves under test.
const version = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
).version;

export default defineConfig({
  define: { __QUANTVEC_VERSION__: JSON.stringify(version) },
  test: {
    include: ['test/**/*.test.ts', 'src/**/*.test.ts'],
    // The suite includes CPU-heavy numerical tests (rotation QR, scipy-reference
    // beta). Under full parallelism workers can saturate the CPU, so give every test
    // generous headroom — tiny async (fs) tests must not starve on the default 5s.
    testTimeout: 30000,
    coverage: {
      provider: 'v8',
      all: true,
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/*.d.ts', 'src/node.ts'],
      thresholds: {
        lines: 90,
        functions: 90,
        statements: 90,
        branches: 90,
      },
    },
  },
});
