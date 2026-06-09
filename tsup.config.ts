import { readFileSync } from 'node:fs';
import { defineConfig } from 'tsup';

// Single source of truth for the version: package.json (injected via `define` below).
const version = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
).version;

// Dual ESM/CJS + .d.ts. The published artifact is runtime-agnostic; Node-only
// helpers are isolated in the `./node` entry so the core stays isomorphic.
export default defineConfig({
  entry: { index: 'src/index.ts', node: 'src/node.ts' },
  format: ['esm', 'cjs'],
  define: { __QUANTVEC_VERSION__: JSON.stringify(version) },
  // Node-aware tsconfig so the node entry's `node:*` imports resolve during .d.ts
  // generation; the base tsconfig keeps the core's own typecheck isomorphic.
  tsconfig: 'tsconfig.build.json',
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: 'es2022',
  outExtension({ format }) {
    return { js: format === 'cjs' ? '.cjs' : '.js' };
  },
});
