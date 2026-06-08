import { defineConfig } from 'tsup';

// Dual ESM/CJS + .d.ts. The published artifact is runtime-agnostic; Node-only
// helpers are isolated in the `./node` entry so the core stays isomorphic.
export default defineConfig({
  entry: { index: 'src/index.ts', node: 'src/node.ts' },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: 'es2022',
  outExtension({ format }) {
    return { js: format === 'cjs' ? '.cjs' : '.js' };
  },
});
