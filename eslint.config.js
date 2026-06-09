import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettier from 'eslint-config-prettier';

export default tseslint.config(
  {
    ignores: [
      'dist',
      'build',
      'node_modules',
      'local',
      '.agents',
      'docs',
      'coverage',
      'benchmarks/datasets',
      'benchmarks/results',
      'assembly',
      'site',
      'scripts',
      'benchmarks/**/*.mjs',
      'src/wasm/wasm-binary.ts',
      '*.config.*',
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  prettier,
  {
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': 'error',
    },
  },
);
