// quantvec — data-oblivious, zero-training vector quantization & search.
//
// Clean-room implementation of Google Research's TurboQuant (arXiv:2504.19874)
// with the RaBitQ unbiased-estimator correction (arXiv:2405.12497).
//
// This is the isomorphic entry point (no `node:*` imports). Node-only filesystem
// helpers live behind the `quantvec/node` subpath export (src/node.ts).

// Replaced at build/test time with package.json's `version` (see tsup.config.ts and
// vitest.config.ts `define`), so the version has a single source of truth and is never
// hand-synced.
declare const __QUANTVEC_VERSION__: string;
/** Library version (sourced from package.json). */
export const VERSION: string = __QUANTVEC_VERSION__;

// ── Core positional index ─────────────────────────────────────────────────────
export { TurboQuantIndex, IndexError } from './index/turboquant-index';
export type {
  TurboQuantIndexOptions,
  IndexSearchOptions,
  IvfOptions,
} from './index/turboquant-index';

// ── Stable id-keyed index ─────────────────────────────────────────────────────
export { IdMapIndex, IdMapError } from './index/id-map-index';
export type { IdMapSearchOptions, IdSearchResult } from './index/id-map-index';

// ── Ergonomic layer (qdrant-inspired: payloads + filter DSL) ────────────────────
export { Collection, createCollection } from './ergonomic/collection';
export { FilterError } from './ergonomic/filter';
export type {
  CollectionConfig,
  Point,
  SearchHit,
  SearchParams,
  Filter,
  Condition,
  MatchCondition,
  RangeCondition,
  HasIdCondition,
} from './ergonomic/types';

// ── Shared types ──────────────────────────────────────────────────────────────
export type { Distance } from './core/metrics';
export type { Bits } from './core/codebook';
export type { SearchResult } from './core/search';
export type { IdType } from './io/serialize';

// ── Typed errors that can escape the public API (for `instanceof`/`.code` checks) ─
export { EncodeError } from './core/encode';
export { SearchError } from './core/search';
export { DeserializeError } from './io/serialize';
