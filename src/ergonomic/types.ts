// Public types for the qdrant-inspired ergonomic layer (see ./collection, ./filter).

import type { Distance } from '../core/metrics';
import type { IdType } from '../io/serialize';

/** A stored point: an id, its vector, and an optional typed payload. */
export interface Point<P = unknown, Id extends IdType = number | string> {
  /** Stable external id. */
  id: Id;
  /** The dense vector (length = collection `size`). */
  vector: Float32Array | number[];
  /** Optional payload stored alongside the vector (returned by search, filterable). */
  payload?: P;
}

/** A search hit: the matched id, its metric score, and (optionally) its payload. */
export interface SearchHit<P = unknown, Id extends IdType = number | string> {
  /** The matched point's id. */
  id: Id;
  /** Metric value (similarity for cosine/dot; squared distance for euclidean). */
  score: number;
  /** The payload, present when `withPayload` is not false. */
  payload?: P;
}

/** Configuration for {@link createCollection}. */
export interface CollectionConfig {
  /** Vector space: dimension (`size`, a positive multiple of 8) and ranking `distance`. */
  vectors: { size: number; distance: Distance };
  /** Quantizer settings. */
  quantization: { bits: 2 | 3 | 4 };
  /** RNG seed for the frozen rotation (default 0). */
  seed?: number;
  /** Enable TQ+ calibration (default false; data-dependent — see TurboQuantIndex). */
  calibrate?: boolean;
}

/** Per-search parameters. */
export interface SearchParams<Id extends IdType = number | string> {
  /** Maximum number of hits to return (default 10). */
  limit?: number;
  /** Structured payload/id filter; only matching points are scanned. */
  filter?: Filter<Id>;
  /** Include each hit's payload (default true). */
  withPayload?: boolean;
}

// ── Filter DSL (qdrant-inspired) ───────────────────────────────────────────────

/** Exact-match on a payload key. */
export interface MatchCondition {
  key: string;
  match: { value: string | number | boolean };
}
/** Numeric range on a payload key (any subset of bounds; all are inclusive/exclusive as named). */
export interface RangeCondition {
  key: string;
  range: { gt?: number; gte?: number; lt?: number; lte?: number };
}
/** Restrict to a set of ids. */
export interface HasIdCondition<Id extends IdType = number | string> {
  hasId: Id[];
}
/** One filter clause: a leaf condition or a nested {@link Filter}. */
export type Condition<Id extends IdType = number | string> =
  | MatchCondition
  | RangeCondition
  | HasIdCondition<Id>
  | Filter<Id>;

/**
 * A boolean combination of conditions (qdrant semantics): a point matches when every
 * `must` clause matches AND (no `should` clauses, or at least one matches) AND no
 * `must_not` clause matches.
 */
export interface Filter<Id extends IdType = number | string> {
  must?: Condition<Id>[];
  should?: Condition<Id>[];
  must_not?: Condition<Id>[];
}
