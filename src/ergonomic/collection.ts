// Collection — the qdrant-inspired ergonomic layer over IdMapIndex (../index).
//
// It pairs the id-keyed quantized index with a payload store and the filter DSL
// (./filter), exposing upsert / search / delete by id with structured filtering and
// typed payloads. Filtering compiles to a predicate that the core scan applies as a
// per-vector allowlist, so it reuses the same (WASM or scalar) search path.

import { IdMapIndex } from '../index/id-map-index';
import type { IdMapSearchOptions } from '../index/id-map-index';
import type { TurboQuantIndexOptions } from '../index/turboquant-index';
import type { Distance } from '../core/metrics';
import type { IdType } from '../io/serialize';
import { compileFilter } from './filter';
import type { CollectionConfig, Point, SearchHit, SearchParams } from './types';

/**
 * A payload-aware vector collection. Add points with {@link upsert}, query with
 * {@link search} (structured filter + payloads), and remove with {@link delete}.
 * `P` is the payload type; ids are `number | string` by default (`bigint` opt-in).
 */
export class Collection<P = unknown, Id extends IdType = number | string> {
  readonly #index: IdMapIndex<Id>;
  readonly #payloads: Map<Id, P>;

  constructor(config: CollectionConfig) {
    const opts: TurboQuantIndexOptions = {
      dim: config.vectors.size,
      bits: config.quantization.bits,
      metric: config.vectors.distance,
    };
    if (config.seed !== undefined) opts.seed = config.seed;
    if (config.calibrate !== undefined) opts.calibrate = config.calibrate;
    this.#index = new IdMapIndex<Id>(opts);
    this.#payloads = new Map();
  }

  /** Number of stored points. */
  get size(): number {
    return this.#index.size;
  }

  /** Vector dimension. */
  get dim(): number {
    return this.#index.dim;
  }

  /** Configured distance metric. */
  get distance(): Distance {
    return this.#index.metric;
  }

  /**
   * Insert or replace points (by id). An id already present is updated: its old vector
   * and payload are removed first, then re-added. A point without a `payload` clears any
   * stored payload for that id.
   *
   * @throws {IdMapError} on duplicate ids within the batch, or `EncodeError` on a
   *   non-finite/zero vector (a malformed vector mid-batch may leave prior versions removed).
   */
  upsert(points: readonly Point<P, Id>[]): void {
    for (const p of points) {
      if (this.#index.has(p.id)) {
        this.#index.remove(p.id);
        this.#payloads.delete(p.id);
      }
    }
    this.#index.addWithIds(
      points.map((p) => p.id),
      points.map((p) =>
        p.vector instanceof Float32Array ? p.vector : Float32Array.from(p.vector),
      ),
    );
    for (const p of points) {
      if (p.payload !== undefined) this.#payloads.set(p.id, p.payload);
    }
  }

  /**
   * Return the `limit` best matches for `vector`, best-first, optionally restricted by
   * a structured `filter` and carrying each hit's `payload`. An empty collection returns `[]`.
   */
  search(vector: Float32Array | number[], params: SearchParams<Id> = {}): SearchHit<P, Id>[] {
    if (this.#index.size === 0) return [];
    const query = vector instanceof Float32Array ? vector : Float32Array.from(vector);
    const limit = params.limit ?? 10;
    const withPayload = params.withPayload !== false;

    const opts: IdMapSearchOptions<Id> = {};
    if (params.filter !== undefined) {
      const predicate = compileFilter(params.filter);
      opts.filter = (id) => predicate(id, this.#payloads.get(id));
    }

    const res = this.#index.search(query, limit, opts);
    const hits: SearchHit<P, Id>[] = new Array(res.ids.length);
    for (let i = 0; i < res.ids.length; i++) {
      const id = res.ids[i]!;
      const hit: SearchHit<P, Id> = { id, score: res.scores[i]! };
      if (withPayload) {
        const payload = this.#payloads.get(id);
        if (payload !== undefined) hit.payload = payload;
      }
      hits[i] = hit;
    }
    return hits;
  }

  /** Remove one id or a list of ids (unknown ids are ignored). */
  delete(ids: Id | Id[]): void {
    const list = Array.isArray(ids) ? ids : [ids];
    for (const id of list) {
      if (this.#index.has(id)) {
        this.#index.remove(id);
        this.#payloads.delete(id);
      }
    }
  }

  /** Whether `id` is present. */
  has(id: Id): boolean {
    return this.#index.has(id);
  }

  /** The payload stored for `id`, or undefined. */
  get(id: Id): P | undefined {
    return this.#payloads.get(id);
  }
}

/** Create a {@link Collection}. See {@link CollectionConfig}. */
export function createCollection<P = unknown, Id extends IdType = number | string>(
  config: CollectionConfig,
): Collection<P, Id> {
  return new Collection<P, Id>(config);
}
