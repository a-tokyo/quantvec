// Collection — the qdrant-inspired ergonomic layer over IdMapIndex (../index).
//
// It pairs the id-keyed quantized index with a payload store and the filter DSL
// (./filter), exposing upsert / search / delete by id with structured filtering and
// typed payloads. Filtering compiles to a predicate that the core scan applies as a
// per-vector allowlist, so it reuses the same (WASM or scalar) search path.

import { IdMapError, IdMapIndex } from '../index/id-map-index';
import type { IdMapSearchOptions } from '../index/id-map-index';
import type { TurboQuantIndexOptions } from '../index/turboquant-index';
import { validateVectorBatch } from '../core/encode';
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
    if (config.ivf !== undefined) opts.ivf = config.ivf;
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
   * The whole batch is validated up front — a duplicate id within the batch or an
   * invalid vector (wrong length, non-finite, or zero) leaves the collection
   * completely unchanged.
   *
   * @throws {IdMapError} `'DUPLICATE_ID'` on duplicate ids within the batch.
   * @throws {EncodeError} `'INVALID_LENGTH'`/`'ZERO_VECTOR'` on an invalid vector.
   */
  upsert(points: readonly Point<P, Id>[]): void {
    const vecArr = points.map((p) =>
      p.vector instanceof Float32Array ? p.vector : Float32Array.from(p.vector),
    );
    const seen = new Set<Id>();
    for (const p of points) {
      if (seen.has(p.id)) {
        throw new IdMapError('DUPLICATE_ID', `duplicate id ${String(p.id)} in upsert batch`);
      }
      seen.add(p.id);
    }
    validateVectorBatch(vecArr);

    for (const p of points) {
      if (this.#index.has(p.id)) {
        this.#index.remove(p.id);
        this.#payloads.delete(p.id);
      }
    }
    this.#index.addWithIds(
      points.map((p) => p.id),
      vecArr,
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
    if (params.nprobe !== undefined) opts.nprobe = params.nprobe;
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
