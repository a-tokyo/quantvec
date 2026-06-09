// IdMapIndex — stable external-id layer over the positional TurboQuantIndex.
//
// Why this exists: TurboQuantIndex is purely positional — a vector's slot is its
// insertion order, and swapRemove renumbers the moved row. That is fast and compact
// but unusable as a database where callers hold their own ids. IdMapIndex wraps one
// TurboQuantIndex and maintains a bijection id ↔ slot so callers add, search, and
// remove by *their* id; the O(1) swapRemove is hidden behind the map (when the last
// row is moved into the gap, its id is re-pointed to the new slot).
//
// Ids default to `number` (safe ≤ 2^53) but `string` and `bigint` are supported via
// the generic parameter and the serializer's per-id tagging (../io/serialize). Id
// types are validated at the boundary so the serializer can trust them.
//
// Persistence: toBytes()/fromBytes() reuse the positional index's payload snapshot
// (../index/turboquant-index) plus the id column, through the single versioned
// format in ../io/serialize. The bytes are treated as untrusted on the read path.

import type { Bits } from '../core/codebook';
import { validateVectorBatch } from '../core/encode';
import type { Distance } from '../core/metrics';
import { deserializeIndex, serializeIndex } from '../io/serialize';
import type { IdType } from '../io/serialize';
import { TurboQuantIndex } from './turboquant-index';
import type { IndexSearchOptions, TurboQuantIndexOptions } from './turboquant-index';

/** Discriminated, code-tagged error for the id-keyed index. */
export class IdMapError extends Error {
  readonly code:
    | 'INVALID_LENGTH'
    | 'INVALID_VECTOR'
    | 'COUNT_MISMATCH'
    | 'INVALID_ID_TYPE'
    | 'DUPLICATE_ID'
    | 'UNKNOWN_ID'
    | 'EMPTY'
    | 'WRONG_KIND';
  constructor(code: IdMapError['code'], message: string) {
    super(message);
    this.name = 'IdMapError';
    this.code = code;
  }
}

/** Per-query options for {@link IdMapIndex.search}. */
export interface IdMapSearchOptions<Id extends IdType = number> {
  /** Override the index's default metric for this query. */
  metric?: Distance;
  /**
   * Optional allowlist predicate: a candidate is scanned only if `filter(id)` is
   * truthy. Evaluated once per stored vector to build a slot mask; for heavy filtering
   * over large indexes the qdrant-style filter DSL (ergonomic layer) will be cheaper.
   */
  filter?: (id: Id) => boolean;
}

/** Result of {@link IdMapIndex.search}: external ids best-first plus aligned metric values. */
export interface IdSearchResult<Id extends IdType> {
  /** External ids of the k best hits, best-first. */
  ids: Id[];
  /** Reported metric values aligned with `ids` (see ../core/search). */
  scores: Float32Array;
}

/** True for values the index accepts as an external id (rejects NaN/±Infinity numbers). */
function isIdType(v: unknown): v is IdType {
  const t = typeof v;
  return (t === 'number' && Number.isFinite(v)) || t === 'string' || t === 'bigint';
}

/** Length of an array-like vector, or `-1` if it is not array-like. */
function vecLen(v: unknown): number {
  if (v == null || typeof (v as { length?: unknown }).length !== 'number') return -1;
  return (v as { length: number }).length;
}

/**
 * A flat quantized index keyed by stable external ids.
 *
 * Wraps a {@link TurboQuantIndex} and keeps an id↔slot bijection so vectors are
 * added, searched, and removed by id. `remove` is O(1): it swap-removes the
 * positional slot and re-points the moved row's id. Ids default to `number`;
 * `string`/`bigint` are opt-in via the `Id` type parameter.
 */
export class IdMapIndex<Id extends IdType = number> {
  /** The positional store this index is a stable-id view over. */
  #index: TurboQuantIndex;
  /** External id living at each positional slot (parallel to the inner storage). */
  readonly #idForSlot: Id[];
  /** Reverse lookup id → positional slot. */
  readonly #slotForId: Map<Id, number>;

  /**
   * @param options index parameters (dim/bits/metric/seed), forwarded to the inner
   *   {@link TurboQuantIndex}.
   * @param prebuilt @internal A reconstructed inner index (deserialization only). When
   *   supplied it is adopted as-is and `options` is ignored — avoids rebuilding the
   *   rotation/codebook twice on load.
   */
  constructor(options: TurboQuantIndexOptions, prebuilt?: TurboQuantIndex) {
    this.#index = prebuilt ?? new TurboQuantIndex(options);
    this.#idForSlot = [];
    this.#slotForId = new Map();
  }

  /** Live vector count. */
  get size(): number {
    return this.#index.size;
  }

  /** Vector dimension d. */
  get dim(): number {
    return this.#index.dim;
  }

  /** Quantizer bit-width. */
  get bits(): Bits {
    return this.#index.bits;
  }

  /** Default ranking metric. */
  get metric(): Distance {
    return this.#index.metric;
  }

  /** RNG seed of the frozen rotation. */
  get seed(): number {
    return this.#index.seed;
  }

  /** Whether a TQ+ calibration was fit and is in effect. */
  get calibrated(): boolean {
    return this.#index.calibrated;
  }

  /** Whether `id` is currently present. */
  has(id: Id): boolean {
    return this.#slotForId.has(id);
  }

  /** A fresh array of all stored ids, in current slot order. */
  ids(): Id[] {
    return this.#idForSlot.slice();
  }

  /** Remove all vectors and ids, resetting to empty (backing capacity is retained). */
  clear(): void {
    this.#index.clear();
    this.#idForSlot.length = 0;
    this.#slotForId.clear();
  }

  /**
   * Add `vectors` under the given `ids` (one id per vector, same order). Vectors may
   * be a flat `Float32Array` of m·dim values, or an array of per-vector
   * `Float32Array`/`number[]`. All ids AND vector shapes are validated up front, so a
   * structural error (bad id, duplicate, wrong length) aborts before any mutation. A
   * value error in the encoder (a non-finite or zero vector) can still surface mid-batch
   * as an `EncodeError`, leaving earlier vectors of the same call added.
   *
   * @throws {IdMapError} `'INVALID_LENGTH'` (flat buffer/vector not dim),
   *   `'INVALID_VECTOR'` (element not array-like), `'COUNT_MISMATCH'` (ids vs vectors),
   *   `'INVALID_ID_TYPE'`, or `'DUPLICATE_ID'`.
   * @throws {EncodeError} (re-thrown) on a non-finite or zero vector.
   */
  addWithIds(ids: readonly Id[], vectors: Float32Array | number[][] | Float32Array[]): void {
    const flat = vectors instanceof Float32Array;
    if (flat && vectors.length % this.dim !== 0) {
      throw new IdMapError(
        'INVALID_LENGTH',
        `flat buffer length ${vectors.length} is not a multiple of dim ${this.dim}`,
      );
    }
    const m = flat ? vectors.length / this.dim : vectors.length;
    if (ids.length !== m) {
      throw new IdMapError('COUNT_MISMATCH', `got ${ids.length} ids for ${m} vectors`);
    }

    // Validate every id (type + uniqueness) and, for the array form, every vector shape
    // before touching storage → a structural error leaves the index unchanged.
    const seen = new Set<Id>();
    for (let j = 0; j < m; j++) {
      const id = ids[j]!;
      if (!isIdType(id)) {
        throw new IdMapError(
          'INVALID_ID_TYPE',
          `id at ${j} must be a finite number, string, or bigint`,
        );
      }
      if (seen.has(id) || this.#slotForId.has(id)) {
        throw new IdMapError('DUPLICATE_ID', `duplicate id ${String(id)}`);
      }
      seen.add(id);
      if (!flat) {
        const len = vecLen(vectors[j]);
        if (len < 0) {
          throw new IdMapError('INVALID_VECTOR', `vector ${j} is not a Float32Array or number[]`);
        }
        if (len !== this.dim) {
          throw new IdMapError('INVALID_LENGTH', `vector ${j} length ${len} != dim ${this.dim}`);
        }
      }
    }

    // Materialize the batch as Float32Array views so TQ+ calibration can be fit from
    // it (this is the first add) before any row is appended one at a time.
    const vecArr: Float32Array[] = new Array(m);
    for (let j = 0; j < m; j++) {
      if (flat) {
        vecArr[j] = vectors.subarray(j * this.dim, (j + 1) * this.dim);
      } else {
        const raw = vectors[j]!;
        vecArr[j] = raw instanceof Float32Array ? raw : Float32Array.from(raw);
      }
    }
    // Validate every vector's values *before* fitting calibration or appending any
    // row, so a non-finite/zero vector anywhere in the batch leaves the index (and
    // id map) completely unchanged — addOne's own checks run too late to undo
    // already-appended rows.
    validateVectorBatch(vecArr);

    this.#index.fitCalibrationFromBatch(vecArr);

    for (let j = 0; j < m; j++) {
      this.#index.addOne(vecArr[j]!);
      const slot = this.#index.size - 1;
      const id = ids[j]!;
      this.#idForSlot.push(id);
      this.#slotForId.set(id, slot);
    }
  }

  /**
   * Search for the k nearest vectors to `query`, best-first, returning the matches'
   * external ids and aligned metric values. An optional `filter` predicate restricts
   * the scan to ids for which it returns truthy.
   *
   * @throws {IdMapError} `'EMPTY'` if the index has no vectors.
   * @throws {SearchError} (re-thrown) on invalid k, query length/finiteness, or a
   *   zero query.
   */
  search(query: Float32Array, k: number, opts: IdMapSearchOptions<Id> = {}): IdSearchResult<Id> {
    if (this.#index.size === 0) {
      throw new IdMapError('EMPTY', 'cannot search an empty index');
    }
    const innerOpts: IndexSearchOptions = {};
    if (opts.metric !== undefined) innerOpts.metric = opts.metric;
    if (opts.filter !== undefined) {
      const filter = opts.filter;
      const mask = new Uint8Array(this.#idForSlot.length);
      for (let slot = 0; slot < this.#idForSlot.length; slot++) {
        if (filter(this.#idForSlot[slot]!)) mask[slot] = 1;
      }
      innerOpts.mask = mask;
    }
    const res = this.#index.search(query, k, innerOpts);
    const ids: Id[] = new Array(res.indices.length);
    for (let i = 0; i < res.indices.length; i++) {
      ids[i] = this.#idForSlot[res.indices[i]!]!;
    }
    return { ids, scores: res.scores };
  }

  /**
   * Remove the vector stored under `id`. O(1): the positional slot is swap-removed
   * and, when a different row was moved into the gap, that row's id is re-pointed to
   * the freed slot.
   *
   * @throws {IdMapError} `'UNKNOWN_ID'` if `id` is not present.
   */
  remove(id: Id): void {
    const slot = this.#slotForId.get(id);
    if (slot === undefined) {
      throw new IdMapError('UNKNOWN_ID', `no vector with id ${String(id)}`);
    }
    const last = this.#index.size - 1;
    this.#index.swapRemove(slot);
    this.#slotForId.delete(id);
    if (slot !== last) {
      const movedId = this.#idForSlot[last]!;
      this.#idForSlot[slot] = movedId;
      this.#slotForId.set(movedId, slot);
    }
    this.#idForSlot.length = last;
  }

  /**
   * Serialize the index (codes + per-slot ids) to a single versioned,
   * bounds-validated byte buffer. Round-trips through {@link IdMapIndex.fromBytes}.
   */
  toBytes(): Uint8Array {
    return serializeIndex({ kind: 'idmap', ...this.#index.toPayload(), ids: this.#idForSlot });
  }

  /**
   * Reconstruct an id-keyed index from bytes produced by {@link toBytes}. The
   * rotation/codebook are rebuilt from the embedded (dim, seed, bits) and the id↔slot
   * maps from the stored id column; the bytes are treated as untrusted.
   *
   * The `Id` type is a caller assertion: the original generic parameter is not stored
   * in the bytes, so it cannot be recovered — pass it explicitly
   * (`IdMapIndex.fromBytes<string>(bytes)`) if you used a non-default id type, and it
   * must match what was serialized.
   *
   * @throws {IdMapError} `'WRONG_KIND'` if the buffer holds a positional index.
   * @throws {DeserializeError} (re-thrown) on a malformed/oversized buffer.
   */
  static fromBytes<Id extends IdType = IdType>(bytes: Uint8Array): IdMapIndex<Id> {
    const parsed = deserializeIndex(bytes);
    if (parsed.kind !== 'idmap') {
      throw new IdMapError(
        'WRONG_KIND',
        `expected an id-keyed index buffer, got '${parsed.kind}' (use TurboQuantIndex.fromBytes)`,
      );
    }
    const idx = new IdMapIndex<Id>(
      { dim: parsed.dim, bits: parsed.bits, metric: parsed.metric, seed: parsed.seed },
      TurboQuantIndex.fromPayload(parsed),
    );
    for (let slot = 0; slot < parsed.ids.length; slot++) {
      const id = parsed.ids[slot]! as Id;
      idx.#idForSlot.push(id);
      idx.#slotForId.set(id, slot);
    }
    return idx;
  }
}
