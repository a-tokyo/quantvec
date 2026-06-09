// TurboQuantIndex — quantvec's positional flat quantized index.
//
// Why this exists: it is the user-facing wrapper that turns the per-vector encode
// pipeline (../core/encode) and the flat scan oracle (../core/search) into a
// stateful, growable database. It owns the frozen rotation (../core/rotation) and
// Lloyd-Max codebook (../core/codebook) — both fully determined by (dim, bits,
// seed) — and the contiguous column storage the scan walks.
//
// Storage layout (matches EncodedDb): a single growable row-major `codes`
// (cap·dim bytes), parallel `scales`/`norms` (cap floats), and a live count `n`.
// add() encodes each vector with one reused scratch buffer and appends; search()
// hands subarray *views* (length n / n·dim) to searchFlat with no copy. Capacity
// doubles when full (amortized O(1) append); the public getters expose the live
// count, never the capacity.
//
// IDs/persistence: this index is purely positional (slot = insertion order, with
// swapRemove for O(1) deletion). Stable ids and the id-keyed API live in
// ../index/id-map-index, which wraps an instance of this class. toBytes/fromBytes
// (single versioned format, ../io/serialize) are wired on below.

import { fitCalibration } from '../core/calibrate';
import type { Calibration } from '../core/calibrate';
import { getCodebook } from '../core/codebook';
import type { Bits, Codebook } from '../core/codebook';
import { createEncodeScratch, encodeVector } from '../core/encode';
import type { EncodeOptions, EncodeScratch } from '../core/encode';
import { scoreMetric } from '../core/metrics';
import type { Distance, QueryNorms } from '../core/metrics';
import { createRotation } from '../core/rotation';
import type { Rotation } from '../core/rotation';
import { buildQueryLut, searchFlat } from '../core/search';
import type { EncodedDb, SearchOptions, SearchResult } from '../core/search';
import { TopK } from '../core/topk';
import { WasmKernel } from '../wasm/kernel';
import { deserializeIndex, serializeIndex } from '../io/serialize';
import type { IndexPayload } from '../io/serialize';

/** Discriminated, code-tagged error for the positional index. */
export class IndexError extends Error {
  readonly code:
    | 'INVALID_DIM'
    | 'INVALID_BITS'
    | 'INVALID_SEED'
    | 'INVALID_LENGTH'
    | 'INVALID_VECTOR'
    | 'INVALID_INDEX'
    | 'EMPTY'
    | 'WRONG_KIND';
  constructor(code: IndexError['code'], message: string) {
    super(message);
    this.name = 'IndexError';
    this.code = code;
  }
}

/** Construction options for {@link TurboQuantIndex}. */
export interface TurboQuantIndexOptions {
  /** Vector dimension d; must be a positive multiple of 8 (rotation/codebook precondition). */
  dim: number;
  /** Quantizer bit-width; one of {2, 3, 4}. Defaults to 4. */
  bits?: Bits;
  /** Default ranking metric for {@link TurboQuantIndex.search}. Defaults to 'cosine'. */
  metric?: Distance;
  /**
   * RNG seed for the frozen rotation (part of the serialized identity). Defaults to 0.
   * Must be finite; the RNG truncates it to an integer, so `3` and `3.7` seed the same
   * rotation.
   */
  seed?: number;
  /**
   * Enable TQ+ per-coordinate calibration (default **false** — opt-in). When the first
   * non-empty add supplies at least {@link CALIBRATION_MIN_SAMPLES} vectors, a
   * per-coordinate map is fit from that batch and frozen for the index's lifetime.
   *
   * Calibration is **data-dependent**: it can lift recall on real embeddings (the paper's
   * regime) but is neutral-to-slightly-negative on well-conditioned data where the random
   * rotation already yields near-canonical coordinates. Enable it only after validating a
   * recall gain on your own data.
   */
  calibrate?: boolean;
  /**
   * Use the WASM scoring kernel when available (default true). It is an exact
   * acceleration with automatic fallback to the pure-TS scan, so disabling it only
   * affects performance, never results. Set false to force the scalar kernel.
   */
  wasm?: boolean;
  /**
   * Use the v128 FastScan kernel for queries (default false; 4-bit only). FastScan is a
   * fast *approximate* SIMD scan that ranks a candidate pool, then rescores the pool
   * exactly — high recall at higher throughput, but (unlike the default exact path) not
   * bit-identical. Ignored when bits ≠ 4 or WebAssembly is unavailable (falls back to
   * the exact scan).
   */
  fastscan?: boolean;
}

/** Per-query options for {@link TurboQuantIndex.search}. */
export interface IndexSearchOptions {
  /** Override the index's default metric for this query. */
  metric?: Distance;
  /** Optional allowlist (length = size): vector j is scanned only if mask[j] is truthy. */
  mask?: Uint8Array | boolean[];
}

/** Initial backing-array capacity before the first growth (kept small; doubles on demand). */
const INITIAL_CAPACITY = 8;

/** Minimum first-batch size to fit TQ+ calibration (smaller first adds stay un-calibrated). */
export const CALIBRATION_MIN_SAMPLES = 1000;

function validateBits(bits: number): asserts bits is Bits {
  if (bits !== 2 && bits !== 3 && bits !== 4) {
    throw new IndexError('INVALID_BITS', `bits must be one of {2, 3, 4}, got ${bits}`);
  }
}

function validateDim(dim: number): void {
  if (!Number.isInteger(dim) || dim <= 0 || dim % 8 !== 0) {
    throw new IndexError('INVALID_DIM', `dim must be a positive multiple of 8, got ${dim}`);
  }
}

function validateSeed(seed: number): void {
  if (!Number.isFinite(seed)) {
    throw new IndexError('INVALID_SEED', `seed must be finite, got ${seed}`);
  }
}

/**
 * Length of an array-like vector argument, throwing a typed error if it is not
 * array-like (e.g. `null`/`undefined` slipping in from untyped JS callers) — keeps a
 * raw `TypeError` from leaking past the boundary (R8).
 */
function vectorLength(v: unknown): number {
  if (v == null || typeof (v as { length?: unknown }).length !== 'number') {
    throw new IndexError(
      'INVALID_VECTOR',
      `expected a Float32Array or number[], got ${v === null ? 'null' : typeof v}`,
    );
  }
  return (v as { length: number }).length;
}

/**
 * A growable, positional flat quantized index over `dim`-dimensional vectors.
 *
 * Encodes each added vector against a frozen (dim, bits, seed) rotation + codebook
 * and stores the compact codes in contiguous column storage; `search` is an O(n)
 * scan over those codes (see ../core/search). Slots are positional and dense;
 * `swapRemove` deletes in O(1) by moving the last row into the gap.
 */
export class TurboQuantIndex {
  /** Vector dimension d. */
  readonly #dim: number;
  /** Quantizer bit-width. */
  readonly #bits: Bits;
  /** Default ranking metric. */
  readonly #metric: Distance;
  /** RNG seed for the rotation (carried into serialization). */
  readonly #seed: number;
  /** Frozen orthonormal rotation (deterministic in dim, seed). */
  readonly #rotation: Rotation;
  /** Frozen Lloyd-Max codebook (deterministic in dim, bits). */
  readonly #codebook: Codebook;
  /** Reused per-encode scratch (one buffer for the whole lifetime). */
  readonly #scratch: EncodeScratch;

  /** Row-major codes, length cap·dim; only the first n·dim bytes are live. */
  #codes: Uint8Array;
  /** Per-vector RaBitQ scales, length cap; only the first n are live. */
  #scales: Float32Array;
  /** Per-vector norms, length cap; only the first n are live. */
  #norms: Float32Array;
  /** Live vector count. */
  #n: number;
  /** Whether TQ+ calibration is enabled (fit from the first eligible batch). */
  readonly #calibrate: boolean;
  /** Frozen TQ+ calibration, or undefined if un-calibrated. */
  #calibration: Calibration | undefined;
  /** Once true, the calibration decision is locked for the index's lifetime. */
  #calibrationFrozen: boolean;
  /** Whether to use the WASM kernel when available. */
  readonly #wasmEnabled: boolean;
  /** Whether to use the FastScan path (4-bit, approximate + rescore) when available. */
  readonly #fastscan: boolean;
  /** Lazily-created WASM kernel: undefined = not tried, null = unavailable. */
  #wasm: WasmKernel | null | undefined;
  /** True when the resident WASM codes are stale (mutation since last upload). */
  #wasmDirty: boolean;

  constructor(options: TurboQuantIndexOptions) {
    const {
      dim,
      bits = 4,
      metric = 'cosine',
      seed = 0,
      calibrate = false,
      wasm = true,
      fastscan = false,
    } = options;
    validateDim(dim);
    validateBits(bits);
    validateSeed(seed);

    this.#dim = dim;
    this.#bits = bits;
    this.#metric = metric;
    this.#seed = seed;
    this.#calibrate = calibrate;
    this.#calibration = undefined;
    this.#calibrationFrozen = false;
    this.#wasmEnabled = wasm;
    this.#fastscan = fastscan && bits === 4;
    this.#wasm = undefined;
    this.#wasmDirty = true;
    this.#rotation = createRotation(dim, seed);
    this.#codebook = getCodebook(dim, bits);
    this.#scratch = createEncodeScratch(dim);

    this.#codes = new Uint8Array(INITIAL_CAPACITY * dim);
    this.#scales = new Float32Array(INITIAL_CAPACITY);
    this.#norms = new Float32Array(INITIAL_CAPACITY);
    this.#n = 0;
  }

  /** Live vector count. */
  get size(): number {
    return this.#n;
  }

  /** Vector dimension d. */
  get dim(): number {
    return this.#dim;
  }

  /** Quantizer bit-width. */
  get bits(): Bits {
    return this.#bits;
  }

  /** Default ranking metric. */
  get metric(): Distance {
    return this.#metric;
  }

  /** RNG seed of the frozen rotation. */
  get seed(): number {
    return this.#seed;
  }

  /** Whether a TQ+ calibration was fit and is in effect. */
  get calibrated(): boolean {
    return this.#calibration !== undefined;
  }

  /** Current backing capacity (number of slots before the next growth). */
  get #capacity(): number {
    return this.#scales.length;
  }

  /** Ensure storage can hold at least `needed` vectors, doubling capacity as required. */
  #ensureCapacity(needed: number): void {
    let cap = this.#capacity;
    if (needed <= cap) return;
    while (cap < needed) cap *= 2;
    const codes = new Uint8Array(cap * this.#dim);
    codes.set(this.#codes.subarray(0, this.#n * this.#dim));
    const scales = new Float32Array(cap);
    scales.set(this.#scales.subarray(0, this.#n));
    const norms = new Float32Array(cap);
    norms.set(this.#norms.subarray(0, this.#n));
    this.#codes = codes;
    this.#scales = scales;
    this.#norms = norms;
  }

  /** Encode one vector (validated upstream) into the slot at `this.#n` and bump the count. */
  #appendOne(vec: Float32Array): void {
    const opts: EncodeOptions = {
      dim: this.#dim,
      bits: this.#bits,
      rotation: this.#rotation,
      codebook: this.#codebook,
      scratch: this.#scratch,
    };
    if (this.#calibration !== undefined) opts.calibration = this.#calibration;
    const encoded = encodeVector(vec, opts);
    const slot = this.#n;
    this.#codes.set(encoded.codes, slot * this.#dim);
    this.#scales[slot] = encoded.scale;
    this.#norms[slot] = encoded.norm;
    this.#n = slot + 1;
    this.#wasmDirty = true;
  }

  /** Normalize an `add`/`addWithIds` batch argument to validated per-vector views. */
  #toVectorArray(vectors: Float32Array | number[][] | Float32Array[]): Float32Array[] {
    if (vectors instanceof Float32Array) {
      if (vectors.length % this.#dim !== 0) {
        throw new IndexError(
          'INVALID_LENGTH',
          `flat buffer length ${vectors.length} is not a multiple of dim ${this.#dim}`,
        );
      }
      const m = vectors.length / this.#dim;
      const out: Float32Array[] = new Array(m);
      for (let j = 0; j < m; j++) out[j] = vectors.subarray(j * this.#dim, (j + 1) * this.#dim);
      return out;
    }
    const out: Float32Array[] = new Array(vectors.length);
    for (let j = 0; j < vectors.length; j++) {
      const raw = vectors[j]!;
      const len = vectorLength(raw);
      if (len !== this.#dim) {
        throw new IndexError('INVALID_LENGTH', `vector ${j} length ${len} != dim ${this.#dim}`);
      }
      out[j] = raw instanceof Float32Array ? raw : Float32Array.from(raw);
    }
    return out;
  }

  /**
   * @internal Fit and freeze TQ+ calibration from the first eligible batch (only when
   * the index is still empty and calibration is enabled with ≥ {@link CALIBRATION_MIN_SAMPLES}
   * vectors); otherwise just locks the (un-calibrated) decision. An empty batch is a
   * no-op so it does not prematurely freeze. Shared by {@link add} and {@link IdMapIndex}.
   */
  fitCalibrationFromBatch(vecs: readonly Float32Array[]): void {
    if (this.#calibrationFrozen || this.#n !== 0 || vecs.length === 0) return;
    if (this.#calibrate && vecs.length >= CALIBRATION_MIN_SAMPLES) {
      const m = vecs.length;
      const samples = new Float32Array(m * this.#dim);
      const { unit, rotated } = this.#scratch;
      for (let j = 0; j < m; j++) {
        const v = vecs[j]!;
        let normSq = 0;
        for (let i = 0; i < this.#dim; i++) normSq += v[i]! * v[i]!;
        const norm = Math.sqrt(normSq);
        if (norm === 0) continue; // zero rows can't be normalized; encode will reject them
        const inv = 1 / norm;
        for (let i = 0; i < this.#dim; i++) unit[i] = v[i]! * inv;
        this.#rotation.apply(unit, rotated);
        samples.set(rotated, j * this.#dim);
      }
      this.#calibration = fitCalibration(samples, m, this.#dim);
    }
    this.#calibrationFrozen = true;
  }

  /**
   * Add one or more vectors. Accepts a flat `Float32Array` of m·dim values (m
   * vectors laid out row-major), or an array of per-vector `Float32Array` /
   * `number[]`. Each vector is encoded and appended in insertion order.
   *
   * @throws {IndexError} `'INVALID_LENGTH'` if a flat buffer is not a multiple of
   *   dim, or an individual vector's length differs from dim; `'INVALID_VECTOR'` if
   *   an element is not array-like.
   * @throws {EncodeError} (re-thrown) on a non-finite or zero vector. Note: a batch is
   *   appended in order, so an encode error mid-batch leaves the preceding vectors added.
   */
  add(vectors: Float32Array | number[][] | Float32Array[]): void {
    const vecs = this.#toVectorArray(vectors);
    this.fitCalibrationFromBatch(vecs); // first eligible batch fits + freezes TQ+
    this.#ensureCapacity(this.#n + vecs.length);
    for (let j = 0; j < vecs.length; j++) this.#appendOne(vecs[j]!);
  }

  /**
   * Add a single vector of length `dim`. A thin convenience over {@link add} for
   * callers that append one row at a time (e.g. {@link IdMapIndex} keeps its id↔slot
   * map in lockstep with each append).
   *
   * @throws {IndexError} `'INVALID_LENGTH'` if `vec.length !== dim`; `'INVALID_VECTOR'`
   *   if `vec` is not array-like.
   * @throws {EncodeError} (re-thrown) on a non-finite or zero vector.
   */
  addOne(vec: Float32Array | number[]): void {
    const len = vectorLength(vec);
    if (len !== this.#dim) {
      throw new IndexError('INVALID_LENGTH', `vector length ${len} != dim ${this.#dim}`);
    }
    this.#ensureCapacity(this.#n + 1);
    this.#appendOne(vec instanceof Float32Array ? vec : Float32Array.from(vec));
  }

  /** Remove all vectors, resetting the live count to 0 (backing capacity is retained). */
  clear(): void {
    this.#n = 0;
    this.#wasmDirty = true;
  }

  /** Build a read-only {@link EncodedDb} view over the live rows (no copy). */
  #db(): EncodedDb {
    const n = this.#n;
    const db: EncodedDb = {
      n,
      dim: this.#dim,
      bits: this.#bits,
      codes: this.#codes.subarray(0, n * this.#dim),
      scales: this.#scales.subarray(0, n),
      norms: this.#norms.subarray(0, n),
      centroids: this.#codebook.centroids,
      rotation: this.#rotation,
    };
    // The codes were encoded with this calibration; search must apply its dual
    // (q_calib + per-query bias) to stay unbiased — see ../core/search.
    if (this.#calibration !== undefined) db.calibration = this.#calibration;
    return db;
  }

  /**
   * Search for the k nearest vectors to `query`, best-first. Returns positional
   * slot indices and the reported metric values (see ../core/search).
   *
   * @throws {IndexError} `'EMPTY'` if the index has no vectors.
   * @throws {SearchError} (re-thrown) on invalid k, query length/finiteness, a
   *   zero query, or a mask length mismatch.
   */
  search(query: Float32Array, k: number, opts: IndexSearchOptions = {}): SearchResult {
    if (this.#n === 0) {
      throw new IndexError('EMPTY', 'cannot search an empty index');
    }
    const metric = opts.metric ?? this.#metric;
    const searchOpts: SearchOptions =
      opts.mask === undefined ? { metric } : { metric, mask: opts.mask };

    if (this.#wasmEnabled) {
      if (this.#wasm === undefined) this.#wasm = WasmKernel.create();
      const kernel = this.#wasm;
      if (
        kernel !== null &&
        this.#fastscan &&
        Number.isInteger(k) &&
        k > 0 &&
        query.length === this.#dim
      ) {
        return this.#searchFastScan(kernel, query, k, searchOpts);
      }
      // Exact WASM acceleration (transparent fallback to the scalar scan). The kernel
      // is created lazily and the codes are uploaded once per mutation, not per query.
      if (kernel !== null) {
        kernel.prepare(this.#n, this.#dim, 1 << this.#bits);
        if (this.#wasmDirty) {
          kernel.uploadCodes(this.#codes.subarray(0, this.#n * this.#dim));
          this.#wasmDirty = false;
        }
        return searchFlat(this.#db(), query, k, searchOpts, (lut, out) => kernel.score(lut, out));
      }
    }
    return searchFlat(this.#db(), query, k, searchOpts);
  }

  /**
   * FastScan search (4-bit): the v128 kernel scans every vector with a u8 LUT to rank a
   * candidate pool (≈4·k), then {@link searchFlat} rescores that pool exactly via a
   * mask — high recall at higher throughput, exact within the pool.
   */
  #searchFastScan(
    kernel: WasmKernel,
    query: Float32Array,
    k: number,
    searchOpts: SearchOptions,
  ): SearchResult {
    const dim = this.#dim;
    const n = this.#n;
    const levels = 16; // 4-bit
    kernel.prepareFastScan(n, dim);
    if (this.#wasmDirty) {
      kernel.uploadBlockedCodes(this.#codes.subarray(0, n * dim));
      this.#wasmDirty = false;
    }

    // Rotate the query; apply the calibration dual to the per-coordinate LUT + bias.
    const qRot = new Float32Array(dim);
    this.#rotation.apply(query, qRot);
    let lutQuery = qRot;
    let biasQ = 0;
    if (this.#calibration !== undefined) {
      const { shift, scale } = this.#calibration;
      const qCalib = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        qCalib[i] = qRot[i]! / scale[i]!;
        biasQ += qRot[i]! * shift[i]!;
      }
      lutQuery = qCalib;
    }
    const valLut = buildQueryLut(lutQuery, this.#codebook.centroids, dim, levels);

    // Quantize the float LUT to u8 with one global affine map so dim·max ≤ 65535 and the
    // u16 accumulator is a monotonic function of the true projection S = ⟨q_calib, c⟩.
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < valLut.length; i++) {
      const x = valLut[i]!;
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    const range = hi - lo;
    const B = Math.min(255, Math.floor(65535 / dim));
    const qScale = range > 0 ? B / range : 0;
    const lut8 = new Uint8Array(dim * 16);
    // Clamp to [0, B]: (valLut[i] - lo) * qScale is mathematically in [0, B], but
    // floating-point rounding can push the max element to B + 1, which would wrap
    // a Uint8Array write to 0 when B === 255 — silently corrupting that LUT entry.
    for (let i = 0; i < dim * levels; i++) {
      const q = Math.round((valLut[i]! - lo) * qScale);
      lut8[i] = q < 0 ? 0 : q > 255 ? 255 : q;
    }
    const acc = new Uint16Array(kernel.fastScanBlocks * 16);
    kernel.fastScan(lut8, acc);

    // Approximate metric per vector (dequantize acc → S), select the candidate pool.
    let qNormSq = 0;
    for (let i = 0; i < dim; i++) qNormSq += query[i]! * query[i]!;
    const norms2: QueryNorms = { qNorm: Math.sqrt(qNormSq), qNormSq };
    const deq = range > 0 ? range / B : 0;
    const mask = searchOpts.mask;
    const poolSize = Math.min(n, Math.max(k * 4, k + 64));
    const pool = new TopK(poolSize);
    for (let v = 0; v < n; v++) {
      if (mask !== undefined && !mask[v]) continue;
      const s = acc[v]! * deq + dim * lo;
      const { rankKey } = scoreMetric(
        searchOpts.metric,
        s - biasQ,
        this.#scales[v]!,
        this.#norms[v]!,
        norms2,
      );
      pool.add(rankKey, v);
    }

    // Exact rescore of the pool: searchFlat scores only the pooled slots (mask) exactly.
    const poolIdx = pool.result().indices;
    const poolMask = new Uint8Array(n);
    for (let i = 0; i < poolIdx.length; i++) poolMask[poolIdx[i]!] = 1;
    return searchFlat(this.#db(), query, k, { metric: searchOpts.metric, mask: poolMask });
  }

  /**
   * Delete the vector at positional slot `i` in O(1) by moving the last row into
   * the gap. The previous last slot's index changes to `i`; all other slots keep
   * their indices. Callers tracking external ids must observe this swap (see
   * ../index/id-map-index).
   *
   * @throws {IndexError} `'INVALID_INDEX'` if `i` is out of range.
   */
  swapRemove(i: number): void {
    if (!Number.isInteger(i) || i < 0 || i >= this.#n) {
      throw new IndexError('INVALID_INDEX', `index ${i} out of range [0, ${this.#n})`);
    }
    const last = this.#n - 1;
    if (i !== last) {
      this.#codes.copyWithin(i * this.#dim, last * this.#dim, (last + 1) * this.#dim);
      this.#scales[i] = this.#scales[last]!;
      this.#norms[i] = this.#norms[last]!;
    }
    this.#n = last;
    this.#wasmDirty = true;
  }

  /**
   * @internal Snapshot the live rows as a serialize {@link IndexPayload} of zero-copy
   * *views* (codes/scales/norms subarrays). Shared by {@link toBytes} and by
   * {@link IdMapIndex} (which adds the id column). The views alias internal storage and
   * are invalidated by the next mutation — not for general use; prefer {@link toBytes}.
   */
  toPayload(): IndexPayload {
    const n = this.#n;
    const payload: IndexPayload = {
      metric: this.#metric,
      bits: this.#bits,
      dim: this.#dim,
      n,
      seed: this.#seed,
      codes: this.#codes.subarray(0, n * this.#dim),
      scales: this.#scales.subarray(0, n),
      norms: this.#norms.subarray(0, n),
    };
    if (this.#calibration !== undefined) payload.calibration = this.#calibration;
    return payload;
  }

  /**
   * @internal Rebuild a positional index from a parsed {@link IndexPayload}. Rebuilds
   * the rotation and codebook from (dim, seed, bits) and copies the codes/scales/norms
   * into fresh backing storage. Shared by {@link fromBytes} and {@link IdMapIndex}.
   */
  static fromPayload(payload: IndexPayload): TurboQuantIndex {
    const idx = new TurboQuantIndex({
      dim: payload.dim,
      bits: payload.bits,
      metric: payload.metric,
      seed: payload.seed,
    });
    idx.#ensureCapacity(payload.n);
    idx.#codes.set(payload.codes);
    idx.#scales.set(payload.scales);
    idx.#norms.set(payload.norms);
    idx.#n = payload.n;
    // Adopt the stored calibration and lock the decision (so later adds don't refit).
    if (payload.calibration !== undefined) idx.#calibration = payload.calibration;
    idx.#calibrationFrozen = true;
    return idx;
  }

  /**
   * Serialize the index to a single versioned, bounds-validated byte buffer (see
   * ../io/serialize). Round-trips through {@link TurboQuantIndex.fromBytes}.
   */
  toBytes(): Uint8Array {
    return serializeIndex({ kind: 'positional', ...this.toPayload() });
  }

  /**
   * Reconstruct a positional index from bytes produced by {@link toBytes}. The
   * rotation and codebook are rebuilt from the embedded (dim, seed, bits); the
   * bytes are treated as untrusted and fully bounds-validated before any bulk read.
   *
   * @throws {IndexError} `'WRONG_KIND'` if the buffer holds an id-keyed index.
   * @throws {DeserializeError} (re-thrown) on a malformed/oversized buffer.
   */
  static fromBytes(bytes: Uint8Array): TurboQuantIndex {
    const parsed = deserializeIndex(bytes);
    if (parsed.kind !== 'positional') {
      throw new IndexError(
        'WRONG_KIND',
        `expected a positional index buffer, got '${parsed.kind}' (use IdMapIndex.fromBytes)`,
      );
    }
    return TurboQuantIndex.fromPayload(parsed);
  }
}
