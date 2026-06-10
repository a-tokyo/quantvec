// Flat quantized nearest-neighbor scan — quantvec's correctness ORACLE.
//
// Why this exists: quantvec's index is a *flat* quantized scan (see
// docs/research/architecture.md "Search path"). Every query rotates once, then
// the kernel walks all n stored vectors, scoring each from its 2–4-bit codes
// against a small per-query lookup table, multiplies by the RaBitQ per-vector
// `scale` (see ./encode), and keeps the best k. This module is the simple,
// obviously-correct scalar reference against which the future AssemblyScript
// v128 kernel is validated (WASM ≡ scalar).
//
// ── The per-query nibble LUT ──────────────────────────────────────────────────
// The query-side score for one database vector is S = ⟨q_rot, c⟩ where q_rot =
// Q·q and c is the vector's reconstruction, i.e. c[i] = centroids[code[i]]. So
//     S = Σ_i q_rot[i] · centroids[code[i]].
// `centroids` has only `levels` (= 2^bits) distinct values, so for a *fixed*
// query the term contributed by coordinate i depends only on the code at i. We
// precompute, once per query, the full table
//     lut[i*levels + code] = q_rot[i] · centroids[code]
// and the inner scan becomes a pure gather-and-add: S = Σ_i lut[i*levels +
// codes[i]] — one table lookup and one add per coordinate, no multiply in the
// hot loop. This is the scalar shape of the "nibble-split LUT" the SIMD kernel
// will block; here it is one flat Float32Array shared across the whole scan.
//
// ── Metric & ranking ──────────────────────────────────────────────────────────
// Each candidate's S, with the stored `scale`/`norm` and the query norms, is
// turned into a (reported value, higher-is-better rankKey) by ./metrics. We push
// rankKey into one size-k min-heap (./topk) so a single heap ranks dot, cosine,
// and euclidean alike; after selection we map the kept rankKeys back to reported
// values. Indices/values come out best-first.
//
// ── Allocation discipline ─────────────────────────────────────────────────────
// One rotated-query buffer and one LUT are allocated per `searchFlat` call (or
// reused via `buildQueryLut`'s `out` param) and shared across all n vectors — the
// scan never allocates per candidate.

import type { Calibration } from './calibrate';
import { scoreMetric } from './metrics';
import type { Distance, QueryNorms } from './metrics';
import type { Rotation } from './rotation';
import { TopK } from './topk';

/** Discriminated, code-tagged error for the search module. */
export class SearchError extends Error {
  readonly code:
    | 'INVALID_K'
    | 'INVALID_DIM'
    | 'INVALID_LENGTH'
    | 'MISMATCH'
    | 'ZERO_QUERY'
    | 'INVALID_MASK'
    | 'INVALID_SLOT';
  constructor(code: SearchError['code'], message: string) {
    super(message);
    this.name = 'SearchError';
    this.code = code;
  }
}

/**
 * A flat quantized database: the column-of-codes layout the scan walks plus the
 * per-vector RaBitQ `scale`/`norm` and the frozen rotation/centroids shared by
 * every vector.
 */
export interface EncodedDb {
  /** Number of stored vectors. */
  n: number;
  /** Vector dimension d (multiple of 8; matches `rotation.dim`). */
  dim: number;
  /** Quantizer bit-width; centroids has 2^bits entries. */
  bits: 2 | 3 | 4;
  /** Row-major codes, length n·dim, one byte per coordinate, each in [0, 2^bits-1]. */
  codes: Uint8Array;
  /** Per-vector RaBitQ scale (scale·⟨q_rot,c⟩ ≈ ⟨q,v⟩), length n. */
  scales: Float32Array;
  /** Per-vector original Euclidean norm ‖v‖, length n. */
  norms: Float32Array;
  /** Reconstruction levels c_0..c_{2^bits−1} (the codebook centroids). */
  centroids: Float32Array;
  /** Frozen orthonormal rotation Q (same dim) applied to the query. */
  rotation: Rotation;
  /**
   * Optional per-coordinate TQ+ calibration the codes were encoded with (length dim
   * each). When present the query is calibrated (q_calib = q_rot / scale) before the
   * LUT and a per-query bias ⟨q_rot, shift⟩ is subtracted from each score; omitted
   * means the un-calibrated path.
   */
  calibration?: Calibration;
}

/** Result of {@link searchFlat}: aligned indices and reported metric values, best-first. */
export interface SearchResult {
  /** Original database indices of the k best hits, best-first. */
  indices: Int32Array;
  /** Reported metric values aligned with `indices` (similarity for dot/cosine, dist² for euclidean). */
  scores: Float32Array;
}

/** Options for {@link searchFlat}. */
export interface SearchOptions {
  /** Nearest-neighbor metric to rank by. */
  metric: Distance;
  /**
   * Optional allowlist: vector j is scanned only if `mask[j]` is truthy. A
   * boolean[] or a Uint8Array (0 = excluded) are both accepted; length must be n.
   * Omit to scan all vectors.
   */
  mask?: Uint8Array | boolean[];
}

/**
 * Invert ./metrics' "higher-is-better" rankKey back to the reported metric value
 * for a kept candidate. dot/cosine report the key itself; euclidean reports the
 * squared distance, which is the negated key (rankKey = −dist²). This is the
 * exact inverse of the `value`↔`rankKey` relation in {@link scoreMetric}, so the
 * scan can rank on keys alone and recover values only for the k survivors.
 */
function mapKeyToValue(metric: Distance, rankKey: number): number {
  return metric === 'euclidean' ? -rankKey : rankKey;
}

/**
 * Build the per-query nibble lookup table
 *     lut[i*levels + code] = qRot[i] · centroids[code]
 * for i in [0, dim) and code in [0, levels). `centroids.length` must equal
 * `levels`; `qRot.length` must equal `dim`. Pass `out` (length dim·levels) to
 * reuse a buffer across queries; otherwise a fresh array is allocated.
 *
 * @throws {SearchError} on length/dimension mismatch.
 */
export function buildQueryLut(
  qRot: Float32Array,
  centroids: Float32Array,
  dim: number,
  levels: number,
  out?: Float32Array,
): Float32Array {
  if (qRot.length !== dim) {
    throw new SearchError('INVALID_LENGTH', `qRot length ${qRot.length} != dim ${dim}`);
  }
  if (centroids.length !== levels) {
    throw new SearchError('MISMATCH', `centroids length ${centroids.length} != levels ${levels}`);
  }
  const size = dim * levels;
  const lut = out ?? new Float32Array(size);
  if (lut.length !== size) {
    throw new SearchError('INVALID_LENGTH', `out length ${lut.length} != dim·levels ${size}`);
  }
  for (let i = 0; i < dim; i++) {
    const qi = qRot[i]!;
    const base = i * levels;
    for (let code = 0; code < levels; code++) lut[base + code] = qi * centroids[code]!;
  }
  return lut;
}

/**
 * Flat scan: rotate the query, build the per-query LUT, score every (unmasked)
 * stored vector S_j = Σ_i lut[i*levels + codes[j*dim+i]], turn S_j into the
 * chosen metric, and return the k best, best-first.
 *
 * Reuses a single rotated-query buffer and a single LUT across the whole scan; no
 * per-candidate allocation. Validates k, the query length/finiteness, a zero
 * query (no direction → cosine is undefined and dot/euclidean degenerate), the
 * per-vector array lengths, the centroid count, and the mask length — all real
 * boundaries a caller can hit, not dead guards.
 *
 * An optional `computeScores` hook (the WASM kernel) fills the projections
 * S[j] = ⟨q_calib, c_j⟩ for all j into a provided buffer; when omitted the scalar
 * loop computes them inline (allocation-free). Either way the metric, per-query
 * calibration bias, mask, and top-k are handled here, so the result is identical.
 *
 * @throws {SearchError} on any failed precondition above.
 */
/**
 * Validate a query vector against `dim` — length, per-element finiteness, and a
 * non-zero norm — throwing the same typed errors for the same bad inputs on every
 * search path. Used by {@link searchFlat}/{@link searchSlots} (via the shared scan
 * preamble) and by the index's IVF branch *before* centroid probing, so a malformed
 * query never reaches the probe arithmetic. Returns the query norms.
 *
 * @throws {SearchError} `'INVALID_LENGTH'` on a wrong-length or non-finite query;
 *   `'ZERO_QUERY'` on a zero query (no direction).
 */
export function validateQuery(query: Float32Array, dim: number): QueryNorms {
  if (query.length !== dim) {
    throw new SearchError('INVALID_LENGTH', `query length ${query.length} != dim ${dim}`);
  }
  let qNormSq = 0;
  for (let i = 0; i < dim; i++) {
    const x = query[i]!;
    if (!Number.isFinite(x)) {
      throw new SearchError('INVALID_LENGTH', `query[${i}] must be finite, got ${x}`);
    }
    qNormSq += x * x;
  }
  if (qNormSq === 0) {
    throw new SearchError('ZERO_QUERY', 'cannot search with a zero query (no direction)');
  }
  return { qNorm: Math.sqrt(qNormSq), qNormSq };
}

/** Per-query state shared by {@link searchFlat} and {@link searchSlots}. */
interface PreparedScan {
  /** The per-query nibble LUT (dim·levels). */
  lut: Float32Array;
  /** Per-query calibration bias ⟨q_rot, shift⟩ (0 when un-calibrated). */
  biasQ: number;
  /** Query norms for ./metrics. */
  norms2: QueryNorms;
  /** 2^bits. */
  levels: number;
}

/**
 * Shared scan preamble: validate every boundary (k, shapes, mask length, query
 * finiteness/zero), rotate the query once, apply the calibration dual, and build
 * the per-query LUT. Both scan entry points run identical validation, so they
 * throw identical typed errors for identical bad inputs.
 *
 * @throws {SearchError} on any failed precondition (see {@link searchFlat}).
 */
function prepareScan(
  db: EncodedDb,
  query: Float32Array,
  k: number,
  opts: SearchOptions,
): PreparedScan {
  const { n, dim, bits, codes, scales, norms, centroids, rotation } = db;
  const levels = 1 << bits;

  // ── Boundary validation ──────────────────────────────────────────────────
  if (!Number.isInteger(k) || k <= 0) {
    throw new SearchError('INVALID_K', `k must be a positive integer, got ${k}`);
  }
  if (rotation.dim !== dim) {
    throw new SearchError('MISMATCH', `rotation.dim ${rotation.dim} != dim ${dim}`);
  }
  const norms2 = validateQuery(query, dim);
  if (codes.length !== n * dim) {
    throw new SearchError('MISMATCH', `codes length ${codes.length} != n·dim ${n * dim}`);
  }
  if (scales.length !== n) {
    throw new SearchError('MISMATCH', `scales length ${scales.length} != n ${n}`);
  }
  if (norms.length !== n) {
    throw new SearchError('MISMATCH', `norms length ${norms.length} != n ${n}`);
  }
  if (centroids.length !== levels) {
    throw new SearchError('MISMATCH', `centroids length ${centroids.length} != levels ${levels}`);
  }
  const { mask } = opts;
  if (mask !== undefined && mask.length !== n) {
    throw new SearchError('INVALID_MASK', `mask length ${mask.length} != n ${n}`);
  }

  // ── Rotate the query once, then build the shared LUT ─────────────────────
  // With TQ+ calibration the codes hold calibrated coordinates, so we score against
  // q_calib = q_rot / scale and subtract the per-query bias ⟨q_rot, shift⟩ from every
  // candidate (the exact dual of the de-calibrated reconstruction used in ./encode).
  const qRot = new Float32Array(dim);
  rotation.apply(query, qRot);
  const { calibration } = db;
  let lutQuery = qRot;
  let biasQ = 0;
  if (calibration !== undefined) {
    const { shift, scale: scaleCal } = calibration;
    const qCalib = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      qCalib[i] = qRot[i]! / scaleCal[i]!;
      biasQ += qRot[i]! * shift[i]!;
    }
    lutQuery = qCalib;
  }
  const lut = buildQueryLut(lutQuery, centroids, dim, levels);

  return { lut, biasQ, norms2, levels };
}

export function searchFlat(
  db: EncodedDb,
  query: Float32Array,
  k: number,
  opts: SearchOptions,
  computeScores?: (lut: Float32Array, out: Float64Array) => void,
): SearchResult {
  const { n, dim, codes, scales, norms } = db;
  const { lut, biasQ, norms2, levels } = prepareScan(db, query, k, opts);
  const { mask, metric } = opts;
  const top = new TopK(k);

  // ── The flat scan ────────────────────────────────────────────────────────
  // Rank by the higher-is-better key; the heap keeps the k best keys + indices.
  // For dot/cosine the reported value equals the key; for euclidean the value is
  // the squared distance −rankKey. We recover values from the kept keys after
  // selection (mapKeyToValue), so the scan stays a pure score-and-add with no
  // per-candidate allocation.
  if (computeScores !== undefined) {
    // WASM path: the kernel fills every S[j] = ⟨q_calib, c_j⟩; we apply the
    // calibration bias, metric, and mask here (the cheap O(n) part).
    const scores = new Float64Array(n);
    computeScores(lut, scores);
    for (let j = 0; j < n; j++) {
      if (mask !== undefined && !mask[j]) continue;
      const { rankKey } = scoreMetric(metric, scores[j]! - biasQ, scales[j]!, norms[j]!, norms2);
      top.add(rankKey, j);
    }
  } else {
    for (let j = 0; j < n; j++) {
      if (mask !== undefined && !mask[j]) continue;
      const base = j * dim;
      let s = 0;
      for (let i = 0; i < dim; i++) s += lut[i * levels + codes[base + i]!]!;
      const { rankKey } = scoreMetric(metric, s - biasQ, scales[j]!, norms[j]!, norms2);
      top.add(rankKey, j);
    }
  }

  const { indices, scores: keys } = top.result();
  const scores = new Float32Array(indices.length);
  for (let i = 0; i < indices.length; i++) scores[i] = mapKeyToValue(metric, keys[i]!);
  return { indices, scores };
}

/**
 * Subset scan: score ONLY the given `slots` (database row indices) — the IVF
 * probed-posting-list scan. Identical query preparation, validation, and error
 * semantics as {@link searchFlat} (same typed errors for the same bad inputs),
 * plus `'INVALID_SLOT'` for a slot outside [0, n). `opts.mask`, when given, is
 * the full n-length allowlist indexed by slot — a probed slot the mask excludes
 * is skipped, exactly like the flat scan.
 *
 * Returns up to k best, best-first; fewer (possibly zero) when the slots/mask
 * yield fewer candidates. A slot listed twice would be scored twice (the heap
 * would then hold duplicates) — callers pass disjoint posting lists, and the
 * IVF bookkeeping guarantees disjointness, so this is not guarded.
 *
 * @throws {SearchError} on any failed precondition above.
 */
export function searchSlots(
  db: EncodedDb,
  query: Float32Array,
  k: number,
  slots: Int32Array,
  opts: SearchOptions,
): SearchResult {
  const { n, dim, codes, scales, norms } = db;
  const { lut, biasQ, norms2, levels } = prepareScan(db, query, k, opts);
  const { mask, metric } = opts;

  const top = new TopK(k);
  for (let t = 0; t < slots.length; t++) {
    const j = slots[t]!;
    if (!Number.isInteger(j) || j < 0 || j >= n) {
      throw new SearchError('INVALID_SLOT', `slot ${j} out of range [0, ${n})`);
    }
    if (mask !== undefined && !mask[j]) continue;
    const base = j * dim;
    let s = 0;
    for (let i = 0; i < dim; i++) s += lut[i * levels + codes[base + i]!]!;
    const { rankKey } = scoreMetric(metric, s - biasQ, scales[j]!, norms[j]!, norms2);
    top.add(rankKey, j);
  }

  const { indices, scores: keys } = top.result();
  const scores = new Float32Array(indices.length);
  for (let i = 0; i < indices.length; i++) scores[i] = mapKeyToValue(metric, keys[i]!);
  return { indices, scores };
}
