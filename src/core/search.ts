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
    | 'INVALID_MASK';
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
export function searchFlat(
  db: EncodedDb,
  query: Float32Array,
  k: number,
  opts: SearchOptions,
  computeScores?: (lut: Float32Array, out: Float64Array) => void,
): SearchResult {
  const { n, dim, bits, codes, scales, norms, centroids, rotation } = db;
  const levels = 1 << bits;

  // ── Boundary validation ──────────────────────────────────────────────────
  if (!Number.isInteger(k) || k <= 0) {
    throw new SearchError('INVALID_K', `k must be a positive integer, got ${k}`);
  }
  if (rotation.dim !== dim) {
    throw new SearchError('MISMATCH', `rotation.dim ${rotation.dim} != dim ${dim}`);
  }
  if (query.length !== dim) {
    throw new SearchError('INVALID_LENGTH', `query length ${query.length} != dim ${dim}`);
  }
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

  // ── Query norms (also rejects a non-finite / zero query) ─────────────────
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
  const qNorm = Math.sqrt(qNormSq);
  const norms2: QueryNorms = { qNorm, qNormSq };

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

  const { metric } = opts;
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
