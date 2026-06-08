// Deterministic orthonormal rotation for quantvec's data-oblivious quantizer.
//
// Why this exists: TurboQuant (arXiv:2504.19874; docs/research/turboquant.md)
// rotates every vector by a single, fixed random orthonormal matrix Q before
// quantizing. After this rotation a random unit vector is uniform on the sphere,
// so each coordinate of Qx follows Beta((d-1)/2,(d-1)/2) on [-1,1] — a known,
// data-independent marginal. That is what lets a precomputed per-coordinate
// Lloyd-Max codebook (see ./codebook) be optimal with NO training data.
//
// The matrix must be reproducible bit-for-bit across runtimes (it is part of the
// serialized index), so Q is derived deterministically from (dim, seed) via the
// seeded RNG in ./rng — never Math.random.
//
// Construction (clean-room, from the standard published form): fill a dim×dim
// matrix A with i.i.d. standard-Gaussian entries, then orthonormalize via
// Householder QR. The Q factor of the QR decomposition of a Gaussian matrix is
// Haar-distributed (uniform) on the orthogonal group — exactly the random
// rotation TurboQuant calls for. We use Householder reflections rather than
// classical Gram–Schmidt because Gram–Schmidt loses orthogonality catastrophically
// to floating-point cancellation, whereas Householder QR is unconditionally
// numerically stable (Golub & Van Loan, "Matrix Computations", §5.2). To make Q
// uniquely determined (and so deterministic across runtimes) we fix the sign of
// each Householder reflection by the sign of the pivot, which pins the signs of
// the R diagonal to be positive — the standard QR uniqueness convention.
//
// Performance note: this dense rotation costs O(d²) per applied vector and O(d³)
// to build. That is fine for validation and the small/medium dimensions exercised
// here. The PLANNED performance path is a structured O(d log d) transform — a
// randomized Hadamard transform (sign-flip diagonal + fast Walsh–Hadamard
// transform), which is also a valid TurboQuant rotation. It is intentionally NOT
// implemented in this wave; the `Rotation` interface below is the seam: a future
// `createHadamardRotation` would implement the same interface and be a drop-in
// replacement. (FWHT requires dim to be a power of two; the multiple-of-8 dim
// validation here is the shared, weaker precondition.)

import { createRng } from './rng';

/** Discriminated, code-tagged error for the rotation module. */
export class RotationError extends Error {
  readonly code: 'INVALID_DIM' | 'INVALID_LENGTH';
  constructor(code: RotationError['code'], message: string) {
    super(message);
    this.name = 'RotationError';
    this.code = code;
  }
}

/**
 * A fixed orthonormal linear map y = Q·x and its inverse Qᵀ·x.
 *
 * Implementations are deterministic and orthonormal (QᵀQ = I), so `apply`
 * preserves Euclidean norm and `applyTranspose` is its exact inverse. The
 * interface is deliberately representation-agnostic: the dense matrix
 * implementation here can be swapped for a future O(d log d) structured
 * transform (randomized Hadamard / FWHT) without changing callers.
 */
export interface Rotation {
  /** Dimension d; both `src` and `dst` must have length d. */
  readonly dim: number;
  /** Write y = Q·x into `dst` (must not alias `src`). */
  apply(src: Float32Array, dst: Float32Array): void;
  /** Write x = Qᵀ·y into `dst` — the inverse of `apply` (must not alias `src`). */
  applyTranspose(src: Float32Array, dst: Float32Array): void;
}

/** Reject dims that are not a positive multiple of 8. */
function validateDim(dim: number): void {
  if (!Number.isInteger(dim) || dim <= 0 || dim % 8 !== 0) {
    throw new RotationError('INVALID_DIM', `dim must be a positive multiple of 8, got ${dim}`);
  }
}

/**
 * Build a Haar-uniform orthonormal dim×dim matrix Q (row-major) deterministically
 * from a Gaussian matrix via Householder QR.
 *
 * Returns Q as a row-major Float32Array of length dim·dim: Q[r*dim + c] is the
 * entry in row r, column c, and y = Q·x is yᵣ = Σ_c Q[r*dim+c]·xᵢ.
 */
function buildQ(dim: number, seed: number): Float32Array {
  const rng = createRng(seed);

  // A holds the matrix being factored, row-major, in double precision (the QR
  // arithmetic is accumulation-heavy; we narrow to Float32 only at the end).
  const a = new Float64Array(dim * dim);
  for (let i = 0; i < a.length; i++) a[i] = rng.nextGaussian();

  // Q accumulates the product of Householder reflections, initialized to I.
  const q = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) q[i * dim + i] = 1;

  // Householder QR: for each column k, reflect A[k:, k] onto the k-th axis.
  // The reflection is H = I - β v vᵀ; we apply it to the trailing block of A
  // and accumulate it into Q. The pivot sign is chosen to avoid cancellation
  // AND to make the result unique (R diagonal positive) → deterministic.
  const v = new Float64Array(dim);
  for (let k = 0; k < dim - 1; k++) {
    // Norm of the sub-column A[k:, k].
    let normSq = 0;
    for (let i = k; i < dim; i++) {
      const e = a[i * dim + k]!;
      normSq += e * e;
    }
    const alpha = Math.sqrt(normSq);
    if (alpha === 0) continue; // Degenerate column (measure-zero); skip reflection.

    const a_kk = a[k * dim + k]!;
    // Choose sign so the pivot of v grows (no cancellation). We negate by the
    // sign of a_kk; the resulting R[k,k] = -sign(a_kk)*alpha. To pin R[k,k] > 0
    // we flip the overall reflection accordingly below.
    const sign = a_kk >= 0 ? 1 : -1;
    const vk = a_kk + sign * alpha;

    // Build Householder vector v = (a_kk + sign*alpha, a[k+1..], ...).
    v[k] = vk;
    for (let i = k + 1; i < dim; i++) v[i] = a[i * dim + k]!;

    // β = 2 / (vᵀv). vᵀv = vk² + Σ_{i>k} a[i,k]².
    let vtv = vk * vk;
    for (let i = k + 1; i < dim; i++) {
      const e = a[i * dim + k]!;
      vtv += e * e;
    }
    if (vtv === 0) continue;
    const beta = 2 / vtv;

    // Apply H to A from the left: A ← A - β v (vᵀ A), columns k..dim-1.
    for (let c = k; c < dim; c++) {
      let s = 0;
      for (let i = k; i < dim; i++) s += v[i]! * a[i * dim + c]!;
      s *= beta;
      for (let i = k; i < dim; i++) a[i * dim + c] -= s * v[i]!;
    }

    // Accumulate H into Q from the right: Q ← Q·H = Q - β (Q v) vᵀ.
    // Q·H affects all rows of Q, columns k..dim-1 (where v is supported).
    for (let r = 0; r < dim; r++) {
      let s = 0;
      for (let i = k; i < dim; i++) s += q[r * dim + i]! * v[i]!;
      s *= beta;
      for (let i = k; i < dim; i++) q[r * dim + i] -= s * v[i]!;
    }
  }

  // Pin uniqueness: the R diagonal is R[k,k] (the post-reflection A[k,k]). To make
  // the decomposition unique (deterministic) with R[k,k] ≥ 0, flip the sign of
  // column k of Q (and conceptually row k of R) wherever R[k,k] < 0. Q's columns
  // are the orthonormal basis; flipping a column keeps Q orthonormal.
  for (let k = 0; k < dim; k++) {
    if (a[k * dim + k]! < 0) {
      for (let r = 0; r < dim; r++) q[r * dim + k] = -q[r * dim + k]!;
    }
  }

  // Narrow to Float32 (the storage/runtime precision for the hot apply path).
  const out = new Float32Array(dim * dim);
  for (let i = 0; i < out.length; i++) out[i] = q[i]!;
  return out;
}

/**
 * Create a deterministic dense orthonormal rotation of the given dimension.
 *
 * The same (dim, seed) always yields the same Q on every runtime. `dim` must be
 * a positive multiple of 8.
 *
 * @throws {RotationError} code `'INVALID_DIM'` if dim is invalid.
 */
export function createDenseRotation(dim: number, seed = 0): Rotation {
  validateDim(dim);
  const q = buildQ(dim, seed);

  function checkLengths(src: Float32Array, dst: Float32Array): void {
    if (src.length !== dim || dst.length !== dim) {
      throw new RotationError(
        'INVALID_LENGTH',
        `src and dst must have length ${dim}, got src=${src.length}, dst=${dst.length}`,
      );
    }
  }

  return {
    dim,
    // y = Q·x : yᵣ = Σ_c Q[r,c]·xᵢ (row r dotted with x).
    apply(src: Float32Array, dst: Float32Array): void {
      checkLengths(src, dst);
      for (let r = 0; r < dim; r++) {
        let s = 0;
        const base = r * dim;
        for (let c = 0; c < dim; c++) s += q[base + c]! * src[c]!;
        dst[r] = s;
      }
    },
    // x = Qᵀ·y : xᵢ = Σ_r Q[r,c]·yᵣ (column c dotted with y). Since Q is
    // orthonormal this is the exact inverse of `apply`.
    applyTranspose(src: Float32Array, dst: Float32Array): void {
      checkLengths(src, dst);
      for (let c = 0; c < dim; c++) dst[c] = 0;
      for (let r = 0; r < dim; r++) {
        const yr = src[r]!;
        const base = r * dim;
        for (let c = 0; c < dim; c++) dst[c] += q[base + c]! * yr;
      }
    },
  };
}
