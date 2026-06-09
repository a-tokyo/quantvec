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
// to build. For power-of-two dimensions we instead use {@link createHadamardRotation}
// — a randomized Hadamard transform (a few rounds of sign-flip + fast Walsh–Hadamard
// transform), an exact orthonormal map that costs O(d·log d) per apply and ~nothing to
// build, with equal-or-better recall (measured). FWHT is exact only at a power-of-two
// length (no zero-padding/truncation), so {@link createRotation} dispatches to the
// Hadamard rotation when dim is a power of two and to the dense rotation otherwise.

import { fwht, isPow2 } from './fwht';
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

/** Rounds of (sign-flip + normalized FWHT) in the randomized Hadamard rotation. */
const HADAMARD_ROUNDS = 3;

/**
 * Create a deterministic orthonormal rotation via a Randomized Hadamard Transform:
 * `HADAMARD_ROUNDS` rounds of (random ±1 diagonal sign-flip + normalized FWHT). This
 * is an exact orthonormal map (norm-preserving; `applyTranspose` is its exact inverse)
 * costing O(d·log d) per apply and ~O(d) to build — no dense matrix.
 *
 * `dim` MUST be a power of two so the FWHT is exact (no zero-padding/truncation, which
 * would lose energy and degrade recall). Use {@link createRotation} to pick this when
 * applicable and fall back to {@link createDenseRotation} otherwise.
 *
 * @throws {RotationError} code `'INVALID_DIM'` if dim is not a power of two.
 */
export function createHadamardRotation(dim: number, seed = 0): Rotation {
  if (!isPow2(dim)) {
    throw new RotationError(
      'INVALID_DIM',
      `Hadamard rotation requires a power-of-two dim, got ${dim}`,
    );
  }
  const rng = createRng(seed);
  const signs: Float64Array[] = [];
  for (let r = 0; r < HADAMARD_ROUNDS; r++) {
    const s = new Float64Array(dim);
    for (let i = 0; i < dim; i++) s[i] = rng.nextFloat() < 0.5 ? -1 : 1;
    signs.push(s);
  }
  const invSqrt = 1 / Math.sqrt(dim);
  const buf = new Float64Array(dim);

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
    // y = (∏_{r} F·D_r) x — for each round apply the sign diagonal then the
    // normalized FWHT (F = fwht / √dim, an orthonormal involution).
    apply(src: Float32Array, dst: Float32Array): void {
      checkLengths(src, dst);
      for (let i = 0; i < dim; i++) buf[i] = src[i]!;
      for (let r = 0; r < HADAMARD_ROUNDS; r++) {
        const s = signs[r]!;
        for (let i = 0; i < dim; i++) buf[i] *= s[i]!;
        fwht(buf);
        for (let i = 0; i < dim; i++) buf[i] *= invSqrt;
      }
      for (let i = 0; i < dim; i++) dst[i] = buf[i]!;
    },
    // Exact inverse: the transpose of (∏ F·D_r) is (∏ D_r·F) in reverse order; since
    // F and each D_r are symmetric orthonormal, undo the rounds back-to-front.
    applyTranspose(src: Float32Array, dst: Float32Array): void {
      checkLengths(src, dst);
      for (let i = 0; i < dim; i++) buf[i] = src[i]!;
      for (let r = HADAMARD_ROUNDS - 1; r >= 0; r--) {
        fwht(buf);
        const s = signs[r]!;
        for (let i = 0; i < dim; i++) buf[i] = buf[i]! * invSqrt * s[i]!;
      }
      for (let i = 0; i < dim; i++) dst[i] = buf[i]!;
    },
  };
}

/**
 * Build the fastest exact orthonormal rotation for `dim`: a Randomized Hadamard
 * transform ({@link createHadamardRotation}, O(d·log d)) when `dim` is a power of two,
 * else the dense Householder rotation ({@link createDenseRotation}, O(d²)). The choice
 * is a deterministic function of `dim`, so a serialized index rebuilds the identical
 * rotation from its stored `dim`/`seed` with no extra format field.
 */
export function createRotation(dim: number, seed = 0): Rotation {
  return isPow2(dim) ? createHadamardRotation(dim, seed) : createDenseRotation(dim, seed);
}
