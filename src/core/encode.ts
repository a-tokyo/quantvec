// The per-vector encode pipeline for quantvec: normalize → rotate → quantize →
// reconstruct → store the RaBitQ per-vector scale.
//
// Why this exists: this is the heart of quantvec's database side. It turns a raw
// vector into (a) a compact array of per-coordinate codes and (b) a single
// per-vector `scale` float that makes the cheap query-time code score an
// (approximately unbiased) estimate of the true dot product. It composes the
// frozen rotation (./rotation) and codebook (./codebook) built in earlier waves.
//
// The pipeline (TurboQuant Algorithm 1, arXiv:2504.19874, docs/research/turboquant.md
// for steps 2-4; RaBitQ, arXiv:2405.12497, docs/research/rabitq.md for step 5):
//
//   1. norm = ‖v‖₂; unit direction o = v / norm.
//   2. o_rot = Q·o            (rotate; ‖o_rot‖ = 1 since Q is orthonormal).
//   3. codes[i] = quantizeCoord(o_rot[i], boundaries)        ∈ [0, 2^bits-1].
//   4. c[i] = centroids[codes[i]]   (reconstructed rotated direction; NOT
//                                    renormalized — we keep the quantized vector).
//   5. inner = ⟨o_rot, c⟩;  scale = norm / inner.
//
// ── Why scale = norm / ⟨o_rot, c⟩ (the RaBitQ correction; derive here) ─────────
// At query time the cheap score is S = ⟨q_rot, c⟩ where q_rot = Q·q and c is the
// stored reconstruction. We want an estimate of the TRUE inner product ⟨q, v⟩.
//
//   • Q is orthonormal, so ⟨q, v⟩ = ⟨q_rot, v_rot⟩ = norm · ⟨q_rot, o_rot⟩.
//   • c is a quantization of the unit direction o_rot. RaBitQ's key approximation
//     is that the reconstruction error is (in expectation, after the random
//     rotation) orthogonal to the true direction, so projecting c back onto o_rot
//     recovers the direction up to the factor ⟨o_rot, c⟩:
//         ⟨q_rot, o_rot⟩ ≈ ⟨q_rot, c⟩ / ⟨o_rot, c⟩.
//   • Substituting:
//         ⟨q, v⟩ ≈ norm · ⟨q_rot, c⟩ / ⟨o_rot, c⟩ = (norm / ⟨o_rot, c⟩) · S.
//
// So storing scale = norm / ⟨o_rot, c⟩ makes `scale · S` an (approximately
// unbiased) estimator of ⟨q, v⟩ — verified empirically in encode.test.ts. The
// single per-vector float is all the query loop needs after scoring the codes.
//
// ── Per-coordinate empirical calibration (TQ+) — optional ─────────────────────
// TurboQuant+ refines accuracy by remapping each coordinate onto the canonical
// marginal before quantizing (see ./calibrate). When a `calibration` is supplied we
// quantize cal_i = (o_rot_i + shift_i)·scale_i, and the reconstruction used for the
// RaBitQ inner product is the *de-calibrated* level r_i = c_i/scale_i − shift_i, so
// step 5 is unchanged: inner = ⟨o_rot, r⟩. With the identity calibration this is
// exactly the un-calibrated pipeline above (shift 0, scale 1 → r_i = c_i).

import type { Calibration } from './calibrate';
import { quantizeCoord } from './codebook';
import type { Rotation } from './rotation';

/** Discriminated, code-tagged error for the encode module. */
export class EncodeError extends Error {
  readonly code: 'INVALID_DIM' | 'INVALID_BITS' | 'INVALID_LENGTH' | 'ZERO_VECTOR' | 'MISMATCH';
  constructor(code: EncodeError['code'], message: string) {
    super(message);
    this.name = 'EncodeError';
    this.code = code;
  }
}

/** The frozen codebook a vector is quantized against (from ./codebook). */
export interface EncodeCodebook {
  /** Decision boundaries t_0..t_n on [-1,1] (length 2^bits + 1). */
  boundaries: Float32Array;
  /** Reconstruction levels c_0..c_{n-1} (length 2^bits). */
  centroids: Float32Array;
}

/** Reusable scratch buffers so a hot encode loop avoids per-call allocation. */
export interface EncodeScratch {
  /** Unit direction o = v/‖v‖ (length dim). */
  unit: Float32Array;
  /** Rotated unit direction o_rot = Q·o (length dim). */
  rotated: Float32Array;
}

/** Options for {@link encodeVector}. */
export interface EncodeOptions {
  /** Vector dimension d (must match vec.length, rotation.dim, and the codebook). */
  dim: number;
  /** Quantizer bit-width; the codebook must have 2^bits centroids. */
  bits: 2 | 3 | 4;
  /** Frozen orthonormal rotation Q (same dim). */
  rotation: Rotation;
  /** Frozen per-(dim,bits) Lloyd-Max codebook. */
  codebook: EncodeCodebook;
  /** Optional reusable scratch buffers (allocated on demand if omitted). */
  scratch?: EncodeScratch;
  /** Optional per-coordinate TQ+ calibration (length dim each); identity if omitted. */
  calibration?: Calibration;
}

/** The encoded representation of one database vector. */
export interface EncodedVector {
  /** Per-coordinate quantizer codes, length dim, each in [0, 2^bits-1]. */
  codes: Uint8Array;
  /** RaBitQ per-vector scale: scale·⟨q_rot, c⟩ ≈ ⟨q, v⟩. */
  scale: number;
  /** Original Euclidean norm ‖v‖ (stored for metric reconstruction). */
  norm: number;
}

/**
 * Allocate reusable scratch buffers for {@link encodeVector} at dimension `dim`.
 * Pass the result via `opts.scratch` to encode many vectors without per-call
 * allocation.
 */
export function createEncodeScratch(dim: number): EncodeScratch {
  return { unit: new Float32Array(dim), rotated: new Float32Array(dim) };
}

function validateBits(bits: number): asserts bits is 2 | 3 | 4 {
  if (bits !== 2 && bits !== 3 && bits !== 4) {
    throw new EncodeError('INVALID_BITS', `bits must be one of {2, 3, 4}, got ${bits}`);
  }
}

/**
 * Encode one database vector into per-coordinate codes plus the RaBitQ scale and
 * the stored norm, exactly per the pipeline in this file's header.
 *
 * The zero vector (‖v‖ = 0) has no direction to quantize and is rejected with a
 * typed `ZERO_VECTOR` error; callers that want to keep zero rows should filter or
 * substitute them upstream (documented choice — we never emit a NaN scale).
 *
 * @throws {EncodeError} on dimension/bits/length mismatch, non-finite input, or a
 *   zero/degenerate vector.
 */
export function encodeVector(vec: Float32Array, opts: EncodeOptions): EncodedVector {
  const { dim, bits, rotation, codebook } = opts;

  // ── Boundary validation (guard clauses) ──────────────────────────────────
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new EncodeError('INVALID_DIM', `dim must be a positive integer, got ${dim}`);
  }
  validateBits(bits);
  if (vec.length !== dim) {
    throw new EncodeError('INVALID_LENGTH', `vec length ${vec.length} != dim ${dim}`);
  }
  if (rotation.dim !== dim) {
    throw new EncodeError('MISMATCH', `rotation.dim ${rotation.dim} != dim ${dim}`);
  }
  const levels = 1 << bits;
  if (codebook.centroids.length !== levels || codebook.boundaries.length !== levels + 1) {
    throw new EncodeError(
      'MISMATCH',
      `codebook must have ${levels} centroids and ${levels + 1} boundaries for ${bits}-bit, ` +
        `got ${codebook.centroids.length}/${codebook.boundaries.length}`,
    );
  }
  if (
    opts.calibration !== undefined &&
    (opts.calibration.shift.length !== dim || opts.calibration.scale.length !== dim)
  ) {
    throw new EncodeError(
      'MISMATCH',
      `calibration shift/scale must have length ${dim}, got ` +
        `${opts.calibration.shift.length}/${opts.calibration.scale.length}`,
    );
  }

  // ── Step 1: norm and unit direction ──────────────────────────────────────
  let normSq = 0;
  for (let i = 0; i < dim; i++) {
    const x = vec[i]!;
    if (!Number.isFinite(x)) {
      throw new EncodeError('INVALID_LENGTH', `vec[${i}] must be finite, got ${x}`);
    }
    normSq += x * x;
  }
  const norm = Math.sqrt(normSq);
  if (norm === 0) {
    throw new EncodeError('ZERO_VECTOR', 'cannot encode a zero vector (no direction to quantize)');
  }

  const scratch = opts.scratch ?? createEncodeScratch(dim);
  const { unit, rotated } = scratch;
  if (unit.length !== dim || rotated.length !== dim) {
    throw new EncodeError(
      'MISMATCH',
      `scratch buffers must have length ${dim}, got unit=${unit.length}, rotated=${rotated.length}`,
    );
  }
  const invNorm = 1 / norm;
  for (let i = 0; i < dim; i++) unit[i] = vec[i]! * invNorm;

  // ── Step 2: rotate the unit direction (‖o_rot‖ = 1) ──────────────────────
  rotation.apply(unit, rotated);

  // ── Steps 3-5: quantize each coordinate, reconstruct, accumulate ⟨o_rot, r⟩ ─
  // Without calibration r = c (the centroid). With TQ+ calibration we quantize the
  // calibrated coordinate and accumulate against the de-calibrated reconstruction
  // r_i = c_i/scale_i − shift_i, so the RaBitQ scale below is unchanged in form.
  const { boundaries, centroids } = codebook;
  const codes = new Uint8Array(dim);
  let inner = 0;
  const calibration = opts.calibration;
  if (calibration === undefined) {
    for (let i = 0; i < dim; i++) {
      const code = quantizeCoord(rotated[i]!, boundaries);
      codes[i] = code;
      inner += rotated[i]! * centroids[code]!;
    }
  } else {
    const { shift, scale: scaleCal } = calibration;
    for (let i = 0; i < dim; i++) {
      const code = quantizeCoord((rotated[i]! + shift[i]!) * scaleCal[i]!, boundaries);
      codes[i] = code;
      inner += rotated[i]! * (centroids[code]! / scaleCal[i]! - shift[i]!);
    }
  }

  // ── RaBitQ scale = norm / ⟨o_rot, c⟩ ─────────────────────────────────────
  // `inner` is the projection of the unit direction o_rot onto its own
  // reconstruction c. The codebook is symmetric with monotonically increasing
  // centroids and its central decision boundary at 0, so quantizeCoord(x) returns
  // a centroid with the same sign as x — every term o_rot[i]·c[i] ≥ 0, hence
  // inner > 0 for a unit direction (empirically inner ∈ [0.55, 1.34]). No clamp
  // branch is needed, and none would be reachable or testable (greenfield: no
  // dead guards).
  const scale = norm / inner;

  return { codes, scale, norm };
}

/**
 * Query-side projection score S = Σ_i centroids[codes[i]] · qRot[i] = ⟨q_rot, c⟩.
 *
 * This is the cheap score the index multiplies by the per-vector `scale` to
 * estimate ⟨q, v⟩ (see this file's header). It is the simple, correct reference
 * kernel; the nibble-split LUT / SIMD-blocked fast path is a later wave.
 *
 * @throws {EncodeError} `'MISMATCH'` if codes and qRot lengths differ.
 */
export function scoreCodes(codes: Uint8Array, centroids: Float32Array, qRot: Float32Array): number {
  if (codes.length !== qRot.length) {
    throw new EncodeError('MISMATCH', `codes length ${codes.length} != qRot length ${qRot.length}`);
  }
  let s = 0;
  for (let i = 0; i < codes.length; i++) s += centroids[codes[i]!]! * qRot[i]!;
  return s;
}
