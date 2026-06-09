// TQ+ per-coordinate calibration for quantvec.
//
// Why this exists: after the random rotation each coordinate is *theoretically*
// distributed like the canonical coordinate marginal (≈ N(0, 1/d)); the fixed
// Lloyd-Max codebook (see ./codebook) is optimal for exactly that distribution. On
// real data the empirical per-coordinate spread drifts from the canonical one, so a
// cheap per-coordinate affine map (the TurboQuant+ refinement) that rescales each
// rotated coordinate onto the canonical marginal recovers some recall at no extra
// storage per vector beyond the two calibration vectors.
//
// Map: cal_i = (rot_i + shift_i) · scale_i, fit so the empirical [5%, 95%] range of
// coordinate i lands on the canonical [5%, 95%] range. This composes with quantvec's
// RaBitQ estimator (the de-calibrated reconstruction r_i = c_i/scale_i − shift_i is
// what the inner product is taken against; see ./encode and ./search), and reduces
// to the identity (shift 0, scale 1) — i.e. the un-calibrated pipeline — for any
// degenerate coordinate, so it is always safe to apply.

import { coordQuantile } from './beta';

/** Per-coordinate affine calibration (length `dim` each), applied after rotation. */
export interface Calibration {
  /** Additive shift per coordinate. */
  shift: Float32Array;
  /** Multiplicative scale per coordinate (never 0). */
  scale: Float32Array;
}

/** Discriminated, code-tagged error for the calibration module. */
export class CalibrateError extends Error {
  readonly code: 'INVALID_LENGTH' | 'INVALID_DIM';
  constructor(code: CalibrateError['code'], message: string) {
    super(message);
    this.name = 'CalibrateError';
    this.code = code;
  }
}

/** Lower/upper sampling probabilities for the robust range match. */
const P_LO = 0.05;
const P_HI = 0.95;
/** Below this empirical range a coordinate is treated as degenerate → identity map. */
const RANGE_EPS = 1e-6;

/** Identity calibration (shift 0, scale 1) — equivalent to no calibration. */
export function identityCalibration(dim: number): Calibration {
  const shift = new Float32Array(dim); // zeros
  const scale = new Float32Array(dim).fill(1);
  return { shift, scale };
}

/**
 * Fit a {@link Calibration} from `m` rotated unit-vector samples laid out row-major
 * in `rotated` (length m·dim). For each coordinate the empirical 5th/95th percentiles
 * are mapped linearly onto the canonical coordinate marginal's 5th/95th percentiles
 * (`coordQuantile`, from the rotation's known distribution). A coordinate whose
 * empirical range is ~0 maps to the identity.
 *
 * @throws {CalibrateError} on a non-positive dim or a length that is not m·dim.
 */
export function fitCalibration(rotated: Float32Array, m: number, dim: number): Calibration {
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new CalibrateError('INVALID_DIM', `dim must be a positive integer, got ${dim}`);
  }
  if (!Number.isInteger(m) || m <= 0 || rotated.length !== m * dim) {
    throw new CalibrateError(
      'INVALID_LENGTH',
      `rotated length ${rotated.length} must equal m·dim = ${m * dim}`,
    );
  }

  // Canonical targets are symmetric about 0 and identical for every coordinate.
  const canonLo = coordQuantile(P_LO, dim);
  const canonHi = coordQuantile(P_HI, dim);
  const canonRange = canonHi - canonLo;

  const loIdx = Math.floor(P_LO * (m - 1));
  const hiIdx = Math.floor(P_HI * (m - 1));

  const shift = new Float32Array(dim);
  const scale = new Float32Array(dim);
  const column = new Float32Array(m);

  for (let d = 0; d < dim; d++) {
    for (let i = 0; i < m; i++) column[i] = rotated[i * dim + d]!;
    column.sort();
    const qLo = column[loIdx]!;
    const qHi = column[hiIdx]!;
    const empRange = qHi - qLo;
    if (empRange < RANGE_EPS) {
      shift[d] = 0;
      scale[d] = 1;
      continue;
    }
    const s = canonRange / empRange;
    scale[d] = s;
    shift[d] = canonLo / s - qLo;
  }

  return { shift, scale };
}
