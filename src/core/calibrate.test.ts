import { describe, expect, it } from 'vitest';
import { fitCalibration, identityCalibration } from './calibrate';
import type { CalibrateError } from './calibrate';
import { coordQuantile } from './beta';

function catchError<T>(fn: () => T): unknown {
  try {
    fn();
  } catch (e) {
    return e;
  }
  return undefined;
}

describe('identityCalibration', () => {
  it('is all-zero shift and all-one scale', () => {
    const c = identityCalibration(8);
    expect(Array.from(c.shift)).toEqual(new Array(8).fill(0));
    expect(Array.from(c.scale)).toEqual(new Array(8).fill(1));
  });
});

describe('fitCalibration — validation', () => {
  it('rejects a bad dim', () => {
    expect(
      (catchError(() => fitCalibration(new Float32Array(0), 1, 0)) as CalibrateError).code,
    ).toBe('INVALID_DIM');
  });

  it('rejects a length that is not m·dim', () => {
    expect(
      (catchError(() => fitCalibration(new Float32Array(7), 2, 4)) as CalibrateError).code,
    ).toBe('INVALID_LENGTH');
  });
});

describe('fitCalibration — affine map', () => {
  const M = 1000;
  const DIM = 4;

  /** Build samples: coord 0 = linspace(-1,1) (wide), coord 1 = constant (degenerate). */
  function makeRotated(): Float32Array {
    const r = new Float32Array(M * DIM);
    for (let i = 0; i < M; i++) {
      r[i * DIM + 0] = -1 + (2 * i) / (M - 1); // wide, already sorted
      r[i * DIM + 1] = 0.25; // constant → degenerate
      r[i * DIM + 2] = -0.1 + (0.2 * i) / (M - 1); // narrow
      r[i * DIM + 3] = ((i * 2654435761) % 1000) / 1000 - 0.5; // pseudo-random spread
    }
    return r;
  }

  it('maps each coordinate’s empirical 5/95 percentile onto the canonical marginal', () => {
    const r = makeRotated();
    const c = fitCalibration(r, M, DIM);
    const canonLo = coordQuantile(0.05, DIM);
    const canonHi = coordQuantile(0.95, DIM);

    const loIdx = Math.floor(0.05 * (M - 1));
    const hiIdx = Math.floor(0.95 * (M - 1));
    // Reconstruct coord 0's percentiles (it's already sorted ascending).
    const qLo0 = -1 + (2 * loIdx) / (M - 1);
    const qHi0 = -1 + (2 * hiIdx) / (M - 1);

    // (q + shift) * scale should land on the canonical percentiles.
    expect((qLo0 + c.shift[0]!) * c.scale[0]!).toBeCloseTo(canonLo, 4);
    expect((qHi0 + c.shift[0]!) * c.scale[0]!).toBeCloseTo(canonHi, 4);
  });

  it('leaves a degenerate (constant) coordinate as the identity', () => {
    const c = fitCalibration(makeRotated(), M, DIM);
    expect(c.shift[1]).toBe(0);
    expect(c.scale[1]).toBe(1);
  });

  it('expands a narrow coordinate (scale > 1) and shrinks a wide one (scale < 1)', () => {
    const c = fitCalibration(makeRotated(), M, DIM);
    expect(c.scale[2]).toBeGreaterThan(1); // narrow [-0.1,0.1] → widened to canonical
    expect(c.scale[0]).toBeLessThan(1); // wide [-1,1] → shrunk to canonical
  });

  it('is deterministic', () => {
    const a = fitCalibration(makeRotated(), M, DIM);
    const b = fitCalibration(makeRotated(), M, DIM);
    expect(Array.from(a.scale)).toEqual(Array.from(b.scale));
    expect(Array.from(a.shift)).toEqual(Array.from(b.shift));
  });
});
