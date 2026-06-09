import { describe, expect, it } from 'vitest';
import { fwht, isPow2, nextPow2 } from './fwht';

describe('isPow2 / nextPow2', () => {
  it('classifies powers of two', () => {
    for (const n of [1, 2, 4, 8, 16, 1024]) expect(isPow2(n)).toBe(true);
    for (const n of [0, 3, 6, 7, 768, 1536, -2, 2.5]) expect(isPow2(n)).toBe(false);
  });

  it('rounds up to the next power of two', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(8)).toBe(8);
    expect(nextPow2(9)).toBe(16);
    expect(nextPow2(768)).toBe(1024);
    expect(nextPow2(1536)).toBe(2048);
  });
});

describe('fwht', () => {
  it('rejects a non-power-of-two length', () => {
    expect(() => fwht(new Float64Array(3))).toThrow(RangeError);
  });

  it('maps a unit impulse to all-ones', () => {
    const a = new Float64Array([1, 0, 0, 0]);
    fwht(a);
    expect(Array.from(a)).toEqual([1, 1, 1, 1]);
  });

  it('is an involution up to the scale n (H·H = n·I)', () => {
    const a = Float64Array.from([3, -1, 4, 1, -5, 9, -2, 6]);
    const original = Array.from(a);
    fwht(a);
    fwht(a);
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(original[i]! * a.length, 9);
  });

  it('satisfies Parseval: ‖H·x‖² = n·‖x‖²', () => {
    const a = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const inSq = a.reduce((s, v) => s + v * v, 0);
    fwht(a);
    const outSq = a.reduce((s, v) => s + v * v, 0);
    expect(outSq).toBeCloseTo(a.length * inSq, 6);
  });
});
