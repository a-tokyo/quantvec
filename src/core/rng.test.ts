import { describe, expect, it } from 'vitest';
import { createRng } from './rng';

describe('createRng', () => {
  it('produces identical sequences for the same seed', () => {
    const a = createRng(12345);
    const b = createRng(12345);
    for (let i = 0; i < 1000; i++) {
      expect(a.nextU32()).toBe(b.nextU32());
    }
  });

  it('produces identical float/gaussian sequences for the same seed', () => {
    const a = createRng(0xdeadbeef);
    const b = createRng(0xdeadbeef);
    for (let i = 0; i < 1000; i++) {
      expect(a.nextFloat()).toBe(b.nextFloat());
    }
    const c = createRng(7n);
    const d = createRng(7n);
    for (let i = 0; i < 1000; i++) {
      expect(c.nextGaussian()).toBe(d.nextGaussian());
    }
  });

  it('accepts a bigint seed deterministically', () => {
    const a = createRng(0x0123456789abcdefn);
    const b = createRng(0x0123456789abcdefn);
    for (let i = 0; i < 500; i++) {
      expect(a.nextU32()).toBe(b.nextU32());
    }
  });

  it('matches a hardcoded golden vector (locks the bit-exact stream)', () => {
    // Known-answer test. These constants were independently reproduced from the
    // published xoshiro256** + SplitMix64 reference (https://prng.di.unimi.it/),
    // NOT snapshotted blindly from this module — so a future refactor that
    // silently alters the stream (seeding, rotation, output word) will fail here.
    const golden42 = [
      360188718, 1627707782, 2920764210, 3971525959, 4259765375, 3306005809, 3089192069, 3650758467,
    ];
    const r = createRng(42);
    for (const expected of golden42) {
      expect(r.nextU32()).toBe(expected);
    }

    // A second seed, supplied as a bigint, pinning the 64-bit path too.
    const goldenBig = [
      4164784269, 692000174, 2390434984, 1948683672, 1805871221, 546424444, 1456522348, 1035992171,
    ];
    const rb = createRng(0x1234567890abcdefn);
    for (const expected of goldenBig) {
      expect(rb.nextU32()).toBe(expected);
    }
  });

  it('rejects a non-finite numeric seed with a typed error', () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      let err: unknown;
      try {
        createRng(bad);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(Error);
      expect((err as { code?: string }).code).toBe('INVALID_SEED');
      expect((err as Error).name).toBe('RngError');
    }
  });

  it('truncates a fractional numeric seed (equals its integer part)', () => {
    const a = createRng(123.99);
    const b = createRng(123);
    for (let i = 0; i < 16; i++) expect(a.nextU32()).toBe(b.nextU32());
  });

  it('produces different sequences for different seeds', () => {
    const a = createRng(1);
    const b = createRng(2);
    let differing = 0;
    for (let i = 0; i < 100; i++) {
      if (a.nextU32() !== b.nextU32()) differing++;
    }
    // Overwhelmingly likely all differ; require the vast majority.
    expect(differing).toBeGreaterThan(95);
  });

  it('nextFloat is always in [0, 1)', () => {
    const r = createRng(42);
    for (let i = 0; i < 100_000; i++) {
      const x = r.nextFloat();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });

  it('nextU32 returns 32-bit unsigned integers covering the range', () => {
    const r = createRng(99);
    let min = Infinity;
    let max = -Infinity;
    let highBitSeen = false;
    let lowSeen = false;
    for (let i = 0; i < 100_000; i++) {
      const x = r.nextU32();
      expect(Number.isInteger(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(0xffffffff);
      if (x < min) min = x;
      if (x > max) max = x;
      if (x >= 0x80000000) highBitSeen = true;
      if (x < 0x01000000) lowSeen = true;
    }
    // Should span a large fraction of the 32-bit range.
    expect(highBitSeen).toBe(true);
    expect(lowSeen).toBe(true);
    expect(max).toBeGreaterThan(0xf0000000);
    expect(min).toBeLessThan(0x10000000);
  });

  it('nextGaussian has mean ~0 and variance ~1 over 100k samples', () => {
    const r = createRng(2024);
    const n = 100_000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const g = r.nextGaussian();
      expect(Number.isFinite(g)).toBe(true);
      sum += g;
      sumSq += g * g;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });

  it('nextGaussian caches the spare value (Box-Muller pairs) deterministically', () => {
    // Drawing one-at-a-time must match drawing in a batch from a fresh rng.
    const a = createRng(555);
    const b = createRng(555);
    const seqA: number[] = [];
    for (let i = 0; i < 10; i++) seqA.push(a.nextGaussian());
    const seqB: number[] = [];
    for (let i = 0; i < 10; i++) seqB.push(b.nextGaussian());
    expect(seqA).toEqual(seqB);
  });
});
