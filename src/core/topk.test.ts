import { describe, expect, it } from 'vitest';
import { TopK } from './topk';
import { createRng } from './rng';

/** Brute-force reference: keep the k largest scores, sorted descending. */
function bruteForce(scores: number[], k: number): { scores: number[]; indices: number[] } {
  const pairs = scores.map((s, i) => ({ s, i }));
  // Stable-ish sort: by score desc, ties broken by index asc.
  pairs.sort((a, b) => (b.s - a.s !== 0 ? b.s - a.s : a.i - b.i));
  const top = pairs.slice(0, Math.min(k, pairs.length));
  return { scores: top.map((p) => p.s), indices: top.map((p) => p.i) };
}

describe('TopK', () => {
  it('throws a typed error on invalid k', () => {
    const cases = [0, -1, -5, 1.5, NaN, Infinity];
    for (const bad of cases) {
      let err: unknown;
      try {
        new TopK(bad);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(Error);
      expect((err as { code?: string }).code).toBe('INVALID_K');
    }
  });

  it('returns empty result when nothing added', () => {
    const t = new TopK(5);
    const r = t.result();
    expect(r.scores.length).toBe(0);
    expect(r.indices.length).toBe(0);
    expect(r.scores).toBeInstanceOf(Float32Array);
    expect(r.indices).toBeInstanceOf(Int32Array);
  });

  it('handles n < k (returns all, sorted descending)', () => {
    const t = new TopK(10);
    t.add(3, 0);
    t.add(1, 1);
    t.add(2, 2);
    const r = t.result();
    expect(Array.from(r.scores)).toEqual([3, 2, 1]);
    expect(Array.from(r.indices)).toEqual([0, 2, 1]);
  });

  it('handles k = 1 (single best)', () => {
    const t = new TopK(1);
    t.add(-5, 0);
    t.add(10, 1);
    t.add(7, 2);
    const r = t.result();
    expect(Array.from(r.scores)).toEqual([10]);
    expect(Array.from(r.indices)).toEqual([1]);
  });

  it('keeps the k largest with ties present', () => {
    const t = new TopK(3);
    // Several equal scores.
    t.add(5, 0);
    t.add(5, 1);
    t.add(5, 2);
    t.add(5, 3);
    t.add(1, 4);
    const r = t.result();
    expect(r.scores.length).toBe(3);
    // All kept scores must be 5 (the largest tier).
    expect(Array.from(r.scores)).toEqual([5, 5, 5]);
    // Indices must be a subset of the tied 5-valued entries.
    for (const idx of r.indices) {
      expect([0, 1, 2, 3]).toContain(idx);
    }
  });

  it('matches brute force on random data for various n, k', () => {
    const rng = createRng(20260608);
    const configs: Array<[number, number]> = [
      [1, 1],
      [5, 1],
      [5, 5],
      [5, 10],
      [100, 1],
      [100, 7],
      [100, 100],
      [1000, 13],
      [1000, 50],
      [37, 32],
    ];
    for (const [n, k] of configs) {
      const scores: number[] = [];
      for (let i = 0; i < n; i++) {
        // Mix of positive/negative, with occasional repeats for ties.
        scores.push(Math.round((rng.nextFloat() * 200 - 100) * 4) / 4);
      }
      const t = new TopK(k);
      for (let i = 0; i < n; i++) t.add(scores[i]!, i);
      const got = t.result();
      const want = bruteForce(scores, k);

      // The multiset of kept scores must match exactly.
      const gotScores = Array.from(got.scores);
      expect(gotScores).toEqual(want.scores);

      // Scores must be sorted descending.
      for (let i = 1; i < gotScores.length; i++) {
        expect(gotScores[i - 1]!).toBeGreaterThanOrEqual(gotScores[i]!);
      }

      // Each returned (score, index) pair must be valid: the score at that
      // original index matches.
      for (let i = 0; i < got.indices.length; i++) {
        expect(scores[got.indices[i]!]).toBe(got.scores[i]!);
      }
      // Returned indices must be unique.
      expect(new Set(Array.from(got.indices)).size).toBe(got.indices.length);
    }
  });

  it('ignores NaN scores (never leaks NaN into results)', () => {
    // NaN must not be admitted while the heap is still filling...
    const t = new TopK(3);
    t.add(NaN, 0);
    t.add(5, 1);
    t.add(NaN, 2);
    t.add(3, 3);
    t.add(4, 4);
    t.add(NaN, 5);
    const r = t.result();
    expect(Array.from(r.scores)).toEqual([5, 4, 3]);
    expect(Array.from(r.indices)).toEqual([1, 4, 3]);
    for (const s of r.scores) expect(Number.isNaN(s)).toBe(false);

    // ...nor once the heap is full (the old `score <= root` test was false for
    // NaN, so NaN slipped past the eviction check).
    const t2 = new TopK(2);
    t2.add(10, 0);
    t2.add(20, 1);
    t2.add(NaN, 2);
    const r2 = t2.result();
    expect(Array.from(r2.scores)).toEqual([20, 10]);
    expect(Array.from(r2.indices)).toEqual([1, 0]);

    // All-NaN input yields an empty result, not a NaN-filled one.
    const t3 = new TopK(3);
    for (let i = 0; i < 5; i++) t3.add(NaN, i);
    expect(t3.result().scores.length).toBe(0);
  });

  it('result() is repeatable and non-destructive', () => {
    const t = new TopK(3);
    for (let i = 0; i < 20; i++) t.add(i, i);
    const a = t.result();
    const b = t.result();
    expect(Array.from(a.scores)).toEqual(Array.from(b.scores));
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
    expect(Array.from(a.scores)).toEqual([19, 18, 17]);
  });
});
