import { describe, expect, it } from 'vitest';
import { scoreMetric } from './metrics';
import type { Distance, QueryNorms } from './metrics';

describe('scoreMetric — dot', () => {
  it('value and rankKey both equal scale·S (estimated dot product)', () => {
    const q: QueryNorms = { qNorm: 3, qNormSq: 9 };
    const r = scoreMetric('dot', 2, 1.5, 4, q);
    // scale·S = 1.5 * 2 = 3
    expect(r.value).toBeCloseTo(3, 12);
    expect(r.rankKey).toBe(r.value);
  });

  it('ranks DESCENDING (larger dot ⇒ larger key)', () => {
    const q: QueryNorms = { qNorm: 1, qNormSq: 1 };
    const a = scoreMetric('dot', 5, 1, 1, q);
    const b = scoreMetric('dot', 2, 1, 1, q);
    expect(a.rankKey).toBeGreaterThan(b.rankKey);
  });
});

describe('scoreMetric — cosine', () => {
  it('value = (scale·S)/(qNorm·norm); rankKey equals value', () => {
    const q: QueryNorms = { qNorm: 2, qNormSq: 4 };
    const r = scoreMetric('cosine', 3, 2, 5, q);
    // scale·S = 6; / (qNorm·norm)=10 → 0.6
    expect(r.value).toBeCloseTo(0.6, 12);
    expect(r.rankKey).toBe(r.value);
  });

  it('a self-aligned unit query/vector yields cosine ≈ 1', () => {
    // q and v both unit, perfectly aligned: estDot = scale·S = 1, norms = 1.
    const q: QueryNorms = { qNorm: 1, qNormSq: 1 };
    const r = scoreMetric('cosine', 1, 1, 1, q);
    expect(r.value).toBeCloseTo(1, 12);
  });

  it('ranks DESCENDING (larger cosine ⇒ larger key)', () => {
    const q: QueryNorms = { qNorm: 1, qNormSq: 1 };
    const a = scoreMetric('cosine', 0.9, 1, 1, q);
    const b = scoreMetric('cosine', 0.1, 1, 1, q);
    expect(a.rankKey).toBeGreaterThan(b.rankKey);
  });
});

describe('scoreMetric — euclidean', () => {
  it('value = qNormSq + norm² − 2·scale·S (squared distance)', () => {
    const q: QueryNorms = { qNorm: Math.sqrt(14), qNormSq: 14 };
    // pick numbers so value is exact: norm=3 → norm²=9; scale·S=2·5=10
    const r = scoreMetric('euclidean', 5, 2, 3, q);
    // 14 + 9 - 2*10 = 3
    expect(r.value).toBeCloseTo(3, 12);
  });

  it('matches the exact ‖q−v‖² identity for known vectors', () => {
    // q = [1,2,3], v = [4,0,1]. ‖q−v‖² = 9+4+4 = 17.
    // ⟨q,v⟩ = 4+0+3 = 7. ‖q‖²=14, ‖v‖²=17(=16+0+1). Wait recompute: ‖v‖²=16+0+1=17.
    // 14 + 17 - 2*7 = 17. Feed estDot=7 via scale=1, S=7; norm=√17.
    const q: QueryNorms = { qNorm: Math.sqrt(14), qNormSq: 14 };
    const norm = Math.sqrt(17);
    const r = scoreMetric('euclidean', 7, 1, norm, q);
    expect(r.value).toBeCloseTo(17, 6);
  });

  it('rankKey = −value so the max-heap selects the SMALLEST distance', () => {
    const q: QueryNorms = { qNorm: 1, qNormSq: 1 };
    const near = scoreMetric('euclidean', 0.9, 1, 1, q); // small dist²
    const far = scoreMetric('euclidean', 0.1, 1, 1, q); // larger dist²
    expect(near.value).toBeLessThan(far.value); // near is genuinely closer
    expect(near.rankKey).toBeGreaterThan(far.rankKey); // but ranks higher
    expect(near.rankKey).toBe(-near.value);
  });

  it('identical query and vector give dist² ≈ 0', () => {
    // q == v, both norm 1, aligned: estDot = 1, qNormSq = 1, norm² = 1.
    const q: QueryNorms = { qNorm: 1, qNormSq: 1 };
    const r = scoreMetric('euclidean', 1, 1, 1, q);
    expect(r.value).toBeCloseTo(0, 12);
  });
});

describe('scoreMetric — covers every Distance variant', () => {
  it('returns a finite value+rankKey for all metrics', () => {
    const q: QueryNorms = { qNorm: 2, qNormSq: 4 };
    for (const m of ['dot', 'cosine', 'euclidean'] as Distance[]) {
      const r = scoreMetric(m, 1.5, 1.2, 3, q);
      expect(Number.isFinite(r.value)).toBe(true);
      expect(Number.isFinite(r.rankKey)).toBe(true);
    }
  });
});
