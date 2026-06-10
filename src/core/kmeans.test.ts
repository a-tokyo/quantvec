import { describe, expect, it } from 'vitest';
import { kmeans, nearestCentroid, KMeansError } from './kmeans';
import type { KMeansResult } from './kmeans';
import { createRng } from './rng';

/** m·dim row-major blob data: `centers.length` tight gaussian blobs. */
function blobs(
  centers: number[][],
  perBlob: number,
  dim: number,
  seed: number,
  noise = 0.05,
): { data: Float32Array; m: number; blobOf: Int32Array } {
  const rng = createRng(seed);
  const m = centers.length * perBlob;
  const data = new Float32Array(m * dim);
  const blobOf = new Int32Array(m);
  let r = 0;
  for (let b = 0; b < centers.length; b++) {
    for (let j = 0; j < perBlob; j++, r++) {
      blobOf[r] = b;
      for (let i = 0; i < dim; i++) {
        data[r * dim + i] = centers[b]![i]! + rng.nextGaussian() * noise;
      }
    }
  }
  return { data, m, blobOf };
}

function run(
  data: Float32Array,
  m: number,
  k: number,
  dim: number,
  seed: number,
  spherical = false,
): KMeansResult {
  return kmeans(data, m, { k, dim, rng: createRng(seed), spherical });
}

describe('kmeans — validation', () => {
  const data = new Float32Array(8 * 4);

  it('rejects a bad dim', () => {
    let err: unknown;
    try {
      kmeans(data, 8, { k: 2, dim: 0, rng: createRng(1), spherical: false });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(KMeansError);
    expect((err as KMeansError).code).toBe('INVALID_DIM');
  });

  it('rejects a length that is not m·dim', () => {
    let err: unknown;
    try {
      kmeans(data, 7, { k: 2, dim: 4, rng: createRng(1), spherical: false });
    } catch (e) {
      err = e;
    }
    expect((err as KMeansError).code).toBe('INVALID_LENGTH');
  });

  it.each([1, 9, 2.5])('rejects k = %s outside [2, m] (m = 8)', (k) => {
    let err: unknown;
    try {
      kmeans(data, 8, { k, dim: 4, rng: createRng(1), spherical: false });
    } catch (e) {
      err = e;
    }
    expect((err as KMeansError).code).toBe('INVALID_K');
  });
});

describe('kmeans — clustering behavior', () => {
  it('recovers two well-separated blobs exactly', () => {
    const { data, m, blobOf } = blobs(
      [
        [10, 0, 0, 0],
        [-10, 0, 0, 0],
      ],
      50,
      4,
      11,
    );
    const { centroids, assignments } = run(data, m, 2, 4, 7);
    // Every row in the same blob lands in the same cluster, blobs in different ones.
    const clusterOfBlob = [assignments[0]!, assignments[50]!];
    expect(clusterOfBlob[0]).not.toBe(clusterOfBlob[1]);
    for (let r = 0; r < m; r++) expect(assignments[r]).toBe(clusterOfBlob[blobOf[r]!]!);
    // Centroids sit on the blob centers.
    for (const c of clusterOfBlob) {
      expect(Math.abs(Math.abs(centroids[c! * 4]!) - 10)).toBeLessThan(0.5);
    }
  });

  it('is deterministic: same data + seed yields identical centroids and assignments', () => {
    const { data, m } = blobs(
      [
        [5, 5],
        [-5, 5],
        [0, -5],
      ],
      40,
      2,
      21,
    );
    const a = run(data, m, 3, 2, 99);
    const b = run(data, m, 3, 2, 99);
    expect(Array.from(a.centroids)).toEqual(Array.from(b.centroids));
    expect(Array.from(a.assignments)).toEqual(Array.from(b.assignments));
    expect(a.iterations).toBe(b.iterations);
  });

  it('repairs empty clusters: k=3 over 2 tight blobs still yields 3 non-empty clusters', () => {
    const { data, m } = blobs(
      [
        [10, 0],
        [-10, 0],
      ],
      30,
      2,
      31,
      0.5,
    );
    const { centroids, assignments } = run(data, m, 3, 2, 13);
    const counts = [0, 0, 0];
    for (let r = 0; r < m; r++) counts[assignments[r]!]!++;
    expect(counts.every((c) => c > 0)).toBe(true);
    for (const x of centroids) expect(Number.isFinite(x)).toBe(true);
  });

  it('spherical mode returns unit-norm centroids and clusters directions', () => {
    // Two antipodal direction blobs on the unit circle (pre-normalized rows).
    const raw = blobs(
      [
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
      ],
      40,
      4,
      41,
      0.02,
    );
    for (let r = 0; r < raw.m; r++) {
      let n = 0;
      for (let i = 0; i < 4; i++) n += raw.data[r * 4 + i]! ** 2;
      const inv = 1 / Math.sqrt(n);
      for (let i = 0; i < 4; i++) raw.data[r * 4 + i] = raw.data[r * 4 + i]! * inv;
    }
    const { centroids, assignments } = run(raw.data, raw.m, 2, 4, 5, true);
    for (let c = 0; c < 2; c++) {
      let n = 0;
      for (let i = 0; i < 4; i++) n += centroids[c * 4 + i]! ** 2;
      expect(Math.sqrt(n)).toBeCloseTo(1, 6);
    }
    expect(assignments[0]).not.toBe(assignments[40]);
  });

  it('terminates within the iteration cap and reports iterations', () => {
    const { data, m } = blobs(
      [
        [3, 0],
        [-3, 0],
      ],
      25,
      2,
      51,
    );
    const res = kmeans(data, m, {
      k: 2,
      dim: 2,
      rng: createRng(3),
      spherical: false,
      maxIterations: 4,
    });
    expect(res.iterations).toBeGreaterThanOrEqual(1);
    expect(res.iterations).toBeLessThanOrEqual(4);
  });

  it('handles k === m (every point its own centroid)', () => {
    const data = Float32Array.from([0, 0, 5, 0, 0, 5, 5, 5]);
    const { assignments } = run(data, 4, 4, 2, 17);
    expect(new Set(Array.from(assignments)).size).toBe(4);
  });

  it('handles duplicate rows (zero D² mass in seeding) without NaN', () => {
    const data = new Float32Array(6 * 2).fill(1); // 6 identical points
    const { centroids } = run(data, 6, 2, 2, 23);
    for (const x of centroids) expect(Number.isFinite(x)).toBe(true);
  });
});

describe('nearestCentroid', () => {
  it('matches brute force for both affinities', () => {
    const rng = createRng(77);
    const dim = 8;
    const k = 5;
    const centroids = new Float32Array(k * dim).map(() => rng.nextGaussian());
    const v = new Float32Array(dim).map(() => rng.nextGaussian());
    for (const spherical of [false, true]) {
      let best = 0;
      let bestKey = -Infinity;
      for (let c = 0; c < k; c++) {
        let dot = 0;
        let d2 = 0;
        for (let i = 0; i < dim; i++) {
          dot += v[i]! * centroids[c * dim + i]!;
          d2 += (v[i]! - centroids[c * dim + i]!) ** 2;
        }
        const key = spherical ? dot : -d2;
        if (key > bestKey) {
          bestKey = key;
          best = c;
        }
      }
      expect(nearestCentroid(v, 0, centroids, k, dim, spherical)).toBe(best);
    }
  });
});
