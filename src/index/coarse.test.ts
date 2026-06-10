import { describe, expect, it } from 'vitest';
import { CoarseQuantizer, defaultNprobe, IVF_TRAIN_SAMPLE_PER_LIST } from './coarse';
import { createRng } from '../core/rng';

const DIM = 8;

function gaussianVecs(n: number, seed: number, dim = DIM): Float32Array[] {
  const rng = createRng(seed);
  return Array.from({ length: n }, () => {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = rng.nextGaussian();
    return v;
  });
}

/** Assert the slot-membership invariants over every live slot. */
function assertInvariants(cq: CoarseQuantizer, liveSlots: number): void {
  const snapshot = cq.listForSlotSnapshot(liveSlots);
  // probe with nprobe = nlist returns every live slot exactly once.
  const probeQuery = new Float32Array(DIM).fill(1);
  const all = cq.probe(probeQuery, cq.nlist);
  expect(all.length).toBe(liveSlots);
  expect(new Set(Array.from(all)).size).toBe(liveSlots);
  for (const s of all) {
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThan(liveSlots);
    expect(snapshot[s]).toBeGreaterThanOrEqual(0);
    expect(snapshot[s]).toBeLessThan(cq.nlist);
  }
}

describe('defaultNprobe', () => {
  it('is 1/8 of nlist, at least 1', () => {
    expect(defaultNprobe(1)).toBe(1);
    expect(defaultNprobe(8)).toBe(1);
    expect(defaultNprobe(64)).toBe(8);
    expect(defaultNprobe(100)).toBe(13);
  });
});

describe('CoarseQuantizer — training', () => {
  it('is deterministic for the same vectors and seed', () => {
    const vecs = gaussianVecs(80, 1);
    const a = CoarseQuantizer.train(vecs, 4, 1, 'cosine', DIM, 42);
    const b = CoarseQuantizer.train(vecs, 4, 1, 'cosine', DIM, 42);
    expect(Array.from(a.centroids)).toEqual(Array.from(b.centroids));
  });

  it('spherical metrics produce unit-norm centroids; euclidean does not renormalize', () => {
    const vecs = gaussianVecs(60, 2).map((v) => {
      for (let i = 0; i < DIM; i++) v[i] = v[i]! * 3 + 5; // off-origin, non-unit
      return v;
    });
    const sph = CoarseQuantizer.train(vecs, 3, 1, 'dot', DIM, 7);
    for (let c = 0; c < 3; c++) {
      let n = 0;
      for (let i = 0; i < DIM; i++) n += sph.centroids[c * DIM + i]! ** 2;
      expect(Math.sqrt(n)).toBeCloseTo(1, 5);
    }
    const l2 = CoarseQuantizer.train(vecs, 3, 1, 'euclidean', DIM, 7);
    let off = 0;
    for (let c = 0; c < 3; c++) {
      let n = 0;
      for (let i = 0; i < DIM; i++) n += l2.centroids[c * DIM + i]! ** 2;
      if (Math.abs(Math.sqrt(n) - 1) > 0.1) off++;
    }
    expect(off).toBeGreaterThan(0); // means sit near the data, far from unit norm
  });

  it('caps the training sample (IVF_TRAIN_SAMPLE_PER_LIST) without error on big batches', () => {
    const nlist = 2;
    const vecs = gaussianVecs(IVF_TRAIN_SAMPLE_PER_LIST * nlist + 50, 3);
    const cq = CoarseQuantizer.train(vecs, nlist, 1, 'cosine', DIM, 5);
    expect(cq.centroids.length).toBe(nlist * DIM);
  });

  it('skips zero rows in training', () => {
    const vecs = gaussianVecs(40, 4);
    vecs[3] = new Float32Array(DIM); // zero row
    const cq = CoarseQuantizer.train(vecs, 4, 1, 'cosine', DIM, 9);
    for (const x of cq.centroids) expect(Number.isFinite(x)).toBe(true);
  });
});

describe('CoarseQuantizer — assignment & probing', () => {
  it('assign matches brute-force nearest centroid for both affinities', () => {
    for (const metric of ['cosine', 'euclidean'] as const) {
      const vecs = gaussianVecs(64, 5);
      const cq = CoarseQuantizer.train(vecs, 8, 2, metric, DIM, 11);
      const rng = createRng(6);
      for (let t = 0; t < 20; t++) {
        const v = new Float32Array(DIM);
        for (let i = 0; i < DIM; i++) v[i] = rng.nextGaussian();
        let best = 0;
        let bestKey = -Infinity;
        for (let c = 0; c < 8; c++) {
          let dot = 0;
          let d2 = 0;
          for (let i = 0; i < DIM; i++) {
            dot += v[i]! * cq.centroids[c * DIM + i]!;
            d2 += (v[i]! - cq.centroids[c * DIM + i]!) ** 2;
          }
          const key = metric === 'euclidean' ? -d2 : dot;
          if (key > bestKey) {
            bestKey = key;
            best = c;
          }
        }
        expect(cq.assign(v)).toBe(best);
      }
    }
  });

  it('probe(nprobe = nlist) returns all live slots; smaller nprobe returns a subset', () => {
    const vecs = gaussianVecs(100, 7);
    const cq = CoarseQuantizer.train(vecs, 8, 2, 'cosine', DIM, 13);
    for (let s = 0; s < vecs.length; s++) cq.addSlot(s, vecs[s]!);
    const all = cq.probe(vecs[0]!, 8);
    expect(all.length).toBe(100);
    const some = cq.probe(vecs[0]!, 2);
    expect(some.length).toBeLessThan(100);
    const allSet = new Set(Array.from(all));
    for (const s of some) expect(allSet.has(s)).toBe(true);
  });
});

describe('CoarseQuantizer — posting-list bookkeeping', () => {
  function build(n: number, seed: number): { cq: CoarseQuantizer; vecs: Float32Array[] } {
    const vecs = gaussianVecs(n, seed);
    const cq = CoarseQuantizer.train(vecs, 4, 1, 'cosine', DIM, 17);
    for (let s = 0; s < n; s++) cq.addSlot(s, vecs[s]!);
    return { cq, vecs };
  }

  it('swapRemove of the last slot (i === last)', () => {
    const { cq } = build(10, 8);
    cq.swapRemove(9, 9);
    assertInvariants(cq, 9);
  });

  it('swapRemove of an interior slot renumbers the moved last slot', () => {
    const { cq } = build(10, 9);
    cq.swapRemove(3, 9);
    assertInvariants(cq, 9);
  });

  it('invariant fuzz: random interleaved adds and removes', () => {
    const rng = createRng(99);
    const vecs = gaussianVecs(400, 10);
    const cq = CoarseQuantizer.train(vecs.slice(0, 50), 4, 1, 'cosine', DIM, 19);
    let n = 0;
    let next = 0;
    for (let op = 0; op < 300; op++) {
      if (n === 0 || (rng.nextFloat() < 0.6 && next < vecs.length)) {
        cq.addSlot(n, vecs[next]!);
        next++;
        n++;
      } else {
        const i = Math.min(n - 1, Math.floor(rng.nextFloat() * n));
        cq.swapRemove(i, n - 1);
        n--;
      }
    }
    assertInvariants(cq, n);
  });

  it('clear empties postings but keeps centroids; adds work after clear', () => {
    const { cq, vecs } = build(20, 11);
    const before = Array.from(cq.centroids);
    cq.clear();
    expect(cq.probe(vecs[0]!, 4).length).toBe(0);
    expect(Array.from(cq.centroids)).toEqual(before);
    cq.addSlot(0, vecs[5]!);
    assertInvariants(cq, 1);
  });

  it('fromState round-trips the snapshot', () => {
    const { cq } = build(30, 12);
    const snapshot = cq.listForSlotSnapshot(30);
    const restored = CoarseQuantizer.fromState(cq.centroids, snapshot, 4, 1, 'cosine', DIM);
    expect(Array.from(restored.listForSlotSnapshot(30))).toEqual(Array.from(snapshot));
    assertInvariants(restored, 30);
    // Mutations on the restored quantizer keep the invariants.
    restored.swapRemove(2, 29);
    assertInvariants(restored, 29);
  });
});
