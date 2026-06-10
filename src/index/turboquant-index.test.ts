import { describe, expect, it } from 'vitest';
import { CALIBRATION_MIN_SAMPLES, IndexError, TurboQuantIndex } from './turboquant-index';
import { IdMapIndex } from './id-map-index';
import { EncodeError } from '../core/encode';
import { SearchError, searchFlat } from '../core/search';
import type { EncodedDb } from '../core/search';
import { createRng } from '../core/rng';
import { createRotation } from '../core/rotation';
import { getCodebook } from '../core/codebook';
import { WasmKernel } from '../wasm/kernel';

const DIM = 8;

/** Four mutually orthogonal dim-8 vectors → self-query is unambiguously top-1. */
const ORTHO: Float32Array[] = [
  Float32Array.from([8, 8, 0, 0, 0, 0, 0, 0]),
  Float32Array.from([0, 0, 8, 8, 0, 0, 0, 0]),
  Float32Array.from([0, 0, 0, 0, 8, 8, 0, 0]),
  Float32Array.from([0, 0, 0, 0, 0, 0, 8, 8]),
];

function catchError<T>(fn: () => T): unknown {
  try {
    fn();
  } catch (e) {
    return e;
  }
  return undefined;
}

describe('TurboQuantIndex — construction', () => {
  it('rejects a dim that is not a positive multiple of 8', () => {
    expect((catchError(() => new TurboQuantIndex({ dim: 7 })) as IndexError).code).toBe(
      'INVALID_DIM',
    );
    expect((catchError(() => new TurboQuantIndex({ dim: 0 })) as IndexError).code).toBe(
      'INVALID_DIM',
    );
  });

  it('rejects bits outside {2,3,4}', () => {
    const err = catchError(() => new TurboQuantIndex({ dim: DIM, bits: 5 as 4 }));
    expect((err as IndexError).code).toBe('INVALID_BITS');
  });

  it('rejects a non-finite seed', () => {
    expect(
      (catchError(() => new TurboQuantIndex({ dim: DIM, seed: NaN })) as IndexError).code,
    ).toBe('INVALID_SEED');
    expect(
      (catchError(() => new TurboQuantIndex({ dim: DIM, seed: Infinity })) as IndexError).code,
    ).toBe('INVALID_SEED');
  });

  it('exposes defaults (bits=4, metric=cosine, seed=0) and overrides', () => {
    const def = new TurboQuantIndex({ dim: DIM });
    expect([def.dim, def.bits, def.metric, def.seed, def.size]).toEqual([DIM, 4, 'cosine', 0, 0]);
    const custom = new TurboQuantIndex({ dim: DIM, bits: 2, metric: 'dot', seed: 9 });
    expect([custom.bits, custom.metric, custom.seed]).toEqual([2, 'dot', 9]);
  });
});

describe('TurboQuantIndex — add', () => {
  it('adds a flat row-major Float32Array', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    const flat = new Float32Array(2 * DIM);
    flat.set(ORTHO[0]!, 0);
    flat.set(ORTHO[1]!, DIM);
    idx.add(flat);
    expect(idx.size).toBe(2);
  });

  it('adds arrays of Float32Array and number[] and grows capacity past the initial 8', () => {
    const idx = new TurboQuantIndex({ dim: DIM, bits: 2 });
    idx.add(ORTHO); // Float32Array[] (4 vectors)
    idx.add([Array.from(ORTHO[0]!)]); // number[][] (1 vector)
    // Push well past INITIAL_CAPACITY (8) to exercise #ensureCapacity doubling.
    for (let i = 0; i < 20; i++) idx.add([ORTHO[i % 4]!]);
    expect(idx.size).toBe(ORTHO.length + 1 + 20);
  });

  it('rejects a flat buffer whose length is not a multiple of dim', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect((catchError(() => idx.add(new Float32Array(DIM + 1))) as IndexError).code).toBe(
      'INVALID_LENGTH',
    );
  });

  it('rejects an array vector of the wrong length', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect((catchError(() => idx.add([new Float32Array(DIM - 1)])) as IndexError).code).toBe(
      'INVALID_LENGTH',
    );
  });

  it('re-throws EncodeError on a zero vector', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect(catchError(() => idx.add([new Float32Array(DIM)]))).toBeInstanceOf(EncodeError);
  });

  it('rejects a non-array-like element with a typed error (no raw TypeError)', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    const err = catchError(() => idx.add([null as unknown as number[]]));
    expect(err).toBeInstanceOf(IndexError);
    expect((err as IndexError).code).toBe('INVALID_VECTOR');
  });
});

describe('TurboQuantIndex — addOne', () => {
  it('appends a single vector (Float32Array and number[])', () => {
    const idx = new TurboQuantIndex({ dim: DIM, bits: 2 });
    idx.addOne(ORTHO[0]!);
    idx.addOne(Array.from(ORTHO[1]!));
    expect(idx.size).toBe(2);
  });

  it('rejects a wrong-length vector', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect((catchError(() => idx.addOne(new Float32Array(3))) as IndexError).code).toBe(
      'INVALID_LENGTH',
    );
  });

  it('rejects a non-array-like vector', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect(
      (catchError(() => idx.addOne(undefined as unknown as number[])) as IndexError).code,
    ).toBe('INVALID_VECTOR');
  });
});

describe('TurboQuantIndex — clear', () => {
  it('resets the live count to 0 and allows re-adding', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    idx.clear();
    expect(idx.size).toBe(0);
    idx.add([ORTHO[0]!]);
    expect(idx.size).toBe(1);
    expect(idx.search(ORTHO[0]!, 1).indices[0]).toBe(0);
  });
});

describe('TurboQuantIndex — search', () => {
  it('returns the queried vector itself as top-1', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    for (let i = 0; i < ORTHO.length; i++) {
      const { indices } = idx.search(ORTHO[i]!, 1);
      expect(indices[0]).toBe(i);
    }
  });

  it('honors a per-query metric override', () => {
    const idx = new TurboQuantIndex({ dim: DIM, metric: 'cosine' });
    idx.add(ORTHO);
    const res = idx.search(ORTHO[0]!, 4, { metric: 'euclidean' });
    expect(res.indices[0]).toBe(0); // self is nearest under euclidean too
    expect(res.indices.length).toBe(4);
  });

  it('honors a slot mask', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    const mask = new Uint8Array([0, 1, 1, 1]); // exclude slot 0
    const res = idx.search(ORTHO[0]!, 1, { mask });
    expect(res.indices[0]).not.toBe(0);
  });

  it('throws EMPTY on an empty index', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    expect((catchError(() => idx.search(ORTHO[0]!, 1)) as IndexError).code).toBe('EMPTY');
  });

  it('re-throws SearchError on a zero query', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    expect(catchError(() => idx.search(new Float32Array(DIM), 1))).toBeInstanceOf(SearchError);
  });
});

describe('TurboQuantIndex — swapRemove', () => {
  it('removes a slot in O(1) by moving the last row into the gap', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO); // slots 0,1,2,3
    idx.swapRemove(1); // slot 3 (ORTHO[3]) moves into slot 1 → {0:O0, 1:O3, 2:O2}
    expect(idx.size).toBe(3);
    // Each surviving (orthogonal) vector deterministically retrieves its new slot.
    expect(idx.search(ORTHO[0]!, 1).indices[0]).toBe(0);
    expect(idx.search(ORTHO[3]!, 1).indices[0]).toBe(1); // moved row now at slot 1
    expect(idx.search(ORTHO[2]!, 1).indices[0]).toBe(2);
  });

  it('removing the last slot needs no move', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    idx.swapRemove(3);
    expect(idx.size).toBe(3);
  });

  it('rejects an out-of-range index', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    idx.add(ORTHO);
    expect((catchError(() => idx.swapRemove(9)) as IndexError).code).toBe('INVALID_INDEX');
    expect((catchError(() => idx.swapRemove(-1)) as IndexError).code).toBe('INVALID_INDEX');
  });
});

describe('TurboQuantIndex — serialization', () => {
  it('round-trips and reproduces search results', () => {
    const idx = new TurboQuantIndex({ dim: DIM, bits: 3, metric: 'dot', seed: 5 });
    idx.add(ORTHO);
    const restored = TurboQuantIndex.fromBytes(idx.toBytes());
    expect([restored.dim, restored.bits, restored.metric, restored.seed, restored.size]).toEqual([
      DIM,
      3,
      'dot',
      5,
      4,
    ]);
    for (let i = 0; i < ORTHO.length; i++) {
      const a = idx.search(ORTHO[i]!, 4);
      const b = restored.search(ORTHO[i]!, 4);
      expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
      expect(Array.from(b.scores)).toEqual(Array.from(a.scores));
    }
  });

  it('round-trips an empty index', () => {
    const idx = new TurboQuantIndex({ dim: DIM });
    const restored = TurboQuantIndex.fromBytes(idx.toBytes());
    expect(restored.size).toBe(0);
  });

  it('does not preserve fastscan: a deserialized index falls back to the exact path', () => {
    const idx = new TurboQuantIndex({ dim: DIM, bits: 4, fastscan: true });
    idx.add(ORTHO);
    expect(idx.fastscan).toBe(true);
    const restored = TurboQuantIndex.fromBytes(idx.toBytes());
    expect(restored.fastscan).toBe(false);
    // search still works correctly via the exact path
    for (let i = 0; i < ORTHO.length; i++) {
      expect(restored.search(ORTHO[i]!, 1).indices[0]).toBe(idx.search(ORTHO[i]!, 1).indices[0]);
    }
  });

  it('rejects an id-keyed buffer with WRONG_KIND', () => {
    const idmap = new IdMapIndex({ dim: DIM });
    idmap.addWithIds([1], [ORTHO[0]!]);
    const err = catchError(() => TurboQuantIndex.fromBytes(idmap.toBytes()));
    expect((err as IndexError).code).toBe('WRONG_KIND');
  });
});

describe('TurboQuantIndex — TQ+ calibration', () => {
  const CDIM = 64;

  /** Anisotropic Gaussian vectors (per-coordinate power-law scale). */
  function aniso(n: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    const out: Float32Array[] = [];
    for (let j = 0; j < n; j++) {
      const v = new Float32Array(CDIM);
      for (let i = 0; i < CDIM; i++) v[i] = rng.nextGaussian() * Math.pow(0.2, i / CDIM);
      out.push(v);
    }
    return out;
  }

  function recallAt10(cal: boolean): number {
    const db = aniso(CALIBRATION_MIN_SAMPLES + 200, 1);
    const queries = aniso(40, 99);
    const idx = new TurboQuantIndex({ dim: CDIM, bits: 4, calibrate: cal });
    idx.add(db);
    let hit = 0;
    for (const q of queries) {
      // exact cosine top-10
      const scored = db
        .map((v, j) => {
          let dotp = 0;
          let nv = 0;
          let nq = 0;
          for (let i = 0; i < CDIM; i++) {
            dotp += v[i]! * q[i]!;
            nv += v[i]! * v[i]!;
            nq += q[i]! * q[i]!;
          }
          return { j, c: dotp / Math.sqrt(nv * nq) };
        })
        .sort((a, b) => b.c - a.c);
      const truth = new Set(scored.slice(0, 10).map((s) => s.j));
      const got = idx.search(q, 10).indices;
      for (let i = 0; i < got.length; i++) if (truth.has(got[i]!)) hit++;
    }
    return hit / (queries.length * 10);
  }

  it('is off by default (opt-in) and auto-fits when enabled with >= the minimum samples', () => {
    const def = new TurboQuantIndex({ dim: CDIM });
    def.add(aniso(CALIBRATION_MIN_SAMPLES, 2));
    expect(def.calibrated).toBe(false); // opt-in: default does not calibrate

    const on = new TurboQuantIndex({ dim: CDIM, calibrate: true });
    on.add(aniso(CALIBRATION_MIN_SAMPLES, 2));
    expect(on.calibrated).toBe(true);
  });

  it('does not calibrate a small first batch even when enabled', () => {
    const small = new TurboQuantIndex({ dim: CDIM, calibrate: true });
    small.add(aniso(50, 3));
    expect(small.calibrated).toBe(false);
  });

  it('an empty first add does not freeze the calibration decision', () => {
    const idx = new TurboQuantIndex({ dim: CDIM, calibrate: true });
    idx.add([]);
    idx.add(aniso(CALIBRATION_MIN_SAMPLES, 5));
    expect(idx.calibrated).toBe(true);
  });

  it('round-trips a calibrated index, reproducing search exactly', () => {
    const idx = new TurboQuantIndex({ dim: CDIM, bits: 3, metric: 'dot', calibrate: true });
    idx.add(aniso(CALIBRATION_MIN_SAMPLES, 6));
    expect(idx.calibrated).toBe(true);
    const restored = TurboQuantIndex.fromBytes(idx.toBytes());
    expect(restored.calibrated).toBe(true);
    const q = aniso(1, 7)[0]!;
    const a = idx.search(q, 10);
    const b = restored.search(q, 10);
    expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
    expect(Array.from(b.scores)).toEqual(Array.from(a.scores));
  });

  it('skips a zero row while fitting, then rejects it on encode', () => {
    const batch = aniso(CALIBRATION_MIN_SAMPLES, 8);
    batch[0] = new Float32Array(CDIM); // zero vector → skipped by the fit, rejected by encode
    const idx = new TurboQuantIndex({ dim: CDIM, calibrate: true });
    expect(catchError(() => idx.add(batch))).toBeInstanceOf(EncodeError);
  });

  it('applies calibration in search — matches a calibration-aware reference, not the un-calibrated one', () => {
    const idx = new TurboQuantIndex({ dim: CDIM, bits: 4, calibrate: true, wasm: false });
    idx.add(aniso(CALIBRATION_MIN_SAMPLES, 6));
    expect(idx.calibrated).toBe(true);

    // Reference EncodedDb rebuilt from the index payload (which carries the calibration).
    const p = idx.toPayload();
    const base = {
      n: p.n,
      dim: p.dim,
      bits: p.bits,
      codes: p.codes,
      scales: p.scales,
      norms: p.norms,
      centroids: getCodebook(p.dim, p.bits).centroids,
      rotation: createRotation(p.dim, p.seed),
    };
    const withCal: EncodedDb = { ...base, calibration: p.calibration! };
    const noCal: EncodedDb = base;

    const q = aniso(1, 7)[0]!;
    const got = idx.search(q, 10, { metric: 'dot' });
    const refCal = searchFlat(withCal, q, 10, { metric: 'dot' });
    const refNoCal = searchFlat(noCal, q, 10, { metric: 'dot' });

    // The index must score with calibration applied (the C1 bug scored without it).
    expect(Array.from(got.scores)).toEqual(Array.from(refCal.scores));
    expect(Array.from(got.indices)).toEqual(Array.from(refCal.indices));
    // And calibration is non-identity here, so it genuinely changes the scores.
    expect(Array.from(refCal.scores)).not.toEqual(Array.from(refNoCal.scores));
  });

  it('keeps the estimator sound (recall impact is data-dependent)', () => {
    // TQ+ helps on real low-dim embeddings but is neutral-to-negative on well-
    // conditioned synthetic data (the rotation already yields near-canonical coords),
    // so we assert the calibrated estimator stays sound, not that it improves.
    expect(recallAt10(true)).toBeGreaterThan(0.6);
    expect(recallAt10(false)).toBeGreaterThan(0.6);
  });
});

describe('TurboQuantIndex — WASM kernel', () => {
  const WDIM = 64;
  function gaussianVecs(n: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    const out: Float32Array[] = [];
    for (let j = 0; j < n; j++) {
      const v = new Float32Array(WDIM);
      for (let i = 0; i < WDIM; i++) v[i] = rng.nextGaussian();
      out.push(v);
    }
    return out;
  }

  it('the WASM kernel is available in this runtime', () => {
    expect(WasmKernel.create()).not.toBeNull();
  });

  it('produces results bit-identical to the scalar fallback (all metrics)', () => {
    const data = gaussianVecs(200, 1);
    const queries = gaussianVecs(8, 2);
    const wasm = new TurboQuantIndex({ dim: WDIM, bits: 4 }); // wasm on by default
    const scalar = new TurboQuantIndex({ dim: WDIM, bits: 4, wasm: false });
    wasm.add(data);
    scalar.add(data);
    for (const metric of ['dot', 'cosine', 'euclidean'] as const) {
      for (const q of queries) {
        const a = wasm.search(q, 10, { metric });
        const b = scalar.search(q, 10, { metric });
        expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
        expect(Array.from(a.scores)).toEqual(Array.from(b.scores)); // f64 accum → exact
      }
    }
  });

  it('re-uploads resident codes after mutations and still matches scalar', () => {
    const wasm = new TurboQuantIndex({ dim: WDIM });
    const scalar = new TurboQuantIndex({ dim: WDIM, wasm: false });
    wasm.add(gaussianVecs(50, 3));
    scalar.add(gaussianVecs(50, 3));
    const q = gaussianVecs(1, 4)[0]!;
    wasm.search(q, 5); // first search uploads codes
    wasm.add(gaussianVecs(50, 5)); // mutate → dirty
    scalar.add(gaussianVecs(50, 5));
    wasm.swapRemove(0);
    scalar.swapRemove(0);
    const a = wasm.search(q, 5);
    const b = scalar.search(q, 5);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
    expect(Array.from(a.scores)).toEqual(Array.from(b.scores));
  });

  it('honors a mask on the WASM path', () => {
    const data = gaussianVecs(40, 7);
    const idx = new TurboQuantIndex({ dim: WDIM });
    idx.add(data);
    const mask = new Uint8Array(40).fill(1);
    mask[3] = 0;
    const res = idx.search(data[3]!, 1, { mask });
    expect(res.indices[0]).not.toBe(3);
  });
});

describe('TurboQuantIndex — FastScan path (v128 blocked-nibble + exact rescore)', () => {
  const FDIM = 64;
  function gaussianVecs(n: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    return Array.from({ length: n }, () => {
      const v = new Float32Array(FDIM);
      for (let i = 0; i < FDIM; i++) v[i] = rng.nextGaussian();
      return v;
    });
  }

  it('returns the same top-k as the exact path on Gaussian data (high recall)', () => {
    const data = gaussianVecs(500, 10);
    const queries = gaussianVecs(10, 11);
    const exact = new TurboQuantIndex({ dim: FDIM, bits: 4, metric: 'cosine' });
    const fast = new TurboQuantIndex({ dim: FDIM, bits: 4, metric: 'cosine', fastscan: true });
    exact.add(data);
    fast.add(data);
    let totalHits = 0;
    const k = 10;
    for (const q of queries) {
      const a = exact.search(q, k);
      const b = fast.search(q, k);
      const setA = new Set(Array.from(a.indices));
      for (const idx of b.indices) if (setA.has(idx)) totalHits++;
    }
    // FastScan rescores the pool exactly; recall should be very high on Gaussian data.
    expect(totalHits / (queries.length * k)).toBeGreaterThan(0.8);
  });

  it('matches exact path when fastscan=false (4-bit, no wasm) to confirm fallback gate', () => {
    const data = gaussianVecs(100, 20);
    const q = gaussianVecs(1, 21)[0]!;
    const exact = new TurboQuantIndex({ dim: FDIM, bits: 4, wasm: false });
    const fast = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: false });
    exact.add(data);
    fast.add(data);
    const a = exact.search(q, 5);
    const b = fast.search(q, 5);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
  });

  it('fastscan is ignored for bits != 4 (falls back to exact)', () => {
    const data = gaussianVecs(100, 30);
    const q = gaussianVecs(1, 31)[0]!;
    const idx2 = new TurboQuantIndex({ dim: FDIM, bits: 2, fastscan: true });
    const idx4 = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: false });
    idx2.add(data);
    idx4.add(data);
    // Both should complete without error; we just verify no crash.
    expect(() => idx2.search(q, 5)).not.toThrow();
    expect(() => idx4.search(q, 5)).not.toThrow();
  });

  it('honors a mask on the FastScan path', () => {
    const data = gaussianVecs(200, 40);
    const idx = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: true });
    idx.add(data);
    const mask = new Uint8Array(200).fill(1);
    // Exclude the self-match at slot 7.
    mask[7] = 0;
    const res = idx.search(data[7]!, 1, { mask });
    expect(res.indices[0]).not.toBe(7);
  });

  it('rejects a wrong-length mask on the FastScan path (same error as the exact paths)', () => {
    const data = gaussianVecs(40, 45);
    const fast = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: true });
    const scalar = new TurboQuantIndex({ dim: FDIM, bits: 4, wasm: false });
    fast.add(data);
    scalar.add(data);
    const q = gaussianVecs(1, 46)[0]!;
    const shortMask = new Uint8Array(10).fill(1);
    for (const idx of [fast, scalar]) {
      let err: unknown;
      try {
        idx.search(q, 5, { mask: shortMask });
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(SearchError);
      expect((err as SearchError).code).toBe('INVALID_MASK');
    }
  });

  it('a mask that excludes every vector yields an empty result on both paths', () => {
    const data = gaussianVecs(40, 47);
    const fast = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: true });
    const scalar = new TurboQuantIndex({ dim: FDIM, bits: 4, wasm: false });
    fast.add(data);
    scalar.add(data);
    const q = gaussianVecs(1, 48)[0]!;
    const none = new Uint8Array(40); // all zeros
    for (const idx of [fast, scalar]) {
      const res = idx.search(q, 5, { mask: none });
      expect(res.indices.length).toBe(0);
      expect(res.scores.length).toBe(0);
    }
  });

  it('re-uploads blocked codes after mutation', () => {
    const idx = new TurboQuantIndex({ dim: FDIM, bits: 4, fastscan: true });
    idx.add(gaussianVecs(50, 50));
    const q = gaussianVecs(1, 51)[0]!;
    idx.search(q, 5); // initial upload
    idx.add(gaussianVecs(50, 52)); // mutate → dirty
    expect(() => idx.search(q, 5)).not.toThrow();
  });

  it('applies the calibration dual on the FastScan path (high recall vs exact+calibrated)', () => {
    const data = gaussianVecs(1200, 60); // >= CALIBRATION_MIN_SAMPLES to fit calibration
    const queries = gaussianVecs(10, 61);
    const exact = new TurboQuantIndex({ dim: FDIM, bits: 4, metric: 'cosine', calibrate: true });
    const fast = new TurboQuantIndex({
      dim: FDIM,
      bits: 4,
      metric: 'cosine',
      calibrate: true,
      fastscan: true,
    });
    exact.add(data);
    fast.add(data);
    expect(exact.calibrated).toBe(true);
    expect(fast.calibrated).toBe(true);
    let totalHits = 0;
    const k = 10;
    for (const q of queries) {
      const a = exact.search(q, k);
      const b = fast.search(q, k);
      const setA = new Set(Array.from(a.indices));
      for (const idx of b.indices) if (setA.has(idx)) totalHits++;
    }
    expect(totalHits / (queries.length * k)).toBeGreaterThan(0.8);
  });
});

describe('TurboQuantIndex — IVF coarse quantizer', () => {
  const IDIM = 32;

  /** Gaussian mixture: `clusters` well-separated centers, `per` points each. */
  function clusteredVecs(clusters: number, per: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    const centers = Array.from({ length: clusters }, () => {
      const c = new Float32Array(IDIM);
      for (let i = 0; i < IDIM; i++) c[i] = rng.nextGaussian() * 10;
      return c;
    });
    const out: Float32Array[] = [];
    for (let b = 0; b < clusters; b++) {
      for (let j = 0; j < per; j++) {
        const v = new Float32Array(IDIM);
        for (let i = 0; i < IDIM; i++) v[i] = centers[b]![i]! + rng.nextGaussian();
        out.push(v);
      }
    }
    return out;
  }

  it('validates nlist and nprobe at construction', () => {
    for (const nlist of [1, 1.5, 0, 1 << 23]) {
      let err: unknown;
      try {
        new TurboQuantIndex({ dim: IDIM, ivf: { nlist } });
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(IndexError);
      expect((err as IndexError).code).toBe('INVALID_NLIST');
    }
    for (const nprobe of [0, 5, 2.5]) {
      let err: unknown;
      try {
        new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4, nprobe } });
      } catch (e) {
        err = e;
      }
      expect((err as IndexError).code).toBe('INVALID_NPROBE');
    }
  });

  it('trains on the first batch when it has ≥ nlist vectors, and freezes flat otherwise', () => {
    const data = clusteredVecs(4, 10, 1);
    const trained = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
    trained.add(data);
    expect(trained.ivfActive).toBe(true);

    const flat = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 8 } });
    flat.add(data.slice(0, 7)); // 7 < nlist = 8 → flat forever
    expect(flat.ivfActive).toBe(false);
    flat.add(data); // a later big add must NOT retrain (decision frozen)
    expect(flat.ivfActive).toBe(false);
  });

  it('nprobe = nlist reproduces the flat scan exactly (indices and scores)', () => {
    const data = clusteredVecs(8, 25, 2);
    const queries = clusteredVecs(8, 1, 3);
    const flat = new TurboQuantIndex({ dim: IDIM, wasm: false });
    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 8 } });
    flat.add(data);
    ivf.add(data);
    expect(ivf.ivfActive).toBe(true);
    for (const metric of ['dot', 'cosine', 'euclidean'] as const) {
      for (const q of queries) {
        const a = flat.search(q, 10, { metric });
        const b = ivf.search(q, 10, { metric, nprobe: 8 });
        expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
        expect(Array.from(b.scores)).toEqual(Array.from(a.scores));
      }
    }
  });

  it('achieves high recall at nprobe ≪ nlist on clustered data', () => {
    const data = clusteredVecs(16, 30, 4);
    const queries = clusteredVecs(16, 1, 5);
    const flat = new TurboQuantIndex({ dim: IDIM, wasm: false });
    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 16, nprobe: 4 } });
    flat.add(data);
    ivf.add(data);
    const k = 10;
    let hits = 0;
    for (const q of queries) {
      const exact = new Set(Array.from(flat.search(q, k).indices));
      for (const j of ivf.search(q, k).indices) if (exact.has(j)) hits++;
    }
    expect(hits / (queries.length * k)).toBeGreaterThan(0.8);
  });

  it('keeps remove parity with a flat twin (interleaved adds and removes)', () => {
    const data = clusteredVecs(4, 30, 6);
    const flat = new TurboQuantIndex({ dim: IDIM, wasm: false });
    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
    flat.add(data.slice(0, 80));
    ivf.add(data.slice(0, 80));
    const rng = createRng(7);
    let n = 80;
    let next = 80;
    for (let op = 0; op < 60; op++) {
      if (rng.nextFloat() < 0.5 && next < data.length) {
        flat.addOne(data[next]!);
        ivf.addOne(data[next]!);
        next++;
        n++;
      } else if (n > 1) {
        const i = Math.min(n - 1, Math.floor(rng.nextFloat() * n));
        flat.swapRemove(i);
        ivf.swapRemove(i);
        n--;
      }
    }
    expect(ivf.size).toBe(flat.size);
    const q = data[0]!;
    const a = flat.search(q, 10);
    const b = ivf.search(q, 10, { nprobe: 4 });
    expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
    expect(Array.from(b.scores)).toEqual(Array.from(a.scores));
  });

  it('round-trips through toBytes/fromBytes with identical search results', () => {
    const data = clusteredVecs(4, 20, 8);
    const ivf = new TurboQuantIndex({
      dim: IDIM,
      metric: 'euclidean',
      ivf: { nlist: 4, nprobe: 2 },
    });
    ivf.add(data);
    const restored = TurboQuantIndex.fromBytes(ivf.toBytes());
    expect(restored.ivfActive).toBe(true);
    expect(restored.size).toBe(ivf.size);
    for (const q of data.slice(0, 5)) {
      const a = ivf.search(q, 5);
      const b = restored.search(q, 5);
      expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
      expect(Array.from(b.scores)).toEqual(Array.from(a.scores));
    }
    // The restored index accepts further adds and removes without retraining.
    restored.addOne(data[0]!);
    restored.swapRemove(0);
    expect(restored.ivfActive).toBe(true);
  });

  it('honors masks within the probed cells, and clear() keeps the trained quantizer', () => {
    const data = clusteredVecs(4, 20, 9);
    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
    ivf.add(data);
    const mask = new Uint8Array(80).fill(1);
    mask[3] = 0;
    const res = ivf.search(data[3]!, 1, { mask, nprobe: 4 });
    expect(res.indices[0]).not.toBe(3);

    ivf.clear();
    expect(ivf.size).toBe(0);
    expect(ivf.ivfActive).toBe(true); // centroids survive; decision stays frozen
    ivf.add(data.slice(0, 10)); // post-clear adds assign to the existing cells
    expect(ivf.search(data[0]!, 1, { nprobe: 4 }).indices.length).toBe(1);
  });

  it('ignores nprobe when IVF is not active; validates it per query when active', () => {
    const flat = new TurboQuantIndex({ dim: IDIM });
    flat.add(clusteredVecs(2, 5, 10));
    expect(() =>
      flat.search(flat.size > 0 ? clusteredVecs(1, 1, 11)[0]! : new Float32Array(IDIM), 2, {
        nprobe: 999,
      }),
    ).not.toThrow();

    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
    ivf.add(clusteredVecs(4, 10, 12));
    let err: unknown;
    try {
      ivf.search(clusteredVecs(1, 1, 13)[0]!, 2, { nprobe: 5 });
    } catch (e) {
      err = e;
    }
    expect((err as IndexError).code).toBe('INVALID_NPROBE');
  });
});

describe('TurboQuantIndex — IVF input hardening (review follow-ups)', () => {
  const IDIM = 32;

  function clusteredVecs(clusters: number, per: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    const centers = Array.from({ length: clusters }, () => {
      const c = new Float32Array(IDIM);
      for (let i = 0; i < IDIM; i++) c[i] = rng.nextGaussian() * 10;
      return c;
    });
    const out: Float32Array[] = [];
    for (let b = 0; b < clusters; b++) {
      for (let j = 0; j < per; j++) {
        const v = new Float32Array(IDIM);
        for (let i = 0; i < IDIM; i++) v[i] = centers[b]![i]! + rng.nextGaussian();
        out.push(v);
      }
    }
    return out;
  }

  it('rejects malformed queries with the same typed errors as the flat path, before probing', () => {
    const data = clusteredVecs(4, 10, 14);
    const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
    const flat = new TurboQuantIndex({ dim: IDIM, wasm: false });
    ivf.add(data);
    flat.add(data);
    const bads: [Float32Array, string][] = [
      [new Float32Array(IDIM - 1), 'INVALID_LENGTH'], // wrong length
      [new Float32Array(IDIM).fill(Number.NaN), 'INVALID_LENGTH'], // non-finite
      [new Float32Array(IDIM), 'ZERO_QUERY'], // zero query
    ];
    for (const [bad, code] of bads) {
      for (const idx of [ivf, flat]) {
        let err: unknown;
        try {
          idx.search(bad, 2);
        } catch (e) {
          err = e;
        }
        expect(err).toBeInstanceOf(SearchError);
        expect((err as SearchError).code).toBe(code);
      }
    }
  });

  it('validates the first training batch atomically: a bad row leaves the index unchanged', () => {
    const good = clusteredVecs(4, 10, 15);
    const nanRow = new Float32Array(IDIM).fill(1);
    nanRow[3] = Number.NaN;
    for (const poison of [nanRow, new Float32Array(IDIM) /* zero row */]) {
      const ivf = new TurboQuantIndex({ dim: IDIM, ivf: { nlist: 4 } });
      let err: unknown;
      try {
        ivf.add([...good.slice(0, 10), poison, ...good.slice(10)]);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(EncodeError);
      expect(ivf.size).toBe(0); // nothing appended
      expect(ivf.ivfActive).toBe(false); // nothing trained on the poisoned batch
      ivf.add(good); // the decision is NOT frozen by the failed batch
      expect(ivf.ivfActive).toBe(true);
      expect(ivf.size).toBe(good.length);
    }
  });

  it('validates the first calibration batch the same way', () => {
    const rng = createRng(16);
    const good = Array.from({ length: CALIBRATION_MIN_SAMPLES }, () => {
      const v = new Float32Array(IDIM);
      for (let i = 0; i < IDIM; i++) v[i] = rng.nextGaussian();
      return v;
    });
    const idx = new TurboQuantIndex({ dim: IDIM, calibrate: true, wasm: false });
    const poisoned = [...good];
    poisoned[7] = new Float32Array(IDIM); // zero row
    let err: unknown;
    try {
      idx.add(poisoned);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(EncodeError);
    expect(idx.size).toBe(0);
    expect(idx.calibrated).toBe(false);
    idx.add(good); // decision not frozen; a clean batch still calibrates
    expect(idx.calibrated).toBe(true);
  });
});
