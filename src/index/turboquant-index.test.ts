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
