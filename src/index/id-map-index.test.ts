import { describe, expect, it } from 'vitest';
import type { IdMapError } from './id-map-index';
import { IdMapIndex } from './id-map-index';
import { CALIBRATION_MIN_SAMPLES, TurboQuantIndex } from './turboquant-index';
import { EncodeError } from '../core/encode';
import { createRng } from '../core/rng';

const DIM = 8;

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

describe('IdMapIndex — add & query by id', () => {
  it('delegates dims/metric and adds vectors under ids (array form)', () => {
    const idx = new IdMapIndex<number>({ dim: DIM, bits: 2, metric: 'dot', seed: 3 });
    expect([idx.dim, idx.bits, idx.metric, idx.seed, idx.size]).toEqual([DIM, 2, 'dot', 3, 0]);
    idx.addWithIds([100, 200, 300, 400], ORTHO);
    expect(idx.size).toBe(4);
    expect(idx.has(200)).toBe(true);
    expect(idx.has(999)).toBe(false);
  });

  it('adds from a flat Float32Array', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    const flat = new Float32Array(2 * DIM);
    flat.set(ORTHO[0]!, 0);
    flat.set(ORTHO[1]!, DIM);
    idx.addWithIds([1, 2], flat);
    expect(idx.size).toBe(2);
  });

  it('adds from number[][] (plain arrays)', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([1, 2], [Array.from(ORTHO[0]!), Array.from(ORTHO[1]!)]);
    expect(idx.size).toBe(2);
    expect(idx.search(ORTHO[1]!, 1).ids[0]).toBe(2);
  });

  it('search returns external ids best-first', () => {
    const idx = new IdMapIndex<string>({ dim: DIM });
    idx.addWithIds(['a', 'b', 'c', 'd'], ORTHO);
    for (let i = 0; i < ORTHO.length; i++) {
      const res = idx.search(ORTHO[i]!, 1);
      expect(res.ids[0]).toBe(['a', 'b', 'c', 'd'][i]);
      expect(res.scores.length).toBe(1);
    }
  });

  it('honors a per-query metric override', () => {
    const idx = new IdMapIndex<number>({ dim: DIM, metric: 'cosine' });
    idx.addWithIds([1, 2, 3, 4], ORTHO);
    const res = idx.search(ORTHO[0]!, 4, { metric: 'euclidean' });
    expect(res.ids[0]).toBe(1);
  });
});

describe('IdMapIndex — validation', () => {
  it('rejects a flat buffer not a multiple of dim', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    expect(
      (catchError(() => idx.addWithIds([1], new Float32Array(DIM + 1))) as IdMapError).code,
    ).toBe('INVALID_LENGTH');
  });

  it('rejects an id/vector count mismatch', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    expect((catchError(() => idx.addWithIds([1], ORTHO)) as IdMapError).code).toBe(
      'COUNT_MISMATCH',
    );
  });

  it('rejects an invalid id type (incl. NaN)', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    const bad = catchError(() => idx.addWithIds([true as unknown as number], [ORTHO[0]!]));
    expect((bad as IdMapError).code).toBe('INVALID_ID_TYPE');
    const nan = catchError(() => idx.addWithIds([NaN], [ORTHO[0]!]));
    expect((nan as IdMapError).code).toBe('INVALID_ID_TYPE');
  });

  it('rejects a non-array-like vector element', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    const err = catchError(() => idx.addWithIds([1], [null as unknown as number[]]));
    expect((err as IdMapError).code).toBe('INVALID_VECTOR');
    expect(idx.size).toBe(0); // structural validation is atomic
  });

  it('rejects a wrong-length vector element atomically', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    const err = catchError(() => idx.addWithIds([1, 2], [ORTHO[0]!, new Float32Array(3)]));
    expect((err as IdMapError).code).toBe('INVALID_LENGTH');
    expect(idx.size).toBe(0); // first vector not committed either
  });

  it('rejects duplicate ids within a batch and against existing ids, without partial mutation', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    expect(
      (catchError(() => idx.addWithIds([1, 1], [ORTHO[0]!, ORTHO[1]!])) as IdMapError).code,
    ).toBe('DUPLICATE_ID');
    expect(idx.size).toBe(0); // aborted before any add

    idx.addWithIds([1], [ORTHO[0]!]);
    expect((catchError(() => idx.addWithIds([1], [ORTHO[1]!])) as IdMapError).code).toBe(
      'DUPLICATE_ID',
    );
    expect(idx.size).toBe(1);
  });

  it('re-throws EncodeError on a zero vector', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    expect(catchError(() => idx.addWithIds([1], [new Float32Array(DIM)]))).toBeInstanceOf(
      EncodeError,
    );
  });

  it('rejects a batch with a zero/non-finite vector anywhere without partial mutation', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    // Vectors 0 and 1 are valid; vector 2 is a zero vector — the whole batch must
    // be rejected before any row or id is appended.
    const zero = new Float32Array(DIM);
    expect(
      (catchError(() => idx.addWithIds([1, 2, 3], [ORTHO[0]!, ORTHO[1]!, zero])) as EncodeError)
        .code,
    ).toBe('ZERO_VECTOR');
    expect(idx.size).toBe(0);
    expect(idx.has(1)).toBe(false);
    expect(idx.has(2)).toBe(false);

    const nonFinite = Float32Array.from(ORTHO[0]!);
    nonFinite[0] = NaN;
    expect(
      (
        catchError(() =>
          idx.addWithIds([1, 2, 3], [ORTHO[0]!, nonFinite, ORTHO[1]!]),
        ) as EncodeError
      ).code,
    ).toBe('INVALID_LENGTH');
    expect(idx.size).toBe(0);
  });

  it('search throws IdMapError EMPTY on an empty index', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    expect((catchError(() => idx.search(ORTHO[0]!, 1)) as IdMapError).code).toBe('EMPTY');
  });
});

describe('IdMapIndex — filter, ids, clear', () => {
  it('restricts the scan to ids passing the filter predicate', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([1, 2, 3, 4], ORTHO);
    // Exclude id 1 (the self-match for ORTHO[0]); top hit must be a different id.
    const res = idx.search(ORTHO[0]!, 1, { filter: (id) => id !== 1 });
    expect(res.ids[0]).not.toBe(1);
    expect(res.ids).toHaveLength(1);
  });

  it('ids() returns a fresh snapshot in slot order', () => {
    const idx = new IdMapIndex<string>({ dim: DIM });
    idx.addWithIds(['a', 'b', 'c'], [ORTHO[0]!, ORTHO[1]!, ORTHO[2]!]);
    const ids = idx.ids();
    expect(ids).toEqual(['a', 'b', 'c']);
    ids.push('mutated'); // snapshot must not affect the index
    expect(idx.ids()).toEqual(['a', 'b', 'c']);
  });

  it('clear() empties the index and its id maps', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([1, 2, 3], [ORTHO[0]!, ORTHO[1]!, ORTHO[2]!]);
    idx.clear();
    expect(idx.size).toBe(0);
    expect(idx.has(1)).toBe(false);
    expect(idx.ids()).toEqual([]);
    // Reusing ids after clear is allowed.
    idx.addWithIds([1], [ORTHO[0]!]);
    expect(idx.has(1)).toBe(true);
  });
});

describe('IdMapIndex — remove', () => {
  it('removes by id and re-points the moved row (swap-remove of a non-last slot)', () => {
    const idx = new IdMapIndex<string>({ dim: DIM });
    idx.addWithIds(['a', 'b', 'c'], [ORTHO[0]!, ORTHO[1]!, ORTHO[2]!]);
    idx.remove('a'); // slot 0; 'c' (last) moves into slot 0
    expect(idx.size).toBe(2);
    expect(idx.has('a')).toBe(false);
    expect(idx.has('b')).toBe(true);
    expect(idx.has('c')).toBe(true);
    // The moved id must still resolve correctly through search.
    expect(idx.search(ORTHO[2]!, 1).ids[0]).toBe('c');
    expect(idx.search(ORTHO[1]!, 1).ids[0]).toBe('b');
  });

  it('removes the last id (no move needed)', () => {
    const idx = new IdMapIndex<string>({ dim: DIM });
    idx.addWithIds(['a', 'b'], [ORTHO[0]!, ORTHO[1]!]);
    idx.remove('b');
    expect(idx.size).toBe(1);
    expect(idx.has('b')).toBe(false);
    // Re-adding a previously-removed id is allowed.
    idx.addWithIds(['b'], [ORTHO[2]!]);
    expect(idx.has('b')).toBe(true);
  });

  it('rejects an unknown id', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([1], [ORTHO[0]!]);
    expect((catchError(() => idx.remove(2)) as IdMapError).code).toBe('UNKNOWN_ID');
  });
});

describe('IdMapIndex — serialization', () => {
  it('round-trips number ids and reproduces search', () => {
    const idx = new IdMapIndex<number>({ dim: DIM, bits: 3, metric: 'dot', seed: 8 });
    idx.addWithIds([11, 22, 33, 44], ORTHO);
    const restored = IdMapIndex.fromBytes<number>(idx.toBytes()); // caller asserts id type
    expect([restored.dim, restored.bits, restored.metric, restored.seed, restored.size]).toEqual([
      DIM,
      3,
      'dot',
      8,
      4,
    ]);
    for (let i = 0; i < ORTHO.length; i++) {
      expect(restored.search(ORTHO[i]!, 1).ids[0]).toBe([11, 22, 33, 44][i]);
    }
    expect(restored.has(33)).toBe(true);
  });

  it('round-trips string and bigint ids', () => {
    const idx = new IdMapIndex<string | bigint>({ dim: DIM });
    idx.addWithIds(['x', 'y', 123456789012345678901234567890n, 'z'], ORTHO);
    const restored = IdMapIndex.fromBytes(idx.toBytes());
    expect(restored.has(123456789012345678901234567890n)).toBe(true);
    expect(restored.search(ORTHO[0]!, 1).ids[0]).toBe('x');
    expect(restored.search(ORTHO[2]!, 1).ids[0]).toBe(123456789012345678901234567890n);
  });

  it('survives a remove before serialization', () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([1, 2, 3], [ORTHO[0]!, ORTHO[1]!, ORTHO[2]!]);
    idx.remove(1);
    const restored = IdMapIndex.fromBytes(idx.toBytes());
    expect(restored.size).toBe(2);
    expect(restored.has(1)).toBe(false);
    expect(restored.has(2)).toBe(true);
    expect(restored.has(3)).toBe(true);
  });

  it('rejects a positional buffer with WRONG_KIND', () => {
    const pos = new TurboQuantIndex({ dim: DIM });
    pos.add([ORTHO[0]!]);
    expect((catchError(() => IdMapIndex.fromBytes(pos.toBytes())) as IdMapError).code).toBe(
      'WRONG_KIND',
    );
  });
});

describe('IdMapIndex — TQ+ calibration', () => {
  it('auto-calibrates on a large first addWithIds and round-trips', () => {
    const cdim = 16;
    const m = CALIBRATION_MIN_SAMPLES + 100;
    const rng = createRng(3);
    const flat = new Float32Array(m * cdim);
    for (let i = 0; i < flat.length; i++) {
      flat[i] = rng.nextGaussian() * Math.pow(0.3, (i % cdim) / cdim);
    }
    const ids = Array.from({ length: m }, (_, i) => i);

    const idx = new IdMapIndex<number>({ dim: cdim, calibrate: true });
    idx.addWithIds(ids, flat);
    expect(idx.calibrated).toBe(true);

    const restored = IdMapIndex.fromBytes<number>(idx.toBytes());
    expect(restored.calibrated).toBe(true);
    expect(restored.size).toBe(m);
    expect(restored.has(0)).toBe(true);
  });
});

describe('IdMapIndex — IVF passthrough', () => {
  const VDIM = 16;

  function vecsAround(center: number, n: number, seed: number): Float32Array[] {
    const rng = createRng(seed);
    return Array.from({ length: n }, () => {
      const v = new Float32Array(VDIM);
      for (let i = 0; i < VDIM; i++) v[i] = center + rng.nextGaussian();
      return v;
    });
  }

  it('trains from the first addWithIds batch and searches by id', () => {
    const data = [...vecsAround(10, 20, 1), ...vecsAround(-10, 20, 2)];
    const ids = data.map((_, i) => 1000 + i);
    const idx = new IdMapIndex<number>({ dim: VDIM, ivf: { nlist: 2 } });
    idx.addWithIds(ids, data);
    expect(idx.ivfActive).toBe(true);
    const res = idx.search(data[5]!, 3, { nprobe: 2 });
    expect(res.ids).toContain(1005);
  });

  it('remove keeps parity with a flat twin; round-trip preserves ivf + id mapping', () => {
    const data = [...vecsAround(10, 15, 3), ...vecsAround(-10, 15, 4)];
    const ids = data.map((_, i) => `p${i}`);
    const ivf = new IdMapIndex<string>({ dim: VDIM, ivf: { nlist: 2 } });
    const flat = new IdMapIndex<string>({ dim: VDIM, wasm: false });
    ivf.addWithIds(ids, data);
    flat.addWithIds(ids, data);
    for (const victim of ['p3', 'p17', 'p0']) {
      ivf.remove(victim);
      flat.remove(victim);
    }
    const a = flat.search(data[5]!, 5);
    const b = ivf.search(data[5]!, 5, { nprobe: 2 });
    expect(b.ids).toEqual(a.ids);

    const restored = IdMapIndex.fromBytes(ivf.toBytes());
    expect(restored.ivfActive).toBe(true);
    expect(restored.search(data[5]!, 5, { nprobe: 2 }).ids).toEqual(b.ids);
  });

  it('filter predicates compose with the probed-cell scan', () => {
    const data = [...vecsAround(10, 20, 5), ...vecsAround(-10, 20, 6)];
    const ids = data.map((_, i) => i);
    const idx = new IdMapIndex<number>({ dim: VDIM, ivf: { nlist: 2 } });
    idx.addWithIds(ids, data);
    const res = idx.search(data[0]!, 10, { nprobe: 2, filter: (id) => id % 2 === 0 });
    expect(res.ids.every((id) => id % 2 === 0)).toBe(true);
    expect(res.ids.length).toBeGreaterThan(0);
  });
});
