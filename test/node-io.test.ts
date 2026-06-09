import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { IndexError } from '../src/index/turboquant-index';
import { TurboQuantIndex } from '../src/index/turboquant-index';
import type { IdMapError } from '../src/index/id-map-index';
import { IdMapIndex } from '../src/index/id-map-index';
import { loadIdMapIndex, loadIndex, readIndexBytes, saveIndex } from '../src/node';

const DIM = 8;
const ORTHO: Float32Array[] = [
  Float32Array.from([8, 8, 0, 0, 0, 0, 0, 0]),
  Float32Array.from([0, 0, 8, 8, 0, 0, 0, 0]),
  Float32Array.from([0, 0, 0, 0, 8, 8, 0, 0]),
];

let dir: string;
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'quantvec-'));
});
afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

async function caught<T>(fn: () => Promise<T>): Promise<unknown> {
  try {
    await fn();
  } catch (e) {
    return e;
  }
  return undefined;
}

describe('quantvec/node — filesystem helpers', () => {
  it('saves and loads a positional index, reproducing search', async () => {
    const idx = new TurboQuantIndex({ dim: DIM, bits: 3, metric: 'dot', seed: 4 });
    idx.add(ORTHO);
    const path = join(dir, 'positional.qv');
    await saveIndex(idx, path);

    const bytes = await readIndexBytes(path);
    expect(bytes).toBeInstanceOf(Uint8Array);

    const restored = await loadIndex(path);
    expect(restored.size).toBe(3);
    expect(restored.search(ORTHO[1]!, 1).indices[0]).toBe(1);
  });

  it('saves and loads an id-keyed index', async () => {
    const idx = new IdMapIndex<number>({ dim: DIM });
    idx.addWithIds([7, 8, 9], ORTHO);
    const path = join(dir, 'idmap.qv');
    await saveIndex(idx, path);

    const restored = await loadIdMapIndex(path);
    expect(restored.size).toBe(3);
    expect(restored.search(ORTHO[2]!, 1).ids[0]).toBe(9);
  });

  it('surfaces WRONG_KIND when the loader and file kind disagree', async () => {
    const pos = new TurboQuantIndex({ dim: DIM });
    pos.add([ORTHO[0]!]);
    const posPath = join(dir, 'p2.qv');
    await saveIndex(pos, posPath);

    const idmap = new IdMapIndex<number>({ dim: DIM });
    idmap.addWithIds([1], [ORTHO[0]!]);
    const idmapPath = join(dir, 'i2.qv');
    await saveIndex(idmap, idmapPath);

    expect(((await caught(() => loadIdMapIndex(posPath))) as IdMapError).code).toBe('WRONG_KIND');
    expect(((await caught(() => loadIndex(idmapPath))) as IndexError).code).toBe('WRONG_KIND');
  });
});
