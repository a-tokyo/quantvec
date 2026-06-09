import { describe, expect, it } from 'vitest';
import type { Collection } from './collection';
import { createCollection } from './collection';
import type { IdMapError } from '../index/id-map-index';

interface Doc {
  tag: string;
  year: number;
}

/** Four orthogonal dim-8 vectors → self-query is unambiguously top-1. */
const V: number[][] = [
  [8, 8, 0, 0, 0, 0, 0, 0],
  [0, 0, 8, 8, 0, 0, 0, 0],
  [0, 0, 0, 0, 8, 8, 0, 0],
  [0, 0, 0, 0, 0, 0, 8, 8],
];

function docs(): Collection<Doc, number> {
  const c = createCollection<Doc, number>({
    vectors: { size: 8, distance: 'cosine' },
    quantization: { bits: 4 },
  });
  c.upsert([
    { id: 1, vector: V[0]!, payload: { tag: 'docs', year: 2021 } },
    { id: 2, vector: V[1]!, payload: { tag: 'blog', year: 2023 } },
    { id: 3, vector: V[2]!, payload: { tag: 'docs', year: 2024 } },
    { id: 4, vector: V[3]!, payload: { tag: 'docs', year: 2019 } },
  ]);
  return c;
}

describe('Collection — upsert & search', () => {
  it('returns hits with id, score, and payload (best-first)', () => {
    const c = docs();
    expect([c.size, c.dim, c.distance]).toEqual([4, 8, 'cosine']);
    const hits = c.search(V[2]!, { limit: 2 });
    expect(hits[0]!.id).toBe(3);
    expect(hits[0]!.payload).toEqual({ tag: 'docs', year: 2024 });
    expect(typeof hits[0]!.score).toBe('number');
    expect(hits.length).toBe(2);
  });

  it('omits payload when withPayload is false', () => {
    const hits = docs().search(V[0]!, { limit: 1, withPayload: false });
    expect(hits[0]!.payload).toBeUndefined();
  });

  it('returns [] for an empty collection', () => {
    const c = createCollection<Doc, number>({
      vectors: { size: 8, distance: 'cosine' },
      quantization: { bits: 4 },
    });
    expect(c.search(V[0]!, { limit: 5 })).toEqual([]);
  });
});

describe('Collection — filtering', () => {
  it('restricts results by a match filter', () => {
    const c = docs();
    // Query nearest to V[1] (id 2, tag 'blog') but require tag 'docs'.
    const hits = c.search(V[1]!, {
      limit: 4,
      filter: { must: [{ key: 'tag', match: { value: 'docs' } }] },
    });
    expect(hits.every((h) => h.payload!.tag === 'docs')).toBe(true);
    expect(hits.map((h) => h.id)).not.toContain(2);
  });

  it('restricts by a range filter', () => {
    const hits = docs().search(V[0]!, {
      limit: 4,
      filter: { must: [{ key: 'year', range: { gte: 2022 } }] },
    });
    expect(hits.map((h) => h.id).sort()).toEqual([2, 3]);
  });

  it('supports hasId and must_not', () => {
    const hits = docs().search(V[0]!, {
      limit: 4,
      filter: {
        must: [{ hasId: [1, 3, 4] }],
        must_not: [{ key: 'tag', match: { value: 'blog' } }],
      },
    });
    expect(hits.map((h) => h.id).sort()).toEqual([1, 3, 4]);
  });
});

describe('Collection — mutation', () => {
  it('upsert replaces an existing id (vector + payload)', () => {
    const c = docs();
    c.upsert([{ id: 2, vector: V[2]!, payload: { tag: 'updated', year: 2030 } }]);
    expect(c.size).toBe(4); // replaced, not added
    expect(c.get(2)).toEqual({ tag: 'updated', year: 2030 });
    // id 2 now lies along V[2]; the top hits for V[2] are ids 2 and 3 (both along it).
    const ids = c.search(V[2]!, { limit: 2 }).map((h) => h.id);
    expect(ids).toContain(2);
  });

  it('delete removes ids (single and array; unknown ignored)', () => {
    const c = docs();
    c.delete(2);
    c.delete([3, 99]);
    expect(c.size).toBe(2);
    expect(c.has(2)).toBe(false);
    expect(c.has(3)).toBe(false);
    expect(c.has(1)).toBe(true);
    expect(c.get(2)).toBeUndefined();
  });

  it('rejects duplicate ids within one upsert batch', () => {
    const c = createCollection<Doc, number>({
      vectors: { size: 8, distance: 'cosine' },
      quantization: { bits: 4 },
    });
    let err: unknown;
    try {
      c.upsert([
        { id: 7, vector: V[0]! },
        { id: 7, vector: V[1]! },
      ]);
    } catch (e) {
      err = e;
    }
    expect((err as IdMapError).code).toBe('DUPLICATE_ID');
  });
});
