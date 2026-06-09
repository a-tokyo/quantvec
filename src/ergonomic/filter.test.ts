import { describe, expect, it } from 'vitest';
import { compileFilter, FilterError } from './filter';
import type { Filter } from './types';

const p = (payload: unknown) => (f: Filter) => compileFilter(f)(1, payload);

describe('compileFilter — leaf conditions', () => {
  it('match: exact payload value', () => {
    const f: Filter = { must: [{ key: 'tag', match: { value: 'docs' } }] };
    expect(p({ tag: 'docs' })(f)).toBe(true);
    expect(p({ tag: 'blog' })(f)).toBe(false);
    expect(p({})(f)).toBe(false);
    expect(p(undefined)(f)).toBe(false);
  });

  it('range: numeric bounds (inclusive/exclusive)', () => {
    const f: Filter = { must: [{ key: 'year', range: { gte: 2020, lt: 2025 } }] };
    expect(p({ year: 2020 })(f)).toBe(true);
    expect(p({ year: 2024 })(f)).toBe(true);
    expect(p({ year: 2019 })(f)).toBe(false);
    expect(p({ year: 2025 })(f)).toBe(false);
    expect(p({ year: 'x' })(f)).toBe(false); // non-number
  });

  it('range: gt and lte bounds', () => {
    const f: Filter = { must: [{ key: 'year', range: { gt: 2020, lte: 2025 } }] };
    expect(p({ year: 2020 })(f)).toBe(false); // gt is exclusive
    expect(p({ year: 2021 })(f)).toBe(true);
    expect(p({ year: 2025 })(f)).toBe(true); // lte is inclusive
    expect(p({ year: 2026 })(f)).toBe(false);
  });

  it('hasId: id membership', () => {
    const f: Filter = { must: [{ hasId: [1, 5, 9] }] };
    expect(compileFilter(f)(5, {})).toBe(true);
    expect(compileFilter(f)(2, {})).toBe(false);
  });
});

describe('compileFilter — boolean combination', () => {
  it('must = AND, should = OR (≥1), must_not = NONE', () => {
    const f: Filter = {
      must: [{ key: 'a', match: { value: 1 } }],
      should: [
        { key: 'b', match: { value: 2 } },
        { key: 'c', match: { value: 3 } },
      ],
      must_not: [{ key: 'd', match: { value: 4 } }],
    };
    expect(p({ a: 1, b: 2 })(f)).toBe(true); // must + a should
    expect(p({ a: 1, c: 3 })(f)).toBe(true); // must + other should
    expect(p({ a: 1 })(f)).toBe(false); // no should matches
    expect(p({ a: 9, b: 2 })(f)).toBe(false); // must fails
    expect(p({ a: 1, b: 2, d: 4 })(f)).toBe(false); // must_not hits
  });

  it('an empty filter matches everything', () => {
    expect(p({ anything: true })({})).toBe(true);
    expect(p(undefined)({})).toBe(true);
  });

  it('nested filter as a condition', () => {
    const f: Filter = {
      must: [
        {
          should: [
            { key: 'x', match: { value: 1 } },
            { key: 'x', match: { value: 2 } },
          ],
        },
      ],
    };
    expect(p({ x: 2 })(f)).toBe(true);
    expect(p({ x: 3 })(f)).toBe(false);
  });

  it('rejects a malformed condition', () => {
    let err: unknown;
    try {
      compileFilter({ must: [{ bogus: true } as unknown as never] });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(FilterError);
    expect((err as FilterError).code).toBe('INVALID_CONDITION');
  });
});
