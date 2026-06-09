// Compile the qdrant-inspired filter DSL (see ./types) into a fast predicate over
// (id, payload). The {@link Collection} turns the predicate into the positional mask
// the core scan understands, so filtering reuses the same kernel path.

import type { IdType } from '../io/serialize';
import type { Condition, Filter, HasIdCondition, MatchCondition, RangeCondition } from './types';

/** Discriminated, code-tagged error for a malformed filter. */
export class FilterError extends Error {
  readonly code: 'INVALID_CONDITION';
  constructor(message: string) {
    super(message);
    this.name = 'FilterError';
    this.code = 'INVALID_CONDITION';
  }
}

/** A compiled filter: true iff the point (id, payload) passes. */
export type FilterPredicate<Id extends IdType> = (id: Id, payload: unknown) => boolean;

/** Read `payload[key]` if payload is an object, else undefined. */
function payloadValue(payload: unknown, key: string): unknown {
  return payload !== null && typeof payload === 'object'
    ? (payload as Record<string, unknown>)[key]
    : undefined;
}

function compileCondition<Id extends IdType>(cond: Condition<Id>): FilterPredicate<Id> {
  if ('match' in cond) {
    const { key, match } = cond as MatchCondition;
    return (_id, payload) => payloadValue(payload, key) === match.value;
  }
  if ('range' in cond) {
    const { key, range } = cond as RangeCondition;
    return (_id, payload) => {
      const x = payloadValue(payload, key);
      if (typeof x !== 'number') return false;
      if (range.gt !== undefined && !(x > range.gt)) return false;
      if (range.gte !== undefined && !(x >= range.gte)) return false;
      if (range.lt !== undefined && !(x < range.lt)) return false;
      if (range.lte !== undefined && !(x <= range.lte)) return false;
      return true;
    };
  }
  if ('hasId' in cond) {
    const set = new Set<IdType>((cond as HasIdCondition<Id>).hasId);
    return (id) => set.has(id);
  }
  if ('must' in cond || 'should' in cond || 'must_not' in cond) {
    return compileFilter(cond as Filter<Id>);
  }
  throw new FilterError(`unrecognized filter condition: ${JSON.stringify(cond)}`);
}

/**
 * Compile a {@link Filter} into a predicate. Semantics: a point passes when every
 * `must` clause passes AND (there are no `should` clauses, or at least one passes) AND
 * no `must_not` clause passes. An empty filter matches everything.
 *
 * @throws {FilterError} on a malformed condition.
 */
export function compileFilter<Id extends IdType>(filter: Filter<Id>): FilterPredicate<Id> {
  const must = (filter.must ?? []).map(compileCondition);
  const should = (filter.should ?? []).map(compileCondition);
  const mustNot = (filter.must_not ?? []).map(compileCondition);

  return (id, payload) => {
    for (const p of must) if (!p(id, payload)) return false;
    if (should.length > 0 && !should.some((p) => p(id, payload))) return false;
    for (const p of mustNot) if (p(id, payload)) return false;
    return true;
  };
}
