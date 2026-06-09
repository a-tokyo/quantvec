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
    const { gt, gte, lt, lte } = range;
    if (gt === undefined && gte === undefined && lt === undefined && lte === undefined) {
      throw new FilterError(`range condition for "${key}" must specify gt, gte, lt, or lte`);
    }
    // An impossible bound (e.g. {gt: 10, lt: 5}, or {gte: 10, lt: 10}) would silently
    // match nothing — almost certainly a mistake, so reject it at compile time.
    const lo = gt ?? gte;
    const hi = lt ?? lte;
    if (lo !== undefined && hi !== undefined) {
      const exclusive = gt !== undefined || lt !== undefined;
      if (lo > hi || (lo === hi && exclusive)) {
        throw new FilterError(
          `range condition for "${key}" can never match: ${JSON.stringify(range)}`,
        );
      }
    }
    return (_id, payload) => {
      const x = payloadValue(payload, key);
      if (typeof x !== 'number') return false;
      if (gt !== undefined && !(x > gt)) return false;
      if (gte !== undefined && !(x >= gte)) return false;
      if (lt !== undefined && !(x < lt)) return false;
      if (lte !== undefined && !(x <= lte)) return false;
      return true;
    };
  }
  if ('hasId' in cond) {
    const ids = (cond as HasIdCondition<Id>).hasId;
    if (ids.length === 0) {
      throw new FilterError('hasId condition must not be empty (would match nothing)');
    }
    const set = new Set<IdType>(ids);
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
