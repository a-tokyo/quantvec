// Per-metric scoring for quantvec's flat scan: turn the RaBitQ estimator's raw
// projection into a ranked, reported distance/similarity value.
//
// Why this exists: the search loop (see ./search) computes, for each database
// vector, a single scalar S = ⟨q_rot, c⟩ (the LUT-summed code score) plus the
// stored per-vector `scale` and `norm`. Every supported metric is a closed-form
// function of (S, scale, norm) and the query's (qNorm, qNormSq) — so the hot loop
// stays a tight typed-array scan and all the metric-specific algebra lives here.
//
// The estimator (derived in ./encode's header): scale·S ≈ ⟨q, v⟩ (true dot
// product). From that single quantity each metric follows:
//
//   • dot       value = scale·S            ≈ ⟨q, v⟩            rank DESCENDING
//   • cosine    value = scale·S/(qNorm·‖v‖) ≈ ⟨q,v⟩/(‖q‖‖v‖)   rank DESCENDING
//   • euclidean value = ‖q‖² + ‖v‖² − 2·scale·S ≈ ‖q−v‖²       rank ASCENDING
//
// (Euclidean uses ‖q−v‖² = ‖q‖² + ‖v‖² − 2⟨q,v⟩; see architecture.md "Distance
// metrics".) The reported value is the metric itself; the RANKING KEY is always
// expressed "higher-is-better" so the single size-k min-heap (./topk) can rank
// every metric without a per-metric heap variant — for euclidean that key is the
// negated distance (smaller distance ⇒ larger key ⇒ kept).
//
// Zero-magnitude inputs: cosine divides by qNorm·‖v‖. A zero query (qNorm = 0)
// has no direction and is rejected with a typed error at the search ENTRY (see
// ./search), not here — so the divide is never reached with qNorm = 0. Database
// norms are likewise > 0 because encodeVector rejects the zero vector. These are
// validated boundary conditions, not silent guards: this module assumes its
// numeric inputs are already valid and stays branch-light.

/** The nearest-neighbor metric a search ranks by. */
export type Distance = 'dot' | 'cosine' | 'euclidean';

/**
 * The per-candidate output of a metric: the value to REPORT to the caller and the
 * "higher-is-better" key to feed the top-k min-heap.
 *
 * For dot/cosine the two coincide (higher similarity is better). For euclidean
 * `value` is the (squared) distance while `rankKey = -value`, so the same
 * max-heap selects the SMALLEST distances.
 */
export interface MetricScore {
  /** The metric value reported to the caller (similarity for dot/cosine, dist² for euclidean). */
  value: number;
  /** Higher-is-better ranking key for the top-k heap. */
  rankKey: number;
}

/**
 * Inputs to a metric score that are constant for one query: its Euclidean norm
 * and squared norm. Precomputed once per `searchFlat` call and reused for every
 * candidate so the hot loop never recomputes them.
 */
export interface QueryNorms {
  /** ‖q‖. */
  qNorm: number;
  /** ‖q‖². */
  qNormSq: number;
}

/**
 * Score one candidate for the given metric from the RaBitQ projection `s` =
 * ⟨q_rot, c⟩, the stored per-vector `scale` and `norm` (= ‖v‖), and the query's
 * precomputed norms.
 *
 * Branch-light: a single `switch` over the (string-literal) metric, each arm a
 * closed form. The estimated dot product `scale·s` is the shared subexpression.
 *
 * Preconditions (validated at the search entry, not re-checked here): `q.qNorm`
 * > 0 and `norm` > 0 for cosine.
 */
export function scoreMetric(
  metric: Distance,
  s: number,
  scale: number,
  norm: number,
  q: QueryNorms,
): MetricScore {
  const estDot = scale * s; // ≈ ⟨q, v⟩
  switch (metric) {
    case 'dot':
      return { value: estDot, rankKey: estDot };
    case 'cosine': {
      const cos = estDot / (q.qNorm * norm);
      return { value: cos, rankKey: cos };
    }
    case 'euclidean': {
      const distSq = q.qNormSq + norm * norm - 2 * estDot;
      return { value: distSq, rankKey: -distSq };
    }
  }
}
