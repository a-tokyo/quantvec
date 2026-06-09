// Fast Walsh–Hadamard Transform (FWHT) primitive for quantvec.
//
// Why this exists: a Randomized Hadamard Transform built from a few rounds of
// (random sign flip + FWHT) is an orthonormal map that spreads a vector's energy
// uniformly across coordinates — the same "make every coordinate look like the
// canonical marginal" property the dense Householder rotation provides (see
// ./rotation), but in O(d·log d) instead of O(d²) per apply and with no O(d³) build.
// It is *exact and norm-preserving only when the length is a power of two* (no
// zero-padding/truncation), which is precisely where ./rotation uses it.

/** True iff `n` is a positive power of two. */
export function isPow2(n: number): boolean {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/** Smallest power of two ≥ `n` (n ≥ 1). */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place, unnormalized Fast Walsh–Hadamard Transform. `a.length` must be a power
 * of two. Applying it twice scales by `a.length` (Hᵀ = H, H·H = n·I); divide by
 * `√n` once to make it the orthonormal transform.
 *
 * @throws {RangeError} if the length is not a power of two.
 */
export function fwht(a: Float64Array): void {
  const n = a.length;
  if (!isPow2(n)) {
    throw new RangeError(`fwht length must be a power of two, got ${n}`);
  }
  for (let len = 1; len < n; len <<= 1) {
    for (let i = 0; i < n; i += len << 1) {
      for (let j = i; j < i + len; j++) {
        const x = a[j]!;
        const y = a[j + len]!;
        a[j] = x + y;
        a[j + len] = x - y;
      }
    }
  }
}
