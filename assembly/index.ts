// quantvec — AssemblyScript scoring kernel (WASM acceleration of the flat scan).
//
// The pure-TS scalar kernel in src/core/search.ts is the correctness ORACLE; this
// module computes the identical per-vector projection
//     S[j] = Σ_i lut[i*levels + codes[j*dim + i]]
// (exact f32 math — same operations, same result) over codes/lut/out laid out in
// linear memory by the JS loader (src/wasm/kernel.ts). The metric, per-vector scale,
// mask, and top-k stay in JS; this kernel is just the O(n·dim) inner sum, which is
// the hot path. Codes are uploaded once per database (re-uploaded only after a
// mutation), so there is no per-query copy — the win is the resident scan in wasm.
//
// Memory layout (all offsets are byte pointers the loader chooses ≥ __heap_base):
//   codes : n·dim u8   ·   lut : dim·levels f32   ·   out : n f64

/** Start of memory the JS side may use (everything below is wasm static data). */
export function heapBase(): i32 {
  return <i32>__heap_base;
}

/** ABI version so the loader can reject a stale inlined binary. */
export function abiVersion(): i32 {
  return 2;
}

/**
 * Fill `out[j] = Σ_i lut[i*levels + codes[j*dim+i]]` for j in [0, n).
 * Pointers are byte offsets into linear memory; `lut` is f32 (4-aligned), `out` is
 * f64 (8-aligned). The sum accumulates in f64 over the same coordinate order as the
 * scalar oracle (src/core/search.ts), so the result is bit-identical, not merely
 * close — the WASM path is an exact acceleration, not an approximation.
 */
export function scoreInto(
  codesPtr: usize,
  n: i32,
  dim: i32,
  levels: i32,
  lutPtr: usize,
  outPtr: usize,
): void {
  for (let j = 0; j < n; j++) {
    let s: f64 = 0;
    const rowBase: usize = codesPtr + <usize>(j * dim);
    for (let i = 0; i < dim; i++) {
      const code = <i32>load<u8>(rowBase + <usize>i);
      s += <f64>load<f32>(lutPtr + ((<usize>(i * levels + code)) << 2));
    }
    store<f64>(outPtr + ((<usize>j) << 3), s);
  }
}
