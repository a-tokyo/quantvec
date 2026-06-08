// quantvec — AssemblyScript SIMD scoring kernel.
//
// The real v128 nibble-LUT scoring kernel is implemented in the WASM-SIMD wave.
// Until then this placeholder keeps `bun run build:wasm` green and documents the
// contract: the pure-TS scalar kernel in `src/core/search.ts` is the correctness
// oracle this module must match within floating-point tolerance.
export function kernelAbiVersion(): i32 {
  return 1;
}
