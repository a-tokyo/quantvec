// WASM kernel loader + per-index scoring handle for quantvec.
//
// Loads the AssemblyScript scoring kernel (assembly/index.ts), inlined as base64
// (./wasm-binary), with full graceful degradation: if WebAssembly is unavailable or
// anything fails, {@link WasmKernel.create} returns null and the index falls back to
// the pure-TS scalar scan (src/core/search.ts) — the kernel is an optimization, never
// a dependency. The kernel computes the SAME projection S[j] = Σ_i lut[i·levels+code]
// as the scalar oracle (exact f32), so results match; the win is the resident-codes
// scan in linear memory (codes uploaded once per mutation, not per query).

import { WASM_BASE64 } from './wasm-binary';

/** Kernel ABI the loader expects (assembly/index.ts `abiVersion`). */
const ABI_VERSION = 2;
/** WebAssembly memory page size (64 KiB). */
const PAGE_BYTES = 65536;

/** Decode the inlined base64 kernel to bytes (atob is universal across runtimes). */
function decodeWasm(): Uint8Array {
  const binary = atob(WASM_BASE64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// undefined = not yet attempted, null = unavailable, Module = compiled once and reused.
let moduleCache: WebAssembly.Module | null | undefined;

/** Compile the kernel module once; null if WebAssembly is unavailable or invalid. */
function getModule(): WebAssembly.Module | null {
  if (moduleCache !== undefined) return moduleCache;
  try {
    if (typeof WebAssembly === 'undefined') {
      moduleCache = null;
    } else {
      const bytes = decodeWasm();
      moduleCache = WebAssembly.validate(bytes) ? new WebAssembly.Module(bytes) : null;
    }
  } catch {
    moduleCache = null;
  }
  return moduleCache;
}

interface KernelExports {
  memory: WebAssembly.Memory;
  heapBase(): number;
  abiVersion(): number;
  scoreInto(
    codesPtr: number,
    n: number,
    dim: number,
    levels: number,
    lutPtr: number,
    outPtr: number,
  ): void;
}

function align4(x: number): number {
  return (x + 3) & ~3;
}
function align8(x: number): number {
  return (x + 7) & ~7;
}

/**
 * A per-index WASM scoring kernel. Holds its own linear memory with the index's codes
 * resident (uploaded once per mutation via {@link uploadCodes}); {@link score} runs the
 * scan for one query's lookup table. Construct with {@link WasmKernel.create} (null if
 * WebAssembly is unavailable).
 */
export class WasmKernel {
  readonly #ex: KernelExports;
  readonly #heapBase: number;
  #n = -1;
  #dim = -1;
  #levels = -1;
  #codesOff = 0;
  #lutOff = 0;
  #outOff = 0;

  private constructor(ex: KernelExports) {
    this.#ex = ex;
    this.#heapBase = ex.heapBase();
  }

  /** Instantiate the kernel, or return null if WebAssembly is unavailable/incompatible. */
  static create(): WasmKernel | null {
    const mod = getModule();
    if (mod === null) return null;
    try {
      const ex = new WebAssembly.Instance(mod).exports as unknown as KernelExports;
      if (typeof ex.scoreInto !== 'function' || ex.abiVersion() !== ABI_VERSION) return null;
      return new WasmKernel(ex);
    } catch {
      return null;
    }
  }

  /** Lay out [codes | lut | out] for (n, dim, levels) and grow memory to fit. */
  prepare(n: number, dim: number, levels: number): void {
    if (n === this.#n && dim === this.#dim && levels === this.#levels) return;
    this.#n = n;
    this.#dim = dim;
    this.#levels = levels;
    this.#codesOff = align4(this.#heapBase);
    this.#lutOff = align4(this.#codesOff + n * dim);
    this.#outOff = align8(this.#lutOff + dim * levels * 4); // out is f64 (8-aligned)
    const end = this.#outOff + n * 8;
    const have = this.#ex.memory.buffer.byteLength;
    // grow() can only fail under OOM, but the same codes already live in JS memory
    // (this.#codes), so a JS allocation would have failed first — i.e. this never
    // returns -1 in practice. We deliberately don't add an untestable guard here.
    if (end > have) this.#ex.memory.grow(Math.ceil((end - have) / PAGE_BYTES));
  }

  /** Write the resident codes (length must be n·dim for the prepared layout). */
  uploadCodes(codes: Uint8Array): void {
    new Uint8Array(this.#ex.memory.buffer, this.#codesOff, this.#n * this.#dim).set(codes);
  }

  /** Write the per-query `lut`, run the scan, and read S[] (f64) into `out` (length n). */
  score(lut: Float32Array, out: Float64Array): void {
    new Float32Array(this.#ex.memory.buffer, this.#lutOff, this.#dim * this.#levels).set(lut);
    this.#ex.scoreInto(
      this.#codesOff,
      this.#n,
      this.#dim,
      this.#levels,
      this.#lutOff,
      this.#outOff,
    );
    out.set(new Float64Array(this.#ex.memory.buffer, this.#outOff, this.#n));
  }
}
