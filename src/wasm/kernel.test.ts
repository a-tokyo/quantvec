import { describe, expect, it } from 'vitest';
import { FASTSCAN_BLOCK, WasmKernel } from './kernel';

describe('WasmKernel — availability', () => {
  it('instantiates in this runtime', () => {
    expect(WasmKernel.create()).not.toBeNull();
  });
});

describe('WasmKernel — FastScan (v128 swizzle) matches the exact u8 sum', () => {
  it('acc[v] = Σ_i lut8[i][code[v][i]] for every vector', () => {
    const kernel = WasmKernel.create()!;
    const dim = 6;
    const n = 37; // spans 3 blocks (16 each), last block partly padded
    // Deterministic codes in [0,15] and a u8 LUT in [0,200] (Σ over dim ≤ 1200 < 65535).
    const codes = new Uint8Array(n * dim);
    for (let i = 0; i < codes.length; i++) codes[i] = (i * 7 + 3) % 16;
    const lut8 = new Uint8Array(dim * 16);
    for (let i = 0; i < lut8.length; i++) lut8[i] = (i * 13 + 5) % 201;

    kernel.prepareFastScan(n, dim);
    kernel.uploadBlockedCodes(codes);
    const acc = new Uint16Array(kernel.fastScanBlocks * FASTSCAN_BLOCK);
    kernel.fastScan(lut8, acc);

    for (let v = 0; v < n; v++) {
      let expected = 0;
      for (let i = 0; i < dim; i++) expected += lut8[i * 16 + codes[v * dim + i]!]!;
      expect(acc[v]).toBe(expected);
    }
  });

  it('re-runs with a fresh LUT (resident codes reused)', () => {
    const kernel = WasmKernel.create()!;
    const dim = 8;
    const n = 16;
    const codes = new Uint8Array(n * dim).map((_, i) => (i * 5 + 1) % 16);
    kernel.prepareFastScan(n, dim);
    kernel.uploadBlockedCodes(codes);
    const lut8 = new Uint8Array(dim * 16).map((_, i) => i % 100);
    const acc = new Uint16Array(kernel.fastScanBlocks * FASTSCAN_BLOCK);
    kernel.fastScan(lut8, acc);
    for (let v = 0; v < n; v++) {
      let expected = 0;
      for (let i = 0; i < dim; i++) expected += lut8[i * 16 + codes[v * dim + i]!]!;
      expect(acc[v]).toBe(expected);
    }
  });
});
