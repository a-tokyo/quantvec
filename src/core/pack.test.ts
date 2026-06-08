import { describe, expect, it } from 'vitest';
import { packCodes, unpackCodes, packedBytes, PackError } from './pack';
import { createRng } from './rng';

/** Build n random valid codes for the given bit-width using a seeded RNG. */
function randomCodes(n: number, bits: 2 | 3 | 4, rng: ReturnType<typeof createRng>): Uint8Array {
  const max = 1 << bits; // exclusive upper bound
  const codes = new Uint8Array(n);
  for (let i = 0; i < n; i++) codes[i] = rng.nextU32() % max;
  return codes;
}

describe('packedBytes', () => {
  it('returns ⌈n·bits/8⌉', () => {
    expect(packedBytes(0, 2)).toBe(0);
    expect(packedBytes(8, 2)).toBe(2); // 16 bits
    expect(packedBytes(1, 3)).toBe(1); // 3 bits → 1 byte
    expect(packedBytes(8, 3)).toBe(3); // 24 bits
    expect(packedBytes(9, 3)).toBe(4); // 27 bits → 4 bytes
    expect(packedBytes(2, 4)).toBe(1); // 8 bits
    expect(packedBytes(7, 4)).toBe(4); // 28 bits → 4 bytes
  });

  it('rejects invalid bits and invalid n', () => {
    for (const bad of [0, 1, 5, 2.5]) {
      let err: unknown;
      try {
        packedBytes(8, bad as 2);
      } catch (e) {
        err = e;
      }
      expect((err as PackError).code).toBe('INVALID_BITS');
    }
    for (const badN of [-1, 1.5, NaN]) {
      let err: unknown;
      try {
        packedBytes(badN, 2);
      } catch (e) {
        err = e;
      }
      expect((err as PackError).code).toBe('INVALID_LENGTH');
    }
  });
});

describe('pack round-trip (the oracle)', () => {
  for (const bits of [2, 3, 4] as const) {
    for (const n of [1, 7, 8, 9, 64, 100]) {
      it(`unpack(pack(codes)) === codes for bits=${bits}, n=${n}`, () => {
        const rng = createRng(1000 * bits + n);
        // Repeat with several random draws to exercise many code patterns.
        for (let trial = 0; trial < 20; trial++) {
          const codes = randomCodes(n, bits, rng);
          const packed = packCodes(codes, bits);
          // packedBytes matches the produced buffer length.
          expect(packed.length).toBe(packedBytes(n, bits));
          const back = unpackCodes(packed, n, bits);
          expect(Array.from(back)).toEqual(Array.from(codes));
        }
      });
    }
  }

  it('round-trips n=0 (empty)', () => {
    for (const bits of [2, 3, 4] as const) {
      const packed = packCodes(new Uint8Array(0), bits);
      expect(packed.length).toBe(0);
      expect(unpackCodes(packed, 0, bits).length).toBe(0);
    }
  });

  it('round-trips the all-max and all-zero patterns', () => {
    for (const bits of [2, 3, 4] as const) {
      const n = 33;
      const maxV = (1 << bits) - 1;
      for (const fill of [0, maxV]) {
        const codes = new Uint8Array(n).fill(fill);
        const back = unpackCodes(packCodes(codes, bits), n, bits);
        expect(Array.from(back)).toEqual(Array.from(codes));
      }
    }
  });
});

describe('packCodes — validation', () => {
  it('throws INVALID_CODE for out-of-range code', () => {
    // For 2-bit, max valid is 3; 4 is out of range.
    const codes = new Uint8Array([0, 1, 4]);
    let err: unknown;
    try {
      packCodes(codes, 2);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(PackError);
    expect((err as PackError).code).toBe('INVALID_CODE');
  });

  it('throws INVALID_CODE for 3-bit code ≥ 8', () => {
    const codes = new Uint8Array([7, 8]);
    let err: unknown;
    try {
      packCodes(codes, 3);
    } catch (e) {
      err = e;
    }
    expect((err as PackError).code).toBe('INVALID_CODE');
  });

  it('throws INVALID_BITS for unsupported width', () => {
    let err: unknown;
    try {
      packCodes(new Uint8Array([0]), 1 as 2);
    } catch (e) {
      err = e;
    }
    expect((err as PackError).code).toBe('INVALID_BITS');
  });
});

describe('unpackCodes — validation', () => {
  it('throws INVALID_LENGTH when buffer too short', () => {
    const packed = new Uint8Array(1); // 8 bits
    let err: unknown;
    try {
      unpackCodes(packed, 9, 2); // needs 18 bits → 3 bytes
    } catch (e) {
      err = e;
    }
    expect((err as PackError).code).toBe('INVALID_LENGTH');
  });

  it('throws INVALID_LENGTH for invalid n', () => {
    for (const badN of [-1, 2.5, NaN]) {
      let err: unknown;
      try {
        unpackCodes(new Uint8Array(8), badN, 2);
      } catch (e) {
        err = e;
      }
      expect((err as PackError).code).toBe('INVALID_LENGTH');
    }
  });

  it('throws INVALID_BITS for unsupported width', () => {
    let err: unknown;
    try {
      unpackCodes(new Uint8Array(1), 1, 5 as 2);
    } catch (e) {
      err = e;
    }
    expect((err as PackError).code).toBe('INVALID_BITS');
  });

  it('tolerates a buffer longer than required', () => {
    const codes = new Uint8Array([1, 2, 3]);
    const packed = packCodes(codes, 2);
    const padded = new Uint8Array(packed.length + 4);
    padded.set(packed);
    expect(Array.from(unpackCodes(padded, 3, 2))).toEqual([1, 2, 3]);
  });
});
