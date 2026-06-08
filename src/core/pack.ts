// Tight bit-packing of per-coordinate quantizer codes for quantvec.
//
// Why this exists: the encoder (see ./encode) turns each vector into a
// Uint8Array of `dim` small codes, one per coordinate, each in [0, 2^bits-1]
// with bits ∈ {2,3,4}. Storing one byte per code wastes 4-6 bits per
// coordinate; for the flat quantized index (docs/research/architecture.md) the
// whole point is that codes are tiny, so we pack them tightly: `bits` per code,
// LSB-first, flowing across byte boundaries.
//
// Layout (LSB-first, little-endian bit order within the stream):
//   code 0 occupies bit positions [0, bits), code 1 [bits, 2*bits), … where bit
//   position p lives in byte ⌊p/8⌋ at intra-byte shift (p mod 8). A code that
//   straddles a byte boundary has its low part in the earlier byte and its high
//   part in the next byte. This is the simplest correct, round-trippable scheme;
//   the SIMD/LUT-blocked nibble layout used by the fast scoring kernel is a
//   LATER wave and is intentionally NOT implemented here. (At 2 and 4 bits the
//   layout never straddles a byte; at 3 bits it does — the cross-byte logic
//   below handles all three uniformly.)
//
// Everything is integer-exact and uses typed arrays; no allocation beyond the
// single output buffer.

/** Discriminated, code-tagged error for the packing module. */
export class PackError extends Error {
  readonly code: 'INVALID_BITS' | 'INVALID_CODE' | 'INVALID_LENGTH';
  constructor(code: PackError['code'], message: string) {
    super(message);
    this.name = 'PackError';
    this.code = code;
  }
}

/** Supported per-coordinate code widths (matches the codebook's `Bits`). */
export type PackBits = 2 | 3 | 4;

/** Reject bit-widths outside {2, 3, 4}. */
function validateBits(bits: number): asserts bits is PackBits {
  if (bits !== 2 && bits !== 3 && bits !== 4) {
    throw new PackError('INVALID_BITS', `bits must be one of {2, 3, 4}, got ${bits}`);
  }
}

/**
 * Number of bytes needed to tightly pack `n` codes of `bits` bits each:
 * ⌈n·bits / 8⌉.
 *
 * @throws {PackError} `'INVALID_BITS'` if bits ∉ {2,3,4}; `'INVALID_LENGTH'` if
 *   `n` is not a non-negative integer.
 */
export function packedBytes(n: number, bits: PackBits): number {
  validateBits(bits);
  if (!Number.isInteger(n) || n < 0) {
    throw new PackError('INVALID_LENGTH', `n must be a non-negative integer, got ${n}`);
  }
  return (n * bits + 7) >> 3;
}

/**
 * Pack `codes` (one value per coordinate, each in [0, 2^bits-1]) into a tightly
 * bit-packed Uint8Array of length {@link packedBytes}(codes.length, bits).
 *
 * Codes are laid out LSB-first and flow across byte boundaries (see file header).
 *
 * @throws {PackError} `'INVALID_BITS'` if bits ∉ {2,3,4}; `'INVALID_CODE'` if any
 *   code is out of range (negative, non-integer, or ≥ 2^bits).
 */
export function packCodes(codes: Uint8Array, bits: PackBits): Uint8Array {
  validateBits(bits);
  const n = codes.length;
  const maxCode = (1 << bits) - 1;
  const out = new Uint8Array(packedBytes(n, bits));

  let bitPos = 0;
  for (let i = 0; i < n; i++) {
    const code = codes[i]!;
    // Uint8Array guarantees an integer in [0,255]; we only need the upper bound
    // against 2^bits, but check both for a precise typed error on bad input.
    if (code < 0 || code > maxCode) {
      throw new PackError(
        'INVALID_CODE',
        `code at index ${i} must be in [0, ${maxCode}] for ${bits}-bit packing, got ${code}`,
      );
    }
    // Write `bits` bits of `code` starting at absolute bit position `bitPos`,
    // splitting across the byte boundary when it straddles one.
    let remaining = bits;
    let value = code;
    let pos = bitPos;
    while (remaining > 0) {
      const byteIndex = pos >> 3;
      const shift = pos & 7; // intra-byte offset
      const free = 8 - shift; // bits left in this byte
      const take = remaining < free ? remaining : free;
      // Mask off the low `take` bits of `value` and place them at `shift`.
      const chunk = value & ((1 << take) - 1);
      out[byteIndex]! |= chunk << shift;
      value >>= take;
      remaining -= take;
      pos += take;
    }
    bitPos += bits;
  }
  return out;
}

/**
 * Inverse of {@link packCodes}: unpack `n` codes of `bits` bits each from a
 * tightly bit-packed buffer.
 *
 * @throws {PackError} `'INVALID_BITS'` if bits ∉ {2,3,4}; `'INVALID_LENGTH'` if
 *   `n` is invalid or `packed` is too short to hold `n` codes.
 */
export function unpackCodes(packed: Uint8Array, n: number, bits: PackBits): Uint8Array {
  validateBits(bits);
  if (!Number.isInteger(n) || n < 0) {
    throw new PackError('INVALID_LENGTH', `n must be a non-negative integer, got ${n}`);
  }
  const need = packedBytes(n, bits);
  if (packed.length < need) {
    throw new PackError(
      'INVALID_LENGTH',
      `packed buffer too short: need ${need} bytes for ${n} ${bits}-bit codes, got ${packed.length}`,
    );
  }

  const out = new Uint8Array(n);
  let bitPos = 0;
  for (let i = 0; i < n; i++) {
    let remaining = bits;
    let pos = bitPos;
    let value = 0;
    let collected = 0; // bits already collected into `value`
    while (remaining > 0) {
      const byteIndex = pos >> 3;
      const shift = pos & 7;
      const avail = 8 - shift;
      const take = remaining < avail ? remaining : avail;
      const chunk = (packed[byteIndex]! >> shift) & ((1 << take) - 1);
      value |= chunk << collected;
      collected += take;
      remaining -= take;
      pos += take;
    }
    out[i] = value;
    bitPos += bits;
  }
  return out;
}
