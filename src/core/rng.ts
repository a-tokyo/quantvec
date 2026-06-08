// Deterministic, isomorphic seeded RNG for quantvec.
//
// The library's quantization is data-oblivious: the random rotation and any
// stochastic step must be reproducible bit-for-bit across Node, Bun, browsers,
// and edge runtimes. We therefore avoid `Math.random` entirely and use
// integer-exact algorithms over BigInt (64-bit) so the generated stream depends
// only on the seed, never on the host's float behavior.
//
// Algorithm: SplitMix64 expands the user seed into the 256-bit state of
// xoshiro256** (Blackman & Vigna, https://prng.di.unimi.it/). xoshiro256** has
// excellent statistical quality and a 2^256 period. Gaussians use Box–Muller
// with a cached spare value.

const MASK64 = (1n << 64n) - 1n;
const MASK32 = 0xffffffffn;

/** A deterministic pseudo-random number generator. */
export interface Rng {
  /** Next uniform 32-bit unsigned integer in [0, 2^32). */
  nextU32(): number;
  /** Next uniform double in [0, 1). */
  nextFloat(): number;
  /** Next standard normal sample (mean 0, variance 1) via Box–Muller. */
  nextGaussian(): number;
}

/** SplitMix64 step — used purely to seed xoshiro256** from a scalar seed. */
function splitmix64(state: bigint): { value: bigint; next: bigint } {
  const next = (state + 0x9e3779b97f4a7c15n) & MASK64;
  let z = next;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  z = (z ^ (z >> 31n)) & MASK64;
  return { value: z, next };
}

function rotl(x: bigint, k: bigint): bigint {
  return ((x << k) | (x >> (64n - k))) & MASK64;
}

/**
 * Create a deterministic RNG seeded from a number or bigint.
 *
 * The same seed always yields the same stream on every runtime. Number seeds
 * are normalized to a 64-bit unsigned integer (the fractional part, if any, is
 * discarded) before expansion.
 */
export function createRng(seed: number | bigint): Rng {
  // Normalize the seed to an unsigned 64-bit BigInt.
  let s: bigint;
  if (typeof seed === 'bigint') {
    s = seed & MASK64;
  } else {
    if (!Number.isFinite(seed)) {
      throw new RngError('INVALID_SEED', `seed must be finite, got ${seed}`);
    }
    // Truncate toward zero, then wrap into the 64-bit range.
    s = BigInt(Math.trunc(seed)) & MASK64;
  }

  // Expand the scalar seed into xoshiro256**'s 256-bit state via SplitMix64.
  // (Recommended seeding procedure for the xoshiro family.)
  let smState = s;
  const st: bigint[] = [];
  for (let i = 0; i < 4; i++) {
    const r = splitmix64(smState);
    smState = r.next;
    st.push(r.value);
  }
  // Avoid the all-zero state (degenerate for xoshiro).
  if (st[0] === 0n && st[1] === 0n && st[2] === 0n && st[3] === 0n) {
    st[0] = 0x9e3779b97f4a7c15n;
  }

  let s0 = st[0]!;
  let s1 = st[1]!;
  let s2 = st[2]!;
  let s3 = st[3]!;

  // One xoshiro256** step → 64-bit output.
  function next64(): bigint {
    const result = (rotl((s1 * 5n) & MASK64, 7n) * 9n) & MASK64;
    const t = (s1 << 17n) & MASK64;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl(s3, 45n);
    return result;
  }

  // Box–Muller spare-value cache.
  let hasSpare = false;
  let spare = 0;

  function nextU32(): number {
    // Take the high 32 bits — these are the highest-quality bits for xoshiro.
    return Number((next64() >> 32n) & MASK32);
  }

  function nextFloat(): number {
    // 53-bit mantissa from the top 53 bits → uniform double in [0, 1).
    const bits = next64() >> 11n; // 64 - 53 = 11
    return Number(bits) / 9007199254740992; // 2^53
  }

  function nextGaussian(): number {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    // Box–Muller: draw u1 in (0,1] to keep log() finite.
    let u1 = nextFloat();
    if (u1 < Number.EPSILON) u1 = Number.EPSILON;
    const u2 = nextFloat();
    const mag = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    spare = mag * Math.sin(angle);
    hasSpare = true;
    return mag * Math.cos(angle);
  }

  return { nextU32, nextFloat, nextGaussian };
}

/** Discriminated, code-tagged error for RNG construction. */
export class RngError extends Error {
  readonly code: 'INVALID_SEED';
  constructor(code: 'INVALID_SEED', message: string) {
    super(message);
    this.name = 'RngError';
    this.code = code;
  }
}
