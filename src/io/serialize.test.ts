import { describe, expect, it } from 'vitest';
import {
  DeserializeError,
  deserializeIndex,
  serializeIndex,
  type IdType,
  type IndexPayload,
  type SerializableIndex,
} from './serialize';
import type { Distance } from '../core/metrics';

const DIM = 8;
const BITS = 2 as const;

/** Build a self-consistent positional payload with deterministic, f32-exact fields. */
function positionalPayload(n: number, metric: Distance = 'cosine', seed = 7): IndexPayload {
  const codes = new Uint8Array(n * DIM);
  for (let i = 0; i < codes.length; i++) codes[i] = i % 4; // values < 2^bits
  const scales = new Float32Array(n);
  const norms = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    scales[i] = 1.5 + i * 0.25; // exact in f32
    norms[i] = 3.5 + i * 0.5;
  }
  return { metric, bits: BITS, dim: DIM, n, seed, codes, scales, norms };
}

/** The 24-byte header field offsets, for tampering. */
const OFF = { version: 4, kind: 5, metric: 6, bits: 7, dim: 8, n: 12 } as const;

/** End of the fixed region (header + PACKED codes + scales + norms). */
function fixedEnd(n: number): number {
  return 24 + (n * DIM * BITS) / 8 + 2 * n * 4;
}

/** Start of the ids section: fixed region + the 1-byte calibration-presence flag
 * (the test payloads carry no calibration, so the flag is a single 0 byte). */
function idsStart(n: number): number {
  return fixedEnd(n) + 1;
}

function expectDeserializeError(bytes: Uint8Array, code: DeserializeError['code']): void {
  let err: unknown;
  try {
    deserializeIndex(bytes);
  } catch (e) {
    err = e;
  }
  expect(err).toBeInstanceOf(DeserializeError);
  expect((err as DeserializeError).code).toBe(code);
}

describe('serialize/deserialize — positional round trip', () => {
  it('round-trips fields for each metric', () => {
    for (const metric of ['dot', 'cosine', 'euclidean'] as const) {
      const p = positionalPayload(3, metric, 11);
      const parsed = deserializeIndex(serializeIndex({ kind: 'positional', ...p }));
      expect(parsed.kind).toBe('positional');
      expect(parsed.metric).toBe(metric);
      expect(parsed.bits).toBe(BITS);
      expect(parsed.dim).toBe(DIM);
      expect(parsed.n).toBe(3);
      expect(parsed.seed).toBe(11);
      expect(Array.from(parsed.codes)).toEqual(Array.from(p.codes));
      expect(Array.from(parsed.scales)).toEqual(Array.from(p.scales));
      expect(Array.from(parsed.norms)).toEqual(Array.from(p.norms));
    }
  });

  it('round-trips an empty index', () => {
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'positional', ...positionalPayload(0) }),
    );
    expect(parsed.n).toBe(0);
    expect(parsed.codes.length).toBe(0);
  });

  it('returns owned copies, not views into the input buffer', () => {
    const bytes = serializeIndex({ kind: 'positional', ...positionalPayload(2) });
    const parsed = deserializeIndex(bytes);
    bytes.fill(0); // mutate the source after parsing
    expect(parsed.codes.some((v) => v !== 0)).toBe(true);
    expect(parsed.scales[0]).toBeCloseTo(1.5);
  });
});

describe('serialize/deserialize — idmap round trip', () => {
  it('round-trips number ids', () => {
    const ids: IdType[] = [10, 20, 30];
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'idmap', ...positionalPayload(3), ids }),
    );
    expect(parsed.kind).toBe('idmap');
    expect((parsed as { ids: IdType[] }).ids).toEqual(ids);
  });

  it('round-trips string and bigint ids (incl. unicode and empty string)', () => {
    const ids: IdType[] = ['alpha', '', '日本語', 9007199254740993n, -42n];
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'idmap', ...positionalPayload(5), ids }),
    );
    expect((parsed as { ids: IdType[] }).ids).toEqual(ids);
  });

  it('round-trips an empty idmap index', () => {
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'idmap', ...positionalPayload(0), ids: [] }),
    );
    expect(parsed.kind).toBe('idmap');
    expect((parsed as { ids: IdType[] }).ids).toEqual([]);
  });
});

describe('deserialize — untrusted input validation', () => {
  const validPos = (): Uint8Array =>
    serializeIndex({ kind: 'positional', ...positionalPayload(2) });
  const validIdmap = (ids: IdType[]): Uint8Array =>
    serializeIndex({ kind: 'idmap', ...positionalPayload(ids.length), ids });

  it('rejects a buffer shorter than the header', () => {
    expectDeserializeError(new Uint8Array(10), 'TOO_SHORT');
  });

  it('rejects a bad magic', () => {
    const b = validPos();
    b[0] = 0x00;
    expectDeserializeError(b, 'BAD_MAGIC');
  });

  it('rejects an unsupported version', () => {
    const b = validPos();
    b[OFF.version] = 2;
    expectDeserializeError(b, 'BAD_VERSION');
  });

  it('rejects an unknown kind', () => {
    const b = validPos();
    b[OFF.kind] = 9;
    expectDeserializeError(b, 'BAD_KIND');
  });

  it('rejects an unknown metric', () => {
    const b = validPos();
    b[OFF.metric] = 5;
    expectDeserializeError(b, 'BAD_METRIC');
  });

  it('rejects bad bits', () => {
    const b = validPos();
    b[OFF.bits] = 7;
    expectDeserializeError(b, 'BAD_BITS');
  });

  it('rejects a dim that is not a positive multiple of 8', () => {
    const b = validPos();
    new DataView(b.buffer).setUint32(OFF.dim, 7, true);
    expectDeserializeError(b, 'BAD_DIM');
  });

  it('rejects a zero dim', () => {
    const b = validPos();
    new DataView(b.buffer).setUint32(OFF.dim, 0, true);
    expectDeserializeError(b, 'BAD_DIM');
  });

  it('rejects a non-finite seed', () => {
    const b = validPos();
    new DataView(b.buffer).setFloat64(16, NaN, true);
    expectDeserializeError(b, 'BAD_SEED');
  });

  it('rejects a positional buffer of the wrong total length', () => {
    const b = validPos();
    expectDeserializeError(new Uint8Array([...b, 0]), 'BAD_LENGTH'); // trailing byte
    expectDeserializeError(b.slice(0, b.length - 1), 'BAD_LENGTH'); // truncated
  });

  it('rejects an idmap whose body region overflows the buffer', () => {
    const b = validIdmap([1, 2]);
    expectDeserializeError(b.slice(0, fixedEnd(2) - 4), 'BAD_LENGTH');
  });

  it('rejects an idmap truncated before an id', () => {
    const b = validIdmap([1, 2]);
    expectDeserializeError(b.slice(0, idsStart(2)), 'BAD_LENGTH'); // no id bytes at all
  });

  it('rejects an idmap with a truncated number id', () => {
    const b = validIdmap([1, 2]);
    expectDeserializeError(b.slice(0, idsStart(2) + 5), 'BAD_LENGTH'); // tag + partial f64
  });

  it('rejects an idmap with trailing bytes after the ids', () => {
    const b = validIdmap([1, 2]);
    expectDeserializeError(new Uint8Array([...b, 0]), 'BAD_LENGTH');
  });

  it('rejects a truncated id length field', () => {
    const b = validIdmap(['hi']); // ...[flag][tag=1][u32 len][utf8]
    // Keep header+body+flag+tag+2 of the 4 length bytes.
    expectDeserializeError(b.slice(0, idsStart(1) + 1 + 2), 'BAD_LENGTH');
  });

  it('rejects an id payload length that exceeds the buffer', () => {
    const b = validIdmap(['hi']);
    new DataView(b.buffer).setUint32(idsStart(1) + 1, 0xffffffff, true); // corrupt len
    expectDeserializeError(b, 'BAD_LENGTH');
  });

  it('rejects an unknown id tag', () => {
    const b = validIdmap([7]);
    b[idsStart(1)] = 99; // tag byte
    expectDeserializeError(b, 'BAD_ID');
  });

  it('rejects a malformed bigint payload', () => {
    const b = validIdmap(['notanumber']); // string id, length 10
    b[idsStart(1)] = 2; // re-tag as bigint → BigInt('notanumber') throws
    expectDeserializeError(b, 'BAD_ID');
  });

  it('rejects a non-canonical bigint encoding', () => {
    const b = validIdmap(['007']); // valid string id "007"
    b[idsStart(1)] = 2; // re-tag as bigint → BigInt('007')=7n, '7' !== '007'
    expectDeserializeError(b, 'BAD_ID');
  });

  it('rejects invalid UTF-8 in a string id', () => {
    const b = validIdmap(['hi']); // [flag][tag=1][u32 len=2]['h','i'] at idsStart(1)
    b[idsStart(1) + 1 + 4] = 0xff; // first utf8 byte → invalid lead byte
    expectDeserializeError(b, 'BAD_ID');
  });

  it('rejects duplicate ids (would break the id↔slot bijection)', () => {
    const bytes = serializeIndex({ kind: 'idmap', ...positionalPayload(2), ids: [1, 1] });
    expectDeserializeError(bytes, 'BAD_ID');
  });

  it('rejects an invalid calibration flag', () => {
    const b = validPos();
    b[fixedEnd(2)] = 2; // calibration flag must be 0 or 1
    expectDeserializeError(b, 'BAD_LENGTH');
  });

  it('rejects a truncated calibration section', () => {
    const cal = { shift: new Float32Array(DIM).fill(0.1), scale: new Float32Array(DIM).fill(1.2) };
    const b = serializeIndex({ kind: 'positional', ...positionalPayload(2), calibration: cal });
    expectDeserializeError(b.slice(0, fixedEnd(2) + 1 + 4), 'BAD_LENGTH'); // flag + 1 float only
  });
});

describe('serialize/deserialize — TQ+ calibration round trip', () => {
  it('round-trips the calibration vectors (positional)', () => {
    const cal = {
      shift: Float32Array.from({ length: DIM }, (_, i) => i * 0.0625),
      scale: Float32Array.from({ length: DIM }, (_, i) => 1 + i * 0.125),
    };
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'positional', ...positionalPayload(3), calibration: cal }),
    );
    expect(parsed.calibration).toBeDefined();
    expect(Array.from(parsed.calibration!.shift)).toEqual(Array.from(cal.shift));
    expect(Array.from(parsed.calibration!.scale)).toEqual(Array.from(cal.scale));
  });

  it('round-trips calibration alongside ids (idmap)', () => {
    const cal = {
      shift: new Float32Array(DIM).fill(0.5),
      scale: new Float32Array(DIM).fill(2),
    };
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'idmap', ...positionalPayload(2), ids: [9, 8], calibration: cal }),
    );
    expect(parsed.kind).toBe('idmap');
    expect((parsed as { ids: IdType[] }).ids).toEqual([9, 8]);
    expect(Array.from(parsed.calibration!.scale)).toEqual(Array.from(cal.scale));
  });

  it('omits the calibration when absent (flag 0)', () => {
    const parsed = deserializeIndex(
      serializeIndex({ kind: 'positional', ...positionalPayload(2) }),
    );
    expect(parsed.calibration).toBeUndefined();
  });

  it('rejects a calibration with a zero scale (would divide by zero at search time)', () => {
    const cal = { shift: new Float32Array(DIM).fill(0.1), scale: new Float32Array(DIM).fill(1.2) };
    const b = serializeIndex({ kind: 'positional', ...positionalPayload(2), calibration: cal });
    // Layout: fixedEnd | 1 flag byte | DIM shift floats | DIM scale floats | ids...
    const scaleStart = fixedEnd(2) + 1 + DIM * 4;
    const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
    view.setFloat32(scaleStart, 0, true); // scale[0] = 0
    expectDeserializeError(b, 'BAD_CALIBRATION');
  });

  it('rejects a calibration with a non-finite shift/scale', () => {
    const cal = { shift: new Float32Array(DIM).fill(0.1), scale: new Float32Array(DIM).fill(1.2) };
    const b = serializeIndex({ kind: 'positional', ...positionalPayload(2), calibration: cal });
    const shiftStart = fixedEnd(2) + 1;
    const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
    view.setFloat32(shiftStart, NaN, true); // shift[0] = NaN
    expectDeserializeError(b, 'BAD_CALIBRATION');
  });
});

describe('serialize — input typing', () => {
  it('accepts the discriminated union without leaking ids on the positional branch', () => {
    const payload: SerializableIndex = { kind: 'positional', ...positionalPayload(1) };
    expect(serializeIndex(payload)).toBeInstanceOf(Uint8Array);
  });
});
