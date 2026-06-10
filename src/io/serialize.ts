// quantvec — versioned, self-describing binary serialization for the index classes.
//
// Why this exists: an index is fully reconstructable from (dim, bits, seed) — which
// regenerate the rotation and codebook — plus the compact per-vector codes/scales/
// norms (and, for the id-keyed index, the external ids). This module packs exactly
// those fields into ONE clean, versioned byte layout (D-010: single format, no
// back-compat readers) and parses them back, treating every input buffer as
// UNTRUSTED: each length is validated against the buffer size BEFORE a single bulk
// read or allocation, so a crafted header can never trigger an OOB read or an
// out-of-memory allocation (premortem T6).
//
// Layout (all multi-byte fields little-endian; header = 24 bytes):
//   [0..4)   magic  "QVEC" (0x51 0x56 0x45 0x43)
//   [4]      version u8  (= 2)
//   [5]      kind    u8  (0 = positional, 1 = idmap)
//   [6]      metric  u8  (0 = dot, 1 = cosine, 2 = euclidean)
//   [7]      bits    u8  (2 | 3 | 4)
//   [8..12)  dim     u32
//   [12..16) n       u32
//   [16..24) seed    f64
//   [24..]   codes   (⌈n·dim·bits/8⌉ bytes, bit-packed) · scales (n·f32) · norms (n·f32)
//   [cali]   flag u8 ∈ {0,1} ; 1 → shift (dim·f32) + scale (dim·f32)
//   [ivf]    flag u8 ∈ {0,1} ; 1 → nlist u32 + nprobe u32
//                                  + centroids (nlist·dim·f32) + listForSlot (n·u32)
//   [idmap]  ids: n × { tag u8 ; 0→f64 | 1→(u32 len + utf8) | 2→(u32 len + utf8 of BigInt) }

import type { Calibration } from '../core/calibrate';
import type { Bits } from '../core/codebook';
import type { Distance } from '../core/metrics';
import { packCodes, unpackCodes } from '../core/pack';

/** External-id value types the id-keyed index supports (number is the default). */
export type IdType = number | string | bigint;

/** Discriminated, code-tagged error for malformed/oversized serialized buffers. */
export class DeserializeError extends Error {
  readonly code:
    | 'TOO_SHORT'
    | 'BAD_MAGIC'
    | 'BAD_VERSION'
    | 'BAD_KIND'
    | 'BAD_METRIC'
    | 'BAD_BITS'
    | 'BAD_DIM'
    | 'BAD_SEED'
    | 'BAD_LENGTH'
    | 'BAD_ID'
    | 'BAD_CALIBRATION'
    | 'BAD_IVF';
  constructor(code: DeserializeError['code'], message: string) {
    super(message);
    this.name = 'DeserializeError';
    this.code = code;
  }
}

/** Fields common to both index kinds (the codebook/rotation are rebuilt from these). */
export interface IndexPayload {
  /** Ranking metric the index defaults to. */
  metric: Distance;
  /** Quantizer bit-width. */
  bits: Bits;
  /** Vector dimension d (positive multiple of 8). */
  dim: number;
  /** Live vector count. */
  n: number;
  /** RNG seed of the frozen rotation. */
  seed: number;
  /** Row-major codes, length n·dim. */
  codes: Uint8Array;
  /** Per-vector RaBitQ scales, length n. */
  scales: Float32Array;
  /** Per-vector norms, length n. */
  norms: Float32Array;
  /** Optional frozen TQ+ calibration (shift/scale, length dim each); absent if un-calibrated. */
  calibration?: Calibration;
  /** Optional frozen IVF coarse-quantizer state; absent while the index is flat. */
  ivf?: IvfPayload;
}

/** Serialized IVF coarse-quantizer state (postings are rebuilt from `listForSlot`). */
export interface IvfPayload {
  /** Number of coarse cells; integer in [2, 2^22]. */
  nlist: number;
  /** Default probe breadth; integer in [1, nlist]. */
  nprobe: number;
  /** Row-major nlist·dim cell centroids. */
  centroids: Float32Array;
  /** Owning cell per live slot, length n (each entry < nlist). */
  listForSlot: Int32Array;
}

/** Serialize input for the positional index. */
export interface PositionalPayload extends IndexPayload {
  kind: 'positional';
}

/** Serialize input for the id-keyed index (carries the external id per slot). */
export interface IdMapPayload extends IndexPayload {
  kind: 'idmap';
  /** External id at each slot, length n (slot order matches `codes`/`scales`). */
  ids: readonly IdType[];
}

/** Tagged union accepted by {@link serializeIndex}. */
export type SerializableIndex = PositionalPayload | IdMapPayload;

/** Parsed positional index (owns fresh copies of the typed arrays). */
export interface DeserializedPositional extends IndexPayload {
  kind: 'positional';
}

/** Parsed id-keyed index (owns fresh copies; `ids` rebuilt in slot order). */
export interface DeserializedIdMap extends IndexPayload {
  kind: 'idmap';
  ids: IdType[];
}

/** Tagged union returned by {@link deserializeIndex}. */
export type DeserializedIndex = DeserializedPositional | DeserializedIdMap;

const MAGIC = 0x51564543; // "QVEC" read big-endian as a u32 (matches the byte order written below)
const VERSION = 2;
const HEADER_BYTES = 24;
/** Same bound the index enforces at construction (see ../index/turboquant-index). */
const MAX_NLIST = 1 << 22;
const FLOAT_BYTES = 4; // width of an f32 scale/norm field
const U32_BYTES = 4; // width of the u32 length prefix on a string/bigint id

const METRIC_TO_BYTE: Record<Distance, number> = { dot: 0, cosine: 1, euclidean: 2 };
const BYTE_TO_METRIC: readonly Distance[] = ['dot', 'cosine', 'euclidean'];

const ID_TAG_NUMBER = 0;
const ID_TAG_STRING = 1;
const ID_TAG_BIGINT = 2;

/** Byte length of one encoded id (1-byte tag + payload). */
function idByteLength(
  id: IdType,
  enc: TextEncoder,
): { tag: number; bytes: number; utf8?: Uint8Array } {
  if (typeof id === 'number') return { tag: ID_TAG_NUMBER, bytes: 1 + 8 };
  if (typeof id === 'string') {
    const utf8 = enc.encode(id);
    return { tag: ID_TAG_STRING, bytes: 1 + U32_BYTES + utf8.length, utf8 };
  }
  // bigint — the only remaining IdType (callers validate id types at the boundary).
  const utf8 = enc.encode(id.toString());
  return { tag: ID_TAG_BIGINT, bytes: 1 + U32_BYTES + utf8.length, utf8 };
}

/**
 * Pack an index payload into a single versioned byte buffer. The inverse of
 * {@link deserializeIndex}. Inputs come from the index classes and are assumed
 * internally consistent (lengths match `n`/`dim`); untrusted validation lives on
 * the read path.
 */
export function serializeIndex(payload: SerializableIndex): Uint8Array {
  const { kind, metric, bits, dim, n, seed, codes, scales, norms } = payload;

  // Codes are stored tightly bit-packed (bits per coordinate). dim is a multiple of
  // 8, so n·dim·bits is a whole number of bytes (no padding waste).
  const packedCodes = packCodes(codes, bits);
  // Calibration section: 1 presence byte, then (if present) shift[dim] + scale[dim] f32.
  const calibration = payload.calibration;
  const caliBytes = 1 + (calibration ? 2 * dim * FLOAT_BYTES : 0);
  // IVF section: 1 presence byte, then (if present) nlist + nprobe u32, centroids, listForSlot.
  const ivf = payload.ivf;
  const ivfBytes = 1 + (ivf ? 2 * U32_BYTES + ivf.nlist * dim * FLOAT_BYTES + n * U32_BYTES : 0);
  const bodyBytes =
    packedCodes.length + (scales.length + norms.length) * FLOAT_BYTES + caliBytes + ivfBytes;

  // For the id-keyed index, pre-encode the ids so we can size the buffer exactly.
  const enc = new TextEncoder();
  const encodedIds = kind === 'idmap' ? payload.ids.map((id) => idByteLength(id, enc)) : undefined;
  const idsBytes = encodedIds ? encodedIds.reduce((sum, e) => sum + e.bytes, 0) : 0;

  const out = new Uint8Array(HEADER_BYTES + bodyBytes + idsBytes);
  const dv = new DataView(out.buffer);

  // ── Header ────────────────────────────────────────────────────────────────
  dv.setUint32(0, MAGIC, false); // big-endian so bytes read as 'Q','V','E','C'
  dv.setUint8(4, VERSION);
  dv.setUint8(5, kind === 'positional' ? 0 : 1);
  dv.setUint8(6, METRIC_TO_BYTE[metric]);
  dv.setUint8(7, bits);
  dv.setUint32(8, dim, true);
  dv.setUint32(12, n, true);
  dv.setFloat64(16, seed, true);

  // ── Body: packed codes, then scales, then norms ─────────────────────────────
  let off = HEADER_BYTES;
  out.set(packedCodes, off);
  off += packedCodes.length;
  for (let i = 0; i < scales.length; i++) {
    dv.setFloat32(off, scales[i]!, true);
    off += FLOAT_BYTES;
  }
  for (let i = 0; i < norms.length; i++) {
    dv.setFloat32(off, norms[i]!, true);
    off += FLOAT_BYTES;
  }

  // ── TQ+ calibration (presence byte, then shift[dim] + scale[dim] if present) ──
  if (calibration) {
    dv.setUint8(off, 1);
    off += 1;
    for (let i = 0; i < dim; i++) {
      dv.setFloat32(off, calibration.shift[i]!, true);
      off += FLOAT_BYTES;
    }
    for (let i = 0; i < dim; i++) {
      dv.setFloat32(off, calibration.scale[i]!, true);
      off += FLOAT_BYTES;
    }
  } else {
    dv.setUint8(off, 0);
    off += 1;
  }

  // ── IVF (presence byte, then nlist/nprobe + centroids + listForSlot if present) ──
  if (ivf) {
    dv.setUint8(off, 1);
    off += 1;
    dv.setUint32(off, ivf.nlist, true);
    off += U32_BYTES;
    dv.setUint32(off, ivf.nprobe, true);
    off += U32_BYTES;
    for (let i = 0; i < ivf.centroids.length; i++) {
      dv.setFloat32(off, ivf.centroids[i]!, true);
      off += FLOAT_BYTES;
    }
    for (let i = 0; i < ivf.listForSlot.length; i++) {
      dv.setUint32(off, ivf.listForSlot[i]!, true);
      off += U32_BYTES;
    }
  } else {
    dv.setUint8(off, 0);
    off += 1;
  }

  // ── Ids (idmap only) ────────────────────────────────────────────────────────
  if (encodedIds) {
    const ids = (payload as IdMapPayload).ids;
    for (let i = 0; i < encodedIds.length; i++) {
      const e = encodedIds[i]!;
      dv.setUint8(off, e.tag);
      off += 1;
      if (e.tag === ID_TAG_NUMBER) {
        dv.setFloat64(off, ids[i] as number, true);
        off += 8;
      } else {
        const utf8 = e.utf8!;
        dv.setUint32(off, utf8.length, true);
        off += U32_BYTES;
        out.set(utf8, off);
        off += utf8.length;
      }
    }
  }

  return out;
}

/** Read a u32 length and verify the payload it describes fits within `limit`. */
function readBoundedLength(dv: DataView, off: number, limit: number): number {
  if (off + U32_BYTES > limit) {
    throw new DeserializeError('BAD_LENGTH', 'truncated id length field');
  }
  const len = dv.getUint32(off, true);
  if (off + U32_BYTES + len > limit) {
    throw new DeserializeError('BAD_LENGTH', `id payload length ${len} exceeds buffer`);
  }
  return len;
}

/**
 * Parse a buffer produced by {@link serializeIndex} back into a typed payload,
 * treating the input as fully untrusted: the magic, version, and every field/length
 * are validated against the buffer size before any bulk read or allocation.
 *
 * @throws {DeserializeError} on a malformed, truncated, or oversized buffer.
 */
export function deserializeIndex(bytes: Uint8Array): DeserializedIndex {
  if (bytes.length < HEADER_BYTES) {
    throw new DeserializeError('TOO_SHORT', `buffer ${bytes.length} < header ${HEADER_BYTES}`);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (dv.getUint32(0, false) !== MAGIC) {
    throw new DeserializeError('BAD_MAGIC', 'not a quantvec index buffer');
  }
  const version = dv.getUint8(4);
  if (version !== VERSION) {
    throw new DeserializeError('BAD_VERSION', `unsupported format version ${version}`);
  }
  const kindByte = dv.getUint8(5);
  if (kindByte !== 0 && kindByte !== 1) {
    throw new DeserializeError('BAD_KIND', `unknown index kind ${kindByte}`);
  }
  const metricByte = dv.getUint8(6);
  if (metricByte > 2) {
    throw new DeserializeError('BAD_METRIC', `unknown metric ${metricByte}`);
  }
  const metric = BYTE_TO_METRIC[metricByte]!;
  const bits = dv.getUint8(7);
  if (bits !== 2 && bits !== 3 && bits !== 4) {
    throw new DeserializeError('BAD_BITS', `bits must be one of {2,3,4}, got ${bits}`);
  }
  const dim = dv.getUint32(8, true);
  if (dim <= 0 || dim % 8 !== 0) {
    throw new DeserializeError('BAD_DIM', `dim must be a positive multiple of 8, got ${dim}`);
  }
  const n = dv.getUint32(12, true);
  const seed = dv.getFloat64(16, true);
  // The seed reconstructs the rotation RNG; a non-finite value would crash that
  // rebuild with a non-DeserializeError, so reject it here on the untrusted path.
  if (!Number.isFinite(seed)) {
    throw new DeserializeError('BAD_SEED', `seed must be finite, got ${seed}`);
  }

  // Validate the fixed body region fits BEFORE allocating anything sized by n.
  // Codes are bit-packed; compute the byte count with Math (not `>>3`) so a crafted
  // huge n can't 32-bit-overflow the size check and slip past the bounds guard.
  const codeCount = n * dim;
  const codesBytes = Math.ceil((codeCount * bits) / 8);
  const fixedEnd = HEADER_BYTES + codesBytes + 2 * n * FLOAT_BYTES;
  if (fixedEnd > bytes.length) {
    throw new DeserializeError('BAD_LENGTH', `body ${fixedEnd} exceeds buffer ${bytes.length}`);
  }

  // Bulk reads — fresh, owned copies (the float loops read LE explicitly).
  let off = HEADER_BYTES;
  const codes = unpackCodes(bytes.subarray(off, off + codesBytes), codeCount, bits);
  off += codesBytes;
  const scales = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    scales[i] = dv.getFloat32(off, true);
    off += FLOAT_BYTES;
  }
  const norms = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    norms[i] = dv.getFloat32(off, true);
    off += FLOAT_BYTES;
  }

  // ── TQ+ calibration (presence byte, then shift[dim] + scale[dim] if present) ──
  if (off + 1 > bytes.length) {
    throw new DeserializeError('BAD_LENGTH', 'truncated before calibration flag');
  }
  const caliFlag = dv.getUint8(off);
  off += 1;
  let calibration: Calibration | undefined;
  if (caliFlag === 1) {
    // Bounds-check the whole calibration block before allocating dim-sized arrays.
    if (off + 2 * dim * FLOAT_BYTES > bytes.length) {
      throw new DeserializeError('BAD_LENGTH', 'calibration section exceeds buffer');
    }
    const shift = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      shift[i] = dv.getFloat32(off, true);
      off += FLOAT_BYTES;
    }
    const scale = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      scale[i] = dv.getFloat32(off, true);
      off += FLOAT_BYTES;
    }
    // scale is a divisor in the search-time calibration dual (../core/search,
    // ../index/turboquant-index #searchFastScan): a zero or non-finite entry would
    // turn into Infinity/NaN scores. Validate untrusted input up front.
    for (let i = 0; i < dim; i++) {
      if (!Number.isFinite(shift[i]!) || !Number.isFinite(scale[i]!) || scale[i] === 0) {
        throw new DeserializeError(
          'BAD_CALIBRATION',
          `calibration coordinate ${i} has invalid shift/scale (${shift[i]}/${scale[i]})`,
        );
      }
    }
    calibration = { shift, scale };
  } else if (caliFlag !== 0) {
    throw new DeserializeError('BAD_LENGTH', `invalid calibration flag ${caliFlag}`);
  }

  // ── IVF (presence byte, then nlist/nprobe + centroids + listForSlot if present) ──
  if (off + 1 > bytes.length) {
    throw new DeserializeError('BAD_LENGTH', 'truncated before ivf flag');
  }
  const ivfFlag = dv.getUint8(off);
  off += 1;
  let ivf: IvfPayload | undefined;
  if (ivfFlag === 1) {
    if (off + 2 * U32_BYTES > bytes.length) {
      throw new DeserializeError('BAD_IVF', 'truncated before ivf nlist/nprobe');
    }
    const nlist = dv.getUint32(off, true);
    off += U32_BYTES;
    const nprobe = dv.getUint32(off, true);
    off += U32_BYTES;
    if (nlist < 2 || nlist > MAX_NLIST) {
      throw new DeserializeError('BAD_IVF', `nlist must be in [2, 2^22], got ${nlist}`);
    }
    if (nprobe < 1 || nprobe > nlist) {
      throw new DeserializeError('BAD_IVF', `nprobe must be in [1, nlist=${nlist}], got ${nprobe}`);
    }
    // Bounds-check the whole section BEFORE allocating anything sized by nlist/n.
    const ivfBody = nlist * dim * FLOAT_BYTES + n * U32_BYTES;
    if (off + ivfBody > bytes.length) {
      throw new DeserializeError('BAD_IVF', 'ivf section exceeds buffer');
    }
    const centroids = new Float32Array(nlist * dim);
    for (let i = 0; i < centroids.length; i++) {
      const x = dv.getFloat32(off, true);
      if (!Number.isFinite(x)) {
        throw new DeserializeError('BAD_IVF', `centroid coordinate ${i} is not finite`);
      }
      centroids[i] = x;
      off += FLOAT_BYTES;
    }
    const listForSlot = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const l = dv.getUint32(off, true);
      if (l >= nlist) {
        throw new DeserializeError(
          'BAD_IVF',
          `listForSlot[${i}] = ${l} out of range [0, ${nlist})`,
        );
      }
      listForSlot[i] = l;
      off += U32_BYTES;
    }
    ivf = { nlist, nprobe, centroids, listForSlot };
  } else if (ivfFlag !== 0) {
    throw new DeserializeError('BAD_IVF', `invalid ivf flag ${ivfFlag}`);
  }

  const base: IndexPayload = { metric, bits: bits as Bits, dim, n, seed, codes, scales, norms };
  if (calibration !== undefined) base.calibration = calibration;
  if (ivf !== undefined) base.ivf = ivf;
  if (kindByte === 0) {
    if (off !== bytes.length) {
      throw new DeserializeError('BAD_LENGTH', `${bytes.length - off} trailing bytes after body`);
    }
    return { kind: 'positional', ...base };
  }

  // ── Ids (idmap) — parse n ids, bounds-checking every read, exact consumption ─
  // `fatal` so crafted invalid UTF-8 is rejected, not silently turned into U+FFFD.
  const dec = new TextDecoder('utf-8', { fatal: true });
  const ids: IdType[] = [];
  // The format guarantees one unique id per slot; reject a crafted buffer whose
  // ids collide, which would otherwise silently break IdMapIndex's id↔slot bijection.
  const seen = new Set<IdType>();
  for (let i = 0; i < n; i++) {
    if (off + 1 > bytes.length) {
      throw new DeserializeError('BAD_LENGTH', `truncated before id ${i}`);
    }
    const tag = dv.getUint8(off);
    off += 1;
    let id: IdType;
    if (tag === ID_TAG_NUMBER) {
      if (off + 8 > bytes.length) {
        throw new DeserializeError('BAD_LENGTH', `truncated number id ${i}`);
      }
      id = dv.getFloat64(off, true);
      off += 8;
    } else if (tag === ID_TAG_STRING || tag === ID_TAG_BIGINT) {
      const len = readBoundedLength(dv, off, bytes.length);
      off += U32_BYTES;
      let str: string;
      try {
        str = dec.decode(bytes.subarray(off, off + len));
      } catch {
        throw new DeserializeError('BAD_ID', `id ${i} is not valid UTF-8`);
      }
      off += len;
      if (tag === ID_TAG_STRING) {
        id = str;
      } else {
        let big: bigint;
        try {
          big = BigInt(str);
        } catch {
          throw new DeserializeError(
            'BAD_ID',
            `id ${i} is not a valid bigint: ${JSON.stringify(str)}`,
          );
        }
        // Enforce the canonical decimal we write, so each id has exactly one encoding
        // (rejects "0x10", " 7 ", "007", "" → 0n, etc.).
        if (big.toString() !== str) {
          throw new DeserializeError(
            'BAD_ID',
            `id ${i} bigint is not canonical decimal: ${JSON.stringify(str)}`,
          );
        }
        id = big;
      }
    } else {
      throw new DeserializeError('BAD_ID', `unknown id tag ${tag} at id ${i}`);
    }
    if (seen.has(id)) {
      throw new DeserializeError('BAD_ID', `duplicate id at slot ${i}: ${String(id)}`);
    }
    seen.add(id);
    ids.push(id);
  }
  if (off !== bytes.length) {
    throw new DeserializeError('BAD_LENGTH', `${bytes.length - off} trailing bytes after ids`);
  }
  return { kind: 'idmap', ...base, ids };
}
