// quantvec/node — Node-only convenience helpers (filesystem save/load).
//
// Isolated from the core so the main entry stays isomorphic (no `node:*` imports
// leak into browser/Workers builds). These thin wrappers persist the runtime-
// agnostic toBytes()/fromBytes() to disk via `node:fs/promises`.

import { readFile, writeFile } from 'node:fs/promises';
import { IdMapIndex } from './index/id-map-index';
import { TurboQuantIndex } from './index/turboquant-index';
import type { IdType } from './io/serialize';

/** Anything that can serialize itself to the versioned byte format. */
export interface BytesSerializable {
  toBytes(): Uint8Array;
}

/** Write an index's serialized bytes to `path`. */
export async function saveIndex(index: BytesSerializable, path: string): Promise<void> {
  await writeFile(path, index.toBytes());
}

/**
 * Read a file into a plain, owned `Uint8Array`. The copy is deliberate: `readFile`
 * returns a `Buffer` whose memory may come from a shared pool, and the deserializer
 * builds zero-copy views over its input — we hand it bytes it can safely own.
 */
export async function readIndexBytes(path: string): Promise<Uint8Array> {
  return new Uint8Array(await readFile(path));
}

/**
 * Load a positional {@link TurboQuantIndex} from `path`.
 *
 * @throws {IndexError} `'WRONG_KIND'` if the file holds an id-keyed index.
 * @throws {DeserializeError} on a malformed/oversized file.
 */
export async function loadIndex(path: string): Promise<TurboQuantIndex> {
  return TurboQuantIndex.fromBytes(await readIndexBytes(path));
}

/**
 * Load an id-keyed {@link IdMapIndex} from `path`.
 *
 * @throws {IdMapError} `'WRONG_KIND'` if the file holds a positional index.
 * @throws {DeserializeError} on a malformed/oversized file.
 */
export async function loadIdMapIndex(path: string): Promise<IdMapIndex<IdType>> {
  return IdMapIndex.fromBytes(await readIndexBytes(path));
}
