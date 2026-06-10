// IVF coarse quantizer — the cell structure behind TurboQuantIndex's `ivf` mode.
//
// Why this exists: a flat scan is O(n) per query. The inverted-file (IVF) idea
// partitions the corpus into `nlist` cells around k-means centroids; a query
// ranks the centroids, probes only the `nprobe` nearest cells, and scans just
// those cells' members — sublinear work at large n. This class owns the cell
// state: the trained centroids (frozen, like the rotation and TQ+ calibration)
// and the posting lists that map cells → live slots, kept in lockstep with the
// index's swap-remove storage.
//
// Affinity is metric-consistent with the index: cosine/dot rank cells by
// argmax ⟨v, c⟩ over unit centroids (spherical k-means); euclidean ranks by
// argmin ‖v − c‖². The quantized scan over the probed slots is ./../core/search
// `searchSlots`, so IVF results at nprobe = nlist are EXACTLY the flat scan's.

import { kmeans, nearestCentroid } from '../core/kmeans';
import type { Distance } from '../core/metrics';
import { createRng } from '../core/rng';
import { TopK } from '../core/topk';

/** Training sample cap: at most this many vectors per cell are fed to k-means. */
export const IVF_TRAIN_SAMPLE_PER_LIST = 64;

/** Default probe breadth: 1/8 of the cells, at least one. */
export function defaultNprobe(nlist: number): number {
  return Math.max(1, Math.ceil(nlist / 8));
}

/** Domain separator XORed into the index seed so the k-means RNG stream never
 *  aliases the rotation's stream derived from the same seed ("IVF1" in ASCII). */
const IVF_SEED_DOMAIN = 0x49564631n;

/**
 * The trained cell structure: centroids plus posting-list bookkeeping.
 *
 * Slot membership invariants (maintained by addSlot/swapRemove/clear, fuzzed in
 * coarse.test.ts): for every live slot s,
 *   postings[listForSlot[s]][posForSlot[s]] === s
 * and Σ_l |postings[l]| equals the number of live slots.
 */
export class CoarseQuantizer {
  readonly nlist: number;
  readonly dim: number;
  readonly metric: Distance;
  /** Resolved default probe breadth (validated by the index). */
  readonly defaultNprobe: number;
  /** Row-major nlist·dim centroids (unit rows for cosine/dot). */
  readonly centroids: Float32Array;

  /** Cell members: slot ids per list (push/pop O(1)). */
  #postings: number[][];
  /** Slot → owning list (grown by doubling alongside the index's capacity). */
  #listForSlot: Int32Array;
  /** Slot → position inside its posting list (for O(1) removal). */
  #posForSlot: Int32Array;

  private constructor(
    centroids: Float32Array,
    nlist: number,
    nprobe: number,
    metric: Distance,
    dim: number,
  ) {
    this.nlist = nlist;
    this.dim = dim;
    this.metric = metric;
    this.defaultNprobe = nprobe;
    this.centroids = centroids;
    this.#postings = Array.from({ length: nlist }, () => []);
    this.#listForSlot = new Int32Array(0);
    this.#posForSlot = new Int32Array(0);
  }

  /** Whether cells use the spherical (dot) affinity. */
  get #spherical(): boolean {
    return this.metric !== 'euclidean';
  }

  /**
   * Train centroids from the first batch. The caller (TurboQuantIndex)
   * guarantees `vecs.length >= nlist` and per-vector length === dim. Training
   * samples at most {@link IVF_TRAIN_SAMPLE_PER_LIST}·nlist rows (partial
   * Fisher–Yates over a domain-separated RNG derived from the index seed), so
   * the k-means cost is bounded regardless of the first batch's size. Zero-norm
   * rows are skipped the same way the calibration fitter skips them — encode
   * rejects them moments later anyway.
   */
  static train(
    vecs: readonly Float32Array[],
    nlist: number,
    nprobe: number,
    metric: Distance,
    dim: number,
    seed: number,
  ): CoarseQuantizer {
    const rng = createRng((BigInt(Math.trunc(seed)) & ((1n << 64n) - 1n)) ^ IVF_SEED_DOMAIN);
    const m = vecs.length;
    const sampleSize = Math.min(m, IVF_TRAIN_SAMPLE_PER_LIST * nlist);

    // Partial Fisher–Yates: pick `sampleSize` distinct row indices deterministically.
    const order = new Int32Array(m);
    for (let i = 0; i < m; i++) order[i] = i;
    for (let i = 0; i < sampleSize; i++) {
      const j = i + Math.min(m - 1 - i, Math.floor(rng.nextFloat() * (m - i)));
      const t = order[i]!;
      order[i] = order[j]!;
      order[j] = t;
    }

    const spherical = metric !== 'euclidean';
    const data = new Float32Array(sampleSize * dim);
    let rows = 0;
    for (let s = 0; s < sampleSize; s++) {
      const v = vecs[order[s]!]!;
      let normSq = 0;
      for (let i = 0; i < dim; i++) normSq += v[i]! * v[i]!;
      if (normSq === 0) continue; // zero rows can't be normalized; encode will reject them
      const base = rows * dim;
      if (spherical) {
        const inv = 1 / Math.sqrt(normSq);
        for (let i = 0; i < dim; i++) data[base + i] = v[i]! * inv;
      } else {
        data.set(v, base);
      }
      rows++;
    }

    const { centroids } = kmeans(data.subarray(0, rows * dim), rows, {
      k: nlist,
      dim,
      rng,
      spherical,
    });
    return new CoarseQuantizer(centroids, nlist, nprobe, metric, dim);
  }

  /** Rebuild from deserialized state; postings are reconstructed from `listForSlot`. */
  static fromState(
    centroids: Float32Array,
    listForSlot: Int32Array,
    nlist: number,
    nprobe: number,
    metric: Distance,
    dim: number,
  ): CoarseQuantizer {
    const cq = new CoarseQuantizer(centroids, nlist, nprobe, metric, dim);
    for (let slot = 0; slot < listForSlot.length; slot++) {
      cq.#growTo(slot + 1);
      const list = listForSlot[slot]!;
      cq.#listForSlot[slot] = list;
      cq.#posForSlot[slot] = cq.#postings[list]!.length;
      cq.#postings[list]!.push(slot);
    }
    return cq;
  }

  /** Nearest cell for `vec` under the index's metric (raw, unrotated vector). */
  assign(vec: Float32Array): number {
    return nearestCentroid(vec, 0, this.centroids, this.nlist, this.dim, this.#spherical);
  }

  /** Assign `vec` and record `slot` (the index's next row) in that cell's list. */
  addSlot(slot: number, vec: Float32Array): void {
    const list = this.assign(vec);
    this.#growTo(slot + 1);
    this.#listForSlot[slot] = list;
    this.#posForSlot[slot] = this.#postings[list]!.length;
    this.#postings[list]!.push(slot);
  }

  /**
   * Mirror the index's swap-remove of slot `i` (the row at slot `last = n−1`
   * moved into the gap). Two memberships are patched, in this order:
   *
   *   A) drop slot i from its own list by swap-pop: pop the list's tail; if the
   *      tail wasn't i itself, the tail fills i's hole (and its posForSlot moves).
   *   B) when i !== last, renumber `last` → `i` in last's list — reading last's
   *      position AFTER step A, because step A may have just moved it.
   */
  swapRemove(i: number, last: number): void {
    // Step A — remove slot i from its posting list.
    const li = this.#listForSlot[i]!;
    const p = this.#posForSlot[i]!;
    const listI = this.#postings[li]!;
    const tail = listI.pop()!;
    if (tail !== i) {
      listI[p] = tail;
      this.#posForSlot[tail] = p;
    }

    // Step B — slot `last` now lives at slot `i`.
    if (i !== last) {
      const lj = this.#listForSlot[last]!;
      const pj = this.#posForSlot[last]!;
      this.#postings[lj]![pj] = i;
      this.#listForSlot[i] = lj;
      this.#posForSlot[i] = pj;
    }
  }

  /** Empty every posting list. The trained centroids survive — clearing data
   *  does not unfreeze the training decision (same contract as calibration). */
  clear(): void {
    for (const list of this.#postings) list.length = 0;
  }

  /** Snapshot of slot → list for the first `n` slots (the serialized form). */
  listForSlotSnapshot(n: number): Int32Array {
    return this.#listForSlot.slice(0, n);
  }

  /**
   * Rank all cells against `query` (metric-consistent, higher-is-better) and
   * concatenate the top-`nprobe` cells' slots into one array for `searchSlots`.
   */
  probe(query: Float32Array, nprobe: number): Int32Array {
    const { nlist, dim, centroids } = this;
    const spherical = this.#spherical;
    const top = new TopK(nprobe);
    for (let c = 0; c < nlist; c++) {
      let key: number;
      if (spherical) {
        let dot = 0;
        const base = c * dim;
        for (let i = 0; i < dim; i++) dot += query[i]! * centroids[base + i]!;
        key = dot;
      } else {
        let d2 = 0;
        const base = c * dim;
        for (let i = 0; i < dim; i++) {
          const d = query[i]! - centroids[base + i]!;
          d2 += d * d;
        }
        key = -d2;
      }
      top.add(key, c);
    }

    const { indices: lists } = top.result();
    let total = 0;
    for (const l of lists) total += this.#postings[l]!.length;
    const slots = new Int32Array(total);
    let off = 0;
    for (const l of lists) {
      const list = this.#postings[l]!;
      for (let t = 0; t < list.length; t++) slots[off++] = list[t]!;
    }
    return slots;
  }

  /** Grow the slot-indexed arrays to hold at least `needed` slots (doubling). */
  #growTo(needed: number): void {
    if (needed <= this.#listForSlot.length) return;
    const cap = Math.max(8, needed, this.#listForSlot.length * 2);
    const nl = new Int32Array(cap);
    nl.set(this.#listForSlot);
    this.#listForSlot = nl;
    const np = new Int32Array(cap);
    np.set(this.#posForSlot);
    this.#posForSlot = np;
  }
}
