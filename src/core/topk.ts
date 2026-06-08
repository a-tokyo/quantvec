// Bounded top-k selection: keep the k items with the largest scores.
//
// The search path scans every stored vector and must retain only the best k
// candidates without buffering all n scores. A size-k binary **min-heap** does
// this in O(n log k) time and O(k) space: the heap root is the smallest of the
// kept scores, so a new candidate is admitted only if it beats the root, which
// it then replaces. Final extraction sorts the k kept items by score
// descending (best first).

/** Sorted top-k result. `scores[i]` is the score of `indices[i]`. */
export interface TopKResult {
  /** Kept scores, descending (best first). */
  scores: Float32Array;
  /** Original indices, aligned with `scores`. */
  indices: Int32Array;
}

/** Discriminated, code-tagged error for invalid TopK construction. */
export class TopKError extends Error {
  readonly code: 'INVALID_K';
  constructor(code: 'INVALID_K', message: string) {
    super(message);
    this.name = 'TopKError';
    this.code = code;
  }
}

/**
 * Bounded best-k tracker backed by a size-k min-heap.
 *
 * Usage: `const t = new TopK(k); for (...) t.add(score, index); t.result();`
 * Higher score is better. `result()` is non-destructive and may be called
 * repeatedly.
 */
export class TopK {
  private readonly k: number;
  /** Heap of scores; `heapScore[0]` is the smallest kept score. */
  private readonly heapScore: Float64Array;
  /** Heap of indices, aligned with `heapScore`. */
  private readonly heapIndex: Int32Array;
  /** Number of items currently in the heap (≤ k). */
  private size = 0;

  constructor(k: number) {
    if (!Number.isInteger(k) || k <= 0) {
      throw new TopKError('INVALID_K', `k must be a positive integer, got ${k}`);
    }
    this.k = k;
    this.heapScore = new Float64Array(k);
    this.heapIndex = new Int32Array(k);
  }

  /** Offer a (score, index) candidate. Kept only if among the top k so far. */
  add(score: number, index: number): void {
    if (this.size < this.k) {
      // Heap not yet full: insert and sift up.
      let i = this.size++;
      this.heapScore[i] = score;
      this.heapIndex[i] = index;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (this.heapScore[parent]! <= this.heapScore[i]!) break;
        this.swap(parent, i);
        i = parent;
      }
      return;
    }
    // Heap full: only admit if it beats the current minimum (the root).
    if (score <= this.heapScore[0]!) return;
    this.heapScore[0] = score;
    this.heapIndex[0] = index;
    this.siftDown(0);
  }

  /** Extract the kept items sorted by score descending (best first). */
  result(): TopKResult {
    const n = this.size;
    // Copy out, then sort descending by score (ties broken by smaller index,
    // matching a stable sort-and-slice reference).
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    const score = this.heapScore;
    const idx = this.heapIndex;
    const orderArr = Array.from(order);
    orderArr.sort((a, b) => {
      const ds = score[b]! - score[a]!;
      if (ds !== 0) return ds;
      return idx[a]! - idx[b]!;
    });
    const scores = new Float32Array(n);
    const indices = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const j = orderArr[i]!;
      scores[i] = score[j]!;
      indices[i] = idx[j]!;
    }
    return { scores, indices };
  }

  private swap(a: number, b: number): void {
    const s = this.heapScore[a]!;
    this.heapScore[a] = this.heapScore[b]!;
    this.heapScore[b] = s;
    const ix = this.heapIndex[a]!;
    this.heapIndex[a] = this.heapIndex[b]!;
    this.heapIndex[b] = ix;
  }

  private siftDown(start: number): void {
    const n = this.size;
    let i = start;
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let smallest = i;
      if (left < n && this.heapScore[left]! < this.heapScore[smallest]!) smallest = left;
      if (right < n && this.heapScore[right]! < this.heapScore[smallest]!) smallest = right;
      if (smallest === i) break;
      this.swap(i, smallest);
      i = smallest;
    }
  }
}
