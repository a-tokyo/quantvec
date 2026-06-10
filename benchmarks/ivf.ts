// quantvec — IVF coarse-quantizer benchmark: speedup vs recall against the flat scan.
//
// Measures the IVF value proposition on clustered data (the regime IVF exists for):
// a seeded Gaussian-mixture corpus, an exact float32 cosine ground truth, a flat
// TurboQuantIndex baseline, and the same index with `ivf` enabled swept across
// nprobe values. At nprobe = nlist the IVF results are exactly the flat scan's
// (the searchSlots oracle), so the sweep shows the recall/QPS trade-off cleanly.
//
// Self-contained and deterministic. Emits a human table, `METRIC key=value` lines,
// and a JSON results file. Run: `npm run bench:ivf` (env: DIM, N, NQ, CLUSTERS, NLIST).

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { TurboQuantIndex } from '../src/index/turboquant-index';

// ── Deterministic PRNG + Gaussian (mulberry32 + Box–Muller) ────────────────────
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gaussian mixture: `clusters` centers (sigma 5), unit-sigma points around them. */
function makeClustered(
  count: number,
  dim: number,
  clusters: number,
  rng: () => number,
): { vectors: Float32Array[]; centers: Float32Array[] } {
  const centers = Array.from({ length: clusters }, () => {
    const c = new Float32Array(dim);
    for (let i = 0; i < dim; i++) c[i] = gaussian(rng) * 5;
    return c;
  });
  const vectors: Float32Array[] = [];
  for (let j = 0; j < count; j++) {
    const c = centers[j % clusters]!;
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = c[i]! + gaussian(rng);
    vectors.push(v);
  }
  return { vectors, centers };
}

// ── Exact cosine top-k ground truth ────────────────────────────────────────────
function normalized(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  const inv = 1 / Math.sqrt(s);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! * inv;
  return out;
}

function exactTopK(db: Float32Array[], query: Float32Array, k: number): number[] {
  const q = normalized(query);
  const scored = db.map((v, idx) => {
    const u = normalized(v);
    let dot = 0;
    for (let i = 0; i < u.length; i++) dot += u[i]! * q[i]!;
    return { idx, dot };
  });
  scored.sort((a, b) => b.dot - a.dot);
  return scored.slice(0, k).map((s) => s.idx);
}

function recallAt(approx: Int32Array, exact: number[], k: number): number {
  const truth = new Set(exact.slice(0, k));
  let hit = 0;
  for (let i = 0; i < Math.min(k, approx.length); i++) if (truth.has(approx[i]!)) hit++;
  return hit / k;
}

// ── Benchmark one configuration ────────────────────────────────────────────────
interface Row {
  label: string;
  nprobe?: number;
  recall1: number;
  recall10: number;
  recall100: number;
  qps: number;
  speedupVsFlat: number;
}

function benchIndex(
  label: string,
  index: TurboQuantIndex,
  queries: Float32Array[],
  exact: number[][],
  nprobe?: number,
): Omit<Row, 'speedupVsFlat'> {
  const K = 100;
  const opts = nprobe === undefined ? {} : { nprobe };
  const approxAll: Int32Array[] = [];
  const tSearch = performance.now();
  for (const q of queries) approxAll.push(index.search(q, K, opts).indices);
  const searchSecs = (performance.now() - tSearch) / 1000;

  let r1 = 0;
  let r10 = 0;
  let r100 = 0;
  for (let i = 0; i < queries.length; i++) {
    r1 += recallAt(approxAll[i]!, exact[i]!, 1);
    r10 += recallAt(approxAll[i]!, exact[i]!, 10);
    r100 += recallAt(approxAll[i]!, exact[i]!, 100);
  }
  const nq = queries.length;
  const row: Omit<Row, 'speedupVsFlat'> = {
    label,
    recall1: r1 / nq,
    recall10: r10 / nq,
    recall100: r100 / nq,
    qps: nq / searchSecs,
  };
  if (nprobe !== undefined) row.nprobe = nprobe;
  return row;
}

// ── Run ────────────────────────────────────────────────────────────────────────
function main(): void {
  const dim = Number(process.env.DIM ?? 768);
  const n = Number(process.env.N ?? 20000);
  const nq = Number(process.env.NQ ?? 200);
  const clusters = Number(process.env.CLUSTERS ?? 64);
  const nlist = Number(process.env.NLIST ?? 128);
  const rng = mulberry32(42);

  process.stdout.write(
    `quantvec IVF benchmark — dim=${dim} n=${n} queries=${nq} clusters=${clusters} nlist=${nlist} (cosine, 4-bit)\n`,
  );

  const { vectors: db } = makeClustered(n, dim, clusters, rng);
  const { vectors: queries } = makeClustered(nq, dim, clusters, mulberry32(7));
  process.stdout.write('computing exact float32 ground truth…\n');
  const exact = queries.map((q) => exactTopK(db, q, 100));

  // Flat baseline (scalar path — the IVF scan is scalar too, so the comparison is fair).
  const flat = new TurboQuantIndex({ dim, bits: 4, metric: 'cosine', seed: 1, wasm: false });
  flat.add(db);
  const flatRow: Row = { ...benchIndex('flat', flat, queries, exact), speedupVsFlat: 1 };

  // IVF index: trained from the same single batch.
  const tTrain = performance.now();
  const ivf = new TurboQuantIndex({ dim, bits: 4, metric: 'cosine', seed: 1, ivf: { nlist } });
  ivf.add(db);
  const trainSecs = (performance.now() - tTrain) / 1000;
  if (!ivf.ivfActive) throw new Error('IVF did not train — first batch smaller than nlist?');

  const sweep = [1, 2, 4, 8, 16, nlist].filter((p, i, arr) => arr.indexOf(p) === i && p <= nlist);
  const rows: Row[] = [flatRow];
  for (const nprobe of sweep) {
    const r = benchIndex(`ivf@${nprobe}`, ivf, queries, exact, nprobe);
    rows.push({ ...r, speedupVsFlat: r.qps / flatRow.qps });
  }

  // Human table.
  process.stdout.write(`\nbuild+train: ${trainSecs.toFixed(2)}s for ${n} vectors\n`);
  process.stdout.write('\nconfig    | recall@1 | recall@10 | recall@100 |    QPS | speedup\n');
  process.stdout.write('----------|----------|-----------|------------|--------|--------\n');
  for (const r of rows) {
    process.stdout.write(
      `${r.label.padEnd(9)} |  ${r.recall1.toFixed(3)}   |   ${r.recall10.toFixed(3)}   |   ${r.recall100.toFixed(
        3,
      )}    | ${Math.round(r.qps).toString().padStart(6)} | ${r.speedupVsFlat.toFixed(2)}x\n`,
    );
  }

  // METRIC lines (autoresearch protocol).
  process.stdout.write('\n');
  process.stdout.write(`METRIC flat_qps=${Math.round(flatRow.qps)}\n`);
  for (const r of rows) {
    if (r.nprobe === undefined) continue;
    process.stdout.write(`METRIC ivf_recall_at10_nprobe${r.nprobe}=${r.recall10.toFixed(4)}\n`);
    process.stdout.write(`METRIC ivf_qps_nprobe${r.nprobe}=${Math.round(r.qps)}\n`);
  }

  // JSON results.
  const outDir = join(process.cwd(), 'benchmarks', 'results');
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, `ivf-d${dim}.json`);
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        dim,
        n,
        nq,
        clusters,
        nlist,
        metric: 'cosine',
        bits: 4,
        trainSecs,
        generatedAt: new Date().toISOString(),
        rows,
      },
      null,
      2,
    ),
  );
  process.stdout.write(`\nwrote ${outPath}\n`);
}

main();
