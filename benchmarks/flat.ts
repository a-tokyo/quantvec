// quantvec — flat-index quality & throughput benchmark.
//
// Measures the core value proposition: how much recall the data-oblivious TurboQuant
// quantizer keeps vs an exact float32 brute-force scan, at 2/3/4 bits, plus encode
// throughput, query throughput, and the actual serialized compression ratio.
//
// Self-contained and deterministic (seeded PRNG, synthetic clustered embeddings that
// mimic the anisotropic structure of real text/image embeddings) so numbers are
// reproducible in CI and across runtimes. Emits a human table, `METRIC key=value`
// lines (for the autoresearch loop), and a JSON results file.
//
// Run: `bun run benchmarks/flat.ts` (or `npx tsx benchmarks/flat.ts`).

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { TurboQuantIndex } from '../src/index/turboquant-index';
import type { Bits } from '../src/core/codebook';

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

/**
 * Generate `count` vectors with a per-coordinate standard-deviation following a power
 * law `sigma_i = anisotropy^(i/dim)`. `anisotropy = 1` is the isotropic Gaussian worst
 * case (no neighborhood structure — Wave-4-validated lower bound on recall); values in
 * (0,1) add the anisotropic spectrum real embeddings exhibit (a few dominant
 * directions), which the random rotation handles and which lifts recall above the
 * isotropic floor.
 */
function makeVectors(
  count: number,
  dim: number,
  anisotropy: number,
  rng: () => number,
): Float32Array[] {
  const sigma = new Float32Array(dim);
  for (let i = 0; i < dim; i++) sigma[i] = Math.pow(anisotropy, i / dim);
  const out: Float32Array[] = [];
  for (let j = 0; j < count; j++) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = gaussian(rng) * sigma[i]!;
    out.push(v);
  }
  return out;
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

// ── Benchmark one bit-width ────────────────────────────────────────────────────
interface Row {
  bits: Bits;
  recall1: number;
  recall10: number;
  recall100: number;
  encodeVecPerSec: number;
  qps: number;
  fastScanQps?: number;
  bytesPerVector: number;
  compressionVsF32: number;
}

function benchBits(
  bits: Bits,
  db: Float32Array[],
  queries: Float32Array[],
  exact: number[][],
  dim: number,
): Row {
  const index = new TurboQuantIndex({ dim, bits, metric: 'cosine', seed: 1 });

  const tEncode = performance.now();
  index.add(db);
  const encodeSecs = (performance.now() - tEncode) / 1000;

  const K = 100;
  const approxAll: Int32Array[] = [];
  const tSearch = performance.now();
  for (const q of queries) approxAll.push(index.search(q, K).indices);
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

  const bytes = index.toBytes().length;
  const f32Bytes = db.length * dim * 4;

  return {
    bits,
    recall1: r1 / nq,
    recall10: r10 / nq,
    recall100: r100 / nq,
    encodeVecPerSec: db.length / encodeSecs,
    qps: nq / searchSecs,
    bytesPerVector: bytes / db.length,
    compressionVsF32: f32Bytes / bytes,
  };
}

// ── Run ────────────────────────────────────────────────────────────────────────
function main(): void {
  const dim = Number(process.env.DIM ?? 768);
  const n = Number(process.env.N ?? 5000);
  const nq = Number(process.env.NQ ?? 500);
  const anisotropy = Number(process.env.ANISOTROPY ?? 0.3);
  const rng = mulberry32(42);

  process.stdout.write(
    `quantvec flat benchmark — dim=${dim} n=${n} queries=${nq} anisotropy=${anisotropy} (cosine)\n`,
  );

  const db = makeVectors(n, dim, anisotropy, rng);
  const queries = makeVectors(nq, dim, anisotropy, rng);
  const exact = queries.map((q) => exactTopK(db, q, 100));

  const rows: Row[] = ([2, 3, 4] as Bits[]).map((bits) => benchBits(bits, db, queries, exact, dim));

  // FastScan QPS — 4-bit only.
  {
    const fsIndex = new TurboQuantIndex({
      dim,
      bits: 4,
      metric: 'cosine',
      fastscan: true,
      seed: 1,
    });
    fsIndex.add(db);
    const tFs = performance.now();
    for (const q of queries) fsIndex.search(q, 100);
    const fsSecs = (performance.now() - tFs) / 1000;
    const row4 = rows.find((r) => r.bits === 4)!;
    row4.fastScanQps = queries.length / fsSecs;
  }

  // Human table.
  process.stdout.write(
    '\nbits | recall@1 | recall@10 | recall@100 | encode (vec/s) |    QPS | fastScan QPS | bytes/vec | compression\n',
  );
  process.stdout.write(
    '-----|----------|-----------|------------|----------------|--------|--------------|-----------|------------\n',
  );
  for (const r of rows) {
    const fsCol =
      r.fastScanQps !== undefined
        ? Math.round(r.fastScanQps).toString().padStart(12)
        : '           —';
    process.stdout.write(
      `  ${r.bits}  |  ${r.recall1.toFixed(3)}   |   ${r.recall10.toFixed(3)}   |   ${r.recall100.toFixed(
        3,
      )}    | ${Math.round(r.encodeVecPerSec).toString().padStart(14)} | ${Math.round(r.qps)
        .toString()
        .padStart(
          6,
        )} | ${fsCol} | ${r.bytesPerVector.toFixed(1).padStart(9)} | ${r.compressionVsF32.toFixed(2)}x\n`,
    );
  }

  // METRIC lines (autoresearch protocol).
  process.stdout.write('\n');
  for (const r of rows) {
    process.stdout.write(`METRIC recall_at10_${r.bits}bit=${r.recall10.toFixed(4)}\n`);
    process.stdout.write(`METRIC qps_${r.bits}bit=${Math.round(r.qps)}\n`);
    process.stdout.write(`METRIC compression_${r.bits}bit=${r.compressionVsF32.toFixed(2)}\n`);
  }
  {
    const row4 = rows.find((r) => r.bits === 4)!;
    if (row4.fastScanQps !== undefined) {
      process.stdout.write(`METRIC fastscan_qps_4bit=${Math.round(row4.fastScanQps)}\n`);
    }
  }

  // JSON results (under the repo's benchmarks/results, relative to the run cwd).
  const outDir = join(process.cwd(), 'benchmarks', 'results');
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, `flat-d${dim}.json`);
  writeFileSync(
    outPath,
    JSON.stringify(
      { dim, n, nq, anisotropy, metric: 'cosine', generatedAt: new Date().toISOString(), rows },
      null,
      2,
    ),
  );
  process.stdout.write(`\nwrote ${outPath}\n`);
}

main();
