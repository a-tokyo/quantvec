// Real-dataset benchmark on SIFT-small (10k × 128-d, 100 queries, 100-NN L2 ground
// truth). Reports recall@{1,10,100} against the *dataset's own* ground truth (not a
// self-computed one), plus encode/query throughput and serialized compression, at
// 2/3/4 bits. dim=128 is a power of two, so the FWHT rotation + WASM kernel are active.
//
// Setup: `node benchmarks/download-siftsmall.mjs` then `bun run benchmarks/real.ts`.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { TurboQuantIndex } from '../src/index/turboquant-index';
import type { Bits } from '../src/core/codebook';

const DATA = join(process.cwd(), 'benchmarks', 'datasets', 'siftsmall');

/** Read a .fvecs file: repeated [int32 dim][dim × float32]. */
function readFvecs(path: string): { dim: number; vectors: Float32Array[] } {
  const buf = readFileSync(path);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const vectors: Float32Array[] = [];
  let off = 0;
  let dim = 0;
  while (off < buf.byteLength) {
    dim = dv.getInt32(off, true);
    off += 4;
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      v[i] = dv.getFloat32(off, true);
      off += 4;
    }
    vectors.push(v);
  }
  return { dim, vectors };
}

/** Read an .ivecs file: repeated [int32 dim][dim × int32]. */
function readIvecs(path: string): number[][] {
  const buf = readFileSync(path);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const rows: number[][] = [];
  let off = 0;
  while (off < buf.byteLength) {
    const dim = dv.getInt32(off, true);
    off += 4;
    const r: number[] = new Array(dim);
    for (let i = 0; i < dim; i++) {
      r[i] = dv.getInt32(off, true);
      off += 4;
    }
    rows.push(r);
  }
  return rows;
}

function recallAt(approx: Int32Array, truth: number[], k: number): number {
  const set = new Set(truth.slice(0, k));
  let hit = 0;
  for (let i = 0; i < Math.min(k, approx.length); i++) if (set.has(approx[i]!)) hit++;
  return hit / k;
}

function main(): void {
  const { dim, vectors: base } = readFvecs(join(DATA, 'siftsmall_base.fvecs'));
  const { vectors: queries } = readFvecs(join(DATA, 'siftsmall_query.fvecs'));
  const groundtruth = readIvecs(join(DATA, 'siftsmall_groundtruth.ivecs'));
  process.stdout.write(
    `SIFT-small — base=${base.length} dim=${dim} queries=${queries.length} (euclidean, dataset ground truth)\n\n`,
  );
  process.stdout.write(
    'bits | recall@1 | recall@10 | recall@100 | encode (vec/s) |   QPS | fastScan QPS | compression\n',
  );
  process.stdout.write(
    '-----|----------|-----------|------------|----------------|-------|--------------|------------\n',
  );

  const rows = ([2, 3, 4] as Bits[]).map((bits) => {
    const index = new TurboQuantIndex({ dim, bits, metric: 'euclidean', seed: 1 });
    const tE = performance.now();
    index.add(base);
    const encS = (performance.now() - tE) / 1000;

    const approx: Int32Array[] = [];
    const tS = performance.now();
    for (const q of queries) approx.push(index.search(q, 100).indices);
    const qps = queries.length / ((performance.now() - tS) / 1000);

    let r1 = 0;
    let r10 = 0;
    let r100 = 0;
    for (let i = 0; i < queries.length; i++) {
      r1 += recallAt(approx[i]!, groundtruth[i]!, 1);
      r10 += recallAt(approx[i]!, groundtruth[i]!, 10);
      r100 += recallAt(approx[i]!, groundtruth[i]!, 100);
    }
    const nq = queries.length;
    const compression = (base.length * dim * 4) / index.toBytes().length;
    const row: {
      bits: Bits;
      recall1: number;
      recall10: number;
      recall100: number;
      encodeVecPerSec: number;
      qps: number;
      fastScanQps?: number;
      compressionVsF32: number;
    } = {
      bits,
      recall1: r1 / nq,
      recall10: r10 / nq,
      recall100: r100 / nq,
      encodeVecPerSec: base.length / encS,
      qps,
      compressionVsF32: compression,
    };
    return row;
  });

  // FastScan QPS — 4-bit only.
  {
    const fsIndex = new TurboQuantIndex({
      dim,
      bits: 4,
      metric: 'euclidean',
      fastscan: true,
      seed: 1,
    });
    fsIndex.add(base);
    const tFs = performance.now();
    for (const q of queries) fsIndex.search(q, 100);
    const fsSecs = (performance.now() - tFs) / 1000;
    const row4 = rows.find((r) => r.bits === 4)!;
    row4.fastScanQps = queries.length / fsSecs;
  }

  // Print rows after FastScan is populated.
  for (const row of rows) {
    const fsCol =
      row.fastScanQps !== undefined
        ? Math.round(row.fastScanQps).toString().padStart(12)
        : '           —';
    process.stdout.write(
      `  ${row.bits}  |  ${row.recall1.toFixed(3)}   |   ${row.recall10.toFixed(3)}   |   ${row.recall100.toFixed(
        3,
      )}    | ${Math.round(row.encodeVecPerSec).toString().padStart(14)} | ${Math.round(row.qps)
        .toString()
        .padStart(5)} | ${fsCol} | ${row.compressionVsF32.toFixed(2)}x\n`,
    );
  }

  process.stdout.write('\n');
  for (const r of rows) {
    process.stdout.write(`METRIC sift_recall_at10_${r.bits}bit=${r.recall10.toFixed(4)}\n`);
    process.stdout.write(`METRIC sift_qps_${r.bits}bit=${Math.round(r.qps)}\n`);
  }
  {
    const row4 = rows.find((r) => r.bits === 4)!;
    if (row4.fastScanQps !== undefined) {
      process.stdout.write(`METRIC sift_fastscan_qps_4bit=${Math.round(row4.fastScanQps)}\n`);
    }
  }
  const outDir = join(process.cwd(), 'benchmarks', 'results');
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    join(outDir, 'sift-small.json'),
    JSON.stringify(
      {
        dataset: 'siftsmall',
        base: base.length,
        dim,
        queries: queries.length,
        metric: 'euclidean',
        generatedAt: new Date().toISOString(),
        rows,
      },
      null,
      2,
    ),
  );
  process.stdout.write(`\nwrote ${join(outDir, 'sift-small.json')}\n`);
}

main();
