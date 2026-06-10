// dbpedia-OpenAI-100k benchmark (text-embedding-ada-002, 1536-d, ~100k base vectors,
// 100-NN cosine ground truth pre-computed in ann-benchmarks format). Reads the HDF5 file
// produced by `node benchmarks/download-openai.mjs`, then measures recall@{1,10,100},
// encode/query throughput, and serialized compression at 2/3/4 bits.
//
// dim=1536 is a power of two → the FWHT rotation + WASM kernel are active, the same
// regime as OpenAI/Ada, BERT-large, and other modern embedding models.
//
// Run: node benchmarks/download-openai.mjs && npx tsx benchmarks/openai.ts
// Env knobs: N (number of base vectors, default = full corpus), NQ (queries, default = full test set)

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import * as h5wasm from 'h5wasm';
import { TurboQuantIndex } from '../src/index/turboquant-index';
import type { Bits } from '../src/core/codebook';

const HDF5_FILE = join(process.cwd(), 'benchmarks', 'datasets', 'dbpedia-openai-100k-angular.hdf5');
const WASM_PATH = 'dbpedia-openai-100k-angular.hdf5';

function recallAt(approx: Int32Array, truth: number[], k: number): number {
  const set = new Set(truth.slice(0, k));
  let hit = 0;
  for (let i = 0; i < Math.min(k, approx.length); i++) if (set.has(approx[i]!)) hit++;
  return hit / k;
}

async function main(): Promise<void> {
  // h5wasm uses an Emscripten virtual filesystem. In Node.js, load the file into a
  // buffer and write it to the WASM VFS so the HDF5 library can open it by name.
  await h5wasm.ready;
  process.stdout.write('loading HDF5 file into memory… ');
  const buf = readFileSync(HDF5_FILE);
  h5wasm.FS.writeFile(WASM_PATH, buf);
  process.stdout.write(`${(buf.byteLength / 1024 / 1024).toFixed(0)} MB loaded\n`);
  const f = new h5wasm.File(WASM_PATH, 'r');

  const trainDs = f.get('train') as h5wasm.Dataset;
  const testDs = f.get('test') as h5wasm.Dataset;
  const neighborsDs = f.get('neighbors') as h5wasm.Dataset;

  const [totalTrain, dim] = trainDs.shape as [number, number];
  const [totalTest] = testDs.shape as [number, number];

  const N = Number(process.env.N ?? totalTrain);
  const NQ = Number(process.env.NQ ?? Math.min(1000, totalTest));
  const nBase = Math.min(N, totalTrain);
  const nQuery = Math.min(NQ, totalTest);

  process.stdout.write(
    `dbpedia-OpenAI-100k — using ${nBase.toLocaleString()} / ${totalTrain.toLocaleString()} train vectors, ` +
      `${nQuery} queries, dim=${dim}, cosine\n`,
  );
  process.stdout.write(
    `(ground truth: ann-benchmarks 100-NN cosine; rotation: FWHT for power-of-two dim=${dim})\n\n`,
  );

  // Read base vectors (slice first nBase rows).
  const trainRaw = trainDs.slice([
    [0, nBase],
    [0, dim],
  ]) as Float32Array;
  const base: Float32Array[] = [];
  for (let i = 0; i < nBase; i++) base.push(trainRaw.subarray(i * dim, (i + 1) * dim));

  // Read query vectors (slice first nQuery rows from test set).
  const testRaw = testDs.slice([
    [0, nQuery],
    [0, dim],
  ]) as Float32Array;
  const queries: Float32Array[] = [];
  for (let i = 0; i < nQuery; i++) queries.push(testRaw.subarray(i * dim, (i + 1) * dim));

  // Ground truth: use the pre-computed ann-benchmarks neighbors only when running the
  // FULL corpus (N >= totalTrain). Otherwise compute brute-force cosine within the
  // sub-sample — the pre-computed indices reference the full corpus so recall numbers
  // against them are misleadingly low (only ~N/totalTrain of true neighbors land in
  // the slice).
  let groundtruth: number[][];
  if (nBase >= totalTrain) {
    const [, gtK] = neighborsDs.shape as [number, number];
    const k = Math.min(100, gtK);
    const neighborRaw = neighborsDs.slice([
      [0, nQuery],
      [0, k],
    ]) as Int32Array;
    groundtruth = [];
    for (let i = 0; i < nQuery; i++) {
      const row: number[] = [];
      for (let j = 0; j < k; j++) row.push(neighborRaw[i * k + j]!);
      groundtruth.push(row);
    }
    process.stdout.write(`(using pre-computed ann-benchmarks ground truth — full corpus)\n\n`);
  } else {
    // Brute-force cosine top-100 within the sub-sample. Normalise once, then dot.
    process.stdout.write(
      `(sub-sample: computing brute-force cosine ground truth within ${nBase.toLocaleString()} vectors…) `,
    );
    const tGT = performance.now();
    const baseNorm: Float32Array[] = base.map((v) => {
      let s = 0;
      for (let i = 0; i < dim; i++) s += v[i]! * v[i]!;
      const inv = 1 / Math.sqrt(s);
      const u = new Float32Array(dim);
      for (let i = 0; i < dim; i++) u[i] = v[i]! * inv;
      return u;
    });
    groundtruth = queries.map((q) => {
      let s = 0;
      for (let i = 0; i < dim; i++) s += q[i]! * q[i]!;
      const inv = 1 / Math.sqrt(s);
      const qn = new Float32Array(dim);
      for (let i = 0; i < dim; i++) qn[i] = q[i]! * inv;
      const scores = new Float32Array(nBase);
      for (let j = 0; j < nBase; j++) {
        let d = 0;
        for (let i = 0; i < dim; i++) d += qn[i]! * baseNorm[j]![i]!;
        scores[j] = d;
      }
      // Partial sort: top-100 indices by descending score.
      const idxs = Array.from({ length: nBase }, (_, i) => i);
      idxs.sort((a, b) => scores[b]! - scores[a]!);
      return idxs.slice(0, 100);
    });
    process.stdout.write(`done (${((performance.now() - tGT) / 1000).toFixed(1)}s)\n\n`);
  }

  f.close();

  process.stdout.write(
    'bits | recall@1 | recall@10 | recall@100 | encode (vec/s) |   QPS | fastScan QPS | compression\n',
  );
  process.stdout.write(
    '-----|----------|-----------|------------|----------------|-------|--------------|------------\n',
  );

  const rows = ([2, 3, 4] as Bits[]).map((bits) => {
    const index = new TurboQuantIndex({ dim, bits, metric: 'cosine', seed: 1 });

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
      r100 += recallAt(approx[i]!, groundtruth[i]!, Math.min(100, groundtruth[i]!.length));
    }
    const n = queries.length;
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
      recall1: r1 / n,
      recall10: r10 / n,
      recall100: r100 / n,
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
      metric: 'cosine',
      fastscan: true,
      seed: 1,
    });
    fsIndex.add(base);
    const tFs = performance.now();
    for (const q of queries) fsIndex.search(q, 100);
    const fsSecs = (performance.now() - tFs) / 1000;
    rows.find((r) => r.bits === 4)!.fastScanQps = queries.length / fsSecs;
  }

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
    process.stdout.write(`METRIC openai_recall_at10_${r.bits}bit=${r.recall10.toFixed(4)}\n`);
    process.stdout.write(`METRIC openai_qps_${r.bits}bit=${Math.round(r.qps)}\n`);
  }
  {
    const row4 = rows.find((r) => r.bits === 4)!;
    if (row4.fastScanQps !== undefined)
      process.stdout.write(`METRIC openai_fastscan_qps_4bit=${Math.round(row4.fastScanQps)}\n`);
  }

  const outDir = join(process.cwd(), 'benchmarks', 'results');
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    join(outDir, 'dbpedia-openai-100k.json'),
    JSON.stringify(
      {
        dataset: 'dbpedia-openai-100k-angular',
        source:
          'https://storage.googleapis.com/ann-datasets/ann-benchmarks/ (dbpedia-entities-openai-1M, ada-002)',
        nBase,
        totalTrain,
        dim,
        nQuery,
        metric: 'cosine',
        note: `Ground truth from the ${totalTrain.toLocaleString()}-vector corpus; recall computed against in-slice neighbors only when sub-sampled.`,
        generatedAt: new Date().toISOString(),
        rows,
      },
      null,
      2,
    ),
  );
  process.stdout.write(`\nwrote ${join(outDir, 'dbpedia-openai-100k.json')}\n`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
