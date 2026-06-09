#!/usr/bin/env node
// Download the GloVe-200 ann-benchmarks HDF5 file (~426 MB).
// Source: https://ann-benchmarks.com — pre-computed 100-NN cosine ground truth included.
// Usage: node benchmarks/download-glove.mjs

import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { get } from 'node:https';
import { pipeline } from 'node:stream/promises';

const OUT_DIR = join(process.cwd(), 'benchmarks', 'datasets');
const OUT_FILE = join(OUT_DIR, 'glove-200-angular.hdf5');
const URL = 'https://ann-benchmarks.com/glove-200-angular.hdf5';
const EXPECTED_MB = 426;

mkdirSync(OUT_DIR, { recursive: true });

if (existsSync(OUT_FILE)) {
  const mb = (statSync(OUT_FILE).size / 1024 / 1024).toFixed(0);
  console.log(`already downloaded: ${OUT_FILE} (${mb} MB)`);
  process.exit(0);
}

console.log(`downloading ${URL} (~${EXPECTED_MB} MB) …`);
console.log(`destination: ${OUT_FILE}`);

function fetchFollowRedirects(url, dest) {
  return new Promise((resolve, reject) => {
    function doGet(currentUrl) {
      get(currentUrl, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          doGet(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${currentUrl}`));
          return;
        }
        const total = parseInt(res.headers['content-length'] ?? '0', 10);
        let received = 0;
        let lastPct = -1;
        res.on('data', (chunk) => {
          received += chunk.length;
          if (total > 0) {
            const pct = Math.floor((received / total) * 100);
            if (pct !== lastPct && pct % 5 === 0) {
              process.stdout.write(
                `\r  ${pct}% (${(received / 1024 / 1024).toFixed(0)} / ${(total / 1024 / 1024).toFixed(0)} MB)`,
              );
              lastPct = pct;
            }
          }
        });
        pipeline(res, createWriteStream(dest)).then(resolve).catch(reject);
      }).on('error', reject);
    }
    doGet(url);
  });
}

await fetchFollowRedirects(URL, OUT_FILE);
const mb = (statSync(OUT_FILE).size / 1024 / 1024).toFixed(0);
console.log(`\ndone — ${OUT_FILE} (${mb} MB)`);
