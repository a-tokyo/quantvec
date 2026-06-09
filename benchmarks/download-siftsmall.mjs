// Download + extract the SIFT-small ANN dataset (10k base × 128-d, 100 queries,
// 100-NN L2 ground truth — ~5 MB) into benchmarks/datasets/siftsmall/ (gitignored).
// Used by benchmarks/real.ts. Requires `curl` and `tar` on PATH.
//
// Run: `node benchmarks/download-siftsmall.mjs`

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(root, 'benchmarks', 'datasets');
const out = join(dir, 'siftsmall');
const URL = 'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz';

if (existsSync(join(out, 'siftsmall_base.fvecs'))) {
  process.stdout.write('siftsmall already present — skipping download.\n');
  process.exit(0);
}

mkdirSync(dir, { recursive: true });
const tarball = join(dir, 'siftsmall.tar.gz');
process.stdout.write(`downloading ${URL} …\n`);
execFileSync('curl', ['-sSL', '--max-time', '120', '-o', tarball, URL], { stdio: 'inherit' });
execFileSync('tar', ['xzf', tarball, '-C', dir], { stdio: 'inherit' });
process.stdout.write(`extracted → ${out}\n`);
