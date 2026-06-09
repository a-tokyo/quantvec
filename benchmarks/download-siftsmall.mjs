// Download + extract the SIFT-small ANN dataset (10k base × 128-d, 100 queries,
// 100-NN L2 ground truth — ~5 MB) into benchmarks/datasets/siftsmall/ (gitignored).
// Used by benchmarks/real.ts. Requires `curl` and `tar` on PATH.
//
// Run: `node benchmarks/download-siftsmall.mjs`

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(root, 'benchmarks', 'datasets');
const out = join(dir, 'siftsmall');

// Official academic source (IRISA/Inria) — only available via FTP; no HTTPS mirror exists.
const DOWNLOAD_URL = 'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz'; // codeql[js/insecure-download] — academic FTP-only source; integrity verified below
const EXPECTED_MD5 = '0b8324a7a82d7f2663d7dcbd57642df7';

if (existsSync(join(out, 'siftsmall_base.fvecs'))) {
  process.stdout.write('siftsmall already present — skipping download.\n');
  process.exit(0);
}

mkdirSync(dir, { recursive: true });
const tarball = join(dir, 'siftsmall.tar.gz');
process.stdout.write(`downloading ${DOWNLOAD_URL} …\n`);
execFileSync('curl', ['-sSL', '--max-time', '120', '-o', tarball, DOWNLOAD_URL], {
  stdio: 'inherit',
});

const actualMd5 = createHash('md5').update(readFileSync(tarball)).digest('hex');
if (actualMd5 !== EXPECTED_MD5) {
  process.stderr.write(`integrity check failed: expected MD5 ${EXPECTED_MD5}, got ${actualMd5}\n`);
  process.exit(1);
}

execFileSync('tar', ['xzf', tarball, '-C', dir], { stdio: 'inherit' });
process.stdout.write(`extracted → ${out}\n`);
