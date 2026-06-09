// Single source of truth: the site renders the repo's canonical markdown.
// This copies each source doc (docs/site/<route>.md) into
// site/content/docs/<route>.md (gitignored) and rewrites links so they resolve on
// the web:
//   - links to another synced doc      -> that doc's site route (/docs/<route>)
//   - links to any other repo file/dir -> absolute GitHub blob/tree URL
// It never edits the source files. Run via `prebuild`/`predev` (and `npm run sync`).
//
// The Overview page (content/docs/index.mdx) is authored directly, not synced.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const SITE = path.resolve(here, '..')
const REPO = path.resolve(SITE, '..')
const OUT = path.join(SITE, 'content', 'docs')
const GH = 'https://github.com/a-tokyo/quantvec'

// src is repo-root-relative; route is the page under /docs. Order mirrors
// content/docs/_meta.js (index/Overview is authored, so it is not listed here).
const DOCS = [
  { src: 'docs/site/getting-started.md', route: 'getting-started' },
  { src: 'docs/site/guide.md', route: 'guide' },
  { src: 'docs/site/architecture.md', route: 'architecture' },
  { src: 'docs/site/serialization.md', route: 'serialization' },
  { src: 'docs/site/api-reference.md', route: 'api-reference' },
  { src: 'docs/site/benchmarks.md', route: 'benchmarks' },
  { src: 'docs/site/roadmap.md', route: 'roadmap' }
]

const norm = p => path.posix.normalize(String(p).replace(/\\/g, '/')).replace(/^\.\//, '')
const routeByRepoPath = new Map(DOCS.map(d => [norm(d.src), `/docs/${d.route}`]))

function rewriteLink(target, srcDir) {
  const m = target.match(/^([^#?]*)([#?].*)?$/)
  const bare = m[1]
  const suffix = m[2] || ''
  if (!bare || /^(https?:|mailto:|\/|#)/.test(bare)) return target

  const relToSrc = norm(path.posix.join(srcDir, bare))
  const relToRoot = norm(bare)
  for (const cand of [relToSrc, relToRoot]) {
    if (routeByRepoPath.has(cand)) return routeByRepoPath.get(cand) + suffix
  }
  // Not a synced doc -> point at GitHub. Prefer the candidate that exists on disk.
  let repoPath = relToRoot
  if (fs.existsSync(path.join(REPO, relToSrc))) repoPath = relToSrc
  else if (fs.existsSync(path.join(REPO, relToRoot))) repoPath = relToRoot
  const abs = path.join(REPO, repoPath)
  const isDir = fs.existsSync(abs) && fs.statSync(abs).isDirectory()
  return `${GH}/${isDir ? 'tree' : 'blob'}/main/${repoPath}${suffix}`
}

function transform(content, srcDir) {
  // inline markdown links: ](target) and ](target "title")
  let out = content.replace(/\]\(([^)\s]+)(\s+"[^"]*")?\)/g, (_full, target, title = '') => {
    return `](${rewriteLink(target, srcDir)}${title})`
  })
  return out
}

fs.mkdirSync(OUT, { recursive: true })
let missing = 0
for (const d of DOCS) {
  const abs = path.join(REPO, d.src)
  if (!fs.existsSync(abs)) {
    console.error(`sync-docs: MISSING SOURCE ${d.src}`)
    missing++
    continue
  }
  const raw = fs.readFileSync(abs, 'utf8')
  const out = transform(raw, path.posix.dirname(norm(d.src)))
  fs.writeFileSync(path.join(OUT, `${d.route}.md`), out)
}
if (missing) process.exitCode = 1
console.log(`sync-docs: wrote ${DOCS.length - missing}/${DOCS.length} docs to site/content/docs/`)
