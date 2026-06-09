// Copy canonical docs from repo-root docs/*.md into site/content/docs/ for Nextra.
// Edit docs/*.md at the repo root — never the copies here (regenerated on dev/build).

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const SITE = path.resolve(here, '..')
const REPO = path.resolve(SITE, '..')
const SRC = path.join(REPO, 'docs')
const OUT = path.join(SITE, 'content', 'docs')

const PAGES = [
  'getting-started',
  'guide',
  'architecture',
  'serialization',
  'api-reference',
  'benchmarks',
  'roadmap'
]

fs.mkdirSync(OUT, { recursive: true })
let missing = 0
for (const page of PAGES) {
  const src = path.join(SRC, `${page}.md`)
  if (!fs.existsSync(src)) {
    console.error(`copy-docs: MISSING SOURCE docs/${page}.md`)
    missing++
    continue
  }
  fs.copyFileSync(src, path.join(OUT, `${page}.md`))
}
if (missing) process.exitCode = 1
console.log(`copy-docs: copied ${PAGES.length - missing}/${PAGES.length} pages`)
