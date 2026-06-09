# quantvec — docs site

[Nextra](https://nextra.site) (Next.js App Router) docs + landing site for **quantvec**. The
documentation pages are **not authored here** — they are synced from the repo's canonical markdown
in `docs/site/*.md` at build time, so the site can't drift from the source.

## Local development

```bash
cd site
npm install
npm run dev        # runs scripts/sync-docs.mjs, then next dev
```

Open http://localhost:3000.

- `npm run sync` — regenerate `content/docs/*.md` from `docs/site/*.md` (also runs automatically before `dev`/`build`).
- `npm run build` — production build (runs the sync first).
- `npm run start` — serve the production build.

## What's authored vs generated

| Authored (committed)                              | Generated (gitignored)                       |
| ------------------------------------------------- | -------------------------------------------- |
| `content/index.mdx` (landing)                     | `content/docs/*.md` (synced from `docs/site/`) |
| `content/_meta.js`, `content/docs/_meta.js` (nav) |                                              |
| `content/docs/index.mdx` (docs overview)          |                                              |
| `scripts/sync-docs.mjs`, app/config               |                                              |

To change a docs page, edit its **source** in `docs/site/<route>.md` at the repo root — never the
generated copy under `content/docs/`. To change which docs are published or their order, edit
`scripts/sync-docs.mjs` and `content/docs/_meta.js`.

### Mermaid diagrams

Fenced ` ```mermaid ` blocks render out of the box (Nextra v4 wires up
`@theguild/remark-mermaid`). See `docs/site/architecture.md` for an example.

## Deploy (Vercel)

Connect `a-tokyo/quantvec` as a Vercel project with:

- **Root Directory:** `site`
- **Framework Preset:** Next.js
- **Build Command:** `npm run build` (the `prebuild` hook runs the doc sync)
- **Install Command:** `npm install`

Auto-deploys on push to `main`.

**Env var (recommended):** set `NEXT_PUBLIC_SITE_URL` to the production URL (e.g. your Vercel domain
or a custom domain). It drives `metadataBase` for absolute SEO URLs — canonical, Open Graph, Twitter
card, the generated OG image, `robots.txt`, and `sitemap.xml`. Defaults to
`https://quantvec.vercel.app` if unset.

SEO is handled via the Next.js Metadata API in `app/layout.tsx` (title template, description,
keywords, Open Graph + Twitter), a generated OG image (`app/opengraph-image.tsx`), favicon
(`app/icon.svg`), plus `app/robots.ts` and `app/sitemap.ts`. Per-page `<title>`s come from each
doc's frontmatter/heading.
