# quantvec — docs site

[Nextra](https://nextra.site) (Next.js App Router) docs + landing site for **quantvec**. Published
doc pages are **single-sourced** from the repo's canonical markdown in `docs/*.md` at the repo root;
`content/docs/*.md` are copied there on `dev`/`build` via `scripts/copy-docs.mjs`.

## Local development

```bash
cd site
npm install
npm run dev
```

Open http://localhost:3000.

- `npm run build` — production build
- `npm run start` — serve the production build

## What's authored vs shared

| Authored in `site/` (committed) | Shared from repo root (`docs/*.md`) |
| ------------------------------- | ----------------------------------- |
| `content/index.mdx` (landing)   | `getting-started.md`, `guide.md`, … |
| `content/_meta.js`, `content/docs/_meta.js` (nav) | copied to `content/docs/*.md` (gitignored) |
| `content/docs/index.mdx` (docs overview) | `scripts/copy-docs.mjs` |

To change a docs page, edit **`docs/<route>.md`** at the repo root — not the copy under
`content/docs/`. To change which docs are published or their order, edit `content/docs/_meta.js`
and the `PAGES` list in `scripts/copy-docs.mjs`.

### Mermaid diagrams

Fenced ` ```mermaid ` blocks render out of the box (Nextra v4 wires up
`@theguild/remark-mermaid`). See `docs/architecture.md` for an example.

## Deploy (Vercel)

Connect `a-tokyo/quantvec` as a Vercel project with:

- **Root Directory:** `site`
- **Framework Preset:** Next.js
- **Build Command:** `npm run build`
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
