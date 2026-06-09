import type { MetadataRoute } from 'next'

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://quantvec.vercel.app'

// Doc routes (mirror of scripts/sync-docs.mjs + the authored overview). Keep in sync
// when adding/removing docs.
const DOC_ROUTES = [
  'getting-started',
  'guide',
  'architecture',
  'serialization',
  'api-reference',
  'benchmarks',
  'roadmap'
]

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date()
  return [
    { url: `${SITE_URL}/`, lastModified: now, changeFrequency: 'weekly', priority: 1 },
    { url: `${SITE_URL}/docs`, lastModified: now, changeFrequency: 'weekly', priority: 0.8 },
    ...DOC_ROUTES.map(d => ({
      url: `${SITE_URL}/docs/${d}`,
      lastModified: now,
      changeFrequency: 'weekly' as const,
      priority: 0.7
    }))
  ]
}
