import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import type { Metadata } from 'next'

const REPO = 'https://github.com/a-tokyo/quantvec'
// Production URL — override on Vercel with NEXT_PUBLIC_SITE_URL once the domain is known.
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://quantvec.vercel.app'
const DESCRIPTION =
  'Data-oblivious, zero-training vector quantization & search for TypeScript. Clean-room TurboQuant + RaBitQ. Runs in Node, browsers, Bun, and edge runtimes.'

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'quantvec',
    template: '%s – quantvec'
  },
  description: DESCRIPTION,
  applicationName: 'quantvec',
  authors: [{ name: 'Ahmed Tokyo', url: 'https://github.com/a-tokyo' }],
  creator: 'Ahmed Tokyo',
  keywords: [
    'vector quantization',
    'vector search',
    'TurboQuant',
    'RaBitQ',
    'approximate nearest neighbor',
    'embeddings',
    'TypeScript',
    'isomorphic',
    'edge',
    'WASM SIMD',
    'RAG'
  ],
  openGraph: {
    type: 'website',
    siteName: 'quantvec',
    title: 'quantvec',
    description: DESCRIPTION,
    url: SITE_URL
  },
  twitter: {
    card: 'summary_large_image',
    title: 'quantvec',
    description: DESCRIPTION
  }
}

const navbar = (
  <Navbar
    logo={<b>quantvec</b>}
    projectLink={REPO}
  />
)

const footer = (
  <Footer>
    Apache-2.0 {new Date().getFullYear()} © <a href="https://ahmedtokyo.com">Ahmed Tokyo</a> · <a href="https://github.com/a-tokyo/quantvec">GitHub</a>.
  </Footer>
)

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head color={{ hue: 265, saturation: 70 }} />
      <body>
        <Layout
          navbar={navbar}
          footer={footer}
          pageMap={await getPageMap()}
          docsRepositoryBase={`${REPO}/tree/main/site`}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
