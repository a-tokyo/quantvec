import { ImageResponse } from 'next/og'

export const alt = 'quantvec — data-oblivious, zero-training vector quantization & search'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

// Generated OG/Twitter card image (Next file convention sets both og:image and twitter:image).
export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: '100%',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          background: '#0a0a0a',
          color: '#ffffff',
          padding: '80px',
          fontFamily: 'sans-serif'
        }}
      >
        <div style={{ display: 'flex', fontSize: 28, letterSpacing: 6, color: '#a78bfa', textTransform: 'uppercase' }}>
          quantvec
        </div>
        <div style={{ display: 'flex', fontSize: 72, fontWeight: 700, lineHeight: 1.05, marginTop: 28, maxWidth: 1000 }}>
          Zero-training vector quantization & search.
        </div>
        <div style={{ display: 'flex', fontSize: 30, color: '#a1a1aa', marginTop: 28, maxWidth: 940 }}>
          Clean-room TurboQuant + RaBitQ for TypeScript. Node, browsers, Bun, and the edge.
        </div>
      </div>
    ),
    { ...size }
  )
}
