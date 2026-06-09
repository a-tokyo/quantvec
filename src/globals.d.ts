// Ambient declarations for the WHATWG Encoding API (TextEncoder/TextDecoder).
//
// These are universal web-platform globals — present in every quantvec target
// runtime (Node 18+, browsers, Bun, Cloudflare Workers) — but they are not part of
// the ES2022 standard lib. We declare them here, narrowly, so the isomorphic core
// can use them WITHOUT pulling in the full DOM or @types/node global surface (the
// core compiles with `types: []`; see tsconfig.json). The node-only entry
// (src/node.ts) compiles separately with `@types/node`, which supplies its own
// declarations — hence this file is excluded from that compilation to avoid a clash.

declare class TextEncoder {
  /** UTF-8 encode a string into a fresh Uint8Array. */
  encode(input?: string): Uint8Array;
}

declare class TextDecoder {
  constructor(label?: string, options?: { fatal?: boolean; ignoreBOM?: boolean });
  /** Decode bytes (UTF-8 by default) into a string. */
  decode(input?: ArrayBuffer | ArrayBufferView): string;
}

/** Decode a base64 string to a binary (latin1) string. Universal global. */
declare function atob(data: string): string;
