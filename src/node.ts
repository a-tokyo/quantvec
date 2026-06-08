// quantvec/node — Node-only convenience helpers (filesystem save/load).
//
// Isolated from the core so the main entry stays isomorphic (no `node:*` imports
// leak into browser/Workers builds). Implemented in the serialization wave; will
// wrap the runtime-agnostic toBytes()/fromBytes() with `node:fs/promises`.
export {};
