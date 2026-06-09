// Minimal ambient `WebAssembly` surface used by the kernel loader (./kernel.ts).
//
// WebAssembly is a universal runtime global (Node, browsers, Bun, Workers) but TS
// only ships its types in the DOM / WebWorker libs — neither the ES2022 lib (the
// isomorphic core, tsconfig.json) nor @types/node (the node-entry build,
// tsconfig.build.json) declares it. Unlike TextEncoder/atob, it is therefore NOT
// supplied by @types/node, so this file is declared narrowly and is included by BOTH
// tsconfig passes (it is not in either `exclude`), with no clash.

declare namespace WebAssembly {
  class Module {
    constructor(bytes: ArrayBuffer | ArrayBufferView);
  }
  class Instance {
    readonly exports: Record<string, unknown>;
    constructor(module: Module, importObject?: Record<string, Record<string, unknown>>);
  }
  class Memory {
    readonly buffer: ArrayBuffer;
    grow(delta: number): number;
  }
  function validate(bytes: ArrayBuffer | ArrayBufferView): boolean;
}
