# quantvec — Decision Log (ADR-style)

Concise record of locked decisions and their rationale. Newest first.

## D-016 · IVF as an option on TurboQuantIndex (not a class); format v2; train-on-first-batch

The IVF coarse quantizer ships as `ivf: { nlist, nprobe? }` on `TurboQuantIndexOptions`, NOT a
separate index class: `IdMapIndex` wraps `TurboQuantIndex` and `Collection` wraps `IdMapIndex`, so a
constructor option gives all three layers full parity (config + per-query `nprobe` + remove + ser/de)
with two passthrough lines each. State lives in `index/coarse.ts` (`CoarseQuantizer`: centroids +
posting lists with slot→list/slot→pos arrays for O(1) swap-remove patching) over `core/kmeans.ts`
(seeded k-means++/Lloyd; spherical for cosine/dot, L2 for euclidean; domain-separated RNG stream from
the index seed). Training mirrors calibration: fit-and-freeze from the first non-empty batch when it
has ≥ `nlist` vectors (the hard k-means floor — predictable from the user's own config; quality
guidance ≥ ~32·nlist lives in docs), else frozen flat forever. The probed-cell scan is the exact
scalar kernel (`searchSlots` in `core/search.ts`, sharing `prepareScan` validation with `searchFlat`)
→ `nprobe = nlist` ≡ flat scan bit-for-bit (the IVF analog of the WASM≡scalar oracle); the
whole-database WASM/FastScan kernels are bypassed while IVF is active (cell-resident kernel = future
wave). Serialization: format `VERSION` 1 → 2 with an always-written ivf presence byte (mirror of the
calibration byte) + `nlist/nprobe/centroids/listForSlot` (postings rebuilt on load); v2-only readers
per D-010 — a v1 reader rejects v2 cleanly with `BAD_VERSION` instead of misparsing. Measured
(20k × 768-d clustered, 4-bit): 11.4× QPS at the flat scan's recall (nprobe = nlist/16).

## D-015 · WASM kernel: exact f64 + resident codes, not approximate FastScan (first)

The WASM acceleration (`assembly/index.ts` + `src/wasm/kernel.ts`) ships as an EXACT kernel: codes
resident in linear memory (uploaded once per mutation via a dirty flag, not per query — a per-query
copy would cost as much as the scan), f64 accumulation in the same coordinate order as the scalar
oracle → **bit-identical** results (verified: `wasm: true` ≡ `wasm: false` for all metrics, after
mutations, with masks). It is base64-inlined (`scripts/inline-wasm.mjs` → `src/wasm/wasm-binary.ts`,
committed) for zero-config isomorphic loading, feature-detected, and falls back to the pure-TS scan
(the kernel is an optimization, never a dependency). On by default (`wasm` option). Measured ~1.3×
query speedup. The approximate **v128 FastScan** (blocked-nibble swizzle + u8 LUT + top-k rescore) is a
larger speedup deferred as a follow-up — it changes the accuracy story (needs rescore), so the exact
kernel lands first. `WebAssembly` is declared in `src/wasm/webassembly.d.ts` (not @types/node-provided,
so included by both tsconfig passes); the no-WASM fallback branches are environment-conditional and not
covered by the Node test env (the overall 90% gate still holds).

## D-014 · FWHT rotation for power-of-two dims (hybrid, exact-only)

`createRotation(dim, seed)` dispatches to a Randomized Hadamard Transform
(`createHadamardRotation`: 3 rounds of random sign-flip + normalized FWHT, exact orthonormal,
O(d·log d) apply, ~O(d) build) when `dim` is a power of two, and to the dense Householder rotation
otherwise. Measured (autoresearch THINK→RUN→MEASURE→DECIDE): at a power-of-two dim the FWHT is
recall-neutral-to-better (e.g. d=1024 4-bit recall@10 0.910 vs dense 0.894) and ~25× faster to encode;
but the pad-to-pow2 **truncation** RHT used for non-power-of-two dims **craters recall** (d=768 4-bit
0.876 → 0.560 — likely why the `turboquant-js` namesake scores so low), so we keep the dense rotation
there. The choice is a deterministic function of `dim`, so a serialized index rebuilds the identical
rotation with no extra format field.

## D-013 · TQ+ calibration composes with RaBitQ; opt-in; data-dependent

TurboQuant+ per-coordinate calibration (`core/calibrate`) is **opt-in** (`calibrate: true`, default
off). When enabled it is fit from the first add of ≥1000 vectors (else frozen identity), frozen, and
serialized. Rather than the paper's QJL two-stage, it composes with quantvec's RaBitQ scale: the encoder
quantizes `(o_rot+shift)·scale` and accumulates the inner product against the _de-calibrated_
reconstruction `c/scale − shift`, and search scores `q_rot/scale` minus the per-query bias
`⟨q_rot,shift⟩` — algebraically reducing to the un-calibrated pipeline when shift=0/scale=1 (so the
default path and all its tests are unchanged). **Measured neutral-to-slightly-negative on synthetic
data** (d=64: 0.927→0.880; d=768: 0.889→0.882) because the random rotation already yields near-canonical
coordinates and the de-calibration mostly amplifies quantization error; turbovec's +0.3–1.4pp gains are
on _real low-dim_ embeddings. Hence default off — correct mechanism, validate before enabling.

## D-012 · Node entry compiled separately so the core stays isomorphic

The core must not depend on Node globals (D: `tsconfig.json` sets `types: []`). But the
Node-only entry `src/node.ts` needs `node:fs/promises` + `@types/node`. We keep the
isomorphic guarantee AND type the Node entry by splitting the typecheck: the base
`tsconfig.json` (excludes `src/node.ts`, `types: []`) proves the core never touches Node
globals; `tsconfig.build.json` (adds `types:["node"]`, excludes the ambient
`src/globals.d.ts`) typechecks the Node entry and drives tsup's `.d.ts` generation.
`TextEncoder`/`TextDecoder` (used by the serializer, universal across all runtimes but
absent from the ES2022 lib) are declared narrowly in `src/globals.d.ts` for the core
compile, and supplied by `@types/node` in the build compile — hence excluded there to
avoid a duplicate-global clash.

## D-011 · Index serialization = one `QVEC` versioned format, untrusted-input-hardened

Both index classes serialize through `io/serialize` into a single little-endian format:
24-byte header (`QVEC` magic, version, kind, metric, bits, dim, n, seed) then
codes·scales·norms (codes **bit-packed** at the true 2/3/4 bits per coordinate via
`core/pack` — dim is a multiple of 8 so `n·dim·bits` is a whole byte count; serialized
size is 7.9–15.7× smaller than float32, on par with native TurboQuant), and for the
id-keyed index a per-id-tagged column supporting
`number` (f64), `string` and `bigint` (u32-length + UTF-8). An index is fully
reconstructable from (dim, bits, seed) — which regenerate rotation + codebook — so only
the compact codes/scales/norms (+ ids) are stored. The read path treats input as
UNTRUSTED: magic, version, and every field/length are validated against the buffer size
**before any bulk read or allocation**, so a crafted header cannot cause an OOB read or
an OOM (premortem T6). `fromBytes` rejects a kind mismatch (`WRONG_KIND`).

## D-010 · Single clean serialization format — no backwards-compat

This is a greenfield project: no released format exists, so the binary index format
ships as ONE clean versioned layout (magic + version byte for future-proofing) with
NO migration/back-compat branches for older versions. If the format ever changes
pre-1.0 we bump and rewrite, not maintain legacy readers.

## D-009 · No tech debt, no backwards-compat, no dead guards (user rule)

Everything is built from scratch, so we never accumulate debt: no TODO/defer notes,
no speculative backwards-compat, no unreachable "defensive" branches kept just in
case. Every line must be reachable and tested, or removed. Reviewers enforce this;
e.g. the W3 `inner<0` clamp was provably unreachable and untested → removed, not kept.

## D-008 · noUncheckedIndexedAccess = false

Core is numeric typed-array hot-loop code; the flag would force `| undefined` on every
TypedArray read. We validate at boundaries (R8) and keep loops clean/fast instead.

## D-007 · Beta distribution implemented in-house (zero runtime deps)

Lloyd-Max codebooks need Beta pdf/cdf/quantile. We implement them ourselves (clean-room,
zero-dependency core, no licensing/runtime weight) and validate against an independent
scipy reference + the paper's distortion bounds. Keeps the published package dependency-free.

## D-006 · Local folder rename deferred

Renaming the live working directory mid-session breaks the shell's cwd and every later
command. Package name, git remote, and GitHub repo are all `quantvec`; the physical folder
`dev/a-tokyo/turbovector` → `quantvec` rename happens as a session-final step.

## D-005 · Toolchain = Bun + Vitest + tsup (NOT bun-native)

Bun for PM/runtime/dev speed; Vitest run on BOTH Node and Bun in CI to prove isomorphism;
tsup for the published dual ESM/CJS + bulletproof `.d.ts`. For a portability-first library,
cross-runtime test coverage and solid type declarations outrank single-runtime dev speed,
so we do not use `bun test`/`bun build`. Published artifact is standard ESM/CJS regardless.

## D-004 · License = Apache-2.0

Explicit patent grant — prudent since this implements Google's published TurboQuant algorithm.

## D-003 · API scope = core TurboQuant index + qdrant-inspired ergonomic layer (both v1)

`TurboQuantIndex` + `IdMapIndex` (faithful TurboQuant index) AND a `Collection`/`Point`/
filter-DSL ergonomic layer informed by qdrant.

## D-002 · WASM-SIMD via AssemblyScript, first-class

v128 SIMD kernel in AssemblyScript (keeps the project TS-coherent, runs in every WASM host).
A pure-TS scalar kernel is the always-on fallback AND the correctness oracle the WASM must match.

## D-001 · Clean-room implementation from the papers only

quantvec is implemented solely from TurboQuant (arXiv:2504.19874) + RaBitQ (arXiv:2405.12497)
and our own design — never referencing any existing library. Code cites equations/theorems,
not external code. Name: `quantvec` (npm-available). GitHub: `a-tokyo/quantvec`.
