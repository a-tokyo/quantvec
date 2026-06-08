# quantvec — Decision Log (ADR-style)

Concise record of locked decisions and their rationale. Newest first.

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
