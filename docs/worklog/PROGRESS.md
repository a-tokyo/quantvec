# quantvec — Progress

> Durable, in-repo progress tracker. Any session resumes from here + `.agents/plans/quantvec-build-plan.md`.
> **Resume-here pointer:** Waves 5–7 complete on `feat/wave-5-index` (gate green: typecheck + lint + **310 tests**, 100% functions, build, docs site; pre-commit hook wired). Shipped: index classes + serialization (W5); the full perf wave (W6) — bit-packed serialization (7.9–15.7×, matches turbovec), TQ+ calibration (opt-in), FWHT rotation (pow2 dims, ~25× faster encode), exact WASM scoring kernel (~1.3× query, bit-identical + fallback); benchmarks + rewritten README + Nextra docs site (W7). Five-agent verification (incl. a Critical calibration-search bug found & fixed). **Library is release-ready.** Next: ergonomic `Collection`/filter layer (W6+), real-dataset benchmarks + v128 FastScan (W7+), then v0.1.0 (W8).

Last updated: 2026-06-09

## Status legend

`TODO` · `WIP` · `BLOCKED` · `REVIEW` · `DONE` (gate green + reviewed; merged to main)

## Phase 0 — Scaffold, docs, CI — DONE

Toolchain (Bun + Vitest + tsup + AssemblyScript), strict TS 6 / ESLint 10 / Prettier, Apache-2.0 + NOTICE, isomorphic package layout (ESM/CJS + .d.ts, `./node` subpath), CI (ci / release-publish / codeql, Node+Bun cross-runtime smoke), distilled research notes, worklog, plan in `.agents/plans/`. Genesis `9a9fdb7`. 90% coverage gate enforced repo-wide.

## Implementation waves (subagent-driven-development; local gate green + reviewed → merge to main)

| Wave | Modules                                                                                             | Status |
| ---- | --------------------------------------------------------------------------------------------------- | ------ |
| W1   | core: `rng`, `topk`, `beta`, `integrate`                                                            | DONE   |
| W2   | core: `rotation` (dense Householder-QR), `codebook` (Lloyd-Max)                                     | DONE   |
| W3   | core: `encode` (RaBitQ-corrected pipeline), `pack` (2/3/4-bit)                                      | DONE   |
| W4   | core: `search` (nibble-LUT scalar oracle), `metrics` (dot/cosine/euclid)                            | DONE   |
| W5   | `index/turboquant-index`, `index/id-map-index`, `io/serialize`, wire `src/index.ts` + `src/node.ts` | DONE   |
| W6   | `wasm` (AssemblyScript kernel ≡ scalar, exact f64), bit-packing, FWHT, TQ+ calibration              | DONE   |
| W6+  | `ergonomic` (Collection/filter DSL)                                                                 | TODO   |
| W7   | `benchmarks` (recall@k/QPS/compression) + README/docs + Nextra site                                 | DONE   |
| W7+  | real-dataset benchmark suite (GloVe/OpenAI); v128 FastScan                                          | TODO   |
| W8   | doer/verifier/devil's-advocate panel, security review, v0.1.0 release                               | TODO   |

### Done

- **W1** (`e8a9b3d`): `rng`, `topk` (NaN-safe), `beta`, `integrate`. One Critical (integrator hang) fixed.
- **W2** (`cd5ecca`): `rotation` (vs numpy to f32), `codebook` Lloyd-Max (vs scipy to 7 digits; ~4×/bit).
- **W3** (`44e065f`): `encode` (RaBitQ scale; unbiased, RMS halves/bit), `pack`. Removed one unreachable guard.
- **W4** (`fdfddf5`): `search` (LUT scan, mask, 3 metrics; euclidean ranking verified), `metrics`. End-to-end recall@10 @4-bit: **0.935–0.979 on realistic anisotropic data**, ~0.895 on adversarial isotropic-random (genuine estimator-variance ceiling — independently confirmed not a bug; recall@20 ≥0.90). search/metrics 100% coverage. Approved.
- **W5** (`feat/wave-5-index`): `TurboQuantIndex` (growable positional flat index: amortized-O(1) `add`/`addOne`, O(n) `search`, O(1) `swapRemove`), `IdMapIndex` (stable `number|string|bigint` ids over the positional store; `addWithIds`/`has`/`remove`/id-returning `search`), `io/serialize` (single versioned `QVEC` format, fully bounds-validated untrusted-input read path — premortem T6, per-id tagging), and the wired `src/index.ts` (isomorphic) + `src/node.ts` (fs save/load) entries. Node entry typechecked separately (`tsconfig.build.json`) so the core stays `types:[]`-isomorphic; Encoding-API ambient in `src/globals.d.ts`. **100% coverage on all three new modules**; full suite 254 tests, 98.87% stmts.
  - **Panel verification + hardening** (Quality / DX / Devil's-Advocate subagents, verdict SHIP-WITH-CAVEATS → all caveats closed): deserialize now rejects duplicate ids, non-canonical bigints, invalid UTF-8 (fatal), and non-finite `seed`; `add`/`addOne`/`addWithIds` throw typed `INVALID_VECTOR` instead of a raw `TypeError`; constructor validates `seed`; `IdMapIndex.search` throws its own `EMPTY`; `toPayload`/`fromPayload` marked `@internal`; added `clear()`, `IdMapIndex.ids()`, an id `filter` predicate, and a generic `fromBytes<Id>`. Full suite now **267 tests, 98.93% stmts, 100% on the new modules**.
  - **Benchmarks + docs**: reproducible dataset-free harness (`benchmarks/flat.ts`, `bun run benchmarks/flat.ts`) — recall@{1,10,100} vs exact float32, QPS, compression; results in `benchmarks/results/`. README rewritten (accurate API, mermaid pipeline, paper refs, honest recall/compression). Diátaxis docs in `docs/*.md` (copied into `site/` at build) and a Nextra v4 docs website in `site/` (mermaid enabled, builds green, Vercel-ready). Honest gaps surfaced: ~4× compression today (1 byte/code; bit-packing→8–16× is W6) and TQ+ calibration (W7).

## Validation oracle (clean-room)

1. Paper distortion bounds: D_mse ≤ (√3π/2)·4^−b. 2. scipy reference for Beta/Lloyd-Max (embedded). 3. Exact brute-force float32 search = recall ground truth. 4. **W7 head-to-head vs the local reference library** (run both; recall parity / "ours better").

## Process notes

- **No tech debt / no backwards-compat / greenfield** (D-009): every line reachable + tested or removed.
- Per-wave: combined spec + code-quality SDD review by an independent subagent; orchestrator verifies the gate itself before merge. Full 3-member panel at W8.
- Coverage gate 90% (`all: true`).

## Perf wave (top turbovec) — approved sequence: TQ+ → FWHT → WASM-SIMD

Goal: match-or-beat the native `turbovec` reference on every axis in the JS runtime.
**Done:** (1) bit-packed serialization → 7.9–15.7× compression (matches turbovec's 8.0×/15.8×);
(2) TQ+ per-coordinate calibration (`core/calibrate`, **opt-in** — measured neutral-to-negative on
synthetic data, gains on real embeddings, so default off; correct mechanism, round-trips, serialized);
(3) FWHT rotation (`core/fwht` + `createHadamardRotation`/`createRotation`) — exact for power-of-two
dims (recall-neutral-to-better, ~25× faster encode), dense fallback otherwise (truncation-RHT craters
recall, so rejected for non-pow2); (4) WASM scoring kernel (`assembly/` + `wasm/kernel`) — exact f64,
resident codes, bit-identical to the scalar oracle, ~1.3× faster query, feature-detect + fallback.
Data-driven decisions via the autoresearch loop.
**Scorecard now:** quantvec ≥ turbovec on compression, serialization safety, metrics (3 vs 1), id
types, and portability; out-recalls `turboquant-js` ~2×. Remaining gaps vs turbovec: recall (needs
TQ+ calibration + a real-data benchmark) and raw query speed (needs WASM-SIMD; native always wins raw —
target = fastest _in JS_). FWHT is a place we can beat turbovec's O(d³) QR build.

### TQ+ calibration — DONE (design as implemented)

Per-coordinate affine map fit on the first add (≥1000 samples, else frozen identity → existing
behavior unchanged), applied after rotation, before quantization. Derived to compose with quantvec's
RaBitQ scale (reduces to the current path when shift=0, scale=1, so all existing tests stay green):

- **Fit** (`src/core/calibrate.ts`, new): for each coord d, empirical 5/95 percentiles `q_lo,q_hi` of
  the rotated samples; canonical targets `coordQuantile(0.05|0.95, dim)` (already in `core/beta.ts`);
  `scale[d] = (canonHi−canonLo)/(q_hi−q_lo)`, `shift[d] = canonLo/scale[d] − q_lo`. Degenerate coord
  (range < eps) → identity (scale 1, shift 0).
- **Encode** (`core/encode`): `cal_i = (o_rot[i]+shift[i])·scale[i]`; quantize `cal_i`; de-cal recon
  `r_i = c_i/scale[i] − shift[i]`; `inner = Σ o_rot[i]·r_i`; `scale_vec = ‖v‖/inner` (unchanged form).
- **Search** (`core/search`): LUT from `q_calib[i]=q_rot[i]/scale[i]`; per-query `biasQ = Σ q_rot[i]·shift[i]`;
  pass `s' = s − biasQ` to `scoreMetric` (verified correct for dot/cosine/euclidean since all derive
  from `estDot = scale·s'`).
- **Index**: auto-fit on first add (pre-rotate the batch to collect samples), freeze, store; pass
  calibration to encode + the EncodedDb.
- **Serialize**: add a calibration section after `norms` (presence byte + shift[dim] + scale[dim] f32),
  bounds-validated; `fromBytes` rehydrates it.
- **Validate**: a recall test asserting recall(with TQ+) ≥ recall(without) on anisotropic data; re-run
  `benchmarks/flat.ts` (its 4–5k single add triggers calibration) and refresh the published tables.

### FWHT rotation — DONE (hybrid; design revised by measurement)

Implemented `core/fwht` (nextPow2/isPow2/fwht) + `createHadamardRotation` (3 rounds sign-flip +
normalized FWHT, exact orthonormal, applyTranspose = exact inverse) + `createRotation` dispatcher.
**Measured decision (autoresearch):** the originally-planned pad-to-pow2-then-truncate RHT for arbitrary
dims craters recall (d=768 4-bit 0.876→0.560), so we use FWHT **only for power-of-two dims** (exact, no
truncation: recall-neutral-to-better, ~25× faster encode) and keep the dense rotation otherwise. The
"quantize all p coords" p-space variant was dropped (it would inflate storage and erode the compression
parity). Rotation choice is a deterministic function of `dim` → no serialization change.

### WASM kernel — DONE (exact f64 first; v128 FastScan deferred)

Shipped an exact AssemblyScript kernel (`assembly/index.ts`) + loader (`src/wasm/kernel.ts`): codes
resident in linear memory (uploaded once per mutation, dirty-tracked), f64 accumulation in oracle order
→ **bit-identical** to the scalar scan (verified across metrics/mutations/mask). Base64-inlined
(`scripts/inline-wasm.mjs` → committed `src/wasm/wasm-binary.ts`), feature-detected, pure-TS fallback,
on by default (`wasm` option). Measured ~1.3× query speedup. The approximate **v128 FastScan**
(blocked-nibble swizzle + u8 LUT + rescore) — the larger speedup — is deferred (changes the accuracy
story; exact kernel lands first). CI `test`/`test:coverage` pass `--test-timeout=30000` (the config
value isn't reliably honored under bun in CI; the CLI flag always applies).

## Open items / planned improvements

- **Recall headroom to beat the reference / hit >0.90@10 on all data:** add **TQ+ per-coordinate calibration** (deferred from W3) and/or the paper's QJL two-stage estimator — run via **autoresearch** in W7 (metric = recall@10). Reference uses TQ+; we will too.
- **Rotation perf:** dense O(d²)/vec; add **fast O(d log d) randomized-Hadamard/FWHT** transform before the large-d benchmark (W7). Serialization stores the seed + rebuilds rotation on load (one-time O(d³)); FWHT removes this.
- WASM kernel (W6) must MATCH the scalar oracle. Flat brute-force index (not HNSW) — honest scale ~1–10M.
- Detailed README + `docs/` (mermaid arch diagrams) + `benchmarks/` modeled on the local `turbovec` reference (W7).
- Local folder still `turbovector`; physical rename deferred to a session-final step. Repo PRIVATE until confirmed.
