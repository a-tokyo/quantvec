# quantvec — Production-grade TurboQuant vector search for TypeScript

## Context

**quantvec** is an open-source, isomorphic TypeScript vector-search library that is a **clean-room implementation of Google Research's TurboQuant algorithm** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni), combined with the RaBitQ unbiased-estimator correction ([arXiv:2405.12497](https://arxiv.org/abs/2405.12497), SIGMOD 2024). It is implemented **solely from the published papers and our own design** — not derived from any existing codebase — and will be published to npm and developed in the open.

**Why:** The JS/TS ecosystem has no production-grade, *data-oblivious, zero-training* vector quantizer. TurboQuant's core property — a random rotation makes every coordinate follow a known Beta distribution, so an optimal per-coordinate scalar quantizer needs **no training and ~zero indexing time** (paper Table 2: ~0.0007–0.0021s vs PQ 37–494s) — is uniquely suited to edge/serverless/browser runtimes where you can't ship a trained codebook or run k-means in-process. quantvec brings near-Shannon-optimal compression and high recall to anywhere JS runs (Node, browser, Cloudflare Workers, React Native), with a WASM-SIMD hot loop for throughput.

**Decisions locked (from user):**
| Decision | Choice |
|---|---|
| Package name | **`quantvec`** (npm-available; verified 404 on registry) |
| Runtime/perf | **WASM-SIMD as a first-class v1 deliverable**, with a pure-TS scalar kernel as the correctness-defining, always-on fallback |
| API scope | **Core TurboQuant index + a qdrant-inspired ergonomic layer**, both in v1 |
| License | **Apache-2.0** (explicit patent grant; implements Google's published algorithm) |

**Open recommendation (overridable):** Use **AssemblyScript** for the WASM-SIMD kernel (v128 SIMD, TS-dialect — keeps the project TypeScript-coherent, tiny `.wasm`, runs in every WASM host). Alternative is Rust→wasm for more speed at the cost of a second toolchain; we can A/B via autoresearch if AssemblyScript SIMD underperforms.

---

## Distilled technical understanding (from the papers only)

### TurboQuant (Google, arXiv:2504.19874)
1. **Random rotation** Π ∈ ℝ^{d×d}, orthonormal, from QR of a Gaussian matrix; **data-independent**. After rotation a unit vector is uniform on the sphere, so each coordinate ~ **Beta((d-1)/2,(d-1)/2)** ≈ N(0,1/d), and coordinates are near-independent in high d. This is why quantization needs **no training data** (the "data-oblivious" claim).
2. **Q_mse (MSE-optimal):** per-coordinate **Lloyd-Max** scalar quantizer for that fixed Beta distribution; codebooks precomputed numerically for b = 2,3,4 bits. **Theorem 1:** D_mse ≤ (√3·π/2)·4^−b ≈ {0.36, 0.117, 0.030, 0.009} for b=1..4 — within **≈2.7×** of the info-theoretic lower bound 4^−b (**Theorem 3**). *These per-bit numbers are a measurable correctness oracle.*
3. **Q_prod (unbiased inner product):** MSE quantizers are biased for dot products (≈2/π at 1 bit); the paper's two-stage fix applies Q_mse at (b-1) bits then a **1-bit Quantized-JL (QJL) transform on the residual**, giving an unbiased estimator with variance ≤ (π/2d)·‖y‖² (**Theorem 2**).
4. **Validated** on KV-cache quantization (quality-neutral at 3.5 bits/channel) and nearest-neighbor search (DBpedia/OpenAI d=1536 & 3072, GloVe d=200): >90% recall@10 at 2–4 bits with ~zero indexing time, beating PQ on recall.

### RaBitQ correction (arXiv:2405.12497)
Same random-rotation preprocessing; corrects the inner-product bias with a **per-vector length-renormalization scale** (an unbiased estimator with a closed-form correction factor) rather than a residual transform. Cheaper to apply and SIMD-friendly via bitwise/popcount scoring.

### quantvec's chosen pipeline (our design)
Per vector: `normalize (store ‖v‖) → rotate → optional per-coordinate calibration → Lloyd-Max quantize for Beta → bit-pack → store per-vector RaBitQ length-renormalization scale`. We adopt **RaBitQ scaling** as the v1 inner-product corrector (simplest correct, SIMD-friendly — R2/R4) and treat the **paper's QJL two-stage as an autoresearch experiment** to evaluate against it. quantvec is a **flat/brute-force quantized index** (linear SIMD scan over 2–4-bit codes — like FAISS `IndexPQFastScan`), not an HNSW graph; sublinear coarse-quantizer/IVF or HNSW is explicit future roadmap, not v1.

### qdrant-inspired ergonomics (public docs)
`Distance` enum (`cosine`/`dot`/`euclid`); point = `{id, vector, payload}`; filter DSL `must`/`should`/`must_not` + `range`/`match`/`hasId`; search params `limit`/`filter`/`oversampling`/`rescore`/`exact`. Our engine underneath; filters compile to the core allowlist/mask.

---

## Target architecture

Single npm package, idiomatic TS, typed arrays throughout (`Float32Array`/`Uint8Array`/`DataView`).

```
quantvec/
├── src/
│   ├── index.ts                 # public exports (+ subpath for Node fs helpers)
│   ├── core/
│   │   ├── rng.ts               # seeded deterministic RNG (xoshiro/PCG)
│   │   ├── rotation.ts          # orthonormal matrix via Householder QR of a Gaussian matrix
│   │   ├── beta.ts              # Beta pdf/cdf/quantile + adaptive Simpson integration
│   │   ├── codebook.ts          # Lloyd-Max boundaries+centroids per (dim,bits)
│   │   ├── encode.ts            # normalize→rotate→calibrate→quantize→pack→scale
│   │   ├── pack.ts              # bit-plane ↔ blocked layout for fast scoring
│   │   ├── search.ts            # nibble-LUT build + scalar scoring + top-k
│   │   ├── topk.ts              # bounded min-heap (named DS — R4)
│   │   └── metrics.ts           # Distance enum: cosine|dot|euclid (via stored norms)
│   ├── wasm/                    # AssemblyScript v128 kernel + feature-detect glue + TS fallback
│   ├── index/
│   │   ├── turboquant-index.ts  # positional index (i64/number indices)
│   │   └── id-map-index.ts      # stable id<->slot mapping
│   ├── ergonomic/
│   │   ├── collection.ts        # createCollection / upsert / search / delete
│   │   ├── filter.ts            # must/should/must_not/range/match/hasId → mask
│   │   └── types.ts             # Point<P>, SearchParams, Filter, typed errors
│   └── io/serialize.ts          # versioned ArrayBuffer format; toBytes/fromBytes (validated)
├── assembly/                    # AssemblyScript source (asc → build/quantvec.wasm)
├── test/  benchmarks/  docs/research/  docs/worklog/  .github/workflows/
├── AGENTS.md  README.md  LICENSE(Apache-2.0)  NOTICE  package.json  tsconfig.json
└── eslint/prettier  vitest.config  asconfig.json  tsup.config
```

**API (v1):**
- Core: `new TurboQuantIndex({dim, bits})`, `.add(vectors)`, `.search(queries, k, {mask})`, `.swapRemove`, `.toBytes()/fromBytes()`; `IdMapIndex` adds `.addWithIds`, `.remove(id)`, `.has(id)`, id-returning search. **ids default to `number`** (safe ≤2^53; bigint/string opt-in — see premortem T4).
- Ergonomic: `createCollection({ vectors:{size, distance}, quantization:{bits} })` → `.upsert([{id, vector, payload}])`, `.search(vector, { limit, filter, oversampling, rescore, exact, withPayload })`, `.delete(...)`.
- Constraints (typed errors at boundaries — R8): `bits ∈ {2,3,4}`, `dim` a positive multiple of 8, reject NaN/Inf/huge magnitudes.

---

## Engineering process

Grounded in the **production-grade** skill throughout (see alignment table below). Three reinforcing loops:

1. **TDD per module (R9).** Contract + tests first. Because this is clean-room with no peer library to diff against, the **trusted references are**: (a) the paper's published distortion bounds/per-bit table (measurable), (b) a small Python/numpy/scipy reference script that computes Beta-distribution Lloyd-Max centroids and quantities independently (standard scientific tooling, not any product), and (c) exact brute-force float32 search for recall ground truth. Assert TS output matches these within tolerance.
2. **autoresearch loop *is* the benchmark-improvement engine (`autoresearch` skill).** The benchmark harness emits `METRIC recall_at10=…`, `METRIC qps=…`, `METRIC compression=…`. autoresearch runs THINK→EDIT→COMMIT→RUN→MEASURE→DECIDE→LOG to push these numbers — tuning calibration on/off, LUT/block layout, WASM kernel variants, and the **paper's QJL two-stage vs RaBitQ scaling**. Primary metric = recall@10 at fixed bit budget; QPS secondary. MAD-based confidence gates keep/discard; ASI logged; kept experiments are clean commits on a dedicated branch; only confidence-validated wins squash-merge to main (premortem E2).
3. **doer/verifier/panel/devil's-advocate verification (agent-skills-harness pattern).** At each milestone, 3 independent subagents (Quality, Utility/DX, Devil's Advocate) score the artifact against a rubric + premortem risks, run the consensus protocol → SHIP / SHIP-WITH-CAVEATS / ITERATE / BLOCK. Plus built-in `/code-review` and `/security-review` on each diff, and the **self-verification gate** before every PR.

---

## Execution orchestration, CI & handoff

This is a **large, multi-session build**. It runs under the `executing-plans` skill with the doer/verifier/panel pattern, isolated worktrees, and durable progress tracking so any session (or agent) can resume cleanly.

### Subagent orchestration
- **Doers — parallel where independent, in isolated worktrees** (`using-git-worktrees` → native `EnterWorktree`, never `git worktree add` when the native tool exists). Dependency graph: `rng→rotation`, `beta→codebook`, `topk` standalone, `encode`⊃{rotation,codebook}, `search`⊃{encode,topk}, `wasm`⊃search(scalar oracle), `index`⊃search, `serialize`⊃index, `ergonomic`⊃index. → **Wave 1 (parallel):** scaffold+CI, rng, beta, topk. **W2:** rotation, codebook. **W3:** encode. **W4:** search (scalar). **W5 (parallel):** index classes, serialize. **W6 (parallel):** wasm kernel, ergonomic layer. **W7:** benchmark harness + autoresearch.
- **Every doer is TDD** (`test-driven-development`): contract + failing tests first, implement to green against the triple oracle (paper bounds, scipy reference, exact search).
- **Panel per wave** (doer/verifier/panel/devil's-advocate): 3 independent verifier subagents — **Quality**, **Utility/DX**, **Devil's Advocate** — receive only the artifact + rubric + premortem risks (no doer context), score independently, run the consensus protocol → **SHIP / SHIP-WITH-CAVEATS / ITERATE / BLOCK**. Use a different model family for the panel where possible.
- **autoresearch** runs the benchmark loop on its own branch (W7+); only confidence-validated wins squash-merge.
- **Orchestrator (lead session, me)** owns the plan, merges only green+SHIP work, and never trusts an agent's "success" report — it independently inspects the VCS diff (per `verification-before-completion`).

### CI as the automated verification gate (`verification-before-completion`)
**Iron Law: no completion claim without fresh verification evidence.** `ci.yml` enforces it on every PR — typecheck, lint, `vitest` + coverage, build TS (tsup) + WASM (asc), type tests. A wave merges to `main` **only when CI is green AND the panel returns SHIP**. Phase-gate and pre-merge claims must cite the actual command output, not "should pass."

### Progress tracking & handoff (under `docs/`)
- `docs/worklog/PROGRESS.md` — committed wave/task board: status, current blocker, "resume-here" pointer.
- `docs/worklog/DECISIONS.md` — committed ADR-style log: the locked decisions, premortem mitigations, and autoresearch wins.
- At session boundaries / context compaction, run the **`handoff`** skill (compaction doc to OS temp, redacted) and drop a one-line pointer in `PROGRESS.md`. A fresh session resumes from `PROGRESS.md` + this plan + the tail of the autoresearch logs.

### Skills map
| Skill | Used for |
|---|---|
| `production-grade` | engineering posture throughout |
| `premortem` | risk pass before/with each major phase |
| `test-driven-development` | contract+tests-first in every doer |
| `executing-plans` | running this plan with review checkpoints |
| `using-git-worktrees` | isolated parallel doer workspaces |
| `verification-before-completion` | phase-gate + pre-merge evidence law |
| `autoresearch` | benchmark-improvement loop |
| `documentation-writer` | Diátaxis docs + paper distillation |
| `typescript-expert` / `typescript-advanced-types` | core types, perf, ergonomics |
| `vite` / `vitest` | build + test + bench |
| `handoff` | cross-session continuity |

---

## Phased implementation plan

**Phase 0 — Scaffold, docs, license, CI skeleton.** `.gitignore` (node_modules, dist, build, datasets, scratch dirs, autoresearch artifacts). `package.json` (name `quantvec`, ESM-first + CJS, `exports`/`types`, `files`, `sideEffects:false`, `engines`). Strict `tsconfig`. eslint+prettier flat config + husky/lint-staged (pre-commit gates wired in first commit — R8). vitest config. **Apache-2.0** `LICENSE` + `NOTICE` (cite TurboQuant + RaBitQ papers only). `docs/research/`: download the TurboQuant + RaBitQ PDFs; write distilled `turboquant.md`, `rabitq.md`, `architecture.md` (Diátaxis, via `documentation-writer`). `AGENTS.md` (overview, repo map, skills table, setup, conventions, key concepts, testing, constraints & stop-signals, git/PR workflow). Set GitHub repo + package name to `quantvec`.

**Phase 1 — Core math (TDD).** `rng` → `rotation` (Householder QR) → `beta` + `codebook` (validated vs scipy reference + paper distortion table) → `topk`.

**Phase 2 — Encode.** Full pipeline + input validation as typed discriminated-union errors (R8/R14).

**Phase 3 — Search (pure-TS).** LUT build, nibble scoring, per-vector scale, top-k, mask/allowlist. **Correctness gate:** self-query→self across n=32/33/63/64/65 (block boundaries); empirical D_mse within the paper's bound; recall vs exact float32. *This scalar kernel is the oracle the WASM kernel must match.*

**Phase 4 — Index classes + serialization.** `TurboQuantIndex`, `IdMapIndex`, versioned `toBytes/fromBytes` with **bounds-validated deserialization** (premortem T6). Core import path free of Node built-ins; fs helpers behind a subpath export (R15 runtime-coherence).

**Phase 5 — WASM-SIMD kernel.** AssemblyScript v128 scoring kernel; `asc` build; runtime feature-detect with automatic pure-TS fallback; `.wasm` inlined (base64) for zero-config isomorphic loading; explicit wasm-memory cleanup (R15). Test: WASM ≡ scalar within fp tolerance.

**Phase 6 — Ergonomic layer.** `Collection`, `Point<P>`, `Distance` enum, filter DSL→mask (typed-array masks; optional per-key inverted index for filter perf — premortem T4).

**Phase 7 — Benchmark harness.** Download GloVe-200 + DBpedia/OpenAI-1536; exact float32 ground truth; report **recall@{1,10,100}, QPS, p50/p95 latency, encode time, compression, peak memory** for bits 2/3/4 and large-n (≥1M). Baselines = exact brute-force + our pure-TS vs WASM + (optional) a JS-ecosystem peer (e.g. `hnswlib-node`) for context — **not** native FAISS (premortem E1). Emit `METRIC` lines; write `benchmarks/results/*.json` + README table.

**Phase 8 — autoresearch tuning → panel verification → release.** Run autoresearch against recall/QPS; doer/verifier/panel review of code+API+docs; `/security-review`; publish v0.1.0.

---

## Release & CI (modeled on user's apple-signin-auth, modernized)
- `ci.yml`: PR/push → install, typecheck, lint, build TS (tsup) + WASM (asc), vitest+coverage, tsd-style type tests; Node matrix (20/22); actions SHA-pinned (R14).
- `release-publish.yml`: push to `main` after CI green → build, GitHub release (`ncipollo/release-action`), `npm publish`. **v0.1.0 ships with the proven `NPM_TOKEN` path; npm provenance/OIDC trusted publishing is a fast-follow once the package exists** (premortem T5).
- `codeql.yml`: JS/TS analysis. husky + lint-staged pre-commit; `prepublishOnly` builds.

## Build tooling (locked)
- **Bun** = package manager + runtime + script runner (fast local dev). **Vitest** = tests + `vitest bench`, run under **both Node and Bun** in CI to prove isomorphism. **tsup** → published dual ESM/CJS + `.d.ts` (wasm inlined). **AssemblyScript (`asc`)** → `build/quantvec.wasm`.
- *Rationale:* quantvec is a portability-first library, so cross-runtime test coverage (Node+Bun+browser/workers) and bulletproof `.d.ts` outrank single-runtime dev speed — hence Bun for dev velocity but Vitest+tsup for correctness/artifacts, not `bun test`/`bun build`. The published package is standard ESM/CJS and runs everywhere regardless of our dev toolchain.

---

## Verification (end-to-end)
1. `npm test` — all unit/integration tests green, incl. self-query-returns-self at block boundaries and codebook-vs-scipy match.
2. **Distortion-bound check** — measured D_mse ≤ (√3π/2)·4^−b and recall@10 >90% at 2–4 bits (the paper's regime).
3. **WASM ≡ scalar** within fp tolerance; pure-TS fallback works when SIMD is unavailable.
4. `npm run bench` produces the recall/QPS/compression/memory table.
5. **Isomorphic smoke** — runs in Node, a browser bundle, and `workerd`/Cloudflare Workers.
6. **Panel + security** — consensus ≥ target, no blocked dimension; `/security-review` clean; `npm publish --dry-run` shows only intended files.

---

## Risk Mitigations (Pre-Mortem — deep, before implementation)

*"Imagine it's 3 months out and quantvec failed. Why?" Verified findings (Tiger / Elephant / Paper Tiger), mitigations folded into the phases above.*

### Tigers (clear threats)
1. **[HIGH] Silent numerical error in Beta Lloyd-Max codebook / rotation.** Clean-room with no peer library to diff → a subtle Beta-CDF/quantile or QR bug degrades recall invisibly. *mitigation_checked: no numerical oracle in v1 of the plan.* **Fix (Phase 1/3):** triple oracle — paper distortion table, independent scipy reference, exact-search recall ground truth.
2. **[HIGH] WASM-SIMD scope creep stalls the project.** Toolchain plumbing (asc, inline, feature-detect) before correctness is proven. *mitigation_checked: "from the start" risks WASM becoming a correctness dependency.* **Fix:** WASM is wired in Phase 0 but the **scalar kernel lands first as the correctness oracle (Phase 3)**; WASM is an optimization layer that must match it (Phase 5), never a blocker.
3. **[MEDIUM] Scale expectations — index is brute-force O(n) scan, not HNSW.** Users may expect sublinear search at 10–100M. *mitigation_checked: plan didn't state search complexity.* **Fix:** README/AGENTS.md state quantvec is a flat quantized index (great to ~1–10M with WASM); IVF/HNSW coarse quantizer is roadmap, not v1.
4. **[MEDIUM] Payload-filter + bigint-id perf in JS.** Full-scan filters and bigint ops are slow/heavy. *mitigation_checked: bigint named but cost not addressed.* **Fix (Phase 6):** ids default to `number`; bigint/string opt-in; optional per-key inverted index; masks in typed arrays.
5. **[MEDIUM] OIDC/provenance config blocks first publish.** Trusted publishing needs npm-side setup not doable from code. **Fix:** ship v0.1.0 via classic `NPM_TOKEN`; add provenance as fast-follow.
6. **[MEDIUM] Untrusted deserialization in `fromBytes`/`load`.** Crafted length/offset fields → OOM/OOB (the real security surface of a library — R7/R8). *mitigation_checked: load path had no input validation.* **Fix (Phase 4):** validate magic, version, and every length against buffer size before allocating; treat input as untrusted.

### Elephants (unspoken)
1. **[HIGH] "Faster than native FAISS" is not the win in JS.** The honest value prop is **portability + zero-training + recall**, not raw-speed supremacy. **Fix:** position and benchmark accordingly (E1 baselines = JS peers + our own kernels), set framing in README up front.
2. **[MEDIUM] autoresearch noise/mess on a public repo.** **Fix:** dedicated branch, gitignored artifacts, squash only confidence-validated wins.
3. **[MEDIUM] Clean-room provenance discipline.** Implementation must derive from the papers, not copied structure, to keep the Apache-2.0 story honest. **Fix:** code/docs cite equations & theorems only.

### Paper Tigers (looked scary, verified fine)
- Cross-language byte-parity of rotation — irrelevant; clean-room defines its own deterministic rotation, validated by recall + distortion bounds.
- f32 vs f64 precision — TS Math is f64, typed arrays f32; well-understood, matches embedding domain.
- `dim` multiple-of-8 — a documented, validated constraint, not a risk.

### Checklist gaps closed
- **Testing:** added large-n (≥1M) + peak-memory benchmark (Phase 7).
- **Security:** added untrusted-deserialization hardening (T6) and `/security-review` gate (Phase 8).

---

## Production-grade alignment
| Rule | How this plan satisfies it |
|---|---|
| R1 plan-of-plans | This document; problem classified **Type C** (novel algorithm) → plan-more/validate-more |
| R2 simplest-correct | RaBitQ scaling for v1; QJL deferred to autoresearch, not bundled speculatively |
| R3 official-first deps | Minimal deps; Beta-dist via vetted lib or own impl, licenses checked; isomorphic |
| R4 ACM-grade | Householder QR, Lloyd-Max, adaptive Simpson, bounded min-heap, nibble-LUT — named |
| R7 security-by-plan | Untrusted-deserialization hardening (T6); Security-impact line per PR; `/security-review` |
| R8 unified standards | Strict TS, typed boundary errors, pre-commit gates in first commit, validate-at-borders |
| R9 TDD vs trusted reference | Tests-first; oracle = paper bounds + scipy reference + exact search |
| R11 evergreen docs | Diátaxis docs in `docs/`, README table, distilled paper notes |
| R14 functional spine | Discriminated-union errors, cost-aware SHA-pinned CI, benchmark metrics as observability |
| R15 runtime-coherence | WASM feature-detect + fallback, explicit wasm-memory cleanup, no Node built-ins in core path |
| Self-verification gate | Run before each PR |
