# AGENTS.md — quantvec

Agent-facing contract for this repo. Humans: see `README.md`.

## Project overview

**quantvec** is an open-source, isomorphic TypeScript vector-quantization & nearest-neighbor
search library — a **clean-room implementation** of Google Research's **TurboQuant**
(arXiv:2504.19874) with the **RaBitQ** unbiased-estimator correction (arXiv:2405.12497).
Data-oblivious, zero-training, ~zero indexing time; runs in Node, browsers, Bun, and edge
runtimes. Published to npm as `quantvec`.

**Clean-room rule (non-negotiable):** implement only from the published papers and our own
design. Cite equations/theorems in code/docs — never reference or copy any existing library.

## Repository structure

```
src/
  core/        rng, rotation (Householder QR), beta, codebook (Lloyd-Max),
               encode, pack, search (nibble-LUT scalar kernel), topk, metrics
  index/       turboquant-index (positional), id-map-index (stable ids)
  ergonomic/   collection, filter DSL (must/should/must_not/range/match), types
  io/          serialize (versioned, bounds-validated)
  wasm/        AssemblyScript-kernel glue + feature-detect + pure-TS fallback
  index.ts     public exports        node.ts  Node-only fs helpers (subpath ./node)
assembly/      AssemblyScript v128 SIMD source (asc → build/quantvec.wasm)
test/          vitest unit/integration       benchmarks/  recall@k / QPS / compression harness
docs/research/ downloaded papers + distilled notes
docs/worklog/  PROGRESS.md (task board) · DECISIONS.md (ADR log)
.agents/       skills/ (production-grade, autoresearch, premortem, subagent-driven-development,
               test-driven-development, verification-before-completion, …) · plans/
local/         scratch / reference — gitignored, NOT part of the project
```

## Setup & commands

```bash
bun install          # deps (Bun is the package manager / runtime / runner)
bun run typecheck    # tsc --noEmit
bun run lint         # eslint
bun run test         # vitest run
bun run build        # tsup → dist (ESM + CJS + .d.ts);  bun run build:wasm → asc
bun run bench        # vitest bench
```

## Conventions

- TypeScript strict; `verbatimModuleSyntax` (use `import type`). Typed arrays in hot paths.
- Validate at boundaries (reject NaN/Inf/huge; `dim` a positive multiple of 8; `bits ∈ {2,3,4}`);
  errors are discriminated unions with string-literal codes.
- Core entry stays isomorphic — no `node:*` imports outside `src/node.ts`.
- Match `.prettierrc.json` + eslint; pre-commit runs format → lint → typecheck.

## Engineering process (skills)

- **production-grade** posture throughout; run the self-verification gate before every commit.
- **subagent-driven-development**: serial implementer per task → **spec-compliance review** →
  **code-quality review** → loop until approved. Never run parallel implementers in one workspace.
- **Milestone panel** (doer/verifier/devil's-advocate): 3 independent scorers (Quality, Utility/DX,
  adversarial) see only the artifact + rubric + premortem; consensus → SHIP / CAVEATS / ITERATE / BLOCK.
- **premortem** before each major wave; **autoresearch** drives benchmark improvement (METRIC loop).
- **using-git-worktrees** for isolated parallel work; **handoff** at session boundaries.

## Key concepts

- **Data-oblivious quantization:** random rotation → coordinates ~ Beta((d-1)/2,(d-1)/2) → optimal
  per-coordinate Lloyd-Max scalar quantizer, no training.
- **RaBitQ scale:** per-vector length-renormalization correction → unbiased inner-product estimate.
- **Flat quantized index:** search is a SIMD linear scan over 2–4-bit codes (not HNSW) — fast
  because codes are tiny; honest scale target ~1–10M vectors with the WASM kernel.
- **Validation oracle:** paper distortion bounds + independent scipy reference + exact-search recall.

## Verification (Iron Law)

No completion claim without fresh verification evidence. A wave is DONE only when **CI is green
AND the panel returns SHIP**. The orchestrator inspects the VCS diff — never trusts an agent's report.

## Constraints & stop signals

- STOP and escalate on: architectural forks with multiple valid approaches, recall below the paper's
  bounds you can't explain, or any clean-room-provenance doubt.
- Do NOT flip the GitHub repo to public without explicit confirmation.
- Do NOT rename the live working directory mid-session.
- Commit/push only when asked; never start feature work directly on `main` — branch first.
