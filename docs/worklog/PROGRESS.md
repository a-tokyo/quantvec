# quantvec — Progress

> Durable, in-repo progress tracker. Any session resumes from here + `.agents/plans/quantvec-build-plan.md`.
> **Resume-here pointer:** Phase 0 complete (scaffold + CI + docs committed). Next: **Wave 1** — core math `rng`, `beta`, `topk` via subagent-driven-development (implementer → spec review → code-quality review).

Last updated: 2026-06-08

## Status legend

`TODO` · `WIP` · `BLOCKED` · `REVIEW` (in panel) · `DONE` (CI green + panel SHIP)

## Phase 0 — Scaffold, docs, CI

| Item                                                            | Status | Notes                                                       |
| --------------------------------------------------------------- | ------ | ----------------------------------------------------------- |
| Bun + Vitest + tsup + AssemblyScript toolchain                  | DONE   | bun 1.3.14, TS 6, vitest 4, eslint 10, asc 0.28             |
| package.json / tsconfig / configs                               | DONE   | clean baseline: typecheck✓ lint✓ build✓ (ESM+CJS+dts) test✓ |
| .gitignore (ignores `local/`, datasets, autoresearch artifacts) | DONE   |                                                             |
| LICENSE (Apache-2.0) + NOTICE                                   | DONE   | clean-room attribution to TurboQuant + RaBitQ papers        |
| Papers downloaded                                               | DONE   | `docs/research/*.pdf`                                       |
| Distilled research notes                                        | DONE   | turboquant.md, rabitq.md, architecture.md                   |
| README.md                                                       | DONE   | honest positioning (portability, not "faster than FAISS")   |
| AGENTS.md                                                       | DONE   | repo contract + process                                     |
| docs/worklog (PROGRESS + DECISIONS)                             | DONE   | this file + DECISIONS.md                                    |
| CI workflows (ci / release-publish / codeql)                    | DONE   | Bun+Node, cross-runtime smoke, NPM_TOKEN publish path       |
| git remote → a-tokyo/quantvec + genesis commit                  | DONE   | genesis commit on main, pushed                              |

## Implementation waves (subagent-driven-development; CI-green + panel-SHIP to merge)

| Wave | Modules                                                                                      | Status |
| ---- | -------------------------------------------------------------------------------------------- | ------ |
| W1   | core: `rng`, `beta`, `topk`                                                                  | TODO   |
| W2   | core: `rotation` (Householder QR), `codebook` (Lloyd-Max, validated vs scipy + paper bounds) | TODO   |
| W3   | core: `encode` (normalize→rotate→calibrate→quantize→pack→scale)                              | TODO   |
| W4   | core: `search` (nibble-LUT scalar kernel = correctness oracle)                               | TODO   |
| W5   | `index/turboquant-index`, `index/id-map-index`, `io/serialize` (validated deserialization)   | TODO   |
| W6   | `wasm` (AssemblyScript v128 kernel ≡ scalar), `ergonomic` (Collection/filter DSL)            | TODO   |
| W7   | `benchmarks` harness (recall@k/QPS/compression) + autoresearch tuning                        | TODO   |
| W8   | panel verification, security review, v0.1.0 release                                          | TODO   |

## Validation oracle (clean-room)

1. Paper distortion bounds: D_mse ≤ (√3π/2)·4^−b ≈ {0.36, 0.117, 0.030, 0.009} for b=1..4.
2. Independent scipy reference for Beta Lloyd-Max codebooks (standard tooling, not any product).
3. Exact brute-force float32 search = recall ground truth; target recall@10 > 90% at 2–4 bits.

## Open items / risks (see plan premortem)

- WASM kernel must MATCH the scalar oracle, never become a correctness dependency.
- Index is brute-force O(n) scan (flat quantized index), not HNSW — set scale expectations in README.
- Local folder still named `turbovector`; physical rename to `quantvec` deferred to a session-final step.
- Repo is PRIVATE; flip to public only on explicit confirmation.
