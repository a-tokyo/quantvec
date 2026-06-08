# quantvec — Progress

> Durable, in-repo progress tracker. Any session resumes from here + `.agents/plans/quantvec-build-plan.md`.
> **Resume-here pointer:** Wave 1 merged to `main` (`e8a9b3d`). Next: **Wave 2** — `rotation` + `codebook` via subagent-driven-development (implementer → review → verify gate → merge).

Last updated: 2026-06-08

## Status legend

`TODO` · `WIP` · `BLOCKED` · `REVIEW` · `DONE` (gate green + reviewed; merged to main)

## Phase 0 — Scaffold, docs, CI — DONE

Toolchain (Bun + Vitest + tsup + AssemblyScript), strict TS 6 / ESLint 10 / Prettier, Apache-2.0 + NOTICE, isomorphic package layout (ESM/CJS + .d.ts, `./node` subpath), CI (ci / release-publish / codeql, Node+Bun cross-runtime smoke), distilled research notes, worklog, plan in `.agents/plans/`. Genesis `9a9fdb7`. 90% coverage gate enforced repo-wide.

## Implementation waves (subagent-driven-development; local gate green + reviewed → merge to main)

| Wave | Modules                                                                                      | Status |
| ---- | -------------------------------------------------------------------------------------------- | ------ |
| W1   | core: `rng`, `topk`, `beta`, `integrate`                                                     | DONE   |
| W2   | core: `rotation` (dense Householder-QR; FWHT-fast as perf follow-up), `codebook` (Lloyd-Max) | WIP    |
| W3   | core: `encode` (normalize→rotate→calibrate→quantize→pack→scale)                              | TODO   |
| W4   | core: `search` (nibble-LUT scalar kernel = correctness oracle)                               | TODO   |
| W5   | `index/turboquant-index`, `index/id-map-index`, `io/serialize` (validated)                   | TODO   |
| W6   | `wasm` (AssemblyScript v128 kernel ≡ scalar), `ergonomic` (Collection/filter DSL)            | TODO   |
| W7   | `benchmarks` (recall@k/QPS/compression vs reference) + README table + autoresearch           | TODO   |
| W8   | doer/verifier/devil's-advocate panel, security review, v0.1.0 release                        | TODO   |

### W1 — done (merged `e8a9b3d`)

- `rng` xoshiro256\*\* (deterministic, golden-vector tested), `topk` size-k min-heap (NaN-safe), `beta` (continued-fraction CDF, inverse-CDF, coordinate density), `integrate` sound terminating adaptive-Simpson (fail-fast on non-finite, spike-aware). Coverage 97% stmt / 92% br / 100% fn / 99% ln. Spec + code-quality reviewed; one Critical (integrator hang) fixed.

## Validation oracle (clean-room)

1. Paper distortion bounds: D_mse ≤ (√3π/2)·4^−b ≈ {0.36, 0.117, 0.030, 0.009} for b=1..4.
2. Independent scipy reference for Beta Lloyd-Max codebooks (embedded constants — node-free tests).
3. Exact brute-force float32 search = recall ground truth; target recall@10 > 90% at 2–4 bits.
4. **W7 head-to-head vs the local reference library** (run both; recall parity / "ours better").

## Process notes

- Per-wave: SDD two-stage review (spec → code-quality); orchestrator verifies the gate itself before merge. Full 3-member doer/verifier/devil's-advocate **panel** reserved for W8 (and the W4 core-algorithm milestone).
- Coverage gate 90% (`all: true`) enforced in `vitest.config.ts` + CI.

## Open items / risks (see plan premortem)

- **Rotation perf:** dense O(d²)/vec is fine for small d & validation; a **fast O(d log d) structured transform (randomized Hadamard / FWHT)** is required before the large-d benchmark (W7) — tracked as the key perf optimization (autoresearch can A/B dense vs FWHT recall).
- WASM kernel must MATCH the scalar oracle, never become a correctness dependency.
- Flat brute-force index (not HNSW) — honest scale target ~1–10M; set expectations in README.
- Local folder still `turbovector`; physical rename to `quantvec` deferred to a session-final step.
- Repo PRIVATE; flip to public only on explicit confirmation.
