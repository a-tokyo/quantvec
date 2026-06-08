# quantvec — Progress

> Durable, in-repo progress tracker. Any session resumes from here + `.agents/plans/quantvec-build-plan.md`.
> **Resume-here pointer:** Wave 4 merged to `main` (`fdfddf5`) — core algorithm complete & validated end-to-end. Next: **Wave 5** — index classes (`TurboQuantIndex`, `IdMapIndex`) + serialization + wire `src/index.ts` public API + `src/node.ts` fs helpers.

Last updated: 2026-06-08

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
| W5   | `index/turboquant-index`, `index/id-map-index`, `io/serialize`, wire `src/index.ts` + `src/node.ts` | WIP    |
| W6   | `wasm` (AssemblyScript v128 kernel ≡ scalar), `ergonomic` (Collection/filter DSL)                   | TODO   |
| W7   | `benchmarks` (recall@k/QPS/compression vs reference) + detailed README/docs + autoresearch          | TODO   |
| W8   | doer/verifier/devil's-advocate panel, security review, v0.1.0 release                               | TODO   |

### Done

- **W1** (`e8a9b3d`): `rng`, `topk` (NaN-safe), `beta`, `integrate`. One Critical (integrator hang) fixed.
- **W2** (`cd5ecca`): `rotation` (vs numpy to f32), `codebook` Lloyd-Max (vs scipy to 7 digits; ~4×/bit).
- **W3** (`44e065f`): `encode` (RaBitQ scale; unbiased, RMS halves/bit), `pack`. Removed one unreachable guard.
- **W4** (`fdfddf5`): `search` (LUT scan, mask, 3 metrics; euclidean ranking verified), `metrics`. End-to-end recall@10 @4-bit: **0.935–0.979 on realistic anisotropic data**, ~0.895 on adversarial isotropic-random (genuine estimator-variance ceiling — independently confirmed not a bug; recall@20 ≥0.90). search/metrics 100% coverage. Approved.

## Validation oracle (clean-room)

1. Paper distortion bounds: D_mse ≤ (√3π/2)·4^−b. 2. scipy reference for Beta/Lloyd-Max (embedded). 3. Exact brute-force float32 search = recall ground truth. 4. **W7 head-to-head vs the local reference library** (run both; recall parity / "ours better").

## Process notes

- **No tech debt / no backwards-compat / greenfield** (D-009): every line reachable + tested or removed.
- Per-wave: combined spec + code-quality SDD review by an independent subagent; orchestrator verifies the gate itself before merge. Full 3-member panel at W8.
- Coverage gate 90% (`all: true`).

## Open items / planned improvements

- **Recall headroom to beat the reference / hit >0.90@10 on all data:** add **TQ+ per-coordinate calibration** (deferred from W3) and/or the paper's QJL two-stage estimator — run via **autoresearch** in W7 (metric = recall@10). Reference uses TQ+; we will too.
- **Rotation perf:** dense O(d²)/vec; add **fast O(d log d) randomized-Hadamard/FWHT** transform before the large-d benchmark (W7). Serialization stores the seed + rebuilds rotation on load (one-time O(d³)); FWHT removes this.
- WASM kernel (W6) must MATCH the scalar oracle. Flat brute-force index (not HNSW) — honest scale ~1–10M.
- Detailed README + `docs/` (mermaid arch diagrams) + `benchmarks/` modeled on the local `turbovec` reference (W7).
- Local folder still `turbovector`; physical rename deferred to a session-final step. Repo PRIVATE until confirmed.
