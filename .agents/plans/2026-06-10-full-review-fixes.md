# quantvec full review — findings & fix plan

## Context

Full review of quantvec v0.0.2 (clean-room TurboQuant TS lib) covering the core math layer, WASM kernels, public API layers, serialization, tests, docs, and project plumbing. Overall state is strong: all 345 tests pass, CI is comprehensive (typecheck/lint/format/coverage/CodeQL/isomorphism smoke tests across Node ESM/CJS + Bun), docs match the actual exports, serialization is hardened against untrusted input, and the roadmap claims check out.

The review surfaced **two real bugs** (both verified by tracing the code, not just agent suspicion), **one housekeeping issue**, and a few **test gaps**. Several agent-reported "high priority" findings were verified to be non-issues and are recorded below so they don't get re-litigated.

## Verified non-issues (no action — do not churn)

- **FastScan u16 overflow**: `turboquant-index.ts:509` scales the u8 LUT with `B = min(255, floor(65535/dim))`, so `dim·max ≤ 65535`; `i16x8.add` is modular and the accumulator is read back as `Uint16Array` (`kernel.ts:198`), so unsigned sums up to 65535 are exact. The "signed i16 overflow at 32767" concern is wrong.
- **Division by `calibration.scale` in search**: `Calibration` objects are only ever produced by `fitCalibration` (scale ≥ 1) or by deserialization (validates non-zero scale). Users can only pass `calibrate: boolean`. No reachable zero-scale path → no guard needed per the repo's "no dead guards" convention.
- **`uploadBlockedCodes` bounds**: only called internally with `this.#codes.subarray(0, n*dim)` — always correct length.
- **"Bit-identical" WASM claim**: both scalar and WASM paths accumulate f64 over f32 LUT loads in the same coordinate order with IEEE semantics; the claim holds.
- **`k > n`**: already tested (`search.test.ts:322`).
- **dist/, build/, coverage/, datasets**: all correctly gitignored; the inlined `wasm-binary.ts` is the committed artifact by design.

## Fix 1 — Calibrated encode can produce a negative RaBitQ scale (real bug)

**File:** `src/core/encode.ts:231-248`

The comment at `encode.ts:239-247` proves `inner > 0` via sign-matching between each rotated coordinate and its centroid — but that proof only holds for the **uncalibrated** path. On the calibrated path, the code is chosen from `(rotated[i] + shift[i]) * scale[i]` while the term added to `inner` is `rotated[i] * (centroid/scale[i] - shift[i])`; the shift breaks the sign-match.

Reachable scenario: fit calibration on a tight cluster around direction `u` (enough per-coordinate spread to avoid the identity fallback in `fitCalibration`), then `add(-u)`. Per coordinate: `shift_d ≈ -u_d`, calibration scale large, so the reconstruction `centroid/scale - shift ≈ u_d`, giving term `≈ -u_d²`. Summed: `inner ≈ -1` → `scale = norm/inner < 0` → that vector's dot/cosine estimates flip sign and it is silently anti-ranked forever.

**Fix:**
- In `encodeVector` (the calibrated branch's epilogue), guard: `if (!(inner > 0)) throw new EncodeError('DEGENERATE', ...)` — message should say the vector is too far outside the calibrated distribution to encode (mention re-building without `calibrate` as the remedy). Using `!(inner > 0)` also catches NaN.
- Add `'DEGENERATE'` to the `EncodeError` code union at `encode.ts:51`.
- Update the `encode.ts:239-247` comment: scope the sign-match proof to the uncalibrated path; the calibrated path gets the runtime guard (now a *reachable* guard, consistent with the no-dead-guards convention).
- Document the new throw in `TurboQuantIndex.add`/`addOne` JSDoc (`turboquant-index.ts`) and in `docs/api-reference.md` / `docs/guide.md` where calibration (`calibrate: true`) is described. Note: a mid-batch encode throw leaving earlier rows added is already the documented batch semantics — no atomicity change needed.

(Alternative considered and rejected: clamping `inner` to +ε — silently produces a wildly inflated scale, which is just a different corruption. Throwing a typed error matches the library's philosophy.)

**Test:** new case in `src/core/encode.test.ts` (and/or `turboquant-index.test.ts` via `calibrate: true`): construct the cluster + `-u` scenario, assert `EncodeError` with code `'DEGENERATE'`; assert uncalibrated encode of the same vector still succeeds.

## Fix 2 — FastScan path bypasses mask validation (inconsistent API, silent empty results)

**File:** `src/index/turboquant-index.ts` — `search()` at 429-461, `#searchFastScan` at 468-548.

The scalar/exact paths go through `searchFlat`, which throws `SearchError('INVALID_MASK', ...)` when `mask.length !== n` (`search.ts:205-208`). The FastScan path reads `mask[v]` directly at `turboquant-index.ts:531` with no length check: a wrong-length mask makes `mask[v]` undefined → every vector skipped → **silently empty results** instead of the typed error. Same input, different behavior depending on whether the WASM FastScan path was selected.

**Fix:** validate the mask once in `search()` (before path selection, after the EMPTY check): `if (opts.mask !== undefined && opts.mask.length !== this.#n) throw new SearchError('INVALID_MASK', \`mask length ${opts.mask.length} != n ${this.#n}\`)` — mirroring `searchFlat`'s wording exactly. This keeps `#searchFastScan` unchanged and makes all three paths (scalar, WASM-exact, FastScan) consistent. Update the `search()` JSDoc `@throws` (already mentions mask mismatch — verify wording).

**Test:** in `turboquant-index.test.ts`, with a FastScan-eligible index (4-bit, wasm enabled): wrong-length mask → `SearchError` code `'INVALID_MASK'`; and a correct-length restrictive mask returns identical hits to the scalar path.

## Fix 3 — package-lock.json version drift

`package-lock.json` says `"version": "0.0.0"` (both root fields) while `package.json` is `0.0.2`. Resync with `npm install --package-lock-only` and verify only version fields churn. `bun.lock` is untouched.

## Fix 4 — Test gaps (small additions)

- **All-excluded mask**: search with a correct-length mask that excludes every vector — assert consistent behavior (empty `SearchResult`) across scalar and FastScan paths (`search.test.ts` + `turboquant-index.test.ts`).
- **Filter matching nothing**: `Collection.search` with a filter no point satisfies → `[]` (`collection.test.ts`), if not already covered.
- The new tests from Fixes 1–2.

Coverage thresholds are 90% (`vitest.config.ts`) — the new guard lines are covered by their tests.

## Files to modify

- `src/core/encode.ts` (+ `encode.test.ts`) — Fix 1
- `src/index/turboquant-index.ts` (+ `turboquant-index.test.ts`) — Fixes 1 (JSDoc), 2
- `src/core/search.test.ts`, `src/ergonomic/collection.test.ts` — Fix 4
- `docs/api-reference.md`, `docs/guide.md` — document the new `EncodeError` code
- `package-lock.json` — regenerated

## Verification

1. `npm run typecheck && npm run lint`
2. `npx vitest run --coverage --test-timeout=30000` — 345 existing + new tests pass, 90% thresholds met
3. Spot-check the Fix 1 repro manually (calibrated cluster + `-u`) before the guard to confirm it produces a negative scale, then after to confirm the typed throw — proves the test exercises the real path.
