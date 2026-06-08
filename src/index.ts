// quantvec — data-oblivious, zero-training vector quantization & search.
//
// Clean-room implementation of Google Research's TurboQuant (arXiv:2504.19874)
// with the RaBitQ unbiased-estimator correction (arXiv:2405.12497).
//
// The public API surface (TurboQuantIndex, IdMapIndex, createCollection, Distance,
// filter DSL) is built up across the implementation waves; see
// docs/worklog/PROGRESS.md and .agents/plans/quantvec-build-plan.md.

/** Library version. Kept in sync with package.json at release time. */
export const VERSION = '0.0.0';
