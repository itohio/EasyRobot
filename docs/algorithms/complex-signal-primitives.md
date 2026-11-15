---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation]
workstream: W7
module: "[[modulation]]"
summary: "CFO corrector wrong (probe)"
source: [x/math/dsp/modulation/internal/fir.go, x/math/dsp/modulation/internal/dc_blocking.go, x/math/dsp/modulation/internal/phase_rotator.go, x/math/dsp/modulation/internal/cfo.go, x/math/dsp/modulation/internal/helpers.go, x/math/dsp/modulation/internal/processor.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe on a copy (2026-11-09): CFOCorrector(100 Hz) on a 100 Hz complex tone → max |y−1| = 2.000 (ideal ≈0)"
---
# Complex-signal primitives: FIR, DC blocker, phase rotator, CFO corrector (internal, unused)

**What they do:** small `chain.Processor[complex64, complex64]` blocks working in place through `processInPlace` (`internal/helpers.go:12-66`). If `dst` is a `buffer.Viewer`, they process its items and **ignore `src`**; otherwise they copy batch by batch.

| Block | Where | Behaviour | Status |
|---|---|---|---|
| FIR | `internal/fir.go:23-60` | runs I and Q through two `x/math/filter` FIRs | implemented, untested |
| DC blocker | `internal/dc_blocking.go:23-107` | EMA DC estimate subtracted per channel; constructors **panic** on bad args (`:30,52,99`) | implemented, untested |
| Phase rotator | `internal/phase_rotator.go:20-88` | fixed rotation by a phase (cached sin/cos) | implemented, untested |
| CFO corrector | `internal/cfo.go:31-110` | derotation by a fixed frequency offset | **wrong**: sin/cos refreshed only when the accumulator wraps (`:76-86`), so a constant rotation is applied between wraps; probe error 2.0 (antiphase) on a 100 Hz tone |

## Completeness review

- Complete: no. No caller uses any of these blocks. The dual `dst`-as-source semantics make composition with `x/chain` error-prone (see [[pipeline-and-chain]]). Panicking constructors don't suit embedded use.
- Tested: none in repo. Correct: CFO corrector no (probe); the others unknown.

## TODOs

- [ ] CFO corrector: recompute sin/cos each sample (or a complex NCO recurrence with renormalisation) — internal/cfo.go:60-88 #p1 #ER-022
- [ ] Return errors instead of panicking in DC-blocker constructors — internal/dc_blocking.go:30,52,99 #p3
- [ ] One documented semantics for in-place processing (don't silently ignore src) — internal/helpers.go:12-30 #p2
- [ ] Unit tests for each block — internal/*.go #p2

## Links

- Module: [[modulation]] · [[afc-frequency-tracker]] · [[iq-mixer]] · [[W7-filters-codecs-voice]]
