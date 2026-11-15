---
type: algorithm
status: stub
tags: [easyrobot, filter, iir, dsp]
workstream: W7
module: "[[filters]]"
summary: "Falls back to Butterworth"
source: [x/math/filter/iir/iir.go]
tests: [x/math/filter/iir/iir_test.go]
test_result: pass
complete: no
correct: no
evidence: "iir.go:259-283 return Butterworth designs; TODO comments at :260,267,274,281"
---
# Chebyshev type I designs (placeholder)

**What it does:** Nothing Chebyshev-specific: `NewChebyshevLowPass/HighPass/BandPass/BandStop` ignore `rippleDb` and return the corresponding (defective) Butterworth/RBJ design.

**Where:** `x/math/filter/iir/iir.go:259-283` (four `// TODO: Implement proper Chebyshev I design`).

**Inputs → outputs:** order, cutoff(s), fs, ripple dB → `*IIR`.

## Completeness review

- Complete: no (stub).
- Tested: Chebyshev tests in `iir_test.go` only check non-NaN; pass because they exercise the fallbacks. Run: go test -count=1 -json ./x/math/filter/iir (2026-11-09) → ok.
- Correct: no — LP/HP fallbacks inherit [[iir-butterworth-filters]] defects.

## TODOs

- [ ] Implement Chebyshev I (pole placement with ripple ε, SOS cascade) or remove the constructors — iir.go:259-283 #p2

## Links

- Module: [[filters]]
