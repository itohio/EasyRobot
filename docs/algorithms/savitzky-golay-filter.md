---
type: algorithm
status: partial
tags: [easyrobot, filter, dsp]
workstream: W7
module: "[[filters]]"
summary: "Right coefficients, wrong application"
source: [x/math/filter/savgol/savgol.go]
tests: []
test_result: none
complete: yes
correct: no
evidence: "no tests; probe sgprobe: coefficients match textbook (-3,12,17,12,-3)/35 but streaming output on a ramp is wrong"
---
# Savitzky–Golay smoothing filter (streaming)

**What it does:** Least-squares polynomial smoothing: coefficients from `(AᵀA)⁻¹Aᵀ` of a Vandermonde design (via `mat.PseudoInverse`), applied as a centred FIR over a ring buffer.

**Where:** `x/math/filter/savgol/savgol.go`: `New` :26, `Process` :56-84, `ProcessBuffer` :88-101, `computeCoefficients` :112-200.

**Inputs → outputs:** odd window, polynomial order < window → float32 per sample (delay = half window).

## Completeness review

- Complete: smoothing only; no derivative coefficients, no edge (start/end) handling.
- Tested: no test file.
- Correct: coefficients **yes**, streaming **no**.
  - Coefficients: the code takes column `halfWindow` of the pseudo-inverse and multiplies by A (:181-197), which equals the centre row of the hat matrix = textbook SG weights. Probe: impulse response contains −0.0857, 0.343 (=−3/35, 12/35) → values right.
  - Application: same ring-index warm-up gate (:72) and rotated window start (:79, `index − halfWindow + i`) as [[gaussian-smoothing-filter]]. Probe on ramp 0..14 (SG preserves polynomials ≤ order 2, so output should equal input delayed by 2): `0 0.34 0.60 0.77 4 5 3.29 4.29 5.29 9 10 8.29 …` — wrong.
  - Uses `PseudoInverse` of the small normal matrix in float32; ill-conditioned for larger windows/orders (Vandermonde powers) [INFERENCE].

## TODOs

- [ ] Index from oldest sample and gate on count — savgol.go:72,79 #p1 #ER-019
- [ ] Tests: polynomial preservation, known 5/2 coefficients #p2
- [ ] Compute coefficients in float64 with centred/scaled abscissa; add derivative orders #p3

## Links

- Module: [[filters]]
