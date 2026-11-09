---
type: algorithm
status: partial
tags: [easyrobot, gps, geodesy, localization]
workstream: W8
module: "[[gps-geodesy]]"
summary: "TDOA formula used for ranges"
source: [x/math/gps/trilateration.go]
tests: [x/math/gps/trilateration_test.go]
test_result: fail
complete: no
correct: no
evidence: "TestMultilateration_4Points fails (-500 vs 500); probe gpsprobe: target (600,800,400) → (-547.6,-623.0,-452.4)"
---
# Least-squares multilateration (≥4 ranges)

**What it does:** For 3 points delegates to [[sphere-trilateration]]; for ≥4 builds a linear system and solves with `mat.PseudoInverse`.

**Where:** `x/math/gps/trilateration.go`: `Multilateration` :69-94, `multilaterationLS` :97-144.

**Inputs → outputs:** slice of `{Coord, ECEF}` (Coord unused) + ranges → `ECEF`, error on <3 points or length mismatch.

## Completeness review

- Complete: no — no weighting, no residual/DOP output, no iterative refinement (Gauss–Newton).
- Tested: `TestMultilateration_4Points` fails (expects (500,500,500), got (−500,−500,−500)); error-path tests pass. Run: go test -count=1 -json ./x/math/gps/... (2026-11-09) → FAIL 16 pass / 5 fail.
- Correct: **no — wrong model.** Rows `A = 2Pₘ/dₘ − 2P₀/d₀`, `b = dₘ − d₀ − |Pₘ|²/dₘ + |P₀|²/d₀` (:113-125) are the TDOA (time-difference) closed form, not range multilateration. For ranges the linearisation is `2(Pₘ−P₀)·x = |Pₘ|² − |P₀|² − (dₘ² − d₀²)`. Probe (`gpsprobe`, anchors (0,0,0),(1000,0,0),(0,1000,0),(1000,1000,1000), target (600,800,400)): returns (−547.6, −623.0, −452.4). Also divides by ranges (zero range → Inf).

## TODOs

- [ ] Replace with range-difference linearisation + optional Gauss–Newton refinement — trilateration.go:113-125 #p1 #backlog
- [ ] Return residual/GDOP; drop unused `Coord` from the input struct #p3

## Links

- Module: [[gps-geodesy]]
