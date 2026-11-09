---
type: algorithm
status: partial
tags: [easyrobot, gps, geodesy, localization]
workstream: W8
module: "[[gps-geodesy]]"
summary: "Z never computed"
source: [x/math/gps/trilateration.go]
tests: [x/math/gps/trilateration_test.go]
test_result: fail
complete: no
correct: no
evidence: "TestTrilateration_GeneralCase and TableDriven/3D_offset fail; probe gpsprobe: target (600,800,400) → (600,800,0)"
---
# Three-sphere trilateration

**What it does:** Intended: intersect three spheres (centres P1..P3, radii d1..d3). Implemented: builds the in-plane basis ex, ey and solves x, y only.

**Where:** `x/math/gps/trilateration.go:12-61` (`Trilateration`).

**Inputs → outputs:** three `ECEF` points + distances → one `ECEF` point (zero value on degenerate geometry).

## Completeness review

- Complete: **no** — the out-of-plane component `z = ±√(d1² − x² − y²)` along `ez = ex × ey` is never computed (:50-60), and the two-solution ambiguity is not exposed. Result always lies in the plane of the three anchors.
- Tested: `TestTrilateration_*`. Run: go test -count=1 -json ./x/math/gps/... (2026-11-09) → FAIL 16 pass / 5 fail. `TestTrilateration_GeneralCase` (expects (600,800,400), got (615.5,787.1,304.3) in its ECEF-scale setup) and `TableDriven/3D_offset` (Z expected 100, got 0) fail — **implementation defect**; planar cases pass.
- Correct: no. Probe (`gpsprobe`): anchors (0,0,0),(1000,0,0),(0,1000,0), target (600,800,400) → (600, 800, 0).
- float32 at ECEF magnitudes (~6.4e6 m) squares distances to ~4e13 — loses ~metres of precision [INFERENCE]; better done in a local frame.

## TODOs

- [ ] Compute z along ex×ey, return both solutions (or pick via prior) — trilateration.go:50-60 #p1 #backlog
- [ ] Work in anchor-relative coordinates (subtract P1) and float64 #p2

## Links

- Module: [[gps-geodesy]]; related: [[least-squares-multilateration]]
