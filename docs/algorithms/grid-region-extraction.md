---
type: algorithm
status: implemented
tags: [easyrobot, grid]
workstream: W8
module: "[[grid-maps]]"
summary: "Element-wise mask multiply and copying rectangular / circular / rotated-elliptical windows out of…"
source: [x/math/grid/extract.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "no tests for Mask/ExtractRectangle/ExtractCircle/ExtractEllipse"
---
# Grid region extraction and masking

**What it does:** Element-wise mask multiply and copying rectangular / circular / rotated-elliptical windows out of a grid (e.g. local map patch around robot for scan matching).

**Where:** `x/math/grid/extract.go` — `Mask :18`, `ExtractRectangle :47`, `ExtractCircle :79`, `ExtractEllipse :137`.

**Inputs → outputs:** `src mat.Matrix`, coords x=col,y=row, sizes in cells → new `mat.Matrix` (allocated each call; `Mask` reuses `dst` if sized). Outside-shape cells set to `fillValue`.

## Completeness review

- Complete: yes for scope; windows fully outside-bounds rejected (returns nil) rather than clipped/padded — a patch near the map edge can't be extracted.
- Tested: **none** (no `extract_test.go`).
- Correct: unknown. Code issues: `ExtractRectangle`/`ExtractCircle`/`ExtractEllipse` index `src[0]` before any empty check → panic on empty src (`:49`, `:81`, `:139`); `Mask` returns `src` (not `dst`) when mask is invalid (`:19-21`), so callers expecting a copy get aliasing; `ExtractEllipse` uses float `maxRadius` for the in-shape test (`:172-173`) but `int(maxRadius)` for the source offset (`:183-184`) → for non-integer a/b the shape is shifted by the fractional part.

## TODOs

- [ ] Tests for all four functions — x/math/grid/extract.go #p2
- [ ] Empty-src guard — x/math/grid/extract.go:49,81,139 #p3
- [ ] Consistent centre for non-integer ellipse axes — x/math/grid/extract.go:172,183 #p3
- [ ] Clip/pad at map edges instead of returning nil — x/math/grid/extract.go:55,84,150 #p2

## Links

- Module: [[grid-maps]]
