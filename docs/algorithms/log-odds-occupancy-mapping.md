---
type: algorithm
status: partial
tags: [easyrobot, filter, slam, mapping, lidar]
workstream: W8
module: "[[filters]]"
summary: "Hits rarely marked, diagonal ranges wrong"
source: [x/math/filter/slam/mapping.go]
tests: [x/math/filter/slam/slam_test.go]
test_result: pass
complete: yes
correct: no
evidence: "no direct tests; probe slamprobe: 1.23 m hit not marked occupied; 1.0 m diagonal ray marks cell at 1.41 m"
---
# Log-odds occupancy mapping (inverse sensor model)

**What it does:** For each beam, walks grid cells with Bresenham from the robot cell; cells before the measured range get `logit(0.3)` added, the hit cell `logit(0.7)`; log-odds clamped to ±10, converted back to probabilities into the map.

**Where:** `x/math/filter/slam/mapping.go`: constants :9-16, `InverseSensorModel` :31-178, `LogOddsToProbability` :188-190, `UpdateMapFromLogOdds` :193-203, `InverseSensorModelAll` :206-251; wired in `slam.go:402-409` when `WithOnline(true)`.

**Inputs → outputs:** map `mat.Matrix`, pose `[x,y,θ]`, beam angles + ranges (m), resolution, origin, max range → updated log-odds matrix (and map probabilities in place).

## Completeness review

- Complete: no — max-range (no-return) beams are clamped to maxRange and then marked occupied (:86-88 + :131); no beam-width/obstacle thickness model.
- Tested: no direct test; slam tests construct with mapping off. Run: go test -count=1 -json ./x/math/filter/slam (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok.
- Correct: **no** (probe `slamprobe`, res 0.05 m, map initialised to 0.5):
  - Occupied marking requires `distanceGrid ≤ measurementGrid + 0.05` with `distanceGrid` an integer step count (:131, `defaultBeta`=0.05 cells): a 1.23 m return (24.6 cells) never satisfies it, so no cell is marked occupied (cells 42–44 = 0.30 free, 45–46 untouched).
  - Range counted as one unit per Bresenham step (:173) — diagonal steps are √2 long. A 1.0 m beam at 45° marked the hit at cell (40,40), i.e. 1.41 m, instead of ≈(34,34).
  - Probabilities from the input map are re-log-odds'ed each call when no log-odds map is passed (:69-82) — repeated calls compound if the caller passes nil.

## TODOs

- [ ] Track Euclidean distance along the ray (or use DDA/Amanatides–Woo traversal) — mapping.go:173 #p1 #backlog
- [ ] Mark the cell containing the endpoint as occupied (compare cell indices, not float tolerance) — mapping.go:131 #p1 #backlog
- [ ] Skip occupied update for max-range returns — mapping.go:86,131 #p1 #backlog
- [ ] Unit tests: straight and diagonal beams, max-range beams #p2

## Links

- Module: [[filters]]; grid ray casting [[grid-maps]]; [[W8-lidar]]
