---
type: algorithm
status: partial
tags: [easyrobot, gps, geodesy]
workstream: W8
module: "[[gps-geodesy]]"
summary: "Approximate (radius 6 376 500, flat-earth course)"
source: [x/math/gps/gps.go]
tests: [x/math/gps/gps_test.go]
test_result: pass
complete: yes
correct: no
evidence: "haversine/course tests pass; probe gpsprobe: +0.09% radius bias, 100 km course 134.25° vs 133.51°, 0.6 m baseline 0.44 m/90°"
---
# Haversine distance and course

**What it does:** Great-circle distance by haversine on a sphere of radius `EarthRadius`=6 376 500 m; course from a flat-earth approximation `atan2(Δλ·cos φ₁, Δφ)`.

**Where:** `x/math/gps/gps.go`: `HaversineDistance` :89-105, `CourseAndDistanceBetween` :109-135, `CourseAndDistance` :57-67.

**Inputs → outputs:** lat/lon deg (float32) → metres, degrees [0, 360).

## Completeness review

- Complete: yes for sphere distance; course is not the great-circle initial bearing.
- Tested: `TestHaversineDistance`, `_ShortDistance`, `_Equator`, `TestCourseAndDistance`, `_SamePoint`, `_North` pass (tolerances loose). Run: go test -count=1 -json ./x/math/gps/... (2026-11-09) → FAIL 16 pass / 5 fail (these pass).
- Correct: **approximate** (probe `gpsprobe` vs float64 haversine with mean radius 6 371 008.8 m and exact initial bearing): 1 km north: 1001.10 vs 1000.76 m (radius 6 376 500 is 0.09% above the IUGG mean); 110 km SE: 110 069.9 vs 109 975.1 m, course 134.25° vs 133.51°; 0.6 m baseline: 0.44 m / 90° vs 0.61 m / 57° — float32 degree inputs quantise to ~0.4 m at 55° latitude.
- Embedded: float32 throughout; not suitable for sub-metre robot navigation from raw degrees — use a local ENU frame.

## TODOs

- [ ] Use mean radius 6 371 008.8 m (or document choice); exact initial-bearing formula — gps.go:15,127 #p2
- [ ] Local tangent-plane (ENU) helpers with float64 origin for short baselines #p2

## Links

- Module: [[gps-geodesy]]
