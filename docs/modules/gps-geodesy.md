---
type: module
group: "Math and maps"
status: partial
tags: [easyrobot, gps, geodesy, localization]
workstream: W8
packages: [x/math/gps]
test_result: fail
---
# GPS and geodesy (x/math/gps)

**Purpose:** Small geodesy helpers: WGS-84 geodetic→ECEF, haversine distance + course, and range-based trilateration/multilateration.

**Packages:** `x/math/gps` — gps.go 136, trilateration.go 144, errors.go 11 (291 LOC); tests gps_test.go 182, trilateration_test.go 313. No build tags. Deps: `x/math/vec`, `x/math/mat`, `math32`. No in-tree consumers found (grep for the import path outside the package: none); NMEA parsing lives in [[nmea]].

## Public API

- Types: `Coord` (`vec.Vector3D` = lat°, lon°, alt m) with `Lat/Lon/Alt`; `ECEF` (`vec.Vector3D`) with `X/Y/Z`; `CourseAndDistance` (`vec.Vector2D` = [distance, course]) — note the field order is reversed relative to the type name.
- Functions: `ToECEF(Coord) ECEF`, `HaversineDistance(lat1, lon1, lat2, lon2)`, `CourseAndDistanceBetween(...)`, `Trilateration(P1,P2,P3, dA,dB,dC) ECEF`, `Multilateration(points []struct{Coord; ECEF}, distances) (ECEF, error)`; errors `ErrInsufficientPoints`, `ErrMismatchedLengths`.

## Algorithms

<!-- vault:algorithms h=9e5879ed7865 -->

Counts: 0 tested · 1 implemented · 3 partial · 0 stub · 0 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[wgs84-geodetic-to-ecef]] | implemented | fail | Correct to ≤0.7 m (float32); failing tests are wrong |
| [[haversine-distance-and-course]] | partial | pass | Approximate (radius 6 376 500, flat-earth course) |
| [[least-squares-multilateration]] | partial | fail | TDOA formula used for ranges |
| [[sphere-trilateration]] | partial | fail | Z never computed |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, float32, no allocations except `multilaterationLS` (`mat.New` + pseudo-inverse). float32 limits ECEF to ~0.5 m resolution and squares of ECEF magnitudes (~4e13) lose precision; degree inputs quantise to ~0.4 m. TinyGo-friendly in principle [INFERENCE].

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/gps/...` (2026-11-09) → FAIL, 16 pass / 5 fail: `TestToECEF_NorthPole`, `TestToECEF_RoundTrip` (North/South pole) — test expects Z = a, correct is b (gps_test.go:72,166); `TestTrilateration_GeneralCase`, `TestTrilateration_TableDriven/3D_offset` — implementation omits z; `TestMultilateration_4Points` — wrong linear model. Same 5 as baseline 2026-11-08.
- Probe (float64 WGS-84 reference): ECEF errors 0–0.67 m at 6 reference points; trilateration (600,800,400) → (600,800,0); multilateration → (−547.6, −623.0, −452.4); haversine +0.09% vs mean radius.

## API notes (ease of use, composability)

- Typed aliases over `vec.Vector3D` are lightweight and composable with the vec package, but `Coord` vs `ECEF` are both `[3]float32` — easy to mix up; no inverse conversions, so users cannot get back to lat/lon or to a local ENU frame (the frame a robot actually navigates in).
- `Multilateration` takes an anonymous struct slice with an unused `Coord` field — awkward; a `[]ECEF` + `[]float32` signature (or a `Range{Anchor ECEF; D float32}`) would be simpler.
- `CourseAndDistance` stores [distance, course] — name order mismatch.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] ECEF→geodetic and ENU/NED local frames (float64 origin) #p2

## Inferred objectives (proposals)

- **proposal:** a local-frame localisation helper (geodetic origin → ENU, float64 anchor, float32 offsets) so GPS fixes from [[nmea]] can fuse with odometry/IMU in an EKF ([[filters]]).
- **proposal:** reuse fixed multilateration for UWB/acoustic beacons indoors (range sensors) — same maths, local frame.
