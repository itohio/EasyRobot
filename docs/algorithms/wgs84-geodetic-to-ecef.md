---
type: algorithm
status: implemented
tags: [easyrobot, gps, geodesy]
workstream: W8
module: "[[gps-geodesy]]"
summary: "Correct to ≤0.7 m (float32); failing tests are wrong"
source: [x/math/gps/gps.go]
tests: [x/math/gps/gps_test.go]
test_result: fail
complete: yes
correct: yes
evidence: "TestToECEF_NorthPole and TestToECEF_RoundTrip/{North,South}_Pole fail (test expects a, correct is b); probe gpsprobe ≤0.67 m vs float64 WGS-84"
---
# WGS-84 geodetic → ECEF

**What it does:** Converts latitude/longitude (deg) and ellipsoidal height (m) to Earth-centred Earth-fixed X, Y, Z using the prime-vertical radius N = a/√(1−e²sin²φ).

**Where:** `x/math/gps/gps.go`: constants :9-16 (`WGS84Axis`=6378137, `WGS84Ecc`=e²=0.00669438037928458), `Coord`/`ECEF` types :20-54, `ToECEF` :71-85.

**Inputs → outputs:** `Coord{lat°, lon°, h m}` → `ECEF{x,y,z}` m, float32.

## Completeness review

- Complete: forward only — no ECEF→geodetic, no ENU/NED local frames (the "round trip" test only checks forward values).
- Tested: `TestToECEF`, `TestToECEF_NorthPole`, `TestToECEF_RoundTrip`, `TestToECEF_WithAltitude`. Run: go test -count=1 -json ./x/math/gps/... (2026-11-09) → FAIL 16 pass / 5 fail. **Three failures are test defects:** `gps_test.go:72` and `:166` expect Z = a = 6 378 137 m at the poles; correct is Z = (1−e²)·N = a·√(1−e²) = b = 6 356 752.3 m (got 6 356 752).
- Correct: yes within float32 precision. Probe (`tmp/easyrobot/filters/gpsprobe`, float64 reference with e²=0.00669437999014): (0,0,0) exact; poles Z=±6 356 752.0 vs 6 356 752.3, X=−0.3 m; (45,45,0) 0.35 m; Vilnius (54.687157, 25.279652, 112) 0.67 m; Sydney 0.52 m. Error is float32 rounding (ulp at 6.4e6 m = 0.5 m). `WGS84Ecc` differs from the WGS-84 e² in the 9th digit (3.9e-10) — negligible.
- Embedded: float32 is insufficient for sub-metre ECEF work (differences of large coordinates); fine for ~1 m.

## TODOs

- [ ] Fix pole expectations to b = 6 356 752.314 m — gps_test.go:72,166 #p2
- [ ] Add ECEF→geodetic (Bowring/Zhu) and ENU conversion; real round-trip test #p2
- [ ] Offer float64 path or local-tangent-plane deltas for precision #p3

## Links

- Module: [[gps-geodesy]]; NMEA source: [[nmea]]
