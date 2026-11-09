---
type: algorithm
status: partial
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
summary: "estimates CCT (K) and Duv (signed distance from the Planckian locus in CIE 1960 uv) for a spectrum."
source: [x/math/colorscience/temperature.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe: McCamy branch D65 6503.8 K/D50 5001.3 K ok; fallback search A 2781 K (ref 2856), blackbody 10000 K→8125 K; Duv(D65)=0.0080 (ref 0.0032); emissive D65 SPD→6937 K (2026-11-09)"
---
# Correlated colour temperature (CCT) and Duv

**What it does:** estimates CCT (K) and Duv (signed distance from the Planckian locus in CIE 1960 uv) for a spectrum.

**Where:** `temperature.go:16-93` `(*ColorScience).ComputeColorTemperature`; `:97-152` `computeCCTRobertson`; `:155-205` `planckianUV`.

**Inputs → outputs:** `ComputeColorTemperature(spd SPD) (cct, duv float32, err)`.
1. XYZ via `ComputeXYZ`, giving chromaticity x, y and u, v (`:22-45`).
2. McCamy: n = (x − 0.3320)/(0.1858 − y), CCT = 449n³ + 3525n² + 6823.3n + 5520.33 (`:50-55`). The constants match McCamy (1992).
3. If CCT < 3000 or > 7000, replace it with a 1-D search over T ∈ [1000, 20000] that minimises uv distance to `planckianUV(T)` (`:59-64`).
4. Duv = ‖(u,v) − planck(CCT)‖, with the sign taken from the polar angle around the uv origin (`:74-90`).

## Completeness review

- Complete: no.
  - **The input is treated as a reflectance**, because `ComputeXYZ` multiplies it by the configured illuminant (see [[spectral-xyz-integration]]). Passing a lamp SPD gives the CCT of lamp × D65. Probe: emissive D65 gives 6937 K, D50 gives 5864 K, A gives 3267 K (references 6504 / 5003 / 2856). A correct result needs a flat "reflectance" plus the lamp as the instance illuminant (`WithIlluminantSPD`), which is not documented.
  - Default 10° observer: XYZ has Z = NaN, so CCT = NaN with no error (probe).
  - **"Robertson" is not Robertson's method.** There are no isotemperature-line tables (Robertson 1968). It is a ternary-like bisection on a polynomial locus (`:115-149`).
  - `planckianUV` uses Kim et al. (2002) cubic-spline locus fits, but only the **1667–4000 K x(T) branch** for every T up to 40 000 K (`:169-170`; for 4000–25 000 K the coefficients should be −3.0258469e9, 2.1070379e6, 0.2226347e3, 0.240390). It also uses only the **1667–2222 K y(x) branch** for all x (`:185-186`; the 2222–4000 K and 4000–25 000 K branches are missing). One coefficient is mistyped: 0.2343580e6 vs 0.2343589e6 (`:170`). Outside its range the locus is replaced by ad-hoc linear fits (`:189-193`).
  - Duv sign by polar angle about (0,0) is not "above/below the locus" (`:79-90`). It works only by accident near the locus.
- Tested: no tests at all (`grep ComputeColorTemperature *_test.go` → none).
- Correct: no. Probe (2° observer; the spectrum is set as the illuminant and a flat reflector is passed):

| source | CCT got | reference | Duv got | reference |
|---|---|---|---|---|
| D65 (McCamy branch) | 6503.8 K | 6504 K | 0.0080 | +0.0032 |
| D50 (McCamy branch) | 5001.3 K | 5003 K | — | — |
| A (fallback search) | 2781.2 K | 2856 K | 0.0031 | ≈0 |
| blackbody 2000 K | 2002 K | 2000 K | — | — |
| blackbody 4000 K | 4009 K | 4000 K | — | — |
| blackbody 6500 K | 6499 K | 6500 K | 0.0053 | 0 |
| blackbody 10 000 K (fallback search) | 8125 K | 10 000 K | — | — |

  McCamy alone would give ≈2857 K for A; the fallback makes it worse. Duv errors come from the wrong locus branches.

## TODOs

- [ ] Accept emissive SPDs (compute xy from SPD×CMF without illuminant) — `temperature.go:22` #p1 #backlog
- [ ] Replace locus with exact Planck integration over CMF (or full Kim et al. piecewise branches) — `temperature.go:169-193` #p1 #backlog
- [ ] Implement real Robertson (or Ohno 2013) method; drop McCamy→search switch at 3000/7000 K — `temperature.go:59-64,97-152` #p1 #backlog
- [ ] Signed Duv via normal to the locus, not polar angle — `temperature.go:79-90` #p2
- [ ] Tests vs CIE illuminants A/D50/D65 and blackbody CCTs — (no test file) #p2

## Links

- Module: [[colorscience]]
