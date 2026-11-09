---
type: algorithm
status: partial
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "recovers a spectrum X (N wavelengths) from M filtered photodetector readings R, given channel…"
source: [x/math/colorscience/spd.go, x/math/colorscience/reconstruction.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe 8 SG4 channels → 31 wl: undamped pseudo-inverse errors; damped λ=0.01 fits readings (resid 3e-4) but SPD RMS 0.466 on 0.2–1.2 truth; weighted undamped singular; caller weights mutated (2026-11-09)"
---
# Multi-sensor SPD reconstruction (AS734x-style channels)

**What it does:** recovers a spectrum X (N wavelengths) from M filtered photodetector readings R, given channel responsivities S (M×N), by solving R = S·X. Options: Moore-Penrose pseudo-inverse, Tikhonov-damped least squares, or weighted LS. Negative outputs are clipped to 0.

**Where:**
- `spd.go:395-499` `SPD.Reconstruct(responses, useDampedLS, λ)`.
- `spd.go:518-674` `SPD.ReconstructWeighted(responses, weights, useDampedLS, λ)`.
- `reconstruction.go:94-192` `FilterSPDFromChannel` (filter models: Gaussian, super-Gaussian n=4/6, boxcar, from centre + FWHM); `:209-229` `SensorChannelsToResponses`; `:273-306` `SPD.ReconstructFromChannels` (uses weighted LS when any `Uncertainty > 0`); `:334-355` `ReconstructSPDFromChannels`; `:32-54` `ReconstructSPDWithConstraints`.

**Inputs → outputs:** `[]SensorChannel{CenterWL, FWHM, Reading, Uncertainty, FilterShape, Asymmetry}` plus a target wavelength grid gives an SPD. Damped form: `X = (SᵀS + λ²I)⁻¹ SᵀR`, which inverts an N×N matrix (`spd.go:450-474`).

## Completeness review

- Complete: no.
  - **Undamped path fails for the normal case M < N.** `mat.PseudoInverse` → `fp32.Gepseu` returned "pseudo-inverse computation failed" for an 8×31 matrix in the probe (`spd.go:479`). [INFERENCE] The SVD helper does not handle wide matrices; see [[mat-vec]] / [[primitive-kernels]].
  - **Weighted undamped** inverts SᵀW²S, which has rank ≤ M < N, so it is always singular ("matrix is singular", probe; `spd.go:653`).
  - `smoothnessWeight` is accepted and ignored (`reconstruction.go:37`). There is no smoothness or basis prior, which is what makes 8→N reconstruction meaningful (e.g. 2nd-difference Tikhonov, or a PCA basis of natural spectra).
  - Weighting defect: `ReconstructFromChannels` sets w = 1/u² (`reconstruction.go:293`), then `ReconstructWeighted` squares it again (W² = w², `spd.go:611`). The effective weight is 1/u⁴; standard WLS uses 1/σ².
  - **Side effect:** `ReconstructWeighted` normalises the caller's `weights` vector in place (`spd.go:604`). Probe: input [2,4,8,…] becomes [0.25,0.5,1,…].
  - Filter models:
    - Gaussian FWHM→σ factor 2.355 ✓.
    - SG4 factor 2.176 vs exact 2(2 ln 2)^{1/4} = 2.170 (≈0.3 % off).
    - SG6 factor 2.21 vs exact 2.112 (≈4.6 % too narrow) (`reconstruction.go:116`).
    - `Asymmetry` only shifts the centre by 0.1σ and is ignored for the Gaussian shape (`:144-155`).
  - Non-negativity is enforced by post-clipping, not NNLS, so the clipped result no longer satisfies the readings.
  - Cost: damped path is O(N³) per call. `mat.DampedLeastSquares` (`x/math/mat/pseudo_inverse.go:48`) already provides the cheaper M×M form `Sᵀ(SSᵀ + λ²I)⁻¹` and is not reused.
- Tested: none.
- Correct: no as a reconstruction. Probe: synthetic Gaussian bump (peak 1.2, base 0.2) on 400–700/10 nm, 8 SG4 channels with FWHM 20.

| solver | SPD RMS error | readings residual |
|---|---|---|
| undamped | error (pseudo-inverse fails) | — |
| damped λ = 0.01 | 0.466 | 3e-4 |
| damped λ = 0.1 | 0.467 | 9e-3 |
| damped λ = 1 | 0.518 | 0.57 |
| weighted, undamped | error (singular) | — |
| weighted, damped λ = 0.1 | 0.467 | 9e-3 |

The damped solver matches the readings but returns a minimum-norm spectrum: zero between channels instead of the smooth truth.

## TODOs

- [ ] Use M×M damped/pseudo-inverse form (reuse `mat.DampedLeastSquares`) so M<N works and costs O(M³) — `spd.go:443-486` #p1 #backlog
- [ ] Implement `smoothnessWeight` (2nd-difference Tikhonov) or basis-function reconstruction — `reconstruction.go:37` #p2
- [ ] Fix weight double-squaring (w=1/u, W²=w²) and stop mutating caller weights — `reconstruction.go:293`, `spd.go:604,611` #p1 #backlog
- [ ] Correct SG6 FWHM factor (2.112) and apply asymmetry consistently — `reconstruction.go:116,144-155` #p3
- [ ] NNLS instead of clipping — `spd.go:489-493,664-668` #p3
- [ ] Synthetic round-trip tests (known SPD → readings → reconstruct) — (none) #p2

## Links

- Module: [[colorscience]]
