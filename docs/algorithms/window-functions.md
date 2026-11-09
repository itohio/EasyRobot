---
type: algorithm
status: partial
tags: [easyrobot, dsp, window, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "13 tapers exact; `ApplyWindowVector(Rectangular)` overwrites signal with 1; N=1 → NaN"
source: [x/math/dsp/windows.go]
tests: [x/math/dsp/windows_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok (9 window tests PASS, 2026-11-09); probe tmp/easyrobot/audio/dspcore/main.go: 13 tapers match textbook formula ≤2.2e-7, Rectangular apply overwrites signal (2026-11-09)"
---
# Window functions (14 symmetric windows)

**What it does:** multiplies a float32 slice in place by a symmetric window `w[i]`, i = 0..N−1, denominator N−1 (not the DFT-even / periodic form).

**Where:** `x/math/dsp/windows.go`
- Types: `WindowType` (`:12-29`), `WindowParams{Type, Param}` (`:32-35`).
- API: `ApplyWindowVector(v, p)` `:38-41`, `ApplyWindowMatrix(m, p)` (per row) `:44-50`, `CreateWindowVector(n, p, dst)` `:54-59` (reuses `dst` if cap allows), `CreateWindowMatrix(r, c, p, dst)` separable outer product `:63-83`.
- Dispatcher `applyWindow` `:86-122`; formulas `:145-320`.

**Inputs → outputs:** `vecTypes.Vector` / `matTypes.Matrix` (hard-asserted to `vec.Vector` via `View()`); float32 in, float32 out, in place.

| Window | Formula (x = 2πi/(N−1)) | Code | Probe max abs err (N=16) |
|---|---|---|---|
| Rectangular | 1 | :145-150 | — **sets data to 1** (see below) |
| Sine | sin(πi/(N−1)) | :152-157 | 8.7e-8 |
| Lanczos | sinc(2i/(N−1)−1) | :159-168 | 2.2e-7 |
| Triangular (Bartlett, zero ends) | 1−\|2i/(N−1)−1\| | :170-175 | 5.6e-8 |
| Hann | 0.5−0.5cos x | :177-182 | 7.8e-8 |
| Bartlett–Hann | 0.62−0.48\|i/(N−1)−½\|−0.38cos x | :184-196 | 6.1e-8 |
| Hamming | 0.54−0.46cos x | :198-208 | 6.4e-8 |
| Blackman | 0.42−0.5cos x+0.08cos 2x | :210-222 | 9.2e-8 |
| Blackman–Harris (4-term) | 0.35875/0.48829/0.14128/0.01168 | :224-237 | 1.0e-7 |
| Nuttall | 0.355768/0.487396/0.144232/0.012604 | :239-252 | 1.1e-7 |
| Blackman–Nuttall | 0.3635819/0.4891775/0.1365995/0.0106411 | :254-267 | 8.8e-8 |
| Flat top (5-term, MATLAB coeffs) | 0.21557895…0.006947368 | :269-283 | 9.7e-8 |
| Gaussian(σ) | exp(−½((i−a)/(σa))²), a=(N−1)/2; σ≤0 → 0.5 | :285-295 | 1.4e-8 (σ=0.4) |
| Tukey(α) | Hann tapers over αN/2 each side; α≤0 rect, α≥1 Hann | :297-320 | 4.2e-8 (α=0.5) |

## Completeness review

- Complete: yes for the listed set (no Kaiser, DPSS, periodic variants, or coherent/ENBW gain helpers).
- Tested (PASS 2026-11-09): `TestWindows_ApplyToVector` (applied to all-ones: Hann/Blackman edges ≈0, centre > edge), `TestWindows_CreateWindowVector`, `…VectorReuse` (buffer identity), `…ApplyToMatrix` (some value < 1), `…CreateWindowMatrix` (corners ≈0), `…MatrixReuse`, `…AllWindowTypes` (length, non-negative except FlatTop), `…EmptySignal`, `…ParametricWindows` (length only). No test compares coefficients; all tests use all-ones input, so they cannot catch the Rectangular bug.
- Correct: no.
  - **`rectangular` assigns 1.0 instead of leaving data unchanged** (windows.go:147-149). `CreateWindowVector` is fine (it starts from ones), but `ApplyWindowVector(sig, Rectangular)` replaces the signal: probe `[2 3 4 5]` → `[1 1 1 1]`.
  - N = 1 → division by zero (N−1 = 0) → NaN for every non-rectangular window (probe: Hann on `[7]` → `[NaN]`).
  - The 13 tapers match their textbook formulas to float32 precision (probe table).

## TODOs

- [ ] Rectangular must be a no-op in `applyWindow`, not overwrite with 1 — x/math/dsp/windows.go:145-150 #p1 #ER-022
- [ ] N=1 yields NaN (divide by N−1=0); return 1 — x/math/dsp/windows.go:153,160,171,178 #p2
- [ ] Offer periodic (N-denominator) windows for STFT/Welch and expose coherent gain / ENBW for amplitude-correct spectra — x/math/dsp/windows.go:144-320 #p2
- [ ] Coefficient tests vs formula; test Apply on non-constant input — x/math/dsp/windows_test.go #p2
- [ ] `CreateWindowMatrix` allocates row/col windows per call — x/math/dsp/windows.go:66-67 #p3
- [ ] `WindowType.String()` is defined in a test file; move to windows.go — x/math/dsp/windows_test.go:343-364 #p3

## Links

- Module: [[dsp-core]]
- Used before: [[fft-radix2-real]], [[goertzel]]; filter design counterpart: [[filters]]
