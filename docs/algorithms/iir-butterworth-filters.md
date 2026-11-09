---
type: algorithm
status: partial
tags: [easyrobot, filter, iir, dsp]
workstream: W7
module: "[[filters]]"
summary: "LP unstable, HP wrong, order ignored"
source: [x/math/filter/iir/iir.go]
tests: [x/math/filter/iir/iir_test.go]
test_result: pass
complete: yes
correct: no
evidence: "probe dspprobe: LP coefficients a=[1,0.363,5.805] (unstable, NaN output); HP1 b=[0.293,-0.293] a=[1,0.414] vs scipy b=[0.707,-0.707] a=[1,-0.414]"
---
# IIR core + Butterworth low/high-pass designs

**What it does:** Direct-form-II IIR (`New(b0..bn, a1..an)`, `Process`) and "Butterworth" designers: 2nd-order low-pass via bilinear transform, 1st-order high-pass.

**Where:** `x/math/filter/iir/iir.go`: `IIR` :10-22, `New` :27-64 (requires equal-length b and a), `Process` DF-II :75-99, `NewButterworthLowPass` :135-171, `NewButterworthHighPass` :174-191.

**Inputs → outputs:** `order` (ignored), cutoff Hz, fs → `*IIR`; per-sample float32.

## Completeness review

- Complete: no. `order` is ignored — LP is always 2nd order, HP always 1st order (:143, :182). No cascaded second-order sections, so higher orders cannot be built safely.
- Tested: `iir_test.go:55-75` runs the LP 100 samples and only checks for NaN/Inf; HP/BP/BS check the first sample. Run: go test -count=1 -json ./x/math/filter/iir (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok, 13 pass.
- Correct: **no**.
  - Low-pass: poles are scaled by `k = wc/tan(wc/2fs)` (≈2fs, :148-152) and then bilinear-transformed with constant `T=2fs` (:155) — mixing two prewarping conventions puts the analog poles near 2fs regardless of cutoff. Probe: fc=1 kHz @ 8 kHz → `a=[1, 0.363, 5.805]` (|pole|≈2.4, unstable; output NaN), reference `scipy.signal.butter(2,0.25)` is b=[0.0976,0.1953,0.0976], a=[1,−0.9428,0.3333]. The unit test passes only because 2.4¹⁰⁰ ≈ 1e38 is still below float32 max.
  - High-pass: with α=(K−1)/(K+1), K=tan(πfc/fs), the code uses b=(1+α)/2·[1,−1], a=[1,−α] (:187-188); correct is b=(1−α)/2·[1,−1]… i.e. b=[0.7071,−0.7071], a=[1,−0.4142] for fc=fs/8. Probe response: −15.4 dB at the 1 kHz "cutoff", 0 dB only near Nyquist.
- DF-II `Process` itself is correct (validated indirectly by the RBJ biquads, see [[iir-rbj-biquad-bandpass-bandstop]]).

## TODOs

- [ ] Rewrite LP design (prewarp Ωc=2fs·tan(πfc/fs), bilinear with 2fs) and support any order via cascaded SOS — iir.go:135-171 #p1 #ER-019
- [ ] Fix HP coefficients and honour `order` — iir.go:174-191 #p1 #ER-019
- [ ] Response/stability tests against known coefficient sets — iir_test.go #p1 #ER-019
- [ ] Allow len(b) ≠ len(a) in `New` — iir.go:32-41 #p3

## Links

- Module: [[filters]]
- Chebyshev placeholders: [[iir-chebyshev-type1]]
