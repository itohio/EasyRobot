---
type: algorithm
status: partial
tags: [easyrobot, dsp, generator, noise, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "white = uniform (SNR +4.77 dB vs nominal as AWGN); pink diverges at sample ~13; brown Nyquist peak"
source: [x/math/dsp/generators.go]
tests: [x/math/dsp/generators_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok (noise tests assert range / non-empty only, 2026-11-09); probe tmp/easyrobot/audio/dspcore/main.go: white=uniform var 1/3, pink diverges at sample 13, brown has Nyquist peak (2026-11-09)"
---
# Noise generators: white, pink, brown

**What it does (intended):** additive noise sources — white (flat PSD), pink (1/f, −3 dB/oct), brown (1/f², −6 dB/oct) — through the common `Generator` interface.

**Where:** x/math/dsp/generators.go
- `NoiseWhiteGenerator` :221-271 — `amp·(rand.Float32()·2−1)` (:247,252).
- `NoisePinkGenerator` :274-350 — 5th-order all-pole recursion `y = b0·w + b1·y1 + … + b5·y5` with b = {0.99886, −2.49482, 1.44331, −0.44419, 0.05081, −0.00226} (:283-293, :311,324).
- `NoiseBrownGenerator` :353-441 — `acc += 0.02·w`, clipped to ±amp, output `acc − 0.99·prev_output` (:384-401, :405-424).

**Inputs → outputs:** `(amplitude float32)`; fills `dst`. RNG is the global `math/rand` source (no seed injection, shared, not reproducible per generator). `reverse` only reverses write order.

## Completeness review

- Complete: no — no Gaussian (AWGN) source exists in the package; SPEC.md:177-178 lists `NoiseGaussian`/`NoiseUniform` on a `NewSignalGenerator1D` API that does not exist.
- Tested (PASS 2026-11-09): `TestNoiseWhiteGenerator_Generate` (100 samples in [−1,1]); `TestNoisePinkGenerator_Generate` and `TestNoiseBrownGenerator_Generate` assert only `len==100` and `NotEmpty` — they pass on Inf.
- Correct: no.
  - **White is uniform, not Gaussian.** Probe (amp 1, 200 k samples): variance 0.3334 (= amp²/3), max |x| 1.000, kurtosis 1.797 (uniform 1.8; Gaussian 3). [[modulation]] tests (`x/math/dsp/modulation/test_utils/transmit.go`) and [[audio-modem-cli]] (`cmd/audio/common.go`) use it as AWGN with amp = √(noise power) → actual noise power P/3, **true SNR 4.77 dB higher than nominal**, and bounded tails understate bit-error floors. Every SNR/BER figure built on it is optimistic. The modulation roundtrip tests compare receiver-measured SNR to the nominal SNR with ±8 dB tolerance (e.g. `x/math/dsp/modulation/mask/roundtrip_test.go:335`), wide enough to absorb this bias.
  - **Pink is unstable.** The recursion's characteristic polynomial z⁵+2.49482z⁴−1.44331z³+0.44419z²−0.05081z+0.00226 has a real root ≈ −3.03 (|pole| > 1). Probe: x = [−0.89, 2.78, −7.66, 23.5, −71.1, …], |x| > 1e6 at sample 13, −3.5e9 at 20, −5.7e28 at 60 → Inf within a few hundred samples. `cmd/audio gen_noise` and `common.go` noise injection use it.
  - **Brown has the wrong spectrum.** `y[n] = acc[n] − 0.99·y[n−1]` is a pole at z = −0.99 (a Nyquist resonator), not a DC-blocking high-pass. Probe band PSD (per-bin, 65 536 samples, bands as fraction of fs): 0–0.01: 2.7e5, 0.01–0.05: 119, 0.05–0.25: 5.7, 0.25–0.45: 5.6, 0.45–0.5: **1063**. Accumulator clipping at ±amp (:397-401) also makes it a bounded walk, and the first sample of every call bypasses the filter (:388-392, :410-414) → block-size-dependent output.

## TODOs

- [ ] Add a Gaussian (Box–Muller / `rand.NormFloat64`) AWGN generator with σ parameter; switch modulation test noise and `cmd/audio` noise to it — x/math/dsp/generators.go:221-271 #p1 #ER-022
- [ ] White noise used as AWGN with amp=√P: SNR is 4.77 dB optimistic, tails bounded — x/math/dsp/generators.go:247,252; x/math/dsp/modulation/test_utils/transmit.go #p1 #ER-022
- [ ] Pink noise filter unstable (pole ≈ −3.03), diverges by sample ~13 — replace with Kellet/Voss–McCartney or a stable 1/f IIR (e.g. b=[0.049922,−0.095993,0.050613,−0.004408], a=[1,−2.494956,2.017266,−0.522189]) — x/math/dsp/generators.go:283-293,311,324 #p1 #ER-022
- [ ] Brown noise: use leaky integrator `y = 0.99·y + w` (pole +0.99), drop clipping and first-sample special case — x/math/dsp/generators.go:384-424 #p1 #ER-022
- [ ] Per-generator seedable RNG (`*rand.Rand` / `math/rand/v2` source) for reproducible tests — x/math/dsp/generators.go:247 #p2
- [ ] Tests: variance, distribution, PSD slope, finiteness over ≥10⁵ samples — x/math/dsp/generators_test.go:405-436 #p2

## Links

- Module: [[dsp-core]]
- Consumers: [[modulation]] (noise-scenario tests), [[audio-modem-cli]] (`gen_noise`, `--noise` options)
- Workstream: [[W7-filters-codecs-voice]]
