---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, ask]
workstream: W7
module: "[[modulation]]"
summary: "2–3 dB from theory; collapses with clock drift (50 ppm → BER 0.5)"
source: [x/math/dsp/modulation/mask/config.go, x/math/dsp/modulation/mask/modulator.go, x/math/dsp/modulation/mask/demodulator.go, x/math/dsp/modulation/mask/math.go, x/math/dsp/modulation/mask/receiver_impl.go, x/math/dsp/modulation/mask/waveform.go]
tests: [x/math/dsp/modulation/mask/roundtrip_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/modulation/mask → FAIL 2 (TestRoundtrip_NoiseScenarios, TestRoundtrip_Baseband), 4 pass (2026-11-09); BER probe tmp/easyrobot/audio/berprobe"
---
# M-ASK modem (OOK / multi-level amplitude keying)

**What it does:** sends M amplitude levels (`even` or `power_efficient` spacing, `Depth` sets the floor) on a real cosine carrier with a one-symbol envelope. A preamble alternates the highest and lowest level, and the receiver uses its min/max to calibrate gain/offset. Detection is coherent mixing against a free-running local cosine, then quantisation to the nearest level.

**Where:** `mask/config.go:48-70` defaults (fs **22 kHz**, fc 10 kHz, SPS 220 → 100 Bd, 4-ASK, preamble 20), `:262-294` levels; `mask/modulator.go:49-140`; `mask/demodulator.go:103-163` `Process` (fixed windows), `:165-453` `decodeSymbol` (energy gate → "preamble" = first 20 symbols above threshold → min/max AGC → quantise → lock loss after 20 zero symbols), `:455-479` `coherentDetection` (`|Σ x·cos| / N · 2.5`), `:481-507` `envelopeDetectionPassband` (unused); `mask/math.go:8-29` `quantizeAmplitude`. Uses `internal.PowerTracker` ([[signal-power-tracker]]).

**Inputs → outputs:** `BitsReader` → real samples; real samples → `BitsWriter`.

## Completeness review

- **Complete: no.**
  - **No carrier phase recovery.** The detector takes `|I|` against a local cosine (`demodulator.go:455-478`), so a carrier phase offset φ scales every amplitude by cos φ (zero at 90°). It works only because TX and RX generators both start at phase 0 and SPS·fc/fs = 100 cycles is an integer. The receiver advances its oscillator only on decoded symbols (`:466`), and the oscillator itself drifts in float32 (see [[sine-oscillator-generators]]). Probe: clock +50 ppm → BER 0.51 (2-ASK), 0.45 (4-ASK).
  - **"Preamble" isn't detected, only counted:** the first `PreambleLength` symbols above threshold are assumed to be preamble (`:286-385`). A noise burst before the packet corrupts both the calibration and the alignment.
  - The scale constant 2.5 is a fudge (`:470-474`). Natural-binary level mapping (no Gray) for M>2.
  - Config fields `TimingLoopBandwidth`, `AGCWindow`, `CoherentMode`, `LoopBandwidth/Damping` are validated but never used by the demodulator (`config.go:35-45,110-121`; grep shows no other use). `MinimumSNR` subtracts 10·log10(bits) (`config.go:214`) where mpsk/qam add it; the helpers are inconsistent.
  - `CarrierHz<=0` is silently replaced by 1000 Hz (`config.go:76-78`), so baseband mode is impossible. That is why `TestRoundtrip_Baseband` fails ("CarrierFrequency should match config: expected 0, actual 1000").
- **Tested:** 4 pass (`Basics`, `DebugNoise` at 35 dB with a 47-symbol trace, `SingleByte_WithSilence`, `SplitBuffer`). `TestRoundtrip_NoiseScenarios` fails on the SNR-estimate assertion: measured 49.7 dB at a simulated 40 dB, i.e. the estimator **over**-reads (`roundtrip_test.go:335`). It runs at 40/35/min+5/min+3 dB only.
- **Correct: partial.** Probe (gated Gaussian noise, aligned, 3×1000 symbols; theory = coherent unipolar PAM, average-energy Eb, Gray):

| Eb/N0 | 2-ASK BER (theory) | 4-ASK BER (theory) |
|---|---|---|
| 6 dB | 3.9e-2…1.2e-1 (2.3e-2) | 0.26–0.28 (0.11) |
| 8 dB | 7e-3…2.3e-2 (6.0e-3) | 0.14–0.19 (6.7e-2) |
| 10 dB | 1e-3…1.5e-2 (7.8e-4) | 0.11–0.21 (3.4e-2) |
| 12 dB | 0–1e-3 (3.4e-5) | 3.8–6.0e-2 (1.3e-2) |
| 15 dB | 0 (9e-9) | 2.5e-3…2.1e-2 (9.9e-4) |
| 20 dB | 0 | 0 (3e-8) |

  That is about 2–3 dB worse than theory, with a large trial-to-trial spread. The gain is calibrated from 20 noisy preamble extremes (min/max), not an average [INFERENCE]. Offsets: 7 samples → 4-ASK BER 0.12; SPS/2 → 0.23/0.30. Full-noise pipeline (noise in padding): BER 0.05–0.35 at wideband SNR ≤20 dB, 0 at ≥25 dB (section B).

## TODOs

- [ ] Carrier phase-insensitive detection (I/Q magnitude) or carrier recovery; the current |I| collapses with phase offset or clock drift — mask/demodulator.go:455-478 #p1 #ER-022
- [ ] Actually detect the preamble (correlate the alternating pattern) before calibrating — mask/demodulator.go:286-385 #p1 #ER-022
- [ ] Calibrate gain from preamble means per level, not min/max — mask/demodulator.go:329-375 #p2
- [ ] Allow baseband (CarrierHz=0), or fix the test — mask/config.go:76-78; roundtrip_test.go:209 #p2
- [ ] Remove or implement the unused config fields (timing loop, coherent mode, AGC window) — mask/config.go:35-45 #p3
- [ ] Gray-map levels for M>2 — mask/math.go:8-29 #p2
- [ ] Replace the 2.5 fudge with envelope-energy normalisation — mask/demodulator.go:470-474 #p3
- [ ] Unify MinimumEbN0/MinimumSNR across schemes (sign of the bits term) — mask/config.go:183-216 #p2

## Links

- Module: [[modulation]] · [[modem-energy-detector-and-preamble-acquisition]] · [[modem-quality-estimators]] · [[signal-power-tracker]] · [[W7-filters-codecs-voice]]
