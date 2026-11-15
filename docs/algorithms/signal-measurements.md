---
type: algorithm
status: partial
tags: [easyrobot, dsp, measurement, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "SNR constant 20 dB; THD always 0"
source: [x/math/dsp/measure.go]
tests: [x/math/dsp/measure_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 8 measure tests PASS (2026-11-09); probe: SNR = 20.00 dB for any input, THD always 0 (2026-11-09)"
---
# Signal measurements: RMS, peak, mean, variance, "SNR", THD, windowed RMS statistics

**What it does:** time-domain summary statistics of a real or complex signal, plus min/max/avg windowed RMS in dB. (Goertzel lives in the same file → [[goertzel]].)

**Where:** x/math/dsp/measure.go
- `MeasurementResult{RMS, Peak, Mean, Variance, SNR, THD}` :14-21.
- `Measure1D(vecTypes.Vector)` :31-64; `Measure2D(matTypes.Matrix)` :67-100 (flat).
- `Measure1DC` / `Measure2DC` (C64 tensors, statistics of |z|) :251-348.
- `calculateSNR` :148-175.
- `RMSStatistics{MinRMSDb, MaxRMSDb, AvgRMSDb}` :178-182; `CalculateRMSStatistics(vecTypes.Vector)` :186-248; `CalculateRMSStatisticsC` :402-477.

**Inputs → outputs:** float32 accumulators for Measure*; float64 sums for RMS statistics; dB = 20·log10(RMS) (full-scale reference = 1.0). Windowed stats use `n/100` sample windows (min 1), the last window may be partial.

| Quantity | Formula in code | Verdict |
|---|---|---|
| RMS, Peak, Mean | √(Σx²/n), max\|x\|, Σx/n | right (tests) |
| Variance | Σx²/n − mean² in float32 (:55,91) | catastrophic cancellation for large DC offsets [derived] |
| SNR | `10·log10(P / (0.01·P))` (:169-172) | **constant 20 dB** — not a measurement |
| THD | never assigned (:20 "placeholder") | **always 0** |
| Measure1DC/2DC Mean, Variance | of \|z\|, not complex mean | semantic surprise; allocates `realData` (:286,336) |
| RMS stats min/max | over n/100-sample windows | partial last window can dominate min/max [derived] |

## Completeness review

- Complete: no — SNR and THD are placeholders; no SINAD, ENOB, noise floor, crest factor, spectral flatness, or EVM.
- Tested (PASS 2026-11-09): `TestMeasurements_Measure1D` (RMS/Peak/Mean for constant, alternating ±1 and ramp inputs; empty → zeros), `TestMeasurements_Measure2D` (mean, peak), `TestMeasurements_Measure1DC`/`_Measure2DC` (magnitude-based RMS/Peak/Mean), `TestCalculateRMSStatistics`/`…C` (finite, max ≥ min, empty → −Inf). None assert SNR, THD or Variance.
- Correct: no. Probe: `Measure1D([1 2 3 4 5]).SNR = 20.00 dB`, `Measure1D([0.001 −0.002 1000]).SNR = 20.00 dB`, `THD = 0`. Callers that print or gate on SNR get a fixed number. RMS/peak/mean correct per tests.
- Consumers: `CalculateRMSStatistics` in [[audio-modem-cli]] (`gen_*` / `recv_*` level reports). No in-repo caller of `dsp.Measure*` (grep `dsp\.Measure|\.SNR|\.THD`, 2026-11-09); receiver SNRs in [[modulation]] are computed by each demodulator, not here.

## TODOs

- [ ] `calculateSNR` returns a constant 20 dB; remove the field or implement against a reference / spectral noise estimate — x/math/dsp/measure.go:148-175 #p1 #ER-022
- [ ] THD never computed; implement (fundamental via FFT/Goertzel, harmonics 2..k) or remove field — x/math/dsp/measure.go:20 #p2
- [ ] Variance via two-pass or Welford in float64 — x/math/dsp/measure.go:55,91,283,333 #p2
- [ ] Measure1DC/2DC allocate a magnitude copy just for the fake SNR — x/math/dsp/measure.go:286-289,336-339 #p3
- [ ] Merge 1D/2D and real/complex duplicates (four near-identical bodies) — x/math/dsp/measure.go:31-100,251-348 #p3
- [ ] Windowed RMS: fold the partial last window into the previous one; make window count a parameter — x/math/dsp/measure.go:206-227 #p3

## Links

- Module: [[dsp-core]]
- Related: [[goertzel]]; SNR measurement need: [[W7-filters-codecs-voice]], [[modulation]]
