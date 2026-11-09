---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, metrics]
workstream: W7
module: "[[modulation]]"
summary: "cause of 11/13 failures"
source: [x/math/dsp/modulation/transmitter.go, x/math/dsp/modulation/mpsk/demodulator.go, x/math/dsp/modulation/qam/demodulator.go, x/math/dsp/modulation/mfsk/demodulator.go, x/math/dsp/modulation/mask/receiver_impl.go, x/math/dsp/modulation/dsss/demodulator.go]
tests: [x/math/dsp/modulation/mpsk/roundtrip_test.go, x/math/dsp/modulation/qam/roundtrip_test.go, x/math/dsp/modulation/mfsk/roundtrip_test.go, x/math/dsp/modulation/mask/roundtrip_test.go, x/math/dsp/modulation/dsss/roundtrip_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/modulation/... (2026-11-09): 11 of the 13 failing tests fail on Quality/Transmission statistic assertions (SNR, SignalStrength, LockCount); 1 on TransmissionStart (noise false trigger, dsss PacketSizes); 1 on config (mask Baseband)"
---
# Receiver quality estimators (`Receiver.Quality()` SNR / SignalStrength)

**What it does:** each demodulator implements `modulation.Receiver` (`transmitter.go:88-101`), returning `QualityStats{SNR, SignalStrength, PhaseError, FrequencyError, ConstellationError}` and `TransmissionStats`. The intent is to feed [[adaptive-modem-tier-selector]] and diagnostics.

**Where / what each actually measures:**

| Modem | SNR definition (code) | Measured in tests (simulated → reported) | Defect |
|---|---|---|---|
| mpsk | mean window energy while locked / while **not locked** (`demodulator.go:251-255,765-776`) | 10–30 dB → 3.2–4.1 dB | preamble signal energy counted as "silence" before lock |
| qam | ideal power / EVM, averaged since reset (`:400-425,610-623`) | 10–30 dB → 1.5–2.5 dB; long packets 15.6→24.5 dB as length grows | acquisition transient averaged in; per-symbol AGC distorts EVM; SignalStrength −43 dB from trailing zero symbols |
| mfsk | best²/second-best² Goertzel (`:316-328`) | 20–28 dB → 20.3–20.7 dB (saturated) | measures tone leakage, not noise; SignalStrength ≈10 dB below Tx scale (flaky test) |
| mask | power-tracker signal / background snapshot (`receiver_impl.go:9-75`) | 40 dB → 49.7 dB | over-reads |
| dsss | correlation-based (`demodulator.go:274-350,404-430`) | 20–30 dB → 4.9–7.2 dB | under-reads |

`FrequencyError` is never implemented (zero in every modem, e.g. `mpsk/demodulator.go:810-811`).

## Completeness review

- Complete: no. The quantities aren't defined consistently: the tests compare against **wideband** SNR (noise over fs/2), while EVM-style estimators measure post-matched-filter Es/N0, which is 22–26 dB higher at SPS 384. A correct EVM estimator would therefore "fail" the ±8 dB assertion too.
- Tested: the statistic assertions (`InDelta(snrDb, measured, 8)` at `mpsk/roundtrip_test.go:191`, `mfsk/roundtrip_test.go:221`, `mask/roundtrip_test.go:335`, `dsss/roundtrip_test.go:268`, plus qam's; SignalStrength ±10 dB; LockCount>0) cause 11 of the 13 failures: mpsk 3, qam 4, mfsk 1, mask 1, dsss 2. Because they come **before** the payload checks, they hide whether data decoded correctly.
- Correct: no (table above).

## TODOs

- [ ] Define one metric: post-matched-filter Es/N0 from decision-directed EVM over a sliding window that excludes acquisition. Expose wideband SNR separately if needed — all demodulators #p2
- [ ] Fix the tests to compare like with like (convert simulated wideband SNR to Es/N0 using SPS) and to check the payload first — */roundtrip_test.go #p2
- [ ] Implement FrequencyError from PLL/AFC state — mpsk/qam/mfsk demodulators #p3

## Links

- Module: [[modulation]] · [[adaptive-modem-tier-selector]] · [[W7-filters-codecs-voice]]
