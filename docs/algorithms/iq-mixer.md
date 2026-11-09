---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, iq]
workstream: W7
module: "[[modulation]]"
summary: "tests only check the identity"
source: [x/math/dsp/modulation/iq/config.go, x/math/dsp/modulation/iq/modulator.go, x/math/dsp/modulation/iq/demodulator.go]
tests: [x/math/dsp/modulation/iq/roundtrip_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/math/dsp/modulation/iq → ok, 13 pass (2026-11-09); tests only check the algebraic identity real→IQ→real"
---
# IQ mixer (real ↔ complex64 carrier mixing)

**What it does:** the package `iq.Modulator` turns **real** float32 samples into complex64 by multiplying with cos/sin of a carrier: I = x·cos ωt, Q = x·sin ωt, or (x, 0) at baseband. `iq.Demodulator` turns complex64 into real: x = I·cos ωt + Q·sin ωt. Optional cos/sin LUT with interpolation.

**Where:** `iq/modulator.go:34-64` (LUT build), `:66-152` `Process`, `:154-180` `convertDirect`, `:182-219` LUT path; `iq/demodulator.go:66-152`, `:154-183`; `iq/config.go` (`CarrierFrequency` 0 = baseband, `LUTSize` 4096).

**Inputs → outputs:** `chain.Processor[complex64, float32]` (Modulator: real → complex) and `chain.Processor[float32, complex64]` (Demodulator: complex → real). The names are inverted relative to their radio meaning: the "modulator" is a down-mixer and the "demodulator" an up-mixer.

## Completeness review

- **Complete: no.**
  - The down-mixer has **no low-pass filter**, so the 2ω image stays in I/Q. The Q sign convention (`+sin`) is the conjugate of the usual e^{−jωt} down-conversion. The pair is self-consistent only because real→IQ→real gives x·(cos²+sin²) = x.
  - There is no adapter between the I/Q-interleaved `[]float32` produced by [[mpsk-modem]]/[[qam-modem]] and the `complex64` used here and by the `internal` processors. So "QPSK on an audio carrier" can't be composed from existing pieces; `cmd/audio gen_psk` writes baseband I/Q as **stereo L/R** instead ([[audio-modem-cli]]).
  - `Process` allocates `readBuf` and `complexBuf` on every call (`modulator.go:72,121`), which doesn't suit embedded/TinyGo hot paths.
- **Tested:** 13 tests pass: baseband/carrier × direct/LUT roundtrips, sizes 1–1000, estimate checks, reset and empty input. Every roundtrip checks only that real→IQ→real reproduces the input (`roundtrip_test.go:68-120`). Nothing checks that the complex output is a correct analytic or baseband signal.
- **Correct: unknown** for down-conversion use (no LPF, sign convention); **yes** for the identity the tests check.

## TODOs

- [ ] Add the LPF (or a Hilbert/analytic path) to real→IQ, fix the Q sign, and test against a known tone at fc±Δ — iq/modulator.go:154-180 #p2
- [ ] Rename Modulator/Demodulator to DownMixer/UpMixer, or document the inversion — iq/modulator.go:13, demodulator.go #p3
- [ ] Add a float32-interleaved ↔ complex64 adapter so mpsk/qam compose with iq and internal/* — iq, mpsk, qam #p2
- [ ] Remove per-call allocations — iq/modulator.go:72,121; demodulator.go #p3

## Links

- Module: [[modulation]] · [[mpsk-modem]] · [[qam-modem]] · [[complex-signal-primitives]] · [[W7-filters-codecs-voice]]
