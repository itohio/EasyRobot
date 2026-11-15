---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, spread-spectrum]
workstream: W7
module: "[[modulation]]"
summary: "BER 0.2–0.48 even noiseless on long packets (LO drift after ≈0.4 s)"
source: [x/math/dsp/modulation/dsss/config.go, x/math/dsp/modulation/dsss/modulator.go, x/math/dsp/modulation/dsss/demodulator.go, x/math/dsp/modulation/dsss/pulse.go]
tests: [x/math/dsp/modulation/dsss/roundtrip_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/modulation/dsss → FAIL 3 (NoiseScenarios, PacketSizes, ModulationOrders), 1 pass (Basics) (2026-11-09); BER and drift probes"
---
# DSSS / M-ary Walsh code-shift modem

**What it does:** each symbol is sent as one of M spreading codes (bipolar ±1, length `SpreadingFactor`, usually Walsh rows supplied by the caller). For M=2 it sends ±code[0]. Each chip is pulse-shaped and multiplied by a cosine carrier into **real passband**. The receiver multiplies by a free-running local cosine, integrates per chip, correlates with every code and picks the maximum (sign for M=2).

**Where:** `dsss/modulator.go:65-133` emit, `:135-152` `chipValue`, `:154-161` `renderChip` (pulse·amplitude·`dsp.CosineGenerator`); `dsss/demodulator.go:87-138` `Process` (fixed windows from stream start), `:149-211` `processSymbol` (downconvert → energy gate → detect → write), `:213-250` `detectSymbol`/`codeCorrelation`; `dsss/config.go:42-310` analysis helpers (processing gain, BER estimate, code-correlation metrics); `dsss/pulse.go`. Codes come from the caller: the test builds Walsh/Hadamard rows (`roundtrip_test.go:48-96`), and row 0 is all ones, so "BPSK-DSSS" with that code has **no spreading**.

**Inputs → outputs:** `BitsReader` → real samples (no preamble: "code correlation provides sync", `modulator.go:170`); real samples → `BitsWriter`.

## Completeness review

- **Complete: no.**
  - **No code or chip acquisition and no frame sync.** Symbols are cut on a fixed grid from sample 0, and every window above a fixed `EnergyThreshold` produces a symbol (`demodulator.go:155-174`). Noise in the padding (or any offset) produces garbage symbols before the payload. This is the `TransmissionStart must be set` failure (start detected at sample 0) in `TestRoundtrip_NoiseScenarios`/`PacketSizes` for SNR ≤16 dB (probe section B: 1500/1000 symbols decoded).
  - **No carrier recovery; the LO loses coherence.** TX renders the carrier per chip (12-sample blocks) and RX per symbol (96-sample blocks). Both use `dsp.CosineGenerator`, whose float32 time accumulator diverges with block size ([[sine-oscillator-generators]]). Probe `dsssdrift` (noiseless, 2000 BPSK symbols at 1 kBd): 0 errors in 0–0.4 s, then 67/200, 200/200 (fully inverted) by 0.6–0.8 s, and ≈50% after that. In the BER probe, noiseless 1000-symbol packets give BER 0.29 (M=2) and 0.18 (M=4). Short test payloads (≤4 bytes) stay inside the coherent window, which is why `TestRoundtrip_Basics` passes.
  - Processing-gain helpers (`config.go:82-160`) assume a real despreading gain that the M-ary Walsh code-shift design doesn't provide in the same sense (orthogonal signalling, not interference rejection) [INFERENCE].
  - `nextSymbol` compares the bit count returned by `Read` against `BitsPerSymbol` (`modulator.go:104`); this works only because `buffer.Bits.Read` returns bits.
- **Tested:** `TestRoundtrip_Basics` passes (short payloads, no noise). `TestRoundtrip_ModulationOrders` and the high-SNR NoiseScenarios cases fail on the SNR-estimate assertion: measured 4.9–7.2 dB vs 20–30 dB simulated. The low-SNR cases fail on TransmissionStart. Payload equality is never reached in the failing cases. A source TODO says the SNR-based threshold is unwired: `roundtrip_test.go:158` (`// TODO: Set target SNR … demod.SetTargetSNR(snrDb)`).
- **Correct: no.** BER is 0.2–0.48 at every Eb/N0 in sections A–C, set by the LO drift and the missing sync rather than by noise. Theory (antipodal BPSK) would be 2.4e-3 at 6 dB.

## TODOs

- [ ] Phase-continuous NCO (integer or float64 phase accumulator) shared by TX/RX; current LO decorrelates after ≈0.4 s — dsss/modulator.go:158-160, demodulator.go:150-152, x/math/dsp/generators_trig.go:240,258 #p1 #ER-022
- [ ] Code/chip acquisition (sliding correlator over a preamble or sync code) and frame start — dsss/demodulator.go:87-211 #p1 #ER-022
- [ ] Carrier phase recovery, or non-coherent (I/Q) correlation — dsss/demodulator.go:149-153 #p1 #ER-022
- [ ] Wire up the SNR-adaptive threshold (source TODO) — dsss/roundtrip_test.go:158 #p2
- [ ] Ship a code generator (Walsh/Gold) in the package instead of in the test, and warn that Walsh row 0 means no spreading — dsss/roundtrip_test.go:48-96 #p2
- [ ] Fix the estimator; see [[modem-quality-estimators]] — dsss/demodulator.go:274-350,404-430 #p2

## Links

- Module: [[modulation]] · [[modem-energy-detector-and-preamble-acquisition]] · [[sine-oscillator-generators]] · [[W7-filters-codecs-voice]] · CLI: [[audio-modem-cli]] (gen_dsss writes silence)
