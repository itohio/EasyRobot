---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, psk]
workstream: W7
module: "[[modulation]]"
summary: "BPSK/QPSK within ≈1–1.5 dB of theory once acquired; acquisition fails ≤4–6 dB, at QPSK 10°…"
source: [x/math/dsp/modulation/mpsk/config.go, x/math/dsp/modulation/mpsk/modulator.go, x/math/dsp/modulation/mpsk/demodulator.go, x/math/dsp/modulation/mpsk/waveform.go]
tests: [x/math/dsp/modulation/mpsk/roundtrip_test.go, x/math/dsp/modulation/mpsk/noise_test.go, x/math/dsp/modulation/mpsk/snr_limits_test.go, x/math/dsp/modulation/mpsk/phase_detection_test.go, x/math/dsp/modulation/mpsk/pll_reset_test.go, x/math/dsp/modulation/mpsk/long_packet_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/modulation/mpsk → FAIL 3 (TestRoundtrip_NoiseScenarios, TestRoundtrip_SingleSymbol_NoPreamble, TestSNRLimits), 9 pass (2026-11-09); BER probe (2026-11-09)"
---
# M-PSK modem (BPSK/QPSK/8-PSK…)

**What it does:** maps `BitsPerSymbol` (1–8) bits to one of M phases on a **complex baseband** symbol (I/Q interleaved float32), shaped by a one-symbol "raised cosine" envelope; the demodulator integrates each symbol window with the envelope (matched filter), runs AGC, a preamble-trained PLL (`x/math/filter/pll`), and quantises the phase.

**Where:**
- `mpsk/modulator.go:46-84` `Process`, `:119-129` `renderSymbol` (I=env·cos, Q=env·sin), `:138-175` preamble pattern (0,1,2,3,0,2,1,3…).
- `mpsk/demodulator.go:107-169` `Process` (fixed, non-overlapping windows of `2·SamplesPerSymbol` floats from stream start), `:171-580` `decodeSymbol` (energy gate → AGC → preamble/PLL → payload decision-directed PLL), `:582-602` `integrateSymbol`, `:604-648` `quantizePhase`, `:763-833` `SNR`/`Quality`.
- `mpsk/config.go:28-45` defaults (fs 96 kHz, SPS 384 → 250 Bd, preamble 20, amplitude 0.8), `:244-273` `MinimumEbN0`/`MinimumSNR`.

**Inputs → outputs:** `chain.Processor[float32, byte]` modulator: `buffer.BitsReader` (one symbol per byte, `BitsLength()==BitsPerSymbol`) → interleaved I,Q float32 (2 floats per sample). Demodulator: interleaved I/Q → `buffer.BitsWriter`. `CarrierHz` is metadata only — no carrier is generated (`renderSymbol` has no oscillator); a passband audio signal needs [[iq-mixer]] plus a float32↔complex64 adapter that does not exist.

## Completeness review

- **Complete: no.**
  - No symbol timing recovery: the symbol grid is whatever `2·SPS` boundary the stream starts on (`demodulator.go:121-165`). Probe (gated noise, Eb/N0 20 dB): offset 7 samples → BER 0; offset SPS/4 → QPSK BER 0.47, 8-PSK no decode; offset SPS/2 → BPSK BER 0.32.
  - No end-of-transmission detection while active: the silence reset sits inside `if !d.active` (`demodulator.go:213-245`), so trailing silence is decoded as symbol 0 forever (probe: 1000 symbols sent → 1062 decoded with 0.25 s trailing silence). `:219` (`if d.active && …` inside `!d.active`) is dead code.
  - Energy gate uses an **absolute** threshold from `Amplitude` and a hard-wired target SNR of 25 dB (`config.go:206-238`, `demodulator.go:83,179,191`), so detection depends on the received level: scale 0.1 → QPSK/8-PSK decode nothing; scale 0.01 → nothing for all orders (probe section C).
  - Phase ambiguity is resolved only by the preamble; no differential encoding, no unique word or frame sync, no soft outputs for FEC.
  - `estimateSamples` with zero bits returns `PreambleLength*SamplesPerSymbol` without the ×2 for I/Q (`modulator.go:89`).
  - AGC is updated two or three times per symbol (`demodulator.go:275,277→655`, `:321`, `:328`, `:345`, `:352`), so its time constant isn't what `agc.New(1.0, 0.001)` says (`:88`).
  - `quantizePhase` re-checks neighbours after an exact nearest search (`:627-645`), which is redundant.
- **Tested:** `TestPhaseDetection_*`, `TestPLLReset_BetweenPackets`, `TestLongPackets` and `TestRoundtrip_1Byte_Noise` pass. `TestRoundtrip_NoiseScenarios`/`TestSNRLimits` fail on the **SNR-estimate** assertion (`roundtrip_test.go:191`: measured 3.2–4.1 dB vs simulated 10–30 dB) **before** the payload comparison at `:230`, so data correctness at those SNRs isn't checked. `TestRoundtrip_SingleSymbol_NoPreamble` fails because `lockCount` is never incremented when `PreambleLength==0` (`demodulator.go:101`). `TestPhaseDetection_Rotation` passes while logging wrong symbols for 45/90/180/270° offsets: it only asserts equality below 22.5° (`phase_detection_test.go:106-115`).
- **Correct: partial.** Decision path, measured by the probe (Gaussian noise only over the signal, symbol-aligned, 3×1000 symbols; theory is coherent Gray PSK):

| Eb/N0 | BPSK BER (theory) | QPSK BER (theory) | 8-PSK BER (theory) |
|---|---|---|---|
| 4 dB | 1.2e-2, 1.3e-2, **3.5e-1** (1.3e-2) | 2.1e-1…4.8e-1 (1.3e-2) | no decode (4.6e-2) |
| 6 dB | 3–6e-3 (2.4e-3) | ≈1.0e-2 (2.4e-3) | no decode / 0.48 (2.0e-2) |
| 8 dB | 0–1e-3 (1.9e-4) | 0 (1.9e-4) | 0.42–1.0 (6.2e-3) |
| 10 dB | 0 (3.9e-6) | 0, 0, **0.47** (3.9e-6) | 5e-3…5e-2 (1.0e-3) |
| ≥12 dB | 0 | 0 | 0–3e-4 |

  Once acquired, symbol decisions sit within about 1–1.5 dB of theory. The failures are **acquisition** failures (BER ≈0.5 or nothing decoded). Phase/frequency (probe section D): BPSK tolerates 10–30° and 180°, but fails at 90° and at a 2 Hz CFO; QPSK fails at **10°** of static rotation and at 0.5 Hz CFO; 8-PSK fails at every rotation tested. So the PLL loop doesn't pull in. Sample clock +50 ppm is OK; +500 ppm gives QPSK no decode. With noise also over the padding (as in the repo tests), every order false-triggers on noise below about 15–20 dB wideband SNR (section B). The test's own `MinimumSNR` adds `10·log10(bits)` twice (`config.go:257,270`) and treats wideband SNR as Es/N0. With SPS 384 the real Eb/N0 is 22–26 dB above the test's "SNR".

## TODOs

- [ ] Symbol timing recovery (early-late/Gardner on the matched-filter output, or correlation against the preamble) instead of the fixed grid — mpsk/demodulator.go:121-165 #p1 #ER-022
- [ ] End-of-transmission detection while active (trailing silence decoded as symbols) — mpsk/demodulator.go:209-245 #p1 #ER-022
- [ ] Carrier phase acquisition fails at 10° (QPSK) and 90° (BPSK): fix the PLL pull-in, or derotate from the preamble correlation — probe section D; mpsk/demodulator.go:282-355 #p1 #ER-022
- [ ] SNR estimator counts preamble energy as "silence" before lock — mpsk/demodulator.go:251-255,765-776 #p2
- [ ] lockCount stays 0 without a preamble — mpsk/demodulator.go:101 #p2
- [ ] Make the rotation test assert exact symbols, or a resolved ambiguity, for every offset — phase_detection_test.go:106-115 #p2
- [ ] Reorder test assertions so payload equality runs before the SNR-estimate check — roundtrip_test.go:190-233 #p2
- [ ] Fix MinimumSNR (double bits term; wideband vs Es/N0) — mpsk/config.go:252-273 #p2
- [ ] estimateSamples zero-bit branch missing the ×2 for I/Q — mpsk/modulator.go:89 #p3
- [ ] Soft-decision (LLR) output for FEC — mpsk/demodulator.go:512-578 #p2
- [ ] Remove the redundant neighbour check in quantizePhase; update AGC once per symbol — demodulator.go:627-645, 275-352 #p3

## Links

- Module: [[modulation]] · components [[modem-energy-detector-and-preamble-acquisition]], [[modem-quality-estimators]], [[iq-mixer]] · workstream [[W7-filters-codecs-voice]]
