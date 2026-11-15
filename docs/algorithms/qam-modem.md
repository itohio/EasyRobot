---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, qam]
workstream: W7
module: "[[modulation]]"
summary: "4-QAM ≈0.5–1 dB from theory; 16-QAM error floor ≈2e-3 at 15 dB (>5 dB loss); 90/180° ambiguity…"
source: [x/math/dsp/modulation/qam/config.go, x/math/dsp/modulation/qam/modulator.go, x/math/dsp/modulation/qam/demodulator.go, x/math/dsp/modulation/qam/waveform.go, x/math/dsp/modulation/qam/math.go]
tests: [x/math/dsp/modulation/qam/roundtrip_test.go, x/math/dsp/modulation/qam/long_packet_test.go, x/math/dsp/modulation/qam/snr_limits_test.go, x/math/dsp/modulation/qam/pll_reset_test.go, x/math/dsp/modulation/qam/debug_position51_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/modulation/qam → FAIL 4 (TestLongPackets, TestRoundtrip_Basics, TestRoundtrip_NoiseScenarios, TestSNRLimits), 9 pass (2026-11-09); BER probe (2026-11-09)"
---
# Square M-QAM modem (4/16/64/256-QAM)

**What it does:** maps `BitsPerSymbol` (an even number) bits to a square constellation, normalised to unit average power, on **complex baseband** I/Q interleaved float32 with a one-symbol envelope. The demodulator integrates each symbol window against the envelope, applies a magnitude AGC with heuristics, and rotates through a decision-directed PLL. There is no preamble.

**Where:** `qam/waveform.go:23-58` constellation table; `qam/modulator.go:45-125`; `qam/demodulator.go:127-189` `Process` (fixed windows from stream start), `:191-516` `decodeSymbol`, `:518-541` integrate/quantise; `:610-727` `Quality`/`SNR`; `qam/config.go:29-46` defaults (fs 96 kHz, SPS 384, 16-QAM, `PreambleLength: 0` — "QAM doesn't currently support preambles"), `:256-290` theory helpers.

**Inputs → outputs:** like [[mpsk-modem]]: `BitsReader` symbols → interleaved I/Q; I/Q → `BitsWriter`. `CarrierHz` is metadata only.

## Completeness review

- **Complete: no.**
  - **No preamble, so no absolute phase reference.** The 90°/180° ambiguity is never resolved. Probe section D: rotation 10–30° → BER 0 (4-QAM), 7.5e-3 (16-QAM at 30°); rotation 90° or 180° → BER ≈0.47; CFO 0.5 Hz → BER ≈0.48.
  - No symbol timing recovery (fixed grid, `demodulator.go:127-189`). 16-QAM at offset SPS/4 → BER 0.40; offset SPS/2 → ≈0.46 for both orders.
  - **Gray mapping is ineffective.** `axisLevels[gray(i)] = level_i` and point = `axisLevels[gray(bits)]` (`waveform.go:37-54`), so bits→level is natural binary. The demodulator mirrors it (`demodulator.go:535-541`), so the roundtrip is consistent, but adjacent-level errors flip two bits [INFERENCE from code; it contributes to the 16-QAM loss below].
  - Amplitude handling relies on absolute heuristics: AGC is skipped "if magnitude is within 0.1 of an expected constellation magnitude" (`demodulator.go:309-341`), and the energy gate uses absolute thresholds. Probe: scale 0.1 → nothing decoded for 4-QAM or 16-QAM.
  - Same trailing-silence issue as MPSK (symbols decoded after the end; probe 1062/1000).
- **Tested:** 9 pass, including `TestRoundtrip_BitsPerSymbolVariants`, which **skips every case** (log: "skipping bits=2 at 15dB (minimum 15.1dB)", bits 4 and 8 also skipped), so it passes without exercising anything. Failures are all on statistics assertions before payload comparison: `TestRoundtrip_NoiseScenarios`/`TestSNRLimits` measured SNR 1.47–2.48 dB vs simulated 10–30 dB; `TestLongPackets` 15.6/18.6/21.6/24.5 dB for 512/1024/2048/4096 bytes (the estimate improves with length, which points to acquisition transients averaged into an un-windowed EVM); `TestRoundtrip_Basics` SignalStrength −43 dB vs −1.94 dB (trailing zero-magnitude symbols averaged in). See [[modem-quality-estimators]].
- **Correct: partial.** Probe (gated Gaussian noise, aligned, 3×1000 symbols; theory: coherent Gray square QAM):

| Eb/N0 | 4-QAM BER (theory) | 16-QAM BER (theory) |
|---|---|---|
| 4 dB | 1.6e-2…3.8e-1 (1.3e-2) | 0.37–0.41 (5.9e-2) |
| 6 dB | 2–4e-3 (2.4e-3) | 0.11–0.48 (2.8e-2) |
| 8 dB | 5e-4…1e-3 (1.9e-4) | 6.5e-2…0.47 (9.2e-3) |
| 12 dB | 0 (9e-9) | 1.0–1.4e-2 (1.4e-4) |
| 15 dB | 0 | 1.8–4.3e-3 (1.8e-7) |
| 20 dB | 0 | 0–2.5e-4 (1e-19) |

  4-QAM is within about 0.5–1 dB of theory once acquired. **16-QAM has an error floor**: about 2e-3 at 15 dB, where theory gives 2e-7, i.e. more than 5 dB loss. Likely causes: AGC normalising per-symbol magnitude, which carries the information in QAM, plus natural-binary mapping [INFERENCE]. Full-noise run at Eb/N0 20 dB: 16-QAM BER 8.5e-3 (section C).

## TODOs

- [ ] Add a preamble or unique word for phase-ambiguity resolution and frame sync — qam/config.go:39; probe section D #p1 #ER-022
- [ ] Symbol timing recovery — qam/demodulator.go:127-189 #p1 #ER-022
- [ ] Make Gray mapping real: map bits→level via gray, demap via the inverse — qam/waveform.go:37-54, demodulator.go:535-541 #p2
- [ ] Replace per-symbol magnitude AGC and absolute-magnitude heuristics with preamble-referenced gain; 16-QAM floor ≈2e-3 at 15 dB — qam/demodulator.go:278-341 #p1 #ER-022
- [ ] End-of-transmission detection while active — qam/demodulator.go:235-252 #p1 #ER-022
- [ ] TestRoundtrip_BitsPerSymbolVariants skips every case; set the SNR so cases actually run — qam/roundtrip_test.go:64-78 #p2
- [ ] Windowed EVM estimator that excludes acquisition — qam/demodulator.go:400-425,610-623 #p2
- [ ] Rename or fold debug_position51_test.go: it is a valid noiseless full-symbol roundtrip (asserts equality at :122) but is named and logged as a debug dump — qam/debug_position51_test.go:84-122 #p3

## Links

- Module: [[modulation]] · [[mpsk-modem]] · [[modem-energy-detector-and-preamble-acquisition]] · [[modem-quality-estimators]] · [[W7-filters-codecs-voice]]
