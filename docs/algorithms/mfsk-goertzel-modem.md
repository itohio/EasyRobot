---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, fsk]
workstream: W7
module: "[[modulation]]"
summary: "4-FSK at non-coherent theory; acquisition limit ≈4–6 dB; deletes symbols under stress"
source: [x/math/dsp/modulation/mfsk/config.go, x/math/dsp/modulation/mfsk/modulator.go, x/math/dsp/modulation/mfsk/demodulator.go, x/math/dsp/modulation/mfsk/waveform.go]
tests: [x/math/dsp/modulation/mfsk/roundtrip_test.go, x/math/dsp/modulation/mfsk/mfsk_test.go, x/math/dsp/modulation/mfsk/config_test.go]
test_result: flaky
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/math/dsp/modulation/mfsk → FAIL 1 (Test16FSK_Noise/SNR_25dB: SignalStrength −11.99 vs −1.94 dB, tol 10), 6 pass (2026-11-09); baseline 2026-11-08 failed the 22 dB case in 1 of 3 runs; BER probe tmp/easyrobot/audio/berprobe"
---
# M-FSK modem (non-coherent Goertzel tone bank)

**What it does:** sends one of M = 2^`BitsPerSymbol` tones (`CarrierHz + i·ToneSpacingHz`; spacing defaults to the symbol rate, i.e. orthogonal at 1/T) as **real passband** audio with a one-symbol envelope. The receiver runs a Goertzel bank per fixed symbol window and picks the strongest tone (non-coherent). A preamble alternates the lowest and highest tone.

**Where:** `mfsk/modulator.go:46-129`; `mfsk/demodulator.go:71-130` `Process` (fixed windows from stream start), `:132-338` `decodeSymbol` (energy gate, preamble count, PLL update), `:340-373` `detectTone` (`dsp.Goertzel` per tone, best/second-best ratio check), `:410-445` `Quality`; `mfsk/config.go:30-48` defaults (fs 96 kHz, fc 20 kHz, SPS 384 → 250 Bd, 4-FSK, preamble 16).

**Inputs → outputs:** `BitsReader` symbols → real float32 samples; real samples → `BitsWriter`. This is the only modem in the package that produces a real single-channel passband signal end to end, so it is the most "audio-ready".

## Completeness review

- **Complete: no.**
  - **Erasures become deletions.** When the best/second ratio is <1.15 (M>4, `demodulator.go:363-368`), or the magnitude is below `EnergyThreshold·factor` (`:231-234`), `decodeSymbol` returns −1 and the symbol is **dropped** from the output stream. Every later bit shifts, and nothing marks the erasure. Probe: clock +500 ppm on 16-FSK → 895/1000 symbols decoded.
  - No symbol timing recovery: offset SPS/2 → nothing decoded; offsets 7 and SPS/4 are fine (probe section C).
  - The PLL is fed the Goertzel phase of the detected tone (`:236-256`), but its output is never used for detection. That's dead computation; the absolute tone phase depends on window timing anyway.
  - The "high M" branches test `BitsPerSymbol >= 8` (256-FSK) while the comments say 8-FSK (`:225,267,302`). Those branches never run for 8- or 16-FSK.
  - The energy gate is absolute (target SNR fixed at 25 dB, `:66`). Probe: scale 0.1 → nothing decoded.
- **Tested:** `TestRoundtrip_*`, `Test8FSK_Noise`, the bandwidth tests and `TestToneCount_MFSK_Formula` pass. `Test16FSK_Noise` is **flaky by design**: Rx `SignalStrength` is 10·log10(Goertzel magnitude²) (`:427-434`), which sits about 10 dB below the Tx value of 10·log10(A²), right at the 10 dB tolerance (`roundtrip_test.go:227`). Measured −11.83…−11.99 dB vs −1.94 dB. The receiver SNR estimate is best²/second-best² (`:316-328`) and saturates at 20.3–20.7 dB **regardless of the injected SNR** (log: 20.67 at "25 dB", 20.45 at "20 dB"). It measures tone leakage, not noise; see [[modem-quality-estimators]].
- **Correct: yes for the decision path** (probe, gated Gaussian noise, aligned, 3×1000 symbols, theory = non-coherent orthogonal M-FSK):

| Eb/N0 | 4-FSK BER (theory) | 16-FSK BER (theory) |
|---|---|---|
| ≤2 dB | nothing decoded (preamble not acquired) | nothing decoded |
| 4 dB | 5.9e-2, 7.0e-2, none (6.2e-2) | 0.49 / none (1.6e-2) |
| 6 dB | 1.5–1.9e-2 (1.6e-2) | none / 0.37 (1.1e-3) |
| 8 dB | 1–2e-3 (1.7e-3) | 0 (1.3e-5) |
| ≥10 dB | 0–1e-3 | 0 |

  4-FSK matches theory once acquired. The limit is the preamble/energy acquisition, at about 4–6 dB Eb/N0. With noise over the padding as well (section B, test-style wideband SNR 5–40 dB), BER is 0 at every level, because 4- and 16-FSK at SPS 384 have Eb/N0 ≥ 21.8 dB even at 5 dB wideband SNR. MFSK is the only scheme that passes the full-noise pipeline at every wideband SNR tested.

## TODOs

- [ ] Emit erasures (or a best guess plus a reliability flag) instead of dropping symbols — mfsk/demodulator.go:217-234,363-368 #p1 #ER-022
- [ ] Symbol timing recovery or a sliding preamble correlator — mfsk/demodulator.go:71-130 #p1 #ER-022
- [ ] Fix the `BitsPerSymbol >= 8` vs tone-count confusion — mfsk/demodulator.go:225,267,302 #p2
- [ ] Remove the unused PLL or use it for AFC (tone-offset tracking) — mfsk/demodulator.go:65,236-256 #p3
- [ ] SignalStrength scale mismatch with the transmitter (≈10 dB) makes Test16FSK_Noise flaky — mfsk/demodulator.go:427-434; roundtrip_test.go:227 #p2
- [ ] SNR estimator should use off-tone noise bins, not the second-best tone — mfsk/demodulator.go:316-328 #p2
- [ ] Soft output (per-tone energies) for FEC — mfsk/demodulator.go:340-373 #p2

## Links

- Module: [[modulation]] · [[modem-energy-detector-and-preamble-acquisition]] · [[modem-quality-estimators]] · [[W7-filters-codecs-voice]] · DSP Goertzel in [[dsp-core]]
