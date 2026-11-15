---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, synchronisation]
workstream: W7
module: "[[modulation]]"
summary: "main failure source (timing, level, noise false-trigger)"
source: [x/math/dsp/modulation/mpsk/demodulator.go, x/math/dsp/modulation/qam/demodulator.go, x/math/dsp/modulation/mfsk/demodulator.go, x/math/dsp/modulation/mask/demodulator.go, x/math/dsp/modulation/dsss/demodulator.go, x/math/dsp/modulation/mpsk/config.go]
tests: [x/math/dsp/modulation/mpsk/roundtrip_test.go, x/math/dsp/modulation/dsss/roundtrip_test.go]
test_result: fail
complete: no
correct: no
evidence: "BER probe sections A–D (2026-11-09); dsss TransmissionStart failures in go test (2026-11-09)"
---
# Modem energy detector, symbol framing and preamble acquisition (shared pattern)

**What it does:** this is the front end every demodulator re-implements inline; there is no shared type. It cuts the stream into fixed windows of one symbol length from sample 0. It computes mean window energy and compares it with a threshold `EnergyThresholdForSNR(25 dB)`, derived from the **configured** amplitude: 5× the expected noise power, clamped to [2×noise, 0.1×signal]. It then scales that by 0.1 (coarse), 0.5 (during preamble) or 0.3 (active) and declares "active". After that it counts preamble symbols (pattern match with lenient tolerances for PSK, alternating extreme tone/level for FSK/ASK, none for QAM/DSSS) before emitting payload symbols.

**Where:** `mpsk/config.go:206-238` + `mpsk/demodulator.go:107-169,171-264,282-467`; `qam/config.go:299-333` + `qam/demodulator.go:127-275,367-380`; `mfsk/config.go:232-259` + `mfsk/demodulator.go:71-130,132-312`; `mask/config.go:228-260` + `mask/demodulator.go:103-163,165-385`; `dsss/demodulator.go:87-211` (fixed `EnergyThreshold`).

**Inputs → outputs:** sample stream → per-window active/preamble/payload state → symbol decisions.

## Completeness review

- **Complete: no.** Missing in all five modems:
  - **Timing:** no symbol-timing estimate, no fractional-delay interpolation, no sliding search. Alignment holds only when the packet starts on a multiple of SPS from stream start. That is true in the repo tests (0.5 s padding = 125·SPS) but not over an audio link. Probe: offset SPS/2 breaks every scheme.
  - **Level independence:** thresholds are absolute (configured amplitude, fixed 25 dB target). Probe: received level ×0.1 → mpsk-4/8, qam and mfsk decode nothing; ×0.01 → every scheme decodes nothing. An acoustic link changes level by 20–40 dB.
  - **Noise robustness of detection:** noise in the padding at realistic levels triggers "active". BPSK accepts any symbol during the preamble because both symbols are "adjacent" (`mpsk/demodulator.go:381-390`), so preambles complete on noise and payload starts misaligned. Probe section B: with noise over padding, mpsk/qam/mask/dsss fail (BER ≈0.3–0.5 or no decode) at wideband SNR ≤10–20 dB, even though Eb/N0 is ≥25 dB there.
  - **End of frame:** PSK/QAM never leave "active" (reset branch only runs when `!active`, `mpsk/demodulator.go:213`, `qam/demodulator.go:237`); trailing silence becomes symbols.
  - **No unique word / length header / CRC**, so the receiver can't tell payload from garbage. Framing has to come from [[line-coding-and-fec]], which the CLI's encoder path can't currently use ([[audio-modem-cli]]).
- **Tested:** indirectly by every roundtrip test; those place packets symbol-aligned after silence or bounded uniform noise, so none of the above is exercised.
- **Correct: no** for any channel with unknown delay, level or noise floor (probe evidence above). It works only under the exact conditions the tests construct.

## TODOs

- [ ] Replace with one shared, tested synchroniser: noise-floor tracking (CFAR), preamble or chirp correlation (normalised), fractional timing estimate, frame start, end-of-frame by energy drop or length header — all demodulators #p1 #ER-022
- [ ] Remove the absolute-threshold coupling to the configured amplitude (`EnergyThresholdForSNR`) — mpsk/config.go:206-238 (and qam/mfsk/mask) #p1 #ER-022
- [ ] Test harness: random fractional offset, ±40 dB level, Gaussian noise over padding, clock ppm — test_utils/transmit.go #p1 #ER-022

## Links

- Module: [[modulation]] · [[mpsk-modem]] · [[qam-modem]] · [[mfsk-goertzel-modem]] · [[mask-modem]] · [[dsss-walsh-modem]] · unused building blocks [[gardner-timing-recovery]], [[symbol-matched-filter-integrator]] · chirp sync in CLI [[chirp-sync-receiver]] · [[W7-filters-codecs-voice]]
