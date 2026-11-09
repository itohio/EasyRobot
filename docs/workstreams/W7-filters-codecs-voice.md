---
type: workstream
status: proposed
tags: [easyrobot, workstream, dsp, audio-modem, fec]
workstream: W7
---
# W7 — Filters, encoders/decoders, voice/audio pipeline robustness and noise figures

**Owner's goal (O-005, verbatim excerpt):** "ux should review module apis for ease of use and composability - i did some clever things there with filters , decoders(need more robust encoders/decoderes, especially ones over voice! i have some tests, but real audio pipeline not fully satisfy me with noise figures...)"

## What exists

The "real audio pipeline" path today:

```
bytes ─▶ [line code / FEC: x/math/dsp/coding]* ─▶ buffer.Bits(k) ─▶ Modulator (mpsk|qam|mfsk|mask|dsss|tc)
      ─▶ []float32 ─▶ x/marshaller/audio (WAV file or malgo device) ─▶ … channel … ─▶ WAV/device
      ─▶ Demodulator (energy gate → preamble → decisions) ─▶ buffer.Bits ─▶ [decoder]* ─▶ bytes
* via x/chain.Adapter in cmd/audio/common.go:502,542 — currently panics (nil buffer, x/chain/adaper.go:111,138)
```

| Layer | Note | Status (2026-11-09) |
|---|---|---|
| Modems | [[modulation]]: [[mpsk-modem]], [[qam-modem]], [[mfsk-goertzel-modem]], [[mask-modem]], [[dsss-walsh-modem]], [[smpte-ltc-timecode]] | all partial (LTC implemented); tests: mpsk/qam/mask/dsss FAIL, mfsk flaky, tc build-fail |
| Receiver front end | [[modem-energy-detector-and-preamble-acquisition]], [[modem-quality-estimators]] | partial; main failure source |
| Unused sync blocks | [[gardner-timing-recovery]] (0 output), [[afc-frequency-tracker]] (wrong), [[costas-carrier-loop]], [[complex-agc]], [[complex-signal-primitives]] (CFO wrong) | no tests, no callers |
| I/Q, adaptation | [[iq-mixer]], [[adaptive-modem-tier-selector]] | partial / implemented |
| DSP primitives | [[dsp-core]]: [[noise-generators]] ("white" is uniform), [[sine-oscillator-generators]] (float32 phase drift), [[goertzel]], [[chirp-generators]] (correlator not matched), [[pulse-shape-generators]] | mixed |
| Filters used by modems | [[filters]]: `filter/pll` (wrap bug: CFO 1010 Hz vs 10 Hz), `filter/agc` (instantaneous level, inverted α), FIR/IIR (Butterworth LP unstable) | partial |
| Coding / FEC | [[line-coding-and-fec]]: Hamming(7,4) tested; RS **corrects nothing**; CRC non-standard; Gray loses data at odd widths | partial |
| Framing / parsing | [[peg-protocol-parser]] (packet patterns, could frame modem payloads) | partial |
| Audio I/O | [[audio-io]] (WAV codec tested; requested data type ignored → FP32 written as int PCM), [[audio-modem-cli]] (gen/recv per scheme) | partial |
| Channel sounding | [[channel-sounding-and-blind-estimation-specs]], [[chirp-sync-receiver]], [[ofdm-modem]] | spec only / partial |

## Measured vs claimed noise figures

Definitions: *test SNR* = what the repo tests call SNR. That is wideband (noise over fs/2), and nominal: the harness uses uniform noise, so the true SNR is +4.77 dB. *Eb/N0* = SNR + 10·log10(SPS/k) (I/Q) or 10·log10(SPS/2k) (real). Probe = `tmp/easyrobot/audio/berprobe` (Gaussian noise, 3×1000 symbols, 2026-11-09). Theory = coherent Gray PSK/QAM, non-coherent orthogonal FSK, coherent unipolar ASK.

| Scheme (defaults) | Claim in code/tests | What the tests actually run and check | Probe: decision path (noise on signal only) | Probe: full pipeline (noise on padding too) | Offline CLI (`gen`→`recv`, [[audio-modem-cli]]) |
|---|---|---|---|---|---|
| QPSK (96 kHz, 250 Bd) | "min SNR 18.1 dB for BER 1e-5" (`mpsk/noise_test.go:26`); "QPSK should work at ~10 dB" (`snr_limits_test.go:21`) | 10–30 dB test SNR = **37.6–57.6 dB Eb/N0**; failures are the SNR-estimate assertion (reads 3–4 dB), payload never compared | BER 1e-2 @6 dB (theory 2.4e-3), 0 @8 dB → ≈1–1.5 dB loss; acquisition fails ≤4 dB and in 1/3 trials at 10 dB | fails (BER ≈0.47 or nothing) at 5–10 dB wideband (Eb/N0 27–32 dB); 1/3 trials fail at 15 dB (37 dB Eb/N0); OK ≥20 dB | clean OK; `--noise -20` (≈14 dB nominal) fails |
| BPSK | same helpers | — | ≈ theory from 6 dB; acquisition flaky ≤4 dB | fails below 20 dB wideband (44 dB Eb/N0) | stock CLI bit-shifted (chunk packing); one-chunk OK |
| 8-PSK | — | — | acquisition fails ≤8 dB; 5e-3 @10 dB (theory 1e-3) | fails below 15 dB wideband | clean OK |
| 4-QAM | min 15.1 dB | variants test **skips all cases** | ≈0.5–1 dB from theory | fails below 15 dB wideband | fail |
| 16-QAM | min 23.5 dB | runs 25/30 dB only; SNR estimate 1.5 dB | **error floor** 2–4e-3 @15 dB (theory 2e-7) | fails below 15 dB wideband | fail |
| 4-FSK (96 kHz, 250 Bd) | min 15 dB (8/16-FSK log) | SNR estimate saturates at 20.3–20.7 dB whatever the input; flaky SignalStrength tolerance | **at theory** (1.5e-2 @6 dB vs 1.6e-2) | BER 0 at every wideband SNR 5–40 dB (Eb/N0 ≥21.8) | demod OK; stock CLI garbled (chunk packing) |
| 16-FSK | — | — | 0 @≥8 dB; acquisition fails ≤6 dB | BER 0 at 5–40 dB | — |
| 2-/4-ASK (22 kHz, 100 Bd) | runs 40/35/min+5/min+3 dB | SNR estimate over-reads (49.7 @40) | 2–3 dB from theory, high variance | 4-ASK BER ≈0.33 below 25 dB wideband (41 dB Eb/N0) | 4-MASK clean OK (needs `-f`); noise fails |
| DSSS/Walsh (96 kHz, 1 kBd, SF 8) | processing-gain helpers | short payloads only pass | **BER 0.2–0.48 even noiseless** (LO drift after ≈0.4 s) | same | gen writes silence |
| LTC | — | stale tests don't build | clean roundtrip only | untested | OK clean |

Impairments (probe C/D, gated noise at Eb/N0 20 dB). Start offset SPS/2 breaks every scheme. Received level ×0.01 decodes nothing on any scheme (×0.1 already kills PSK-4/8, QAM and FSK). Sample clock +500 ppm: QPSK no decode, 16-FSK loses 105 symbols. Clock +50 ppm: ASK BER ≈0.5. I/Q rotation: QPSK fails at 10°, QAM at 90°/180°, QAM fails at 0.5 Hz CFO.

## Where the bad noise figures come from

1. **Implementation, receiver front end (dominant).** The symbol decisions are close to theory, so the loss isn't in detection. Acquisition is the problem: energy-gated detection with absolute thresholds, preambles that "complete" on noise, no symbol timing recovery, frames that never end. With noise present before the packet, PSK/QAM/ASK need about 15–25 dB wideband SNR (Eb/N0 ≈37–42 dB) where theory needs ≈10 dB Eb/N0, a **gap of ≈25–30 dB** ([[modem-energy-detector-and-preamble-acquisition]]).
2. **Implementation, carrier and oscillators.** float32-time oscillators drift (DSSS dies after 0.4 s; ASK dies at 50 ppm). The `filter/pll` wrap bug and slow pull-in leave QPSK failing at a 10° offset. No CFO tracking anywhere. QAM has no phase reference.
3. **Implementation, coding and framing.** No sync word, length or CRC; RS corrects nothing; the encoder chain panics; the CLI zero-pads each of 8 chunks and shifts bits (`cmd/audio/common.go:304-318`). Hard decisions only (no LLRs), and erasures become deletions (MFSK).
4. **Test design (it hides 1–3 and makes numbers look better than they are).** Uniform noise (+4.77 dB, bounded tails); wideband "SNR" that is 20–26 dB more generous than Eb/N0; packets always symbol-aligned after silence; statistics assertions placed before payload checks; vacuous or skipped cases (qam variants, mpsk rotation >22.5°); the `MinimumSNR` helpers disagree with theory and with each other.
5. **Receiver self-reported figures are wrong.** Every `Quality().SNR` is biased or saturated ([[modem-quality-estimators]]), and the CLI prints "Signal Strength 176 dB" because FP32 is written as int PCM ([[audio-io]]). Any "noise figure" read from these printouts is meaningless.
6. **Waveform choices that a voice channel will reject.** 96 kHz/20 kHz and 22 kHz/10 kHz defaults sit above the 300–3400 Hz voice band. PSK/QAM are baseband I/Q written as stereo L/R. One-symbol pulse shaping has a wide spectrum. Nothing models band-limiting, echo or codec distortion.

## Gaps

- **AGC:** none that works across ±40 dB. Detection depends on the configured amplitude; `filter/agc` is defective and `internal/agc` is clamped to ±20 dB and unused.
- **Timing recovery:** none in any modem; the Gardner block outputs nothing.
- **Carrier recovery:** a PLL exists only in PSK/QAM (bug in `filter/pll` wrap), with no CFO estimation. ASK/DSSS rely on a free-running LO.
- **Channel model:** only (uniform) additive noise. Missing: Gaussian noise, delay or fractional offset, level, ppm drift, band-pass 300–3400 Hz, echo/multipath, clipping, codec emulation.
- **FEC integration:** the encoder path panics; RS is a stub; there is no interleaver, no soft decoding, no erasure handling, and no framing with CRC.
- **Passband PSK/QAM:** no adapter from I/Q float32 to [[iq-mixer]].
- **Measurement:** no BER-vs-Eb/N0 harness in the repo; channel sounding is spec only.
- **Ergonomics:** silent zero returns on a mis-sized `BitsReader`; `Validate()` mutates user config; two I/Q representations; `internal` processors read from dst.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Shared synchroniser: CFAR energy detect, chirp or preamble correlation, fractional timing, frame end — [[modem-energy-detector-and-preamble-acquisition]] #p1 #ER-022
- [ ] Framer (sync word + length + CRC-16/32 standard) between bytes and modem; candidate implementation via [[peg-protocol-parser]] patterns on RX — new #p2
- [ ] Working FEC: real RS decoder or convolutional + Viterbi (soft), plus a block interleaver; erasure/LLR outputs from demodulators — [[line-coding-and-fec]], [[modulation]] #p2
- [ ] Voice-band profile: carriers and tones in 500–3000 Hz at 8/16/48 kHz sample rates; passband PSK via [[iq-mixer]] — mfsk/mpsk configs #p2
- [ ] Port tc stale tests — x/math/dsp/modulation/tc/*_test.go #p1 #ER-022

## Inferred objectives (proposals)

- **Proposal (robust baseline):** a voice-band 4/8-FSK modem (non-coherent, so phase and CFO don't matter, and it already sits at theory) with a linear-chirp preamble for timing and level, a framer with CRC-16, and interleaved Hamming(7,4) or convolutional FEC. Target BER <1e-4 at Eb/N0 ≈10 dB, measured end to end on a channel model and on a real speaker→mic loop.
- **Proposal (rate tier):** passband DBPSK/DQPSK (differential, so phase ambiguity and slow CFO don't matter) with RRC shaping in 300–3400 Hz, used through [[adaptive-modem-tier-selector]] once the quality estimators are fixed.
- **Proposal (process):** every modem PR ships a BER curve (probe-style table) against theory; "noise figure" means Eb/N0 at BER 1e-3/1e-5 plus the acquisition threshold, not receiver-reported SNR.

## Research questions for ideation

1. Which channel is "voice": a stereo cable, speaker→mic in air (reverb, phone AGC, Doppler), a phone or VoIP codec (AMR/Opus suppress non-speech tonals), or a radio voice channel? Each favours a different waveform (FSK vs OFDM vs chirp spread spectrum).
2. For speaker→mic and codec paths, how do FSK (Bell-202-like), OFDM with pilots and cyclic prefix, and chirp spread spectrum (LoRa-style CSS) compare for BER at a given Eb/N0, multipath tolerance, and ESP32 decode cost?
3. Which preamble gives the best joint timing, CFO and level acquisition at low SNR in a 3 kHz band: linear chirp, Zadoff–Chu/CAZAC, or Barker? What false-alarm rate is acceptable?
4. Which FEC is feasible on ESP32 at audio rates: convolutional K=7 soft Viterbi, short LDPC, RS(255,223) with interleaving, or a fountain/rateless code for one-way broadcast?
5. Can speech codecs be survived by shaping data like speech (formant-like tones, codec-aware constellation), as some data-over-voice research does? What rates are realistic (100 b/s vs 1–2 kb/s)?
6. What channel model should the repo standardise on to reproduce field results offline: an impulse response from channel sounding (measure/SPEC.md) plus noise, ppm and band-pass?
7. Fixed-point (Q15) Goertzel/FFT and NCO on ESP32: what cycle budget per symbol is there at 8–16 kHz sampling alongside other tasks?
8. Is it worth unifying the coding layer on packed bits (`buffer.Bits`) instead of one symbol per byte, to support soft values and erasures cleanly?
