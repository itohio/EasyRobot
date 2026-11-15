---
type: research
status: draft
owner: ideation
tags: [easyrobot, research, ideation, W7, modem, dsp, fec, audio]
workstream: W7
created: 2026-11-09
updated: 2026-11-09
---

# W7: data over a "voice" channel: waveform, preamble, FEC, soft bits, channel model

**Questions:** EasyRobot W7 and [[W7-filters-codecs-voice]], whose measured noise figures this note relies on: decisions are near theory, while acquisition and synchronisation lose 25–30 dB.

**The channel is the owner's open question** (pending in `docs/decisions.md`), so this note designs for the likely options.

**Status:** hypotheses and recommendations; nothing built or run.

**Labels:** observed, reported, inferred, `#verify`. DSP textbook facts (Goertzel, LFM chirps, Zadoff–Chu, Barker, Viterbi) are from memory and marked inferred where numbers matter.

**Network (read-only):** 3 searches; no page reads beyond the result excerpts.
- data-over-voice codec research: `rozetyp/voice-codec-modem` repo description; Bilkent GSM speech-like modulation paper excerpt;
- 3GPP TS 26.267 eCall in-band modem (ETSI/TEC excerpts);
- ggwave, liquid-dsp/quiet and FreeDV/codec2 data modes: Rhizomatica Mercury `MODES.md` excerpt; codec2 README excerpts.

## Verdict

1. **The waveform isn't the main problem; the receiver front end is** (W7 measurements). Whichever waveform is chosen, the first work is a shared synchroniser:
   - a CFAR detector;
   - an up-chirp + down-chirp preamble giving timing, CFO and level;
   - an NCO, AGC and framing with CRC;
   - a BER-vs-Eb/N0 harness with a realistic channel model.
2. **Per channel:**
   - non-coherent **M-FSK** for speaker→mic in air and for FM radio voice;
   - **OFDM with pilots + LDPC** (FreeDV-style) for SSB/HF or clean line audio, where rate matters;
   - **speech-shaped or eCall-style** modulation only if the path goes through a cellular/VoIP speech codec, at about 100 b/s.
3. **ESP32 cost doesn't discriminate.** All candidates decode in a small fraction of an ESP32 core at 8–16 kHz. Choose for robustness.
4. **FEC:** CRC-16 framing + convolutional K=7 r=1/2 soft Viterbi + block interleaver first. Outer RS or a fountain code later for erasures and one-way broadcast.
5. **Soft bits:** add a soft path (`[]int8` LLRs, 0 = erasure) beside the byte path rather than replacing it.

## 1. Which "voice" channel (Q1, owner)

| Option | What breaks data | Fits | Expected rate (inferred) |
|---|---|---|---|
| **A. Line audio** (cable, sound card) | little: level, ppm clock offset | anything; OFDM/QAM | kb/s |
| **B. Speaker → mic in air** (robot ↔ robot, phone ↔ robot) | reverb/multipath (tens of ms), phone AGC and noise suppression, speaker non-linearity, Doppler if moving | non-coherent M-FSK with guard time (ggwave-style); chirp/CSS for range | 50–500 b/s |
| **C. Radio voice channel** (FM walkie-talkie / VHF; SSB HF) | FM: pre-emphasis, squelch tails, limited audio band; SSB: frequency offset, fading | FM: AFSK (Bell 202, 1200 Bd, as APRS uses); SSB: codec2/FreeDV OFDM data modes with LDPC (reported: Mercury uses the stock codec2 OFDM data modes plus custom rate-1/3 LDPC modes) | FM ~1200 b/s; SSB tens to ~1,000+ b/s |
| **D. Cellular / VoIP speech codec** (AMR, Opus/SILK, apps with noise suppression) | the codec models speech: steady tones and OFDM are distorted or suppressed | speech-shaped modulation; the **3GPP eCall in-band modem** (TS 26.267) exists exactly to carry the eCall Minimum Set of Data (140 bytes) through speech codecs (reported, spec scope) | ~100 b/s over cellular; a learned speech-shaped modem claims **76 b/s (cellular) to 3,196 b/s (Zoom-class)** (reported, `rozetyp/voice-codec-modem` description, not verified) |

**Recommendation:** design for **B first** (robots and phones near each other), with a profile system that adds A/C/D later. If the owner means D, the work is a different research project (speech-shaped modulation); flag it before investing (owner decision 1).

## 2. Waveform comparison in 300–3400 Hz (Q2)

| Waveform | Multipath / reverb | CFO / Doppler | Codec survival (D) | Rate in 3 kHz | ESP32 decode cost at 8 kHz (inferred) |
|---|---|---|---|---|---|
| **M-FSK (non-coherent), symbol ≥ 20 ms + guard** | good if the symbol ≫ delay spread | good (tone bins ≥ 2× Doppler) | poor on CELP codecs (steady tones get modelled away; noise suppression attacks stationary tones) | 8-FSK at 50 Bd = 150 b/s; multi-tone (ggwave-like) higher | Goertzel: ~2 MAC/sample/tone → 8 tones × 8 kS/s = 128 kMAC/s: negligible |
| **OFDM + cyclic prefix + pilots** | excellent with CP > delay spread | needs CFO estimation (pilots/preamble); Doppler limits the symbol length | poor | 1–3 kb/s (QPSK on ~16–30 carriers) | one 256-point FFT per ~20–40 ms symbol: < 1 % of a core |
| **Chirp spread spectrum (LoRa-style)** | very good (processing gain, dechirp resolves paths) | good: an offset becomes a bin shift, resolved with up/down chirps | moderate (chirps are less "stationary" than tones `#verify`) | low: SF7 over 3 kHz ≈ 160 b/s raw (inferred: 7 bits per 128/3000 s) | dechirp + 128-pt FFT per symbol: negligible |
| **DBPSK/DQPSK passband, RRC** (EasyRobot's rate tier) | needs equalisation beyond mild multipath | differential tolerates slow CFO | poor | 1–2.4 kb/s | negligible; equaliser adds a little |

**Recommendation:** M-FSK (8-FSK, 50 Bd, 400 Hz spacing in 600–3400 Hz) as the **robust baseline**. It already decodes at theory in the repo (W7: 4-FSK and 16-FSK at theory once acquired). OFDM is the **rate tier** for channels A and C-SSB, after the synchroniser exists.

## 3. Acquisition preamble (Q3)

| Preamble | Timing | CFO | Level/AGC | Low-SNR detection | Notes |
|---|---|---|---|---|---|
| Barker-13 | sharp (peak-to-sidelobe 13:1) | poor (correlation collapses under CFO) | no | short, so low energy | too short for 3 kHz at low SNR |
| Zadoff–Chu (CAZAC) | ideal periodic autocorrelation | sensitive: a CFO shifts the peak (timing–CFO ambiguity) | constant envelope helps AGC | good with long sequences | standard for OFDM (LTE PSS); needs coarse CFO first |
| **Linear chirp, up + down pair** | good: average of the two peaks | **CFO = half the difference of the up/down peak times × the sweep rate** (inferred, standard LFM range–Doppler coupling) | constant envelope; known amplitude → level estimate | long duration → high energy and processing gain | **recommended**: one block gives timing, CFO and level |

**Detection:** CFAR on the normalised correlation, with a threshold set for a target false-alarm rate (e.g. 1 per hour of noise at the symbol rate, inferred). This replaces the absolute energy thresholds that W7 identifies as the main failure source.

**Length:** 2 × 100 ms sweeps over 500–3000 Hz give a time-bandwidth product of 250 per chirp, ~24 dB processing gain (inferred).

## 4. FEC feasible on ESP32 at audio rates (Q4)

| Code | Strength | Cost at 1 kb/s on ESP32 (inferred) | Use |
|---|---|---|---|
| **Convolutional K=7 r=1/2, soft Viterbi** | ~5 dB coding gain at BER 1e-5 (textbook) | 64 states × 2 add-compare-selects per bit → 128 k ACS/s: negligible | **first**; needs soft input and an interleaver against bursts |
| Block interleaver (e.g. 16 × 32) | spreads burst errors for Viterbi | memory only | with the convolutional code |
| RS(255,223) / shortened | 16 byte errors or 32 erasures per block | Berlekamp–Massey + Chien: a few ms per block | **outer code** for erasures (lost symbols, collisions). W7 found the repo's RS "corrects nothing", so it needs a real decoder |
| Short LDPC (e.g. codec2's rate-1/3 256/768) | best at low SNR | belief propagation: tens of iterations × edges; fine at < 1 kb/s | rate tier (OFDM), as FreeDV does |
| Fountain / LT / RaptorQ | rateless; ideal for one-way broadcast with packet erasures | XOR-heavy, cheap | broadcast to many robots without ACKs |

## 5. Rates through speech codecs (Q5)

- **D (cellular/VoIP):** plan on **~100 b/s** with robust speech-shaped modulation, a conservative planning figure.
  - eCall's in-band modem is the standardised precedent (TS 26.267/26.268 with reference C code). One MSD data frame (1,120 bits) takes 1,320 ms with the fast modulator or 2,320 ms with the robust one, ≈ 850 / 480 b/s before sync and HARQ (reported from TS 26.267 excerpts).
  - The learned speech-shaped modem's 76–3,196 b/s Pareto range is a claim to test, not to build on (reported).
- **Tones and OFDM through Opus at high bitrates** (music mode/CELT) pass much better than through AMR-NB or noise-suppressed VoIP apps (inferred). Design profile D for the worst case.
- **Practical test:** record the modem through the actual phone path once and replay it offline (§6).

## 6. Channel model to standardise (Q6)

`x/math/dsp/channel` (new), composable stages, each seeded and deterministic:
1. level change and AGC emulation (an attack/release compressor);
2. sample-clock offset (resample by 1 ± ppm) and fractional start delay;
3. band-pass 300–3400 Hz (or a profile);
4. multipath: convolve with an impulse response, either synthetic (exponential decay, RT60 parameter) or **measured** with the repo's channel-sounding spec (`measure/SPEC.md`);
5. additive **Gaussian** noise at a specified Eb/N0. Fix the uniform "white" generator first (W7 TODO);
6. clipping;
7. optionally **replay of recorded real-channel captures** (phone, radio, room) instead of emulating codecs. No codec libraries are needed in the repo.

**"Noise figure" reporting:** Eb/N0 at BER 1e-3 and 1e-5, plus the acquisition threshold (detection probability ≥ 0.99 at the stated false-alarm rate). This reporting format is adopted.

## 7. Fixed-point budget on ESP32 (Q7)

At 16 kS/s, an 8-FSK Goertzel bank is ~256 kMAC/s. A 256-point FFT per OFDM symbol is (N/2)·log₂N = 1,024 radix-2 butterflies per 32 ms. The Viterbi decoder is ~128 k ACS/s. Total < 5 % of one 240 MHz core (inferred). **No fixed-point port is needed for budget reasons.** Use float32 on the S3's FPU, and Q15 only on chips without an FPU.

## 8. Soft and erasure information through byte interfaces (Q8)

Keep the existing byte coders, and add a parallel soft path:

```go
type SoftBits []int8           // LLR per bit: +127 = surely 0, −127 = surely 1, 0 = erasure
type SoftDemodulator interface { DemodulateSoft(samples []float32, dst SoftBits) (SoftBits, Quality) }
type SoftDecoder interface    { DecodeSoft(llr SoftBits, dst []byte) ([]byte, error) }
func HardToSoft(bits []byte) SoftBits   // adapter: ±127, so every hard demodulator plugs into soft decoders
```

- M-FSK LLRs come from the tone energies (log-ratio of the best bins).
- Erasures (a lost symbol or collision) are LLR 0, never deletions. W7 found that MFSK erasures become deletions today.
- This answers W7 Q8 without converting the whole coding layer to packed bits. Packed `buffer.Bits` can stay for the hard path.

## Smallest offline experiment (no hardware)

**`berharness_test` for the voice-band 8-FSK profile** (8 kS/s, 50 Bd, 600–3400 Hz):
- an up+down chirp preamble with CFAR detection;
- CRC-16 framing;
- conv K=7 soft Viterbi with a 16×32 interleaver.

Run through the §6 channel at:
- Eb/N0 2–14 dB (Gaussian);
- ±200 ppm clock offset;
- random start offset;
- −30 to 0 dBFS level;
- band-pass;
- a synthetic 30 ms RT60 room.

**Report:** BER and frame-error rate against Eb/N0 next to theoretical non-coherent 8-FSK; acquisition probability and false alarms per hour of pure noise.

**Done when:** acquisition ≥ 0.99 at Eb/N0 where coded BER ≤ 1e-4, and the result stays within 2 dB of theory + coding gain across all impairments. The synchroniser, NCO, channel package and soft path built here are reused by every other modem.

## Owner decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Which voice channel: **B speaker→mic in air** (recommended first), A line, C radio, D phone/VoIP codec (a separate research track at ~100 b/s) | B first; tell us if D is the real need |
| 2 | Baseline waveform **8-FSK + chirp preamble + K=7 soft Viterbi**; OFDM+LDPC as the rate tier | Yes |
| 3 | Add a **soft-bit path** (`[]int8` LLRs) beside the byte coders | Yes |
| 4 | Record real channel captures (phone, radio, room) once for offline replay, with hardware use approved per session | Yes |
