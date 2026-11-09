---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, dsp, modulation, audio-modem]
workstream: W7
packages: [x/math/dsp/modulation, x/math/dsp/modulation/mpsk, x/math/dsp/modulation/qam, x/math/dsp/modulation/mfsk, x/math/dsp/modulation/mask, x/math/dsp/modulation/dsss, x/math/dsp/modulation/iq, x/math/dsp/modulation/adaptive, x/math/dsp/modulation/tc, x/math/dsp/modulation/internal, x/math/dsp/modulation/test_utils]
test_result: fail
---
# modulation — digital modems for audio-band links

**Purpose:** streaming modulators and demodulators (M-PSK, square QAM, M-FSK, M-ASK, DSSS/Walsh, SMPTE LTC), plus I/Q mixing, an adaptive tier switch, a set of unused synchronisation building blocks, and a shared test harness. They exist to carry bytes over sound (cable, speaker→mic, voice channel). Front end: [[audio-modem-cli]] (`cmd/audio gen_*/recv_*`) and [[audio-io]]. Signal primitives (oscillators, Goertzel, pulse shapes, noise) come from [[dsp-core]]; line codes and FEC from [[line-coding-and-fec]].

**Packages** (non-test Go LOC, no build tags, pure Go, CGO-free):

| Package | LOC | Contents | Signal format |
|---|---|---|---|
| `modulation` (root) | 110 | `Receiver`/`Transmitter` interfaces, `QualityStats`, `TransmissionStats`, `TransmitterParams` (`transmitter.go`) | — |
| `mpsk` | 1497 | [[mpsk-modem]] | complex baseband, I/Q interleaved float32 |
| `qam` | 1392 | [[qam-modem]] | complex baseband, I/Q interleaved float32 |
| `mfsk` | 1039 | [[mfsk-goertzel-modem]] | real passband |
| `mask` | 1329 | [[mask-modem]] | real passband |
| `dsss` | 1043 | [[dsss-walsh-modem]] | real passband |
| `iq` | 530 | [[iq-mixer]] | real ↔ complex64 |
| `adaptive` | 98 | [[adaptive-modem-tier-selector]] | generic |
| `tc` | 844 | [[smpte-ltc-timecode]] | real bi-phase audio |
| `internal` | 1691 | [[gardner-timing-recovery]], [[costas-carrier-loop]], [[afc-frequency-tracker]], [[complex-agc]], [[symbol-matched-filter-integrator]], [[iq-imbalance-corrector]], [[complex-signal-primitives]], [[signal-power-tracker]] — only PowerTracker is used (by mask) | complex64 |
| `test_utils` | 423 | `Transmit` (padding + packets + "AWGN"), `Receive` (chunked streaming + inspection callbacks), `MakeData` | — |
| `ofdm`, `measure` | 0 | spec documents only: [[ofdm-modem]], [[channel-sounding-and-blind-estimation-specs]] | — |

Docs in the tree (untrusted claims): `THEORY.md` (982 lines), `TUNING.md` (729), `BLIND_CHANNEL_ESTIMATION.md`, per-package `SPEC.md`/`IMPLEMENTATION_PLAN.md`, `mpsk/THRESHOLD_TUNING.md`, `dsss/STATISTICS_OVERHEAD.md`, and a lecture PDF (`Lectures_Information_Theory_Coding_Barni_Tondi.pdf`, 8.8k lines extracted).

## Public API

- Every modem has a `Config` with `DefaultConfig()` and `Validate()` (which mutates in place, filling defaults), plus `NewModulator(cfg) (*Modulator, error)` and `NewDemodulator(cfg) (*Demodulator, error)`. DSSS has no `DefaultConfig`, and its caller must supply the spreading codes.
- Both sides are `chain.Processor` ([[pipeline-and-chain]]). Modulator: `Process(dst buffer.Writer[float32], src buffer.Reader[byte]) int`, where `src` **must** be a `buffer.BitsReader` whose `BitsLength()==BitsPerSymbol` (one symbol per byte); otherwise it returns 0 silently. `Process(nil, src)` returns the sample estimate. The demodulator mirrors this with a `buffer.BitsWriter` destination. `Reset()` on both.
- Demodulators implement `modulation.Receiver` (`Quality()`, `Transmission()`, `Params()`), and modulators implement `Transmitter`.
- Theory helpers on each `Config`: `MinimumEbN0`, `MinimumSNR`, `IsSNRSufficient`, `EnergyThresholdForSNR`, `RequiredBandwidth`, plus PLL bandwidth tables in mpsk/qam.
- Composition today: `bytes → buffer.Bits(k) → Modulator → []float32 → audio-io marshaller → WAV/device`, and back. Optional FEC/line coding is wrapped through `x/chain.Adapter` in `cmd/audio/common.go`, which currently crashes ([[audio-modem-cli]]).

## Algorithms

<!-- vault:algorithms h=48d2871a8be7 -->

Counts: 0 tested · 4 implemented · 14 partial · 2 stub · 0 unverified (20 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[adaptive-modem-tier-selector]] | implemented | pass | ok 4 |
| [[complex-agc]] | implemented | none | — |
| [[iq-imbalance-corrector]] | implemented | none | — |
| [[signal-power-tracker]] | implemented | fail | indirect |
| [[afc-frequency-tracker]] | partial | none | probe: 9277 Hz for a 100 Hz offset |
| [[complex-signal-primitives]] | partial | none | CFO corrector wrong (probe) |
| [[costas-carrier-loop]] | partial | none | — |
| [[dsss-walsh-modem]] | partial | fail | BER 0.2–0.48 even noiseless on long packets (LO drift after ≈0.4 s) |
| [[gardner-timing-recovery]] | partial | none | probe: 0 outputs |
| [[iq-mixer]] | partial | pass | tests only check the identity |
| [[mask-modem]] | partial | fail | 2–3 dB from theory; collapses with clock drift (50 ppm → BER 0.5) |
| [[mfsk-goertzel-modem]] | partial | flaky | 4-FSK at non-coherent theory; acquisition limit ≈4–6 dB; deletes symbols under stress |
| [[modem-energy-detector-and-preamble-acquisition]] | partial | fail | main failure source (timing, level, noise false-trigger) |
| [[modem-quality-estimators]] | partial | fail | cause of 11/13 failures |
| [[mpsk-modem]] | partial | fail | BPSK/QPSK within ≈1–1.5 dB of theory once acquired; acquisition fails ≤4–6 dB, at QPSK 10°… |
| [[qam-modem]] | partial | fail | 4-QAM ≈0.5–1 dB from theory; 16-QAM error floor ≈2e-3 at 15 dB (>5 dB loss); 90/180° ambiguity… |
| [[smpte-ltc-timecode]] | partial | build-fail | build-fail (stale tests); copy: ok 18 |
| [[symbol-matched-filter-integrator]] | partial | pass | qam integrate test ok |
| [[channel-sounding-and-blind-estimation-specs]] | stub | none | spec only |
| [[ofdm-modem]] | stub | none | spec only |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, float32 (`chewxy/math32`), CGO-free, no build tags, no goroutines, no reflection. float64 appears only in tc time arithmetic and a generic power switch. Imports `x/buffer`, `x/chain`, `x/math/dsp`, `x/math/filter/{agc,pll,fir}`, `x/math/vec`.
- TinyGo/embedded suitability [INFERENCE, not built]: plausible after cleanup. Steady-state demodulation mostly reuses FIFOs, but `readBuf` grows lazily. dsss allocates a 1-byte slice per symbol (`dsss/demodulator.go:203`), and `iq` and `internal/*` allocate per `Process` call. mask/qam keep `log.Printf` debug paths (7 and 12 call sites) that drag `log`/`fmt` into the binary. The fixed SPS-384 windows at 96 kHz cost 768 floats per I/Q symbol, fine for ESP32 RAM. Goertzel per tone per symbol is O(M·SPS).
- Sample-rate assumptions: defaults use 96 kHz with a 20 kHz carrier (mpsk/qam/mfsk) or 22 kHz/10 kHz (mask), which is above or at the edge of what phone/voice channels pass (300–3400 Hz) and what cheap MEMS mics capture. The CLI overrides carriers to 1–5 kHz.

## Tests and evidence

Command: `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -v ./x/math/dsp/modulation/...` (2026-11-09; log `tmp/easyrobot/audio/mod_v.txt`).

| Package | Result | Failing tests and what actually failed |
|---|---|---|
| adaptive | ok (4) | — |
| dsss | FAIL (1 pass / 3 fail) | NoiseScenarios (SNR estimate 7.2 vs 20–30 dB; TransmissionStart=0 at ≤15 dB), PacketSizes (TransmissionStart), ModulationOrders (SNR estimate 4.9–7.2 vs 20) |
| iq | ok (13) | — |
| mask | FAIL (4 / 2) | NoiseScenarios (SNR estimate 49.7 vs 40), Baseband (CarrierHz forced to 1000) |
| mfsk | FAIL (6 / 1), flaky | Test16FSK_Noise/SNR_25dB SignalStrength −11.99 vs −1.94 (tol 10). Baseline saw the 22 dB case fail 1/3 |
| mpsk | FAIL (9 / 3) | NoiseScenarios and SNRLimits (SNR estimate 3.2–4.1 vs 10–30), SingleSymbol_NoPreamble (LockCount 0) |
| qam | FAIL (9 / 4) | LongPackets (SNR 15.6–21.6 vs 30), Basics (SignalStrength −43 dB), NoiseScenarios and SNRLimits (SNR 1.5–2.5) |
| tc | build failed | stale mpsk API + removed tc.NewModulator in 3 test files; the other 18 tests pass on a scratch copy |

**Test harness noise model** (`test_utils/transmit.go:225-281`): the noise is `dsp.NewNoiseWhiteGenerator(sqrt(P/10^(S/10)))`, which is **uniform** in ±amp (variance amp²/3; see [[noise-generators]]). The real wideband SNR is therefore nominal + 4.77 dB, with bounded tails, which is much kinder than AWGN at low BER. SNR is wideband (noise over fs/2), so Eb/N0 = SNR + 10·log10(SPS/k) (complex) or SNR + 10·log10(SPS/2k) (real). That is +22.8 dB for QPSK and +19.8 dB for 4-FSK and 16-QAM at SPS 384. A test labelled "10 dB" runs QPSK at about **37.6 dB Eb/N0** (theory needs 9.6 dB for 1e-5). `snrDb <= 0` adds no noise at all (`:226`). Packets always start on a symbol boundary after 0.5 s of padding.

**BER probe** (`tmp/easyrobot/audio/berprobe/main.go`, output `ber.txt`, run 2026-11-09 with `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run .`). Gaussian noise; 1000 random symbols ×3 trials; alignment search ±16 symbols. Section A: noise only over the signal (decision path). B: noise over padding too, by wideband SNR. C: offsets, level, clock ppm. D: I/Q phase/CFO. Per-scheme tables are in the algorithm notes, and the W7 summary is in [[W7-filters-codecs-voice]].

## API notes (ease of use, composability)

- **Good:** a uniform `chain.Processor` shape for every modem; the `Process(nil, …)` estimate idiom lets callers pre-size buffers; Receiver/Transmitter interfaces give a common place for diagnostics; `Config.Validate()` fills defaults; the test harness's streaming and chunking are reusable.
- **Sharp edges:**
  1. Modulators silently return 0 when `src` isn't a `BitsReader` of exactly `BitsPerSymbol` bits. A wrong `buffer.NewBits(k)` therefore yields "no samples" with no error (the CLI hits this with `--encoding`: "estimate=0").
  2. Two incompatible sample representations: mpsk/qam use I/Q-interleaved `[]float32`, while iq/internal use `complex64`, and no adapter exists. PSK/QAM are baseband-only, and the CLI writes them as stereo L/R, unusable over a mono voice/acoustic channel.
  3. `Validate()` silently rewrites user values (mask `CarrierHz` 0→1000, preamble forced ≥20; mpsk loop bandwidth from a 25 dB table).
  4. DSSS's config shape differs (no `DefaultConfig`; codes and SamplesPerChip derived by hand).
  5. `internal` processors read from `dst` when it is also a Reader and ignore `src`, and some constructors panic.
  6. Thresholds tie detection to the configured `Amplitude`, so the receiver must know the transmit level.
  7. `Receiver.Quality()` numbers are not comparable across modems ([[modem-quality-estimators]]).
  8. No soft outputs, so FEC can only be hard-decision.
- **Composability with coding:** line codes and FEC ([[line-coding-and-fec]]) are separate `chain.Processor`s over bytes and bits, which is the right shape. But no modem emits erasures or LLRs, no framer exists (sync word + length + CRC), and the chain adapter used to splice them panics.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] One shared synchroniser (CFAR detection, preamble/chirp correlation, fractional timing, frame start/end) used by every demodulator — [[modem-energy-detector-and-preamble-acquisition]] #p1 #ER-022
- [ ] Phase-continuous NCO for every carrier (TX and RX); current float32-time oscillators drift — dsss/mask via x/math/dsp/generators_trig.go:240,258 #p1 #ER-022
- [ ] Fix the test harness: Gaussian noise, fractional start offsets, level changes, ppm drift, and payload checks before statistics — test_utils/transmit.go:225-281 #p1 #ER-022
- [ ] float32-interleaved ↔ complex64 adapter, plus a passband PSK/QAM path through [[iq-mixer]] — mpsk, qam, iq #p2
- [ ] Soft-decision outputs and erasure flags for FEC — all demodulators #p2
- [ ] Framer (sync word, length, CRC) as a chain processor between bytes and modem — new #p2
- [ ] Wire or delete internal/* (only PowerTracker is used); fix the CFO/AFC stale-trig and Gardner zero-output bugs first — internal/*.go #p2
- [ ] Unify theory helpers (MinimumEbN0/MinimumSNR formulas disagree across packages) — mpsk/config.go:244-273, qam/config.go:256-290, mask/config.go:183-216 #p2
- [ ] Remove log.Printf debug paths from hot loops (or put them behind a build tag) — mask/demodulator.go, qam/demodulator.go #p3

## Inferred objectives (proposals)

- **Proposal:** make 4-FSK/8-FSK (non-coherent, phase-agnostic, already at theory once acquired) the robust baseline voice/acoustic modem. Add a chirp preamble for timing and level, a framer with CRC, and an interleaved Hamming or RS FEC. Keep PSK/QAM as higher-rate tiers for clean cable links.
- **Proposal:** rebuild the receiver front end once (sync, AGC, timing, CFO) as shared, tested chain stages using the `internal` package's processor shape, then retire the per-modem inline copies.
- **Proposal:** replace the "SNR within ±8 dB" assertions with BER-vs-Eb/N0 curve tests against theory plus a margin, run on a channel model (Gaussian noise, delay, level, ppm, band-pass 300–3400 Hz, optional echo).
