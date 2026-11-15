---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, audio, modem, cli]
workstream: W7
source: [cmd/audio/]
tests: []
test_result: none
evidence: "go build -o <scratch>/audiobin ./cmd/audio (GOFLAGS=-mod=readonly) → ok; offline gen→recv runs 2026-11-09 under  (table below)"
---
# audio-modem-cli — `cmd/audio`

Cobra CLI (6,091 LOC, 22 .go files, no tests) for audio generation, modem TX/RX over WAV or sound card, and device I/O. Built on [[audio-io]] (WAV/malgo), [[modulation]] (mpsk, mfsk, qam, mask, dsss, tc), [[dsp-core]] generators, [[line-coding-and-fec]] (gray, hamming, rs, manchester) composed with `x/chain` adapters ([[pipeline-and-chain]]). Context: [[W7-filters-codecs-voice]].

Subcommand names use **hyphens** (`gen-psk`, not `gen_psk`).

## Subcommands

| Command | File | Package / config | Key flags | I/O |
|---|---|---|---|---|
| list | list.go | audio-io device enumeration | — | **device** (malgo) — not run |
| record | record.go | audio-io capture (+monitor) | `-o`, `-i/--device-id`, `-r`, `-c`, `-f` | **device** → WAV — not run |
| play | play.go | audio-io WAV → playback | `-i`, `-o/--device-id`, `-c` | WAV → **device** — not run |
| gen-sine | gen_sine.go | dsp sine gens | `-f` list, `-p`, `-c`, `-d`, `-a` dB | WAV |
| gen-noise | gen_noise.go | dsp white/pink/brown | `-t`, `-a` list, `-c`, `-d` | WAV |
| gen-chirp | gen_chirp.go | dsp chirp + ref tone | `--chirp-type`, `--ref-*`, `--delay`, `--signal-channel` | WAV → [[chirp-sync-receiver]] |
| gen-psk / recv-psk | gen_psk.go, recv_psk.go | `mpsk.DefaultConfig` + SampleRate/CarrierHz(5 kHz)/BitsPerSymbol/SamplesPerSymbol(rate/1000) | `-m 2..16`, `--symbol-rate`, `--encoding/--decoding none,gray,hamming,rs`, `--pad`, `--noise`, `--noise-type` | 2-ch "IQ interleaved" WAV or `--device-id` |
| gen-fsk / recv-fsk | gen_fsk.go, recv_fsk.go | `mfsk`, freqs `1000,2000`; gen adds sync bytes AA AA … 55 55 (gen_fsk.go:117-119) | `-f`, `-m`, `--encoding` | mono WAV |
| gen-qam / recv-qam | gen_qam.go, recv_qam.go | `qam`, `-m` overrides `-n/--poles` (gen_qam.go:72-78) | `-m 4..256`, `--encoding` | 2-ch WAV; no padding |
| gen-mask / recv-mask | gen_mask.go, recv_mask.go | `mask` + own line-code/ECC chain (gray/manchester, hamming/rs) | `--modulation-order 2..256`, `--line-code`, `--ecc` | mono WAV or device |
| gen-dsss / recv-dsss | gen_dsss.go, recv_dsss.go | `dsss.Config` raised-cosine, rolloff .35, codes `--code`/`--gen-code` | `-c`, `--spreading-factor`, defaults 96 kHz, 10 kHz carrier | mono WAV (device playback TODO, gen_dsss.go:560) |
| gen-tc / recv-tc | gen_tc.go, recv_tc.go | `tc` LTC | `--frame-rate`, `--timecode` | mono WAV (device playback TODO, gen_tc.go:217) |
| mix | mix.go | block mixer → [[audio-mixer]] | `-i` (repeat), `--offset`, `-o` | WAV (broken) |

There is no separate `noise` command; noise comes from `gen-noise` or from `--noise`/`--pad-type noise` inside `gen-psk/mask/dsss` (common.go:104-178).

### Device safety (checked before running)
`recv-*` with `-i` builds `audio.NewUnmarshaller(audio.WithWAVPath(input))` (e.g. recv_psk.go:106-108; same pattern recv_fsk:151, recv_qam:117, recv_mask:118, recv_dsss:154, recv_chirp:117, recv_tc:88). `demodulateAudio` first asks for a FrameStream, which errors immediately without devices (x/marshaller/audio/unmarshaller.go:606-609), then reads the tensor. `gen-*` with `-o` and no `--device-id` sets only `WithWAVPath`. malgo is initialised only inside `newAudioDevice` (device_malgo.go:62-67). So the offline path opens no device.

## Offline roundtrip results (payload `Hello` = 48 65 6c 6c 6f, 2026-11-09)

Binary built with `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go build -o …/AudioIONotes/audiobin ./cmd/audio`. "One-chunk" = probe copy of cmd/audio in `…/AudioIONotes/onechunk/` with only `defaultChunkParts = 8 → 1` (common.go:190), to isolate the CLI's output-packing bug from the demodulators. Binaries deleted afterwards.

| Scheme / options | Command (gen → recv) | Stock CLI output | One-chunk output | Verdict |
|---|---|---|---|---|
| QPSK (default) | `gen-psk -o psk.wav` → `recv-psk -i psk.wav` | `48656c6c6f00…` | `48656c6c6f00…` | **roundtrip OK** (+ trailing zeros from padding) |
| BPSK `-m 2` | same, `-m 2` both | `H@` + `95b1b1bc00…` (bit-misaligned) | `48656c6c6f00…` | demod OK; CLI packing breaks it |
| 8-PSK `-m 8` | `-m 8` both | `48656c6c6f00…` | `48656c6c6f00…` | **OK** |
| PSK + `--noise -20` (white) | gen `--noise -20` | random hex | random hex | **fail** at ~14 dB nominal SNR |
| PSK + `--noise -10`, pink -20 | | nothing decoded | — | **fail** |
| PSK `--encoding hamming/gray/rs` | gen `--encoding hamming` | gen error: `modulator produced no samples (estimate=0)` | — | **not runnable** |
| PSK `--decoding rs` | `recv-psk -i psk.wav --decoding rs` | **panic** nil deref x/chain/adaper.go:111 via common.go:315 | — | **crash** (#p1) |
| 4-FSK (default) | `gen-fsk -o fsk.wav` → `recv-fsk -i fsk.wav` | `a8`,`aa90`,`8650`,`ll`,`1bd0`,`UT` | `aaaa48656c6c6f5555` | demod OK incl. sync bytes; CLI packing breaks it |
| 2-FSK, 4-tone 1–4 kHz, 100 baud | | garbled | `aaaa48656c6c6f5555` | same |
| QAM-4 / QAM-16 | `gen-qam -n 4 -m 4` → `recv-qam --poles 4 -m 4` | `00…08` | `0000000002` / `0000000000` | **fail** (demod) |
| 4-MASK | `gen-mask -o mask.wav` → `recv-mask -i mask.wav -f 5000` (recv default carrier is 1000 Hz, gen 5000) | `48656c6c6f0000000000` | same | **OK** (needs explicit `-f`) |
| 2-MASK | `--modulation-order 2` both | `H@`,`95b1b1bc…` | `48656c6c6f000000` | demod OK; CLI packing |
| MASK + `--noise -20` | | random hex | random hex | **fail** |
| MASK `--line-code manchester` / `--ecc hamming` | both sides | empty `Decoded:` | empty | **fail** |
| DSSS (default / explicit codes) | `gen-dsss -o dsss.wav [-c FFFF -c 5555 -c 3333 -c 9999]` | gen prints `MAX amplitude: 0.000000`; recv decodes nothing | — | **fail**: TX writes silence |
| LTC timecode | `gen-tc -o tc.wav --timecode 01:02:03:04` → `recv-tc -i tc.wav` | `Timecode: 01:02:03:04` (frame rate printed 0.00) | — | **OK** |
| Chirp | `gen-chirp -d 1s --signal-channel 0 --chirp-type log` → `recv-chirp … --channel 0` | ref found at 48000–60000; correlation WAV saturated | — | partial, see [[chirp-sync-receiver]] |
| mix | `gen-noise -o n.wav -c 2 -a -20 -d 3s`; `mix -i psk.wav -i n.wav -o mixed.wav` | `failed to read input 0 … no devices specified for streaming` | — | **fail**, see [[audio-mixer]] |

Summary: clean-channel roundtrips succeed for PSK (2/4/8), FSK (2/4), MASK (2/4), LTC — but only QPSK, 8-PSK and 4-MASK decode cleanly through the stock CLI. Nothing survives added noise; QAM, DSSS, MASK line-code/ECC and every `--encoding/--decoding` path fail.

## Defects found

1. **CLI output packing** — `processSampleBuffer` splits input into 8 chunks and `flushDecodedSymbols` emits after each chunk; `symbolsToBytes` pads the trailing partial byte with zeros (common.go:304-318, 356-360, 430-432). Any chunk boundary not on a byte boundary corrupts all later output. Breaks BPSK, 2-MASK, all FSK in stock CLI. #p1
2. **`--encoding/--decoding` broken** — `createEncodedModulator`/`createDecodedDemodulator` build `chain.NewAdapter[…, *buffer.Values[byte]]` (common.go:502, 542); the adapter's pointer buffer is never allocated → nil deref at x/chain/adaper.go:111 (and :138 on Reset; peer IOVision probe). Gen side fails earlier with "estimate=0" because `Process(nil, …)` size estimation via the adapter returns 0. FEC/CRC is therefore **not usable** in any gen/recv pair except MASK's own chain (which decodes empty). #p1
3. **gen-dsss writes silence** — `buffer.New(vec.New(expectedSize))` appends after the zero-filled length and the code then slices the original zeros (gen_dsss.go:396-405); also `dataBits.Write(dataBytes)` writes whole bytes into a `bitsPerSymbol`-wide Bits buffer, masking each byte to 2 bits (gen_dsss.go:383-387; compare `bytesToSymbols` in common.go:440). #p1
4. **Un-normalised samples reach demodulators** — gen writes FP32 as int32 PCM; recv converts with `CloneTensorDataTo(FP32, …)` without scaling (common.go:278), so demods see ±1e9 values ("Signal Strength 176 dB"). Amplitude thresholds (MASK, detectors, chirp) are meaningless. Root cause in [[audio-io]] (ignored `WithDestinationType`). #p1
5. **"White" noise is uniform** — `applyNoise`/`applyPadding` use `dsp.NewNoiseWhiteGenerator` (common.go:125,156), which is uniform `amp*(2r-1)` (x/math/dsp/generators.go:226-271), variance amp²/3; `--noise` dB is a peak, not RMS, value. Noise added sample-wise then clamped to [-1,1] (common.go:169-174) although signal is later scaled to int32. #p2
6. **Default mismatches gen vs recv** — mask carrier 5000 vs 1000 Hz (gen_mask.go:59, recv_mask.go:51); dsss symbol rate 500 vs 1000 (gen_dsss.go:179, recv_dsss.go:46) and recv requires `--code` while gen generates them; chirp channels (see note). #p2
7. **`--data-type` ignored** on all gen-* (header shows 32-bit PCM for `INT16` request) — [[audio-io]] TODO. #p2
8. **PSK/QAM write 2-channel "IQ interleaved" WAV** (gen_psk.go:172-208, recv_psk.go:151 expects 2 ch): not a real passband audio signal that would survive a speaker→mic path [INFERENCE from code/print text; not checked in mpsk]. #p2
9. **mix unusable on WAV** — [[audio-mixer]]. #p1

## Pipeline gaps (what a robust voice-band data link still needs)

- **Framing:** none for PSK/QAM/MASK/DSSS — padding is demodulated as data (trailing `00…`), no preamble/sync word, length or end marker. Only FSK prepends AA AA/55 55 (gen_fsk.go:117-119) and recv does not search for it.
- **CRC / FEC:** `x/math/dsp/coding` is imported (gray, hamming, rs, manchester) but the adapter path crashes (defect 2); no CRC anywhere.
- **AGC / level normalisation:** none in the CLI; receivers fed raw int32-scale samples.
- **Timing recovery / carrier sync:** relies on whatever the [[modulation]] demodulators do; the CLI does not test offsets — every roundtrip here is sample-aligned and same-clock. Noise at -20 dB already breaks PSK/MASK, suggesting lock happens on leading padding/noise [INFERENCE].
- **Channel model:** only additive uniform/pink/brown noise inside gen. Missing: sample-rate offset / clock drift (no resampler on read, [[audio-naive-resampling]] is playback-only and aliasing), band-limiting (speaker/mic, telephone 300–3400 Hz), reverb/multipath (could reuse chirp-measured IR), clipping, DC offset, delay. `mix` (the natural place) is broken.
- **BER/SNR measurement:** no tool compares decoded vs sent bits; receivers print "Final SNR" from the demod only.

## Algorithms

<!-- vault:algorithms h=680485ae768e -->

Counts: 0 tested · 0 implemented · 2 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[audio-mixer]] | partial | fail |  |
| [[chirp-sync-receiver]] | partial | none | `gen-chirp` writes silence → reference tone (default 4400 Hz, 250 ms, ref channel) → delay → chirp… |

<!-- /vault:algorithms -->

## Build targets

- Native only: imports [[audio-io]] → malgo/CGO; cobra. `go build -o … ./cmd/audio` ok (slice owner and this run). Not a TinyGo/embedded target.
- Allocations: whole-file tensors, `vec.NewFrom` copies per stage, `[]byte` per flush; fine for a desktop tool.

## API notes (ease of use, composability)

- Every gen/recv repeats ~150 lines of flag parsing, config building and printing; only PSK/FSK/QAM share `createEncodedModulator`/`demodulateAudio`; MASK and DSSS have private pipelines (gen_mask.go, gen_dsss.go:355-405) with different symbol-packing logic — the source of defect 3. A single `scheme` registry (`name → (modCfg, demodCfg, bitsPerSymbol, channels)`) with shared framing/coding/noise stages would remove the default mismatches.
- Flags differ across schemes for the same concept (`--encoding` vs `--line-code/--ecc`; `-n/--poles` vs `-m`; `-c` = channels in gen-noise but `--code` in dsss).
- `demodulateAudio` streaming design (Bits buffer + per-chunk flush) is the right shape for live capture, but must carry a bit-level remainder between flushes instead of byte-padding.
- No machine-readable output (exit code/JSON for decoded payload), so roundtrip checks need text scraping; adding `--expect <hex>` / BER report would make the CLI a regression harness.

## CHIRP_PLAN.md / CONVERSION_ANALYSIS.md vs code

- CHIRP_PLAN Part 1 (ref tone + delay + silences in gen-chirp): implemented (flags at gen_chirp.go:48-64). Part 2: Goertzel sliding detection, cut, correlate, write WAV — implemented; separate `--ref-channel/--signal-channel` for recv not implemented; chunked collection only for devices; "ref amplitude too low: warn" and multi-ref handling absent; plan defaults (ref 100 ms, delay 0, linear) differ from code (250 ms, 1 s, pink-log).
- CONVERSION_ANALYSIS: duplicated scaling and magic numbers — confirmed (see [[audio-sample-format-conversion]]); "no dithering" — confirmed; proposed `x/types/st/convert.go` refactor — not done (no such use in cmd/audio).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Carry partial-byte bits across flushes instead of zero-padding each chunk — cmd/audio/common.go:304-318,356-360,430-432 #p1 #ER-022
- [ ] gen-dsss: use `buffer.New(make([]float32,0,n))` + `Items()` and `bytesToSymbols` — cmd/audio/gen_dsss.go:383-405 #p1 #ER-022
- [ ] Normalise received samples to [-1,1] before demod (Unmarshal with `WithDestinationType(FP32)` or scale in `processTensorInput`) — cmd/audio/common.go:278 #p1 #ER-022
- [ ] Gaussian AWGN option with RMS-referenced SNR for `--noise` — cmd/audio/common.go:142-178, x/math/dsp/generators.go:226-271 #p1 #ER-022
- [ ] QAM roundtrip fails on clean channel — investigate qam demod/CLI config — cmd/audio/gen_qam.go:63-182, recv_qam.go #p1 #ER-022
- [ ] MASK manchester/hamming chains decode empty — cmd/audio/recv_mask.go:130-232 #p1 #ER-022
- [ ] Align gen/recv defaults (mask carrier, dsss symbol rate/codes) — recv_mask.go:51, recv_dsss.go:46-49 #p2
- [ ] Framing: preamble + sync word + length + CRC shared by all schemes #p2
- [ ] Channel-model stage: resampling offset/drift, band-pass, reverb IR, clipping — new, uses [[filters]] #p2
- [ ] BER/`--expect` reporting and golden roundtrip tests (offline, WAV in temp dir) for each scheme — cmd/audio #p2
- [ ] Device playback for gen-dsss / gen-tc — gen_dsss.go:560, gen_tc.go:217 #p3
- [ ] recv-tc prints frame rate 0.00 — recv_tc.go #p3

## Inferred objectives (proposal)

- **Proposal:** turn `cmd/audio` into a thin shell over a library package (e.g. `x/math/dsp/modem`) that composes `framing → FEC/CRC → line code → modulator` and the inverse, with a pluggable channel model; the CLI and Go tests would then share one roundtrip harness that reports BER vs Eb/N0 using true Gaussian noise.
- **Proposal:** prioritise real-passband mono schemes (MFSK, MASK/OOK, BPSK on a carrier) for speaker→mic links over 2-channel IQ WAVs, since only those survive actual audio hardware.
