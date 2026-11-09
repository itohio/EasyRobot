---
type: algorithm
status: partial
tags: [easyrobot, audio]
workstream: W7
module: "[[audio-modem-cli]]"
source: [cmd/audio/mix.go]
tests: []
test_result: fail
complete: no
correct: no
evidence: "go build -o <scratch> ./cmd/audio; `audio mix -i psk.wav -i n.wav -o mixed.wav` → Error: failed to read input 0 (psk.wav): ... no devices specified for streaming (2026-11-09)"
---
# Audio mixer (`audio mix`)

**What it does (intended):** sum N WAV inputs with per-input time offsets into one WAV, block-wise (8192 frames), tracking per-input and output RMS.

**Where:** cmd/audio/mix.go — `runMix` :47-155, `parseOffsets` :157, `rmsAccumulator` :226-250, `mixAudio` :255-508.

**Inputs → outputs:** `-i a.wav -i b.wav [--offset 0:1.5] -o out.wav [--data-type]` → summed WAV. No per-input gain flag.

## Completeness review

- Complete: no — both the metadata probe (mix.go:76-80) and `mixAudio` (:272-276) request a `types.FrameStream` from a WAV-only unmarshaller; [[audio-io]] streams only from devices (x/marshaller/audio/unmarshaller.go:606-609), so every WAV input fails before mixing.
- Tested: no tests. Run: fails as above for any WAV input.
- Correct: no (unreachable). Consequence: no offline way to add a channel impairment to a modem WAV via `mix`; only `gen-*` `--noise` (added inside the generator) works.

## TODOs

- [ ] Read WAV inputs as tensors (or add WAV FrameStream to audio-io) so `mix` works offline — cmd/audio/mix.go:76-80,272-276 #p1 #ER-022
- [ ] Per-input gain (dB) flag to set SNR when mixing signal + noise files — cmd/audio/mix.go:36-39 #p2

## Links

- Module: [[audio-modem-cli]]; [[audio-io]]
