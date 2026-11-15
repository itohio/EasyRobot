---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, audio, marshaller, dsp]
workstream: W7
source: [x/marshaller/audio/]
tests: [x/marshaller/audio/marshaller_test.go, x/marshaller/audio/unmarshaller_test.go]
test_result: pass
evidence: "go test -count=1 -v ./x/marshaller/audio/ → ok, 21 PASS, 0 SKIP (2026-11-09)"
---
# audio-io — `x/marshaller/audio`

Audio marshaller/unmarshaller implementing `x/marshaller/types` (`Marshaller`/`Unmarshaller`) over two back ends: **WAV files** (go-audio/wav, pure Go) and **audio devices** (gen2brain/malgo = miniaudio via CGO). Value type is a 2-D tensor `[frames, channels]` ([[tensor]]) or a `types.FrameStream` (devices only). Used by every subcommand of [[audio-modem-cli]]. See also [[marshallers]], [[W7-filters-codecs-voice]].

## Package (7,022 LOC non-test, 1,005 LOC tests)

| File | LOC | Role |
|---|---|---|
| marshaller.go | 981 | `NewMarshaller`, `Marshal` (Tensor / FrameStream), WAV write, device playback (single/multi/mixed) |
| unmarshaller.go | 1446 | `NewUnmarshaller`, `Unmarshal` into `*Tensor` / `*FrameStream` / `*[]AudioDeviceInfo`; WAV read, multi-WAV channel concat, device capture/stream |
| device_malgo.go | 957 | `//go:build !wasm`; malgo context/device wrapper, callbacks, converter goroutines |
| sample_ring.go | 1709 | typed `sampleRing` + `byteRing` → [[audio-sample-ring-buffer]] |
| tensor_convert.go | 484 | dtype conversion + sample-rate matching → [[audio-sample-format-conversion]], [[audio-naive-resampling]] |
| device_convert.go | 518 | PCM bytes ↔ typed samples, ring↔ring → [[audio-sample-format-conversion]] |
| converter.go | 233 | goroutines pumping bytes chan ↔ sampleRing |
| options.go / config.go | 183 / 91 | `WithWAVPath(s)`, `WithSampleRate`, `WithChannels`, `WithFormat`, `WithAudioDevice`, `WithCaptureDevice`, `WithPlaybackDevice`, `WithFrameSize`, `WithBufferSize`, `WithBestEffort` |
| device.go, device_copy.go, device_helpers.go, tensor_chunk.go, tensor_extract.go, validation.go, types.go, pools.go | small | glue: channel extraction, chunking, validation |

Build tags: only `device_malgo.go` has `!wasm`; `device.go:11-13` calls `newMalgoDevice` with no tag, so a wasm build would not link [INFERENCE, not built].

## How it composes

- Source/sink chosen by options, not by type: `writeTensor` (marshaller.go:131-176) → device IDs present ? playback (or mixed WAV+device) : WAV path(s) ? `writeSingleWAV` : error. `unmarshalTensor` (unmarshaller.go:157-183) mirrors this. Multiple WAV *outputs* are rejected (marshaller.go:175); multiple WAV *inputs* are channel-concatenated (`readMultipleWAV`, unmarshaller.go:428).
- `Unmarshal(&FrameStream)` works only with devices (`unmarshalStream`, unmarshaller.go:606-609 returns "no devices specified for streaming" otherwise). WAV input is tensor-only.
- Constructors do **not** touch malgo; `initMalgoContext` (device_malgo.go:62-67) runs only in `newAudioDevice`. WAV-only use is device-free (verified by reading paths and by running [[audio-modem-cli]] offline).
- Device path: malgo callback → byte chan → `converter` goroutine → `sampleRing` → `ReadFrames` into tensor (and reverse for playback). Goroutines + `workers.WorkerPool` per instance.

## Algorithms

<!-- vault:algorithms h=64b4dfb30ffc -->

Counts: 1 tested · 1 implemented · 1 partial · 0 stub · 1 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[wav-codec]] | tested | pass | 21 pass (value checks, loose) |
| [[audio-sample-format-conversion]] | implemented | none | none direct |
| [[audio-naive-resampling]] | partial | none | changes a tensor's sample rate before device playback when WAV rate ≠ device rate. |
| [[audio-sample-ring-buffer]] | unverified | none | none (device-only) |

<!-- /vault:algorithms -->

## Build targets

- CGO required for malgo (miniaudio C) on native targets; WAV path itself is pure Go but lives in the same package, so importing the package pulls malgo/CGO.
- TinyGo/embedded: unsuitable — CGO, go-audio, goroutines + sync, `any` type switches, per-call tensor allocation, `[]int` intermediate buffers (WAV read appends whole file into `[]int`, unmarshaller.go:257-287).
- Samples are float32 in tensors; WAV write goes via `int` per frame through `audio.IntBuffer` (one encoder call per frame, marshaller.go:243-257) — slow for long files.

## Tests / evidence

- 21 tests, all PASS, none skipped (dsp_v.txt). Testdata WAVs in `testdata/` are **non-empty** (110–880 KB, 2026-11-09 `ls`); the baseline claim "0 bytes" is stale for this checkout.
- Marshaller tests (marshaller_test.go:32-486): write tensor to `t.TempDir()` WAV, read back; assert shape/rate/channels and ≥80 of first 100 samples within tolerance (INT16 quantization, 1% for FP32). FP32 test explicitly accepts reading back INT32 (marshaller_test.go:243).
- Unmarshaller tests: read testdata files, assert shape/dtype; Goertzel at 1 kHz amplitude >0.1 and frequency within 1% (unmarshaller_test.go:410-420); phase-shift file comparison; sample-rate/channel mismatch errors; invalid WAV error.
- Not tested: devices (by design, needs hardware), FrameStream, ring buffer, dtype conversion helpers, resampling, multi-WAV concat, mixed outputs.

## API notes (ease of use, composability)

- `types.WithDestinationType` is silently ignored by `Marshal`: `localOpts` is discarded (marshaller.go:83-84) and bit depth comes from the tensor dtype (marshaller.go:188-201). Every `gen-*` passes `--data-type`, yet output is always the tensor's dtype — FP32 tensors become **32-bit integer PCM** (format 1, verified header `0100 0200 80bb0000 … 2000` on psk.wav). Readers then get INT32 back, not FP32.
- Asymmetric scaling: writer maps float→int with ×32767 / ×2147483647 (marshaller.go:319,327) but reader divides by 32768 / 2147483648 (unmarshaller.go:394-401); `tensor_convert.go:111,135` uses 2147483647/32767. Three different conventions.
- Reading an integer WAV gives an integer tensor; downstream `CloneTensorDataTo(FP32, …)` in cmd/audio casts without normalising, so demodulators see ±1e9 values (recv reports "Signal Strength 176 dB"). The package offers no "give me normalised float32" option; `WithDestinationType(FP32)` on Unmarshal does normalise (`fillTensorFromPCM`, unmarshaller.go:369-401) but cmd/audio does not use it.
- Stream vs tensor split is source-dependent (WAV = tensor only, device = both), so callers write try-stream-then-tensor fallbacks (cmd/audio/common.go:221-233) and `mix` breaks (see [[audio-mixer]]). A WAV FrameStream reader would make files and devices interchangeable for streaming DSP.
- No resampling on read (error on mismatch, unmarshaller.go:242-246) but naive resampling on device playback — inconsistent.
- Option functions are clean and composable; device/WAV mixing via repeated options is a nice touch. No `io.Reader`/`io.Writer` WAV support (the `w`/`r` args are ignored for WAV), so in-memory pipelines must go through temp files.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] WAV FrameStream source so streaming DSP works on files — x/marshaller/audio/unmarshaller.go:606-609 #p2
- [ ] Tests for ring buffer, dtype conversion, resampling (no hardware needed) — x/marshaller/audio/sample_ring.go, tensor_convert.go #p2
- [ ] Use the `io.Writer`/`io.Reader` arguments for WAV (in-memory pipelines) — marshaller.go:78, unmarshaller.go:80 #p3
- [ ] wasm build: `device.go` references `newMalgoDevice` without a stub for wasm — x/marshaller/audio/device.go:11-13 #p3
- [ ] Remaining TODO comments: `sample_ring.go:496` (slow conversion path), `tensor_convert.go:284` (anti-aliasing) #p3

## Inferred objectives (proposal)

- **Proposal:** split into `audio/wav` (pure Go, TinyGo-friendly, io.Reader/Writer) and `audio/device` (CGO malgo), so modem code and tests can depend on WAV without CGO.
- **Proposal:** make float32 normalised [-1,1] the canonical exchange format at the DSP boundary (one option on Unmarshal, honoured dtype on Marshal), which removes the scaling bugs seen in [[audio-modem-cli]].
