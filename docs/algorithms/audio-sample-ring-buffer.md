---
type: algorithm
status: unverified
tags: [easyrobot, audio, buffer]
workstream: W7
module: "[[audio-io]]"
summary: "none (device-only)"
source: [x/marshaller/audio/sample_ring.go, x/marshaller/audio/converter.go]
tests: []
test_result: none
complete: unknown
correct: unknown
evidence: "code reading only; used only by malgo device paths (device_malgo.go:444-471), which need audio hardware (not run)"
---
# Audio sample ring buffer (`sampleRing` / `byteRing`)

**What it does:** blocking, context-cancellable FIFO between the malgo device callback and tensor consumers/producers. `sampleRing` holds typed samples (one `buffer.Ring[T]` behind `any`); `byteRing` holds raw device PCM.

**Where:** x/marshaller/audio/sample_ring.go — `newSampleRing` :39-74, `Write`/`Read` :101-162, `ReadFrames` :219-322 (+ `readFramesWithConversion` :325), `WriteFrames` :408-516 (+ conversion :518), `WriteSamples`/`ReadSamples` and per-dtype `*Blocking`/`*WithBuffer` variants :592-1494, `byteRing` :1496-1709. Pumps: `converter`/`reverseConverter` goroutines (converter.go:11-233). Created in `newMalgoDevice` (device_malgo.go:444-471).

**Inputs → outputs:** tensor `[frames,ch]` or PCM `[]byte` in; same out. Blocking: waits on `notEmpty`/`notFull` 1-slot signal channels; returns `ctx.Err()` on cancel.

## Completeness review

- Complete: unknown — 1,709 lines, mostly 4× duplicated per-dtype copies (int16/int32/float32/uint8) of the same read/write logic; the generic `buffer.Ring[T]` underneath could carry this with one generic function. `TODO: optimize this path` at :496 (dtype-converting write).
- Tested: no tests in package; the ring is reachable only through real devices.
- Correct: unknown. Reading notes: `ReadFrames` locks `sr.mu` around ring ops and signals `notFull` (:261-318); `Write`/`Read` (:101-162) themselves do not lock, so correctness relies on callers taking `sr.mu` (20 `sr.mu.Lock` sites). `readFramesWithConversion` ignores its `offset` argument (:325-329) — harmless today because the slow path is only entered on the first iteration (dtype is fixed), but fragile. Single-slot signalling with spurious wakeups is fine because loops recheck `Len()`.

## TODOs

- [ ] Unit tests (producer/consumer goroutines, cancel, wrap-around) — no hardware needed — x/marshaller/audio/sample_ring.go:219,408 #p2
- [ ] Collapse per-dtype duplicates into generics over `buffer.Ring[T]` — sample_ring.go:592-1494 #p3
- [ ] Remove unused `offset` param or honour it — sample_ring.go:325 #p3

## Links

- Module: [[audio-io]]; [[audio-sample-format-conversion]]
