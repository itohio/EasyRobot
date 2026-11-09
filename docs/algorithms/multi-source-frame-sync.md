---
type: algorithm
status: unverified
tags: [easyrobot, camera, stereo, sync, streams]
workstream: [W4, W5]
module: "[[camera-io]]"
summary: "unverified (goroutine leak, no time alignment)"
source: [x/marshaller/gocv/sync.go, x/marshaller/v4l/stream.go, x/marshaller/v4l/unmarshaller.go]
tests: [x/marshaller/gocv/streams_test.go]
test_result: not-run
complete: no
correct: unknown
evidence: "gocv and v4l packages cannot build on this host (OpenCV missing; go4vl needs linux cgo) (2026-11-09); code read"
---
# Multi-source frame stream synchronisation (gocv and V4L2)

**What it does:** merges N frame sources into one `types.FrameStream`. *Sequential* mode drains source 0, then 1, …; *parallel* mode reads one frame from every source per tick and emits one combined frame (`Tensors` concatenated, per-source metadata under `sources`). Intended for stereo pairs and multi-camera rigs.

**Where:** gocv: `newFrameStream` `sync.go:21-103` (sequential :44-69, lock-step :70-102), `combineFrameItems` :105-145, `errorFrame` :147; V4L2: `MultiStream` `v4l/stream.go:264-530` (`syncSequential` :373, `syncParallel` :409), `multiStreamToFrameStream` `v4l/unmarshaller.go:259-304`.

## Completeness review

- Complete: no.
  - "Sync" is lock-step reading, not time alignment: sources are read one after another in the same goroutine (`sync.go:79-89`), so stereo frames are skewed by one capture latency, and the combined `Timestamp` is `time.Now()` at merge time (`sync.go:106`) — per-source timestamps stay in metadata only. No nearest-timestamp matching or drop policy (the deprecated `x/pipeline/steps/sync.go` had one).
  - Producer goroutine sends on an unbuffered channel without `select` on `ctx` (`sync.go:57, 66, 86, 101`): if the consumer calls `Close()` and stops reading, `done()` cancels the context and closes sources, but the goroutine stays blocked on the send forever (leak).
  - Errors are delivered as frames with `Metadata["error"]` and no tensors (`sync.go:147-154`) — consumers must check every frame.
- Tested: not run (no OpenCV). `streams_test.go` covers multi-glob sequential streams only.
- Correct: unknown.

## TODOs

- [ ] `select` on `ctx.Done()` for every send — x/marshaller/gocv/sync.go:57,66,86,101 #p1 #backlog
- [ ] Timestamp-based pairing (max skew, drop oldest) for stereo — x/marshaller/gocv/sync.go:70-102 #p2
- [ ] Read sources concurrently (grab all, then retrieve) to minimise skew — x/marshaller/gocv/sync.go:79-89 #p2

## Links

- Module: [[camera-io]]; [[gocv-frame-sources]]; [[v4l2-capture-stream]]; stereo consumer [[stereo-sgbm-disparity]]
