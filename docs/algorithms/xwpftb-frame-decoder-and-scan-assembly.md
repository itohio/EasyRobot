---
type: algorithm
status: partial
tags: [easyrobot, devices, lidar, protocol]
workstream: W8
module: "[[lidar]]"
summary: "Test never starts reader, format mismatch, angles > 245.75° wrong"
source: [x/devices/lidar/xwpftb/device.go, x/devices/lidar/xwpftb/decoder.go, x/devices/lidar/xwpftb/DESIGN.md]
tests: [x/devices/lidar/xwpftb/decoder_test.go]
test_result: fail
complete: no
correct: unknown
evidence: "go test -count=1 ./x/devices/lidar/xwpftb → FAIL TestParseAndAssembleFullRotation (timeout), TestCRC16Cumulative pass (2026-11-09); probe tmp/easyrobot/devices/lidarprobe: test stream + Configure → 0 scans; 55AA-format frames → 240-point scan, angles > 245.75° wrong"
---
# XWPFTB frame decoder and scan assembly

**What it does:** Streams bytes from a `devices.Serial`, finds fixed 60-byte frames with header `0x55 0xAA`, type `0x23`, count `0x10`, checks a 16-bit additive checksum, spreads 16 points evenly between start and end angle, and emits a 2×N scan (row 0 mm, row 1 deg) after every 15 frames.

**Where:** `x/devices/lidar/xwpftb/device.go:165-212` (`readLoop`), `:221-314` (`consumeOneFrame`), `:316-~480` (`consumeMeasurement` + emit), checksum `decoder.go:8-14` (`crc16Cumulative`); test helper `BuildMeasurementFrame` `decoder.go:17-54`. Implements `lidar.Device`.

**Inputs → outputs:** `New(ctx, ser, motor|nil, targetPoints, maxPoints)`; host baud **250000** (`drivers/lidar/main.go:117`) but **115200** on RP2040 (`drivers/lidar/main_xwpftb.go:23`) → `OnRead` 2×(15·16=240) scans.

Frame as decoded (`device.go:317-324`): `55 AA | 23 | 10 | speed u16 | start | 16×(dist_L, dist_H, intensity) | end | sum16`. Angle: `((H&0x7F)<<8 | L) − 0x2000` in 0.01° (`:348-355`); distance `((H&0x3F)<<8 | L) · 0.1 mm` (`:407-408`). Comments cite a "README" formula that is not in the repo.

## Which device is this?

Unidentified. DESIGN.md:1-38 describes a **Delta-2G** (3irobotix vacuum lidar, header `0xAA`, variable length, command `0xAD`, 15 slices × 24°, distance ×0.25 mm), and `BuildMeasurementFrame` builds that format — but the decoder implements a different protocol (`55 AA 23 10`, 60-byte frames, 16 points, 0.1 mm). The code was evidently rewritten against a captured stream (heavy `slog` hex dumps, `device.go:178-189,237-306`; `cmd/monitor` exists for this) without updating DESIGN.md or tests. "XWPFTB" is likely a marking on the owner's module; the protocol family should be confirmed from a capture [INFERENCE].

## Completeness review

- Complete: **no.** No wrap/angle-based rotation detection (fixed 15-frame count, `device.go:432-433`) so the first scan after start/resync is misaligned; speed and intensity discarded; `Device Health` frames not handled.
- Tested: `TestCRC16Cumulative` (`decoder_test.go:29-37`) checks 1+2+3=6 — pass. `TestParseAndAssembleFullRotation` (`:39-75`) — **FAIL** "timeout waiting for scan". Two causes, each sufficient:
  1. The test **never calls `dev.Configure`** (`decoder_test.go:57-65`), so `readLoop` never starts (`device.go:145-163`).
  2. The test stream is Delta-2G (`0xAA …`, from `BuildMeasurementFrame`), which `consumeOneFrame` never matches (`device.go:228-250` looks for `55 AA`). Probe: same stream **with** `Configure` → 0 scans.
- Correct: **unknown / suspect.** Probe with frames in the decoder's own format (30 frames, starts 0..336° step 24°): one 240-point scan, distances right (1000 mm), angles exact up to 240°, then **wrong**: 264→296.32, 288→320.32, 312→344.32, 336→8.32. The `&0x7F` mask minus `0x2000` limits angles to −81.92°…245.75°, so either the real encoding differs (scale or offset) or the decoder is wrong — needs a capture. The `&0x3F` distance mask caps range at 1638.3 mm, implausibly short for a room lidar [INFERENCE] — same caveat.
- Other defects (read): `slog.Info` with a hex dump on **every serial read** (`device.go:186-189`) — heavy allocation/log spam on host and unusable on TinyGo; `stepDeg = span/16` (`:371`) puts the last point one step before the end angle (LD06 uses `/(n−1)`); same un-double-buffered scan/race and per-scan `View()` allocation as [[ld06-packet-decoder-and-scan-assembly]]; overflow resets silently (`:374-383`); package-level `min` shadows the Go 1.21 builtin (`:214-219`); motor loop duplicated verbatim from ld06 ([[lidar-motor-point-count-pid]]).

## TODOs

- [ ] Identify the module protocol from a real capture (`cmd/monitor -capture` at 115200/250000) and fix angle/distance scaling — x/devices/lidar/xwpftb/device.go:348-355,407-408 #p1 #ER-024
- [ ] Rewrite the test: call Configure, build frames in the decoder's 55AA format, cover angles > 245°, check values — x/devices/lidar/xwpftb/decoder_test.go:39-75; decoder.go:17-54 #p1 #ER-024
- [ ] Update DESIGN.md to the implemented protocol (or implement Delta-2G as a separate decoder) — x/devices/lidar/xwpftb/DESIGN.md #p2
- [ ] Detect rotation by angle wrap instead of a 15-frame counter — x/devices/lidar/xwpftb/device.go:432-433 #p2
- [ ] Reconcile baud rate 250000 vs 115200 — drivers/lidar/main.go:117; drivers/lidar/main_xwpftb.go:23 #p2
- [ ] Drop per-read Info/hex logging (Debug at most, no hex.EncodeToString in hot path) — x/devices/lidar/xwpftb/device.go:178-200 #p2
- [ ] Remove local `min` (builtin since Go 1.21) — x/devices/lidar/xwpftb/device.go:214-219 #p3

## Links

- Module: [[lidar]] · Workstream: [[W8-lidar]] · Sibling: [[ld06-packet-decoder-and-scan-assembly]] · Motor loop: [[lidar-motor-point-count-pid]] · Capture tool: [[apps]]
