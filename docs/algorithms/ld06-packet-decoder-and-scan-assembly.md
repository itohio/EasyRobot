---
type: algorithm
status: partial
tags: [easyrobot, devices, lidar, protocol]
workstream: W8
module: "[[lidar]]"
summary: "5 defects (VerLen, speed units, int16 angles, wrap test, test stream); never emits a scan"
source: [x/devices/lidar/ld06/device.go, x/devices/lidar/ld06/decoder.go, x/devices/lidar/ld06/DESIGN.md]
tests: [x/devices/lidar/ld06/decoder_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/devices/lidar/ld06 → FAIL TestParseAndAssembleFullRotation (timeout waiting for scan), TestCRC8 pass (2026-11-09); probe → 0 scans for every synthetic stream incl. datasheet-format packets"
---
# LD06 packet decoder and scan assembly

**What it does:** Streams bytes from a `devices.Serial`, finds LDROBOT LD06 measurement packets (header `0x54`), checks CRC-8, linearly interpolates an angle for each point between the packet's start and end angle, accumulates points into a preallocated 2×N matrix (row 0 distance mm, row 1 angle deg) and emits a scan via `OnRead` callback when the angle wraps.

**Where:** `x/devices/lidar/ld06/device.go:166-193` (`readLoop`), `:195-333` (`consumeOnePacket`), `:343-375` (`emitScan`), `:335-341` (`normalizeAngle`); CRC table `decoder.go:10-37`, `crc8` `:31-37`; test helper `BuildMeasurementPacket` `decoder.go:41-96`. Implements `lidar.Device` (`x/devices/lidar/lidar.go:9-36`).

**Inputs → outputs:** `New(ctx, ser devices.Serial, motor devices.PWM|nil, targetPoints, maxPoints int)`; byte stream at 230400 baud (set by caller: `drivers/lidar/main.go:120`, `main_ld06.go:22`) → `OnRead(func(matTypes.Matrix))` with a 2×count view, `Read(dst 2×K) int` copy of last scan.

Packet as decoded (`device.go:208-240`): `0x54 | len | speed u16 | start i16 | len×(dist u16, intensity u8) | end i16 | timestamp u16 | crc8` → `11+3·len` bytes. The real LD06 second byte is **VerLen = `0x2C`** (version 1 in the top 3 bits, 12 points in the low 5 bits) and every frame is 47 bytes [INFERENCE: LD06 datasheet / ldlidar `PKG_VER_LEN 0x2C`, from memory, not checked offline]. The probe's "realistic" packets used byte 1 = 12 (the package's own builder), i.e. they bypass defect 0 below.

## Completeness review

- Complete: **no.** Intensity discarded (`device.go:302`), timestamp and speed not exposed, no out-of-range filtering (distance 0 / low intensity), no per-point timestamps for motion de-skew.
- Tested: `TestCRC8` (`decoder_test.go:30-43`) only checks CRC is non-zero and self-consistent — pass. `TestParseAndAssembleFullRotation` (`:45-109`) feeds 3 packets (10 points, 120° spans, last packet 240°→10°) and expects one scan of 30 points — **FAIL** "timeout waiting for scan" (run 2026-11-09, matches baseline §5).
- Correct: **no.** Five independent defects, each enough to stop scans on real data:
  0. **VerLen read as point count** — `device.go:209` uses the whole byte 1 as `dataLen`; for a real frame (`0x2C`) it expects 44 points / 143 bytes, computes the CRC over the wrong span and drops the frame (`:223-226`), then resyncs byte by byte. Every real LD06 frame would be rejected before any other check [INFERENCE: depends on the VerLen fact above]. Fix: `n := buf[1] & 0x1F` (or require `buf[1] == 0x2C`).
  1. **Speed-plausibility check rejects all realistic packets** — `device.go:274-279`: `maxExpectedDiff = speed·0.01 · len / 4500 · 1.5` treats the speed field as 0.01 Hz. The reference (`lipkg.cpp`, cited at `:273`) and the LD06 datasheet use **degrees per second** (≈3600 at 10 Hz) and compare the span in degrees, giving ≈14.4° for 12 points [INFERENCE: reference semantics from memory]. With the code's units the limit is 0.14° at speed=3600 and at most 2.62° even at speed=65535, while a real 12-point packet spans ≈8–10°. Probe: realistic packets (12 pts, 8.8° span, speed 3600 or 65535) → writeIdx stays 0 (all rejected). The unit test's 120° spans fail the same check.
  2. **Angles decoded as int16** — `device.go:234,239` (`int16(binary.LittleEndian.Uint16(...))`). LD06 angles are unsigned 0.01° (0–35999). Raw values ≥ 32768 (≥ 327.68°) become negative and `normalizeAngle` maps them to `raw/100 − 295.36`, i.e. 327.68°–359.99° decode as 32.32°–64.63°. Probe (datasheet-format uint16 packets): last packet end 359.60° stored as `lastEndAngleDeg=64.24`. DESIGN.md:33 repeats the wrong "int16, can be negative" claim; the test helper writes `int16(math.Round(deg*100))` (`decoder.go:71,84`), which overflows above 327.67°.
  3. **Wrap detection can never fire** — `device.go:250-257` requires `end − lastEnd < −350°`. With defect 2 the largest decodable angle is 327.67°, so the biggest possible drop is ≈ −327.7°: no rotation is ever completed. The reference applies the −350° test per point (successive point angles), not per packet end angle. Probe: 300 datasheet-format packets with spans that pass check 1 (speed 65535, 2° spans) → 0 scans, 3576 points accumulated until overflow reset.
  4. **Test stream cannot satisfy the detector even with 1–3 fixed** — the test's third packet goes 240°→10°: the end-angle drop is −230°, not < −350°, and the fourth packet that would close the rotation is missing; expected count 30 would also include the next rotation's points (the wrap packet is appended before emit, `device.go:290-330`).
- Other defects (read): CRC comment says poly 0x31 (`decoder.go:9`) but the table is poly **0x4D** (probe: table == bitwise 0x4D CRC, `76 == 76`); scan buffer is not double-buffered — the reader goroutine overwrites `mat2xN` while `Read` and the callback view still reference it (`device.go:282-322` writes without the mutex; `:118-121` reads), so a slow `OnRead` consumer sees torn scans and `go test -race` would flag it [INFERENCE: not run with -race]; `View()` allocates a new outer slice every scan and every `Read` (`x/math/mat/mat.go:100-106`), contrary to the "zero-alloc" comment at `device.go:349`; `readLoop` busy-loops when `Read` returns `0, nil` (no sleep, `:166-193`); overflow silently drops the rotation (`:291-297`).

## TODOs

- [ ] Parse byte 1 as VerLen (`&0x1F` points, expect `0x2C`) and make the test builder emit real 47-byte frames — x/devices/lidar/ld06/device.go:209; decoder.go:64 #p1 #ER-024
- [ ] Fix speed check units (deg/s, span in deg, threshold `speed·len/4500·1.5`) or drop the check — x/devices/lidar/ld06/device.go:274-279 #p1 #ER-024
- [ ] Decode start/end angles as uint16 (0–35999) and fix the test helper and DESIGN.md — x/devices/lidar/ld06/device.go:234,239; decoder.go:71,84; DESIGN.md:33 #p1 #ER-024
- [ ] Detect rotation per point (angle drop between successive points) and split the wrapping packet between scans — x/devices/lidar/ld06/device.go:250-257,290-330 #p1 #ER-024
- [ ] Rewrite TestParseAndAssembleFullRotation with datasheet-shaped packets (12 pts, ~9° spans, ≥2 rotations) and value checks (angles monotone, count, distances) — x/devices/lidar/ld06/decoder_test.go:45-109 #p1 #ER-024
- [ ] Add a golden test from a real LD06 capture (record with `cmd/monitor -capture`) — no capture in repo #p2
- [ ] Double-buffer the scan (swap on emit) and guard writes; run tests with `-race` — x/devices/lidar/ld06/device.go:282-375 #p2
- [ ] Expose intensity (3rd row or separate slice) and per-scan timestamp/speed — x/devices/lidar/ld06/device.go:230-231,302 #p2
- [ ] Fix CRC comment (poly 0x4D) — x/devices/lidar/ld06/decoder.go:9 #p3
- [ ] Sleep/yield on zero-byte reads — x/devices/lidar/ld06/device.go:174 #p3

## Links

- Module: [[lidar]] · Workstream: [[W8-lidar]] · Motor loop: [[lidar-motor-point-count-pid]] · Sibling decoder: [[xwpftb-frame-decoder-and-scan-assembly]]
