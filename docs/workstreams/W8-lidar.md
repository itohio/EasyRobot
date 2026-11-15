---
type: workstream
status: proposed
tags: [easyrobot, workstream, lidar, devices]
workstream: W8
---
# W8 — LiDAR module integration

**Goal:** integrate the lidar module, which exists but hasn't been analysed yet.

Evidence date 2026-11-09, HEAD abb5c9cc, darwin/arm64, no hardware. Details: [[lidar]], [[ld06-packet-decoder-and-scan-assembly]], [[xwpftb-frame-decoder-and-scan-assembly]], [[lidar-motor-point-count-pid]], [[vl53l0x-tof-array-scan]].

## What exists — which lidars the code supports

| Driver | Device | Link / baud | Frame | Checksum | Angles | Scan assembly | Status |
|---|---|---|---|---|---|---|---|
| `x/devices/lidar/ld06` | LDROBOT **LD06** DTOF 360° (also LD19/STL-06 family [INFERENCE]) | UART TX-only, 230400 8N1 (`drivers/lidar/main.go:120`, `main_ld06.go:22`) | `54 2C speed(°/s) start(0.01°) 12×(dist mm u16, intensity u8) end ts(ms) crc` = 47 B (decoder assumes `byte1 = n`) | CRC-8 poly 0x4D, table (`decoder.go:10-37`; probe: equals bitwise 0x4D) | linear start→end, last point = end (`device.go:288-311`) | emit when end angle drops by > 350° (`:250-257`) | **partial — cannot produce a scan** |
| `x/devices/lidar/xwpftb` | **unidentified** (DESIGN.md says Delta-2G; code parses a different `55 AA 23 10` 60-byte frame) | UART, 250000 host / 115200 RP2040 (conflict) | `55 AA 23 10 speed start 16×(dist,intensity) end sum16` | 16-bit byte sum | `((H&0x7F)<<8\|L) − 0x2000` ×0.01°, step span/16 | every 15 frames | **partial — angles > 245.75° wrong (probe)** |
| `x/devices/lidar/vl53l0x` (pkg `array`) | N × ST **VL53L0X** ToF on a TCA9548A mux, fixed mounting angles | I²C 0x29 per channel | — | — | per-sensor configured angle | poll all every 50 ms | implemented, untested ([[sensors]]) |

Common output (`x/devices/lidar/lidar.go:9-36`): 2×N `float32` matrix, row 0 distance **mm**, row 1 angle **deg** (0° forward, clockwise per LD06 DESIGN.md), via `OnRead` callback / `Read`. Published by `drivers/lidar` as protobuf `types/devices.LIDARReading{DistancesMm, AnglesDeg, Timestamp ns, Valid}` (`proto/types/devices/sensors.proto`) on dndm intent `LIDARReading@lidar.scan` (RP2040, `drivers/lidar/main_rp2040.go:79-80`) or `--intent` routes (host, via `cmd/display/destination.IntentDestination`). Intensity is dropped everywhere.

**Which module does the owner have?** Not determinable from the code. Both spinning drivers were written for real modules: ld06 follows the LD06 datasheet/ldlidar reference closely; xwpftb was clearly fitted to a captured stream (hex-dump logging, comments cite a "README" formula). A label photo or a 2-second capture decides it.

## Failure analysis: `TestParseAndAssembleFullRotation` ("timeout waiting for scan")

Reproduced 2026-11-09: `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/devices/...` → ld06 FAIL (3.0 s), xwpftb FAIL (1.0 s); CRC tests pass. Both are pure decoder tests (stub serial in memory, no device).

**ld06** — the implementation rejects or never completes every stream; the test is also wrong:
1. Speed plausibility check uses 0.01 Hz units instead of °/s (`device.go:274-279`): limit 0.033° for the test's 120° spans → every test packet skipped (`return 1`). Probe: also rejects realistic 12-point packets at speed 3600 or 65535.
2. Angles parsed as `int16` (`:234,239`): 327.68°–359.99° decode as 32.32°–64.63° (probe: 359.60° → 64.24°).
3. Rotation completes only when the end angle drops by > 350° (`:250-257`); with (2) the max drop is ~327.7° → never fires (probe: 300 accepted packets, 0 scans).
4. The test's wrap packet 240°→10° gives a −230° drop and no closing packet (`decoder_test.go:45-109`), so even a correct per-packet detector would not fire; its expected 30 points mixes rotations.
5. (Real data, by reading) byte 1 of a real frame is VerLen `0x2C`, not a point count (`:209`) → CRC over the wrong length → every real frame dropped [INFERENCE: datasheet from memory].

**xwpftb** — the test is wrong twice:
1. It never calls `dev.Configure()` (`decoder_test.go:57-65`), so the read goroutine never starts (`device.go:145-163`).
2. Its frames come from `BuildMeasurementFrame` (Delta-2G `0xAA` format, `decoder.go:17-54`), which `consumeOneFrame` (looks for `55 AA`, `device.go:228-250`) never matches. Probe: test stream + `Configure` → still 0 scans; frames in the decoder's format → 240-point scan.

## Gaps for integration

- **Decoder correctness** (above) — nothing has ever produced a verified scan; no golden capture in the repo.
- **TinyGo/RP2040 node does not compile**: `targetPoints` (and `maxPoints` for xwpftb) declared twice under `rp2040,ld06` / `rp2040,xwpftb` (go/types check; `main_rp2040.go:41-42`, `main_ld06.go:14`, `main_xwpftb.go:14-15`). Host-only files are tagged `linux || windows`; TinyGo uses `GOOS=linux` for bare-metal [INFERENCE], so `display.go` (GoCV) would be pulled in too. Motor PWM placeholder (`main_rp2040.go:62-68`). The dndm serial transport imports `tarm/serial` (`../dndm/network/serial/serial.go:9`), unlikely to build on TinyGo [INFERENCE]. Board tags: only `rp2040` exists for lidar; no `esp32` variant.
- **Host node** needs OpenCV (always draws), `NewSerialWithConfig` (linux/windows only, F2), so it cannot build on darwin; serial reads poll with `VMIN=0,VTIME=0` and the read loops spin at 100 % CPU when idle (`x/devices/serial_linux.go:86-87`).
- **SLAM link** ([[filters]] `x/math/filter/slam`): `slam.New(rayAngles rad, originX, originY, opts…)` needs a fixed ray set and its `Update(dt, input)` reads row 0 = angles (rad), row 1 = distances (m) (`slam.go:376-395`), `len == numRays` (panics otherwise, `:387-394`). Lidar scans are row 0 mm, row 1 deg, variable N. Missing: an adapter that bins a scan into fixed angular rays (min-range per bin, invalid → max range), swaps rows and converts units; plus lidar→robot mounting transform.
- **Occupancy grid** ([[grid-maps]] `x/math/grid`, and SLAM's log-odds `mapping.go`): ray casting exists; no standalone "scan → occupancy update" entry point outside SLAM, no map publishing over dndm.
- **Timing / motion**: no per-point timestamps, so no de-skew while moving; `LIDARReading.Timestamp` is the host time of emission.
- **Concurrency**: scan buffers are not double-buffered; `OnRead` views are overwritten by the reader goroutine.
- **Darwin build gaps** (baseline F1–F3) affect the surrounding apps: F2 serial (no darwin `NewSerialWithConfig`) blocks every host serial tool incl. `drivers/lidar` and `cmd/monitor`; F1 encoder stub blocks motor (wheel odometry for SLAM); F3 elrs-rx desktop.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Owner: identify the lidar module (label/photo) and record a 5 s raw capture (`cmd/monitor -capture` on Linux, or any serial logger) into `x/devices/lidar/<model>/testdata/` — no capture in repo #p1 #ER-024
- [ ] Make GoCV display optional in the host node; add darwin serial — drivers/lidar/main.go:194-216; x/devices/serial_stub.go #p2
- [ ] Double-buffer scans and expose intensity + timestamps in `LIDARReading` (proto change) — x/devices/lidar/ld06/device.go:343-375; proto/types/devices/sensors.proto #p2

## Inferred objectives (proposals)

- **Proposal (M1, host only, no hardware needed):** golden-capture tests for the owner's lidar; decoder fixed until a capture yields monotone angles over 360° and plausible distances. Acceptance: `go test ./x/devices/lidar/...` passes with value checks.
- **Proposal (M2, owner approves hardware scope):** LD06 on Linux/RPi USB-UART → host node publishes `LIDARReading` over dndm → viewer (GoCV or a headless PNG dump).
- **Proposal (M3):** lidar + wheel odometry → SLAM adapter → occupancy grid; demo: map a room with the wheeled ESP32 rover (shared with W6 obstacle avoidance).
- **Proposal (M4):** MCU bridge: ESP32 or RP2040 forwards raw LD06 frames over dndm/UART or Wi-Fi; decode on host first, decode on MCU only when needed (keeps the MCU build small).

## Research questions for ideation

1. Is the owner's module an LD06/LD19 (DTOF, 230400, 0x54 frames), a Delta-2G/3irobotix (0xAA frames), or a Camsense/other 0x55AA protocol? Which public protocol description matches the `55 AA 23 10` 60-byte frame with `−0x2000` angle offset?
2. For low-cost 2-D SLAM on an ESP32-class robot: scan-matching (ICP / correlative, as in Hector/BreezySLAM) vs the repo's EKF-against-map approach — which fits 450-point scans at 10 Hz in TinyGo memory?
3. Should the bus carry raw polar scans (`LIDARReading`) or pre-binned fixed-ray scans (fits SLAM and NN inputs; smaller messages)?
4. Can lidar scans feed the W6 decision-tree obstacle avoider directly (sector minima as features), shared with camera features from W4/W5?

## Decisions for the owner

- **Which driver to invest in:** (a) fix ld06 only, (b) fix both, (c) replace xwpftb with a protocol-identified driver. **Recommendation:** decide after identifying the module; if it is an LD06, do (a) and park xwpftb with its test skipped.
- **Where to decode:** host vs MCU. **Recommendation:** host first (debuggable, Linux), MCU bridge later.
- **Output format:** keep 2×N mm/deg or move to SI (m, rad) with intensity row. **Recommendation:** add a 3rd row (intensity) and a typed `Scan` struct; keep mm/deg on the wire for compatibility, convert in the SLAM adapter.
