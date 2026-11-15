---
type: algorithm
status: partial
tags: [easyrobot, devices, sensors, optical-flow, spi]
workstream: [W3, W4]
module: "[[sensors]]"
summary: "partial (discards all received bytes)"
source: [x/devices/adns3080/adns3080.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no test files; build/vet/linux-arm64 cross ok (2026-11-09); fake-SPI probe (Transfer always returns 0x17): Read8(ProductID)=0x00, Configure(true) → invalid response, ReadMotion → invalid state. Driver discards every received byte."
---
# ADNS3080 optical-flow sensor driver

**What it does (intended):** SPI driver for the ADNS3080 mouse sensor: reset, product-ID check, config write, motion burst (motion, ΔX, ΔY, SQUAL, shutter) and 30×30 frame capture. No flow-to-velocity conversion (height/FOV scaling) is provided.

**Where:** `x/devices/adns3080/adns3080.go:12-52` registers; `:55-62` `MotionData`; `:77-83` `New(bus SPI, cs, reset Pin)`; `:87-122` `Configure(init)`; `:125-139` `Reset`; `:142-178` `ReadMotion`; `:182-243` `FrameCapture`; `:246-261` `Read8`; `:264-277` `Write8`.

**Inputs → outputs:** `devices.SPI` (mode/clock set by caller), CS and optional reset pins. `ReadMotion() (*MotionData, error)` raw counts (ΔX/ΔY int8, counts per frame at configured CPI); `FrameCapture([]byte ≥900) (int, error)` 6-bit pixels scaled ×4.

**Sequence vs datasheet** (from memory of the ADNS-3080 datasheet — [INFERENCE]): register map 0x00–0x3F and burst addresses 0x40/0x50 match; product ID 0x17 matches; write = addr|0x80, read = addr then ≥50 µs (tSRAD) before clocking data — code waits 20 µs (:255, :272); motion burst order is Motion, ΔX, ΔY, SQUAL, Shutter_Upper, Shutter_Lower, Max_Pixel after ≥75 µs — code reads shutter as low/high (:174), skips Max_Pixel and has no delay after the burst command (:152-158); frame capture = write 0x83 to Frame_Capture, then Pixel_Burst read, first pixel flagged by bit 6 — implemented as such (:192-236).

## Completeness review

- Complete: no — every read discards the byte returned by `Transfer`: `Read8` returns a zero `value` never assigned (:250-260), `ReadMotion` stores an unassigned `b` (:159-166), `FrameCapture` tests an unassigned `pix` (:212-217). Comments acknowledge it ("simplified version", :164-165).
- Tested: no test files. Fake-SPI probe (2026-11-09): with a bus that always returns 0x17, `Read8(ProductID)` = 0x00, `Configure(true)` → `ErrInvalidResponse`, `ReadMotion` → `ErrInvalidState`. By reading, `FrameCapture` would always time out after 100 transfers (:224-228).
- Correct: no — the driver cannot return sensor data on any bus.

**Defects**
- Received SPI bytes discarded in `Read8`, `ReadMotion`, `FrameCapture` (adns3080.go:252-260, 160-166, 213-217) → `Configure(true)` always fails, driver unusable.
- Shutter byte order swapped and Max_Pixel missing in motion burst (:174-175) [INFERENCE].
- Read delay 20 µs < tSRAD 50 µs; no 75 µs delay after motion-burst command (:255, :152) [INFERENCE].
- `ShutterMaxBoundUpper` duplicates 0x1E (:41-42) — one constant is wrong.
- `ReadMotion` allocates a 6-byte buffer and `*MotionData` per call (:148, :177).
- `Configure` writes config twice (`Reset` :134 and :116) with magic 0x19 undocumented.

## TODOs

- [ ] Use the byte returned by `Transfer` (or `Tx`) in `Read8`, `ReadMotion`, `FrameCapture` — x/devices/adns3080/adns3080.go:252 #p1 #backlog
- [ ] Fix motion-burst field order (shutter high/low, Max_Pixel) and inter-byte delays per datasheet — x/devices/adns3080/adns3080.go:150 #p1 #backlog
- [ ] Correct duplicated `ShutterMaxBoundUpper` address — x/devices/adns3080/adns3080.go:42 #p2
- [ ] Fake-SPI unit tests (product-ID check, burst decode, frame-capture sync bit) — x/devices/adns3080/adns3080.go:142 #p2
- [ ] Add flow → velocity helper (counts, CPI, height, FOV) for odometry use — x/devices/adns3080/adns3080.go:55 #p3
- [ ] Zero-alloc `ReadMotionInto(*MotionData)` — x/devices/adns3080/adns3080.go:148 #p3

## Links

- Module: [[sensors]] · Interfaces: [[device-io]] · Possible consumers: [[filters]] (odometry fusion), [[motion-and-gait]]
