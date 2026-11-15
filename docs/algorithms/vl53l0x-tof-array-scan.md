---
type: algorithm
status: partial
tags: [easyrobot, devices, sensors, tof, lidar]
workstream: W8
module: "[[sensors]]"
summary: "Unsynchronised buffer, sensors ≥ 8 alias 0–7"
source: [x/devices/lidar/vl53l0x/array.go, x/devices/lidar/vl53l0x/DESIGN.md]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "no test files; build/vet/linux-arm64 cross ok (2026-11-09); probe with fake I2C + real tca9548a.Router: 4 sensors at [-25,-5,5,25]° → Read n=4 dist=[500 500 500 500] ang=[-25 -5 5 25], min/max -25/25; New accepts 9 angles (ch 0 aliased); compile-time var _ lidar.Device = (*array.Device)(nil) ok; go run -race single run: no race reported."
---
# VL53L0X ToF array scan ("array lidar")

**What it does:** turns N fixed VL53L0X sensors behind a TCA9548A I²C mux into a `lidar.Device`: one sensor per mux channel, each tagged with a configured angle; a 50 ms ticker goroutine polls every sensor in continuous mode and publishes a 2×N matrix (row 0 distance mm, row 1 angle deg). Detailed lidar-family context lives in [[lidar]].

**Where:** `x/devices/lidar/vl53l0x/array.go` (package name **`array`**, DESIGN.md calls the path `x/devices/lidar/array`):
- `:17-51` `Device`, `Config{Router, Angles, SensorAddresses}`.
- `:56-102` `New` — backing `[]float32` 2N, `mat.New(2,N,backing...)`; sensor *i* on `Router.Channel(i % 8)` at address `SensorAddresses[i]` or 0x29.
- `:105-127` `Configure` (router, each `vl53l0x.Configure(init)`, `StartContinuous(50)` if init, start `readLoop` once).
- `:130-137` `Close`; `:141-145` `OnRead`; `:148-166` `Read`; `:169-194` min/max angle; `:197-201` `GetPointCount`.
- `:203-215` `readLoop` (fixed 50 ms ticker); `:217-243` `readScan`.

**Addressing:** no XSHUT sequencing — the driver relies entirely on the mux; all sensors stay at 0x29 unless `SensorAddresses` is set, but [[vl53l0x-ranging-driver]] has no `SetAddress`, so non-default addresses cannot actually be programmed. Effective limit: 8 sensors (one per channel).

**Inputs → outputs:** `New(ctx, Config) (*Device, error)`; `Read(dst 2×K) int` copies min(K,N) columns; `OnRead(func(matTypes.Matrix))` gets a view sharing the internal backing (valid only during the callback). Failed reads → distance 0.

## Completeness review

- Complete: no — single mux only; >8 angles accepted (limit 64, :63) but channel `i%8` aliases sensors with identical address (:77); scan period fixed (:204) and `StartContinuous(50)` period ignored by the driver; no per-point validity/range-status.
- Tested: no test files. Probe (fake I²C emulating register memory, real `tca9548a.NewRouter`): `Configure(true)` → nil; after 180 ms `Read` into 2×4 → n=4, distances 500 mm, angles echo config, `GetMinAngle/GetMaxAngle` −25/25, `GetPointCount` 4; `New` with 9 angles → nil error (aliasing accepted). `go run -race` → no report (single short run; not proof).
- Correct: 2×N layout and angle assignment correct per probe; distances inherit [[vl53l0x-ranging-driver]] defects (no calibration, stale reads).

**Defects / risks**
- Data race by inspection: `readScan` writes `mat2xN` rows without the mutex (array.go:218-229) while `Read` copies them under the mutex (148-165); the lock only guards `count`/`onRead`.
- `Read`/`readScan` type-assert `View().(mat.Matrix)` (161-162, 234) → panic for other `matTypes.Matrix` implementations; `dst` with <2 rows panics.
- `View()` allocates a new row-slice header each scan (234; mat.go:100-105) → 20 allocs/s on TinyGo.
- `Configure(false)` starts the loop without continuous mode → every read `ErrInvalidState` → all zeros (110-125; vl53l0x.go:193-195).
- `Close` cancels ctx and immediately issues `StopContinuous` on all sensors without waiting for an in-flight `readScan` → concurrent I²C through a non-thread-safe `Router` (130-137).
- `Config.Angles` slice aliased, not copied (94); `_ = i` dead code (120).
- Invalid marker 0 is indistinguishable from "very close"; out-of-range ≈8190 mm passes as valid.

## TODOs

- [ ] Guard row writes in `readScan` with the mutex (or double-buffer) — x/devices/lidar/vl53l0x/array.go:218 #p1 #backlog
- [ ] Reject >8 angles until multi-mux/readdressing exists — x/devices/lidar/vl53l0x/array.go:63 #p1 #backlog
- [ ] Wait for `readLoop` exit in `Close` before stopping sensors — x/devices/lidar/vl53l0x/array.go:130 #p2
- [ ] Unit test with fake I²C (2×N layout, failure→0, aliasing guard) — x/devices/lidar/vl53l0x/array.go:217 #p2
- [ ] Configurable scan period; XSHUT-based addressing as alternative to mux — x/devices/lidar/vl53l0x/array.go:204 #p2
- [ ] Avoid per-scan `View()` allocation; accept any `matTypes.Matrix` in `Read` — x/devices/lidar/vl53l0x/array.go:234 #p3
- [ ] Rename package or fix DESIGN.md path (`array` vs `lidar/vl53l0x`) — x/devices/lidar/vl53l0x/array.go:1 #p3

## Links

- Module: [[sensors]] · Family: [[lidar]] · Workstream: [[W8-lidar]] · Driver: [[vl53l0x-ranging-driver]] · Output consumers: [[grid-maps]]
