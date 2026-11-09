---
type: algorithm
status: partial
tags: [easyrobot, devices, sensors, tof, i2c, lidar]
workstream: W8
module: "[[sensors]]"
summary: "partial (no SPAD init/calibration/timing budget)"
source: [x/devices/vl53l0x/vl53l0x.go, x/devices/vl53l0x/tuning.go]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "no test files; build/vet/linux-arm64 cross ok (2026-11-09); fake-I2C probe: Configure(true) → 98 writes, ends 0B=01 01=E8 01=01 01=02 01=E8 (no SYSRANGE_START → no ref calibration); ReadRangeSingle with data-ready never set → 300 mm, err=nil after 120 ms; StartContinuous(50) writes only 00=02."
---
# VL53L0X ranging driver (init, tuning, single/continuous ranging)

**What it does:** "simplified" port of the ST VL53L0X API / Pololu sequence (vl53l0x.go:3): 2V8 I/O, stop-variable capture, MSRC/TCC limit disable, 0.25 MCPS signal-rate limit, default tuning table, GPIO interrupt config, then single-shot or back-to-back continuous ranging returning millimetres from RESULT_RANGE_STATUS+10.

**Where:**
- `x/devices/vl53l0x/vl53l0x.go:19-78` register constants; `:89-109` `Device`, `New` (default 0x29).
- `:112-124` `Configure`/`Reset`; `:127-132` `Connected` (model 0xEE, rev 0xAA).
- `:135-168` `ReadRangeSingle`; `:171-189` `StartContinuous`/`StopContinuous`; `:192-220` `ReadRangeContinuous`.
- `:223-313` `init`; `:316-330` `Calibrate`; `:333-340` `SetSignalRateLimit` (Q9.7); `:343-385` `startRangeSingle`; `:387-399` I²C helpers.
- `x/devices/vl53l0x/tuning.go:5-23` `TuningData` (80 reg/value pairs).

**Inputs → outputs:** `devices.I2C`, 7-bit address. `ReadRangeSingle() (uint16 mm, error)`, `ReadRangeContinuous() (uint16 mm, error)` (`ErrTimeout` if no new sample, non-blocking), `SetSignalRateLimit(MCPS)`.

**Sequence vs ST API / Pololu library** (reference recalled from memory of Pololu `VL53L0X::init` — [INFERENCE]):

| Step | Pololu | Here |
|---|---|---|
| 2V8 I/O, 0x88=0, stop variable via 0x80/0xFF/0x00 dance | yes | yes (:225-263) |
| MSRC_CONFIG_CONTROL \|= 0x12, signal rate 0.25 MCPS | yes | yes (:266-277) |
| SYSTEM_SEQUENCE_CONFIG=0xFF | yes | yes (:280) |
| `getSpadInfo` + reference SPAD map (GLOBAL_CONFIG_SPAD_ENABLES_REF_0..5) | yes | **missing** (constants :65-73 unused) |
| DefaultTuningSettings | yes | table matches as recalled (tuning.go) |
| GPIO interrupt new-sample-ready, clear | yes | yes (:292-304) |
| get/set measurement timing budget | yes | **missing** (`measurementTimingBudget` unused, :93) |
| SEQUENCE_CONFIG=0xE8 | yes | yes (:307) |
| VHV calibration (SEQ=0x01, SYSRANGE_START=0x41, wait, clear) and phase calibration (SEQ=0x02, SYSRANGE_START=0x01) | yes | **stub**: writes SEQ 0x01, 0x02, 0xE8 only (:316-330; probe last writes) |
| startContinuous: stop-variable preamble, period → INTERMEASUREMENT + mode 0x04 | yes | **no preamble, period ignored**, always 0x02 (:171-179) |
| stopContinuous postamble (0x91=0x00) | yes | missing (:182-189) |
| read range at RESULT_RANGE_STATUS+10 (0x1E), clear interrupt | yes | yes (:155, :208) |

## Completeness review

- Complete: no — SPAD management, timing budget and reference calibration absent; no `SetAddress` (I2CSlaveDeviceAddress 0x8A constant unused, :37) so several sensors on one bus require a mux; `ioTimeout` unused (:94); range-status (0x14 bits 6:3) never checked, so out-of-range readings (≈8190/8191 mm [INFERENCE]) are returned as distances.
- Tested: no test files. Fake-I²C probe (`tmp/easyrobot/devices/sensors`, 2026-11-09): `Configure(true)` → nil, 98 register writes; `ReadRangeSingle` with RESULT_INTERRUPT_STATUS never ready → returns 300 mm (stale register content), `err=nil`, after 120 ms; `StartContinuous(50)` last write `00=02`.
- Correct: unknown on hardware; ranging without SPAD init and calibration typically gives degraded/offset results [INFERENCE].

**Defects**
- `ReadRangeSingle` ignores data-ready timeout and returns stale data with nil error (vl53l0x.go:141-151; probe).
- `Calibrate` performs no calibration (vl53l0x.go:316-330).
- SPAD info/ref SPAD map not configured (vl53l0x.go:279-289 gap).
- `StartContinuous` ignores `periodMs`, skips stop-variable preamble (vl53l0x.go:171-179); `StopContinuous` lacks postamble (182-189).
- `init` does not verify model ID before writing (vl53l0x.go:223).
- `read8` allocates per call (vl53l0x.go:392), range reads allocate 2 B (154, 207); busy-poll with 1 ms sleep, up to 2×100 ms blocking per single shot.

## TODOs

- [ ] Return `ErrTimeout` when data-ready never asserts in `ReadRangeSingle` — x/devices/vl53l0x/vl53l0x.go:141 #p1 #backlog
- [ ] Implement VHV + phase reference calibration (`performSingleRefCalibration`) — x/devices/vl53l0x/vl53l0x.go:316 #p1 #backlog
- [ ] Implement SPAD info read + reference SPAD map — x/devices/vl53l0x/vl53l0x.go:279 #p1 #backlog
- [ ] Implement timing budget get/set and timed continuous mode with period — x/devices/vl53l0x/vl53l0x.go:171 #p2
- [ ] Add `SetAddress` (reg 0x8A) for XSHUT-sequenced multi-sensor buses — x/devices/vl53l0x/vl53l0x.go:37 #p2
- [ ] Expose range status / reject out-of-range readings — x/devices/vl53l0x/vl53l0x.go:155 #p2
- [ ] Fake-I²C unit tests asserting init write sequence and timeout behaviour — x/devices/vl53l0x/vl53l0x.go:223 #p2
- [ ] Remove allocations in `read8`/range reads — x/devices/vl53l0x/vl53l0x.go:392 #p3

## Links

- Module: [[sensors]] · Used by: [[vl53l0x-tof-array-scan]] → [[lidar]], [[W8-lidar]] · Interfaces: [[device-io]]
