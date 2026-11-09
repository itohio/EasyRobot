---
type: algorithm
status: implemented
tags: [easyrobot, devices, i2c]
workstream: [W8]
module: "[[actuators]]"
summary: "none (probe)"
source: [x/devices/tca9548a/tca9548a.go, x/devices/tca9548a/router.go]
tests: [x/devices/tca9548a/router_example_test.go]
test_result: none
complete: yes
correct: unknown
evidence: "probe tmp/easyrobot/devices/actprobe (fake I2C, 2026-11-09): ch3 write ×2 → one mux select 0x08 then two writes; ch5 → select 0x20 then Tx. Example test is tinygo-only (not built)."
---
# TCA9548A I2C multiplexer router

**What it does:** Wraps an 8-channel TCA9548A as eight virtual `devices.I2C` buses. Each `ChannelI2C` call selects its channel (cached: no re-select if already active) and forwards to the base bus. Lets several same-address devices (e.g. VL53L0X at 0x29) share one bus.

**Where:** `x/devices/tca9548a/tca9548a.go:39-74` (`Device`, `SetChannel`, `SetChannelMask`, `Reset`), `router.go:8-110` (`Router`, `Channel`, `setChannel` cache, `ChannelI2C.ReadRegister/WriteRegister/Tx`).

**Inputs → outputs:** base `devices.I2C`, mux address → `Router.Channel(n) devices.I2C`.

## Completeness review

- Complete: yes for single-mux use.
- Tested: `router_example_test.go` uses `machine.I2C0` under `tinygo` (not built on host). Probe with a recording bus shows correct select bytes and caching (above).
- Correct: unknown on hardware. Read: no mutex — two goroutines on different channels can interleave select and transaction (`router.go:59-109`); the cache goes stale if anything else writes the mux; cascaded muxes not supported, and `lidar/vl53l0x` maps sensor `i` to channel `i % 8` on one router, so sensors 8–63 alias 0–7 (`x/devices/lidar/vl53l0x/array.go:76`).

A good composability example: `ChannelI2C` is a decorator that satisfies the same interface as the bus it wraps.

## TODOs

- [ ] Add a mutex spanning select+transaction — x/devices/tca9548a/router.go:59-109 #p2
- [ ] Host unit test with a fake bus (select bytes, caching, Reset) — x/devices/tca9548a #p2

## Links

- Module: [[actuators]] · [[device-io]] · ToF array consumer: [[lidar]], [[sensors]]
