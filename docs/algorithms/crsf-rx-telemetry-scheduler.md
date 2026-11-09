---
type: algorithm
status: tested
tags: [easyrobot, devices, crsf, scheduling]
workstream: [W6, W3]
module: "[[crsf-elrs]]"
summary: "round-robins registered telemetry providers and returns the first non-nil frame, at most one per…"
source: [x/devices/crsf/rx/scheduler.go, x/devices/crsf/rx/device.go]
tests: [x/devices/crsf/rx/scheduler_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -json ./x/devices/crsf/rx → TestSchedulerEmpty, TestSchedulerRoundRobin, TestSchedulerAllNil, TestSchedulerLen pass (2026-11-09)"
---
# CRSF rx telemetry slot scheduler (round-robin)

**What it does:** round-robins registered telemetry providers and returns the first non-nil frame, at most one per call; `rx.Device` calls it once per received RC-channels frame so telemetry is paced by the uplink rate (one downlink slot per RC frame, LOW-LEVEL.md:66-68).

**Where:** `x/devices/crsf/rx/scheduler.go:6` `TelemetryFunc func() *crsf.Frame`; `:10-13` `Scheduler{slots, idx}`; `:16-18` `Register`; `:22-35` `Next` (scans ≤ n slots starting at `idx`, advancing `idx` past every slot it calls); `:38-40` `Len`. Wiring: `x/devices/crsf/rx/device.go:31` exported field `Telemetry Scheduler`; `:195` `handleChannels` → `:198-205` `sendScheduledTelemetry` → `enc.WriteFrame`.

**Inputs → outputs:** providers return `*crsf.Frame` or nil ("nothing this slot"). `Next() *crsf.Frame` — nil when empty or all providers nil. O(n) per call, no allocation in the scheduler itself (providers usually allocate via `EncodeX`).

## Completeness review

- Complete: yes for plain round-robin. No priorities/rates per provider, no unregister.
- Tested: `TestSchedulerEmpty` (nil), `TestSchedulerRoundRobin` (Battery, nil, GPS → Battery, GPS (skips nil), Battery (wraps)), `TestSchedulerAllNil`, `TestSchedulerLen`. Pass (2026-11-09). `x/devices/crsf/advanced_test.go:336` `TestSchedulerRoundRobin` in package `crsf` tests a copy of the loop written inside the test, not production code. Wiring into `rx.Device` is untested.
- Correct: yes for the scheduler (values asserted above).

**Defects / risks (wiring)**
- Not goroutine-safe: `Register` appends while the read loop calls `Next` (scheduler.go:17, device.go:202) → data race if providers are added after `Start`.
- Scheduled writes run on the read goroutine and share the `Encoder` buffer with user `SendBattery/SendGPS/SendFrame` calls (device.go:108,140,203; encoder.go:8) — concurrent use corrupts frames [INFERENCE: from code; no `-race` test exists]. `cmd/elrs-rx` does not register providers, so it does not hit this.
- Write errors from scheduled telemetry are dropped (device.go:203).

## TODOs

- [ ] Mutex (or channel) around `Encoder` writes in `rx.Device`/`tx.Device` — x/devices/crsf/rx/device.go:203 #p1 #backlog
- [ ] Make `Register` safe after `Start` (or document "register before Start") — x/devices/crsf/rx/scheduler.go:16 #p2
- [ ] Integration test: channels frames in → one telemetry frame out per RC frame (io.Pipe) — x/devices/crsf/rx/scheduler_test.go:1 #p2
- [ ] Remove the duplicate pseudo-scheduler test in package crsf — x/devices/crsf/advanced_test.go:336 #p3

## Links

- Module: [[crsf-elrs]] · Related: [[crsf-rx-failsafe-detection]], [[crsf-telemetry-frames]]
