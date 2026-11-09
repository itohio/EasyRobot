---
type: algorithm
status: partial
tags: [easyrobot, devices, crsf, safety]
workstream: [W6, W3]
module: "[[crsf-elrs]]"
summary: "marks the link lost when no RC-channels frame has arrived for > 1 s and notifies via callback…"
source: [x/devices/crsf/rx/device.go, x/devices/crsf/decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "No test. Code read 2026-11-09: checkFailsafe only runs when Decoder.ReadFrame returns a non-EOF error; Linux serial Read returns (0,nil) on EAGAIN (x/devices/serial_linux.go:178-180) so no error ever surfaces"
---
# CRSF rx failsafe detection

**What it does:** marks the link lost when no RC-channels frame has arrived for > 1 s and notifies via callback; clears on the next channels frame.

**Where:** `x/devices/crsf/rx/device.go:14` `failsafeTimeout = time.Second`; `:67` `OnFailsafe`; `:97-101` `IsFailsafe`; `:143-162` `readLoop` (calls `checkFailsafe` only in the `ReadFrame` error branch, :152-158); `:177-196` `handleChannels` (records `lastChannelAt`, clears flag, fires `onFailsafe(false)`); `:220-235` `checkFailsafe`.

**Inputs → outputs:** stream of frames → `IsFailsafe() bool`, `onFailsafe(active bool)` edge callbacks. Never active before the first channels frame (`lastChannelAt.IsZero()`, :222).

## Completeness review

- Complete: no — there is no timer; detection piggybacks on read errors.
- Tested: no test (rx has only scheduler tests).
- Correct: no (by code reading):
  - `ReadFrame` returns only when a full frame decodes or the reader errors (decoder.go:20-57). A silent link produces neither: Linux `LinuxSerial.Read` maps EAGAIN to `(0, nil)` (x/devices/serial_linux.go:178-180) so `ensureBytes` spins (decoder.go:60-70); a blocking reader just blocks. Either way `checkFailsafe` never runs → failsafe never asserts when the receiver goes quiet.
  - If other frames keep arriving (e.g. link stats from an RX that lost the TX), `ReadFrame` succeeds, so `checkFailsafe` is again skipped (device.go:160).
  - On `io.EOF` the loop exits without asserting failsafe (device.go:153-155).
  - TinyGo `machine.UART` as used by `cmd/elrs-rx` [INFERENCE: TinyGo `UART.Read` returns `(0, nil)` on an empty buffer, recalled not verified] → same spin, and on the cooperative scheduler the spinning read goroutine may starve `main`'s telemetry ticker.
  - Callbacks are read without synchronisation (device.go:67 vs :229) — register before `Start`.

## TODOs

- [ ] Drive failsafe from a timer/ticker (or deadline-capable reader) independent of `ReadFrame` returning — x/devices/crsf/rx/device.go:151 #p1 #backlog
- [ ] Assert failsafe on EOF/fatal read error before exiting the loop — x/devices/crsf/rx/device.go:153 #p2
- [ ] Unit test with a fake clock/reader: frames stop → callback(true) within ~1 s; resume → callback(false) — x/devices/crsf/rx/device.go:220 #p2

## Links

- Module: [[crsf-elrs]] · Decoder spin: [[crsf-frame-codec-and-crc8]] · Serial transports: [[device-io]]
