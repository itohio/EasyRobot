---
type: algorithm
status: partial
tags: [easyrobot, devices, encoder]
workstream: [W3, W6]
module: "[[actuators]]"
summary: "Table correct, RPM gate 1000× off, F1"
source: [x/devices/encoder/encoder.go, x/devices/encoder/linux.go, x/devices/encoder/stub.go, x/devices/encoder/tinygo.go]
tests: [x/devices/encoder/example_test.go]
test_result: build-fail
complete: no
correct: no
evidence: "go test ./x/devices/encoder (darwin) → build failed 'configurePins(...) (no value) used as value' (2026-11-09); GOOS=windows build → same; GOOS=linux build ok, vet: 'ExampleEncoder refers to unknown identifier: Encoder'; probe tmp/easyrobot/devices/encprobe (copied table) → 100-state Gray sequence = ∓99 counts, RPM gate needs 100 s instead of 100 ms"
---
# Quadrature encoder decoding (4× interrupt, lookup table)

**What it does:** On every edge of A or B, reads both pins, shifts the 2-bit state into a 4-bit history and adds `states[old<<2|new] ∈ {−1,0,+1}` to an atomic position (4× decoding, invalid double-steps count 0). `RPM()` differentiates position over at least `UpdateInterval`.

**Where:** table `x/devices/encoder/encoder.go:20`, `Configure` `:76-108`, `interrupt` `:113-134`, `Position/Reset/RPM/updateRPM` `:137-183`; platform `configurePins`: `linux.go` (no-op, returns error), `stub.go` (no return value — **F1**), `tinygo.go` (pull-ups).

**Inputs → outputs:** `New(pinA, pinB devices.Pin, Config{CountsPerRevolution (default 2048), UpdateInterval (100 ms)})` → `Position() int64` counts, `RPM() float64`.

## Completeness review

- Complete: **no.** No direction/velocity filtering, no index (Z) channel, no timestamped edges for low-speed velocity; `RPM()` only updates when polled.
- Tested: no test; `example_test.go` has no `// Output:` (never runs; infinite loop) and its name `ExampleEncoder` refers to a non-existent `Encoder` type (vet error on linux).
- Builds: **fails on darwin and windows** (`stub.go:7` lacks `error` return; baseline F1, windows confirmed 2026-11-09 by cross-build). linux builds. **TinyGo build would fail** [INFERENCE from source]: `tinygo.go:8-17` calls `pinA.Configure(devices.PinConfig{Mode: devices.PinInputPullup})`, but `devices.Pin` has no `Configure` and `devices.PinConfig`/`PinInputPullup` exist nowhere in `x/devices` (grep), and `machine.Pin` cannot satisfy `devices.Pin` anyway (see [[device-io]]: `SetInterrupt` callback type).
- Correct: table **yes** (probe on a copy: one-direction Gray sequence of 100 states → −99, reverse → +99, bounce/illegal → 0). **RPM gate wrong**: `encoder.go:164` converts a microsecond difference to `time.Duration` (nanoseconds) and compares with 100 ms, so RPM refreshes only after **100 s** (probe: 150 ms elapsed → no update). Linux: each pin's interrupt runs in its own epoll goroutine (`x/devices/pin_linux.go:133-190`), so `oldAB` (`encoder.go:32,119-125`) is updated concurrently without atomics — lost transitions/race.

## TODOs

- [ ] Make `configurePins` in stub.go return `error` (fixes darwin+windows builds of encoder and motor) — x/devices/encoder/stub.go:7 #p1 #ER-025
- [ ] Compare `time.Duration(now-last)*time.Microsecond` (or store `time.Time`) in the RPM gate — x/devices/encoder/encoder.go:164 #p1 #ER-025
- [ ] Define pin configuration in `devices` (or take pre-configured pins) so tinygo.go compiles — x/devices/encoder/tinygo.go:8-17 #p1 #ER-025
- [ ] Serialise the interrupt handler on Linux (single goroutine for both pins or atomic CAS on oldAB) — x/devices/encoder/encoder.go:113-134 #p2
- [ ] Unit test with a fake `devices.Pin` driving Gray sequences + RPM with injected clock; rename example to `ExampleDevice` — x/devices/encoder/example_test.go:11 #p2

## Links

- Module: [[actuators]] · Consumer: [[dc-motor-pid-speed-control]] · [[device-io]] · Odometry users: [[kinematics]]
