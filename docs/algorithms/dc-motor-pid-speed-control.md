---
type: algorithm
status: partial
tags: [easyrobot, devices, actuators, control]
workstream: [W6]
module: "[[actuators]]"
summary: "Ticker race, encoder dependency"
source: [x/devices/motor/motor.go, x/devices/motor/array.go, x/devices/motor/types.go]
tests: []
test_result: build-fail
complete: no
correct: unknown
evidence: "go build ./x/devices/motor (darwin, windows) → fails via encoder F1 (2026-11-09); GOOS=linux build ok; no tests"
---
# DC motor PID speed control (encoder feedback)

**What it does:** A ticker loop (`SamplePeriod`, default 10 ms) reads encoder RPM, runs a [[pid-control]] `PID1D` (output clamped ±`MaxOutput`), and drives one of three H-bridge wirings: `TypeDirPWM` (DIR pin + PWM), `TypeABPWM` (two PWMs), `TypeABDirPWM` (IN1/IN2 pins + PWM enable). `MotorArray` fans speeds out to several motors.

**Where:** `x/devices/motor/motor.go:40-78` (`New`, PWM at 20 kHz), `:161-187` (`SetSpeed`), `:204-266` (`Enable/Disable/controlLoop`), `:269-323` (`update`), `:326-353` (`setPWM`); `array.go:20-128`; config `types.go`.

**Inputs → outputs:** target RPM (signed, clamped to `MaxRPM`) + `*encoder.Device` → PWM duty and direction pins.

## Completeness review

- Complete: **no.** No feed-forward (duty ≈ rpm/MaxRPM), no dead-band/minimum duty, no anti-windup beyond clamp, no brake vs coast choice, no current limit; depends on encoder `RPM()` which effectively never refreshes (see [[quadrature-encoder-decoding]], `encoder.go:164`), so the loop would see 0 RPM and saturate.
- Tested: none. README (265 lines) documents usage but nothing executes it.
- Correct: unknown. Read defects: `controlLoop` reads `m.ticker.C` without the lock while `Disable` sets `m.ticker = nil` (`motor.go:237,262`) → nil-pointer panic race; `pwmDuty = |output|` (`:288`) exceeds 1 if `MaxOutput > 1`, and `Set` errors are ignored (`:295-320`); D term uses `SamplePeriod` not measured dt (`:282`); `Disable` leaves PID integrator state until next `Enable` (reset there, OK).
- Builds on darwin/windows fail only because of the encoder import (F1).

## TODOs

- [ ] Fix ticker nil race (capture ticker channel before starting goroutine) — x/devices/motor/motor.go:237,257-266 #p1 #ER-025
- [ ] Clamp duty to [0,1] and propagate PWM errors — x/devices/motor/motor.go:288-322 #p2
- [ ] Unit test with fake PWM/Pin and a first-order motor model — x/devices/motor #p2
- [ ] Add feed-forward + min duty for low-speed control (needed for wheeled ESP32 rover, W6) — x/devices/motor/motor.go:269-323 #p2

## Links

- Module: [[actuators]] · [[pid-control]] · [[quadrature-encoder-decoding]] · Kinematics consumers: [[kinematics]]
