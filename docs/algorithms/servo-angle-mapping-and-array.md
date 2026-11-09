---
type: algorithm
status: partial
tags: [easyrobot, devices, actuators]
workstream: [W3]
module: "[[actuators]]"
summary: "`Stop` deadlocks, TinyGo-only"
source: [x/devices/servo/types.go, x/devices/servo/array.go, x/devices/servo/array_tinygo.go, x/devices/servo/array_stub.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/devices/actprobe (2026-11-09): AngleToMicroseconds 0/90/180/200° → 1000/1500/2000/2000 µs (1000–2000 µs config); NewServoArray on !tinygo → 'not implemented for this platform'; ServoArray.Stop → deadlock"
---
# Servo angle mapping and servo array

**What it does:** `servo.Motor` maps an angle in [0, MaxAngle] linearly to a pulse width in [MinUs, MaxUs] (and back); `ServoArray` owns one `devices.PWM` channel per servo at 50 Hz and sets all angles at once.

**Where:** `x/devices/servo/types.go:4-100` (`Motor`, `NewMotorConfig`, `WithPin`, `WithMicroseconds`, `AngleToMicroseconds`, `MicrosecondsToAngle`), `array.go:12-87` (`ServoArray`, `Configure`, `Set`, `Stop`), `array_tinygo.go:20-60` (`newServoArray`), `array_stub.go:12-16` (returns error).

**Inputs → outputs:** `[]float32` angles (deg) → `PWM.SetMicroseconds(uint32)` per channel.

## Completeness review

- Complete: **no.** Only TinyGo can construct an array (`array_stub.go:14-15` "PWM servo array not implemented for this platform", Linux `LinuxPWMDevice` exists in [[device-io]] but is not used); no per-servo offset/trim, direction inversion or non-linear calibration — exactly the "helpers to configure joints properly" the owner asks for in W3; no speed/acceleration limiting (done separately by `vaj` in `drivers/manipulator`).
- Tested: none; probe checks the linear map and clamping (values above).
- Correct: **no.** `Stop` takes `a.mu` then calls `Set`, which takes `a.mu` again (`array.go:~76-85` → `:~52`): **self-deadlock** (probe: no return within 500 ms). `Configure` copies motors but does not re-set channels; `uint32(us)` truncates; `NewMotorConfig` recomputes `Default` only when `Default == 90` (`types.go:29-36`). `array_tinygo.go:15-17` returns `machine.Pin` as `devices.Pin`, which does not type-check because `machine.Pin.SetInterrupt` takes `func(machine.Pin)` (see [[device-io]]) [INFERENCE: TinyGo not run].

## TODOs

- [ ] Fix `Stop` deadlock (internal unlocked `set`) — x/devices/servo/array.go:76-85 #p1 #ER-025
- [ ] Add joint calibration (offset, sign, min/max angle, µs per degree) as a reusable `servo.Joint` used by IK output — x/devices/servo/types.go #p2
- [ ] Allow non-TinyGo construction over any `devices.PWMDevice` (Linux sysfs, PCA9685) — x/devices/servo/array_stub.go:12-16 #p2
- [ ] Unit tests with a fake PWMDevice — x/devices/servo #p2

## Links

- Module: [[actuators]] · [[pca9685-pwm-driver]] · [[xiao-samd21-tcc-pwm]] · IK side: [[kinematics]], [[motion-and-gait]] · [[W3-legged-ik]]
