---
type: algorithm
status: unverified
tags: [easyrobot, devices, actuators, tinygo]
workstream: [W3]
module: "[[actuators]]"
summary: "Suspect period units"
source: [x/devices/xiao/xiao_pwm.go, x/devices/xiao/xiao_pwm_stub.go, x/devices/xiao/README.md]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "build tag sam && xiao (TinyGo only); not compiled (no TinyGo). Host build of stub ok (baseline)."
---
# XIAO (SAMD21) TCC PWM device

**What it does:** Implements `devices.PWMDevice` for the Seeed XIAO SAMD21 by mapping pins to TCC0/TCC1 channels, configuring the TCC period for a frequency, and setting compare values for duty or pulse width.

**Where:** `x/devices/xiao/xiao_pwm.go:24-100` (pin→TCC map, "Verified" comments on D8/D9/D10), `:102-152` (`Configure`, `configureTCC`), `:154-191` (`SetFrequency`), `:193-245` (`Channel`), `:247-289` (`Set`, `SetMicroseconds`, `Stop`); stub for other targets returns errors.

**Inputs → outputs:** frequency Hz, duty 0–1 or µs → `machine.TCC.Set(channel, value)`.

## Completeness review

- Complete: no — TCC2 pins commented out (`:95-96`); TCC0 channel ≥ 4 on WO[4..5] maps to CC[0..1] on SAMD21 (TCC0 has 4 CC registers) [INFERENCE], so D6/D7 may alias D9/D3.
- Tested: none (TinyGo only).
- Correct: unknown. Suspect: `machine.PWMConfig.Period` in TinyGo is the period in **nanoseconds** [INFERENCE: TinyGo API from memory], but `configureTCC` passes TCC counts `48 MHz/64/f` (`:139-143`, 15000 for 50 Hz → 15 µs = 66.7 kHz). Duty values use the same self-computed `period` (`:253`) instead of `tcc.Top()`. If TinyGo honours ns, servo pulses are wrong; the "Verified" comments suggest the owner saw servos move, so needs a bench check. `Channel` requires `pin.(machine.Pin)` type assertion (`:198`) — but `machine.Pin` cannot be a `devices.Pin` (see [[device-io]]).

## TODOs

- [ ] Pass `Period: 1e9/frequency` (ns) and use `tcc.Top()` for duty scaling; bench-check 50 Hz/1500 µs on a scope — x/devices/xiao/xiao_pwm.go:130-152,247-258 #p1 #ER-025
- [ ] Verify TCC0 WO[4..7] → CC mapping for D6/D7 — x/devices/xiao/xiao_pwm.go:80-81 #p2

## Links

- Module: [[actuators]] · [[servo-angle-mapping-and-array]] · App: `drivers/manipulator` in [[apps]]
