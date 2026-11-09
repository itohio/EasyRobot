---
type: algorithm
status: tested
tags: [easyrobot, control, pid]
workstream: [W3, W8]
module: "[[pid-control]]"
summary: "Positional PID with derivative on measurement and integral clamping: `I += Ki·e·dt` clamped to…"
source: [x/math/control/pid/pid.go, x/math/control/pid/pid1d.go]
tests: [x/math/control/pid/pid_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/math/control/pid → ok, 8 tests (2026-11-09): term values, componentwise, integral clamp, reset"
---
# PID controller (scalar and vector)

**What it does:** Positional PID with derivative on measurement and integral clamping: `I += Ki·e·dt` clamped to `[min,max]`; `u = clamp(Kp·e + I − Kd·(x−x_prev)/dt)`.

**Where:** `pid/pid1d.go:7-38` (`PID1D`, `New1D` :14, `Reset` :24, `Update` :30); `pid/pid.go:14-48` (`PID` over `vec.Vector`, element-wise).

**Inputs → outputs:** set `Input`, `Target`, call `Update(dt)` → `Output`. Gains public fields; limits private.

## Completeness review

- Complete: yes.
- Tested: `TestNewPanicsOnLengthMismatch`, `TestResetClearsIntegralAndSyncsLastInput`, `TestUpdateComputesPIDTerms`, `TestUpdateIsComponentwise`, `TestIntegralClampingPreventsWindup`, `TestPID1DReset`, `TestPID1DUpdate`, `TestPID1DIntegralClamping`. Pass (2026-11-09).
- Correct: yes (values asserted). Caveats: without `Reset()` the first `Update` sees `lastInput = 0` → derivative kick; the vector `PID.Update` allocates four clones per call (`pid.go:41-45`) — not for MCU loops; `New` panics with `panic(-1)` on length mismatch (`pid.go:17`); `dt = 0` divides by zero.

## TODOs

- [ ] Allocation-free vector Update (scratch buffers in struct) — pid/pid.go:41-45 #p3
- [ ] Guard dt ≤ 0; return error instead of `panic(-1)` — pid/pid.go:17; pid1d.go:35 #p3

## Links

- Module: [[pid-control]] · users: [[path-tracking-pid-controller]], lidar motor speed loop ([[lidar]]), motor speed loop ([[actuators]])
