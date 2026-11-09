---
type: algorithm
status: partial
tags: [easyrobot, devices, lidar, control]
workstream: W8
module: "[[lidar]]"
summary: "Control sign inverted, never wired"
source: [x/devices/lidar/ld06/device.go, x/devices/lidar/xwpftb/device.go, x/math/control/pid/pid1d.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "read only; no test exercises a motor (all tests pass motor=nil). Sign derivation below; never wired: drivers/lidar/main_rp2040.go:64-68 leaves motor nil"
---
# LiDAR motor point-count PID and auto-calibration

**What it does:** Optional closed loop for lidars whose motor is driven by host PWM: keeps the number of points per rotation at a target by adjusting duty. If `targetPoints == 0`, first sweeps duty 0→`minDuty`(0.1)→`maxDuty`(0.8) and sets the target to the midpoint of the extreme point counts seen.

**Where:** identical copies in `x/devices/lidar/ld06/device.go:377-505` and `x/devices/lidar/xwpftb/device.go:~483-611` (`initPID`, `updatePID`, `calibrationLoop`, `updateCalibration`, `controlLoop`); called from `emitScan` (`ld06/device.go:363-369`). PID: `pid.New1D(0.001, 0.0005, 0.0001, 0, 1)` → [[pid-control]] (`x/math/control/pid/pid1d.go:30-39`, error `E = Target − Input`).

**Inputs → outputs:** points per completed scan, rotation period → `devices.PWM.Set(duty ∈ [0,1])`.

## Completeness review

- Complete: **no.** No consumer passes a motor: host app passes `nil` (`drivers/lidar/main.go:151,161`), RP2040 app prints "PWM support for RP2040 not yet implemented" and leaves `motor` nil (`drivers/lidar/main_rp2040.go:62-68`). LD06 normally regulates its own speed (its PWM pin is optional) [INFERENCE: datasheet knowledge].
- Tested: none.
- Correct: **no** (derivation). Points per rotation = sample rate / rotation rate, so raising duty **lowers** the count. With `E = target − points`, a too-low count (motor too fast) gives `E > 0` and the PID **raises** duty: positive feedback, runs to duty 1 (`ld06/device.go:387-405`). Gains need a negative sign (or input = −points). Also: `controlLoop` re-applies the last scan's count every 200 ms with `dt = time since last rotation` (`:480-505`), integrating the same error repeatedly; `updateCalibration` names are inverted (`minPoints` holds the max, `:463-476`); calibration and control loops write `currentDuty` from different goroutines without the mutex.

## TODOs

- [ ] Fix control sign (duty ↑ ⇒ points ↓) and add a unit test with a simulated motor — x/devices/lidar/ld06/device.go:377-405 #p1 #ER-024
- [ ] Deduplicate the motor loop into one helper shared by ld06/xwpftb (e.g. `x/devices/lidar/motor.go`) — x/devices/lidar/ld06/device.go:377-505; x/devices/lidar/xwpftb/device.go:483-611 #p3
- [ ] Wire RP2040 PWM (machine.PWM) or delete the motor path from the app — drivers/lidar/main_rp2040.go:62-68 #p2

## Links

- Module: [[lidar]] · [[pid-control]] · [[W8-lidar]]
