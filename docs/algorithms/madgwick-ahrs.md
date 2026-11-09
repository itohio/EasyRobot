---
type: algorithm
status: tested
tags: [easyrobot, filter, ahrs, imu]
workstream: [W3, W7]
module: "[[filters]]"
summary: "Matches reference equations, converges"
source: [x/math/filter/ahrs/madgwick.go, x/math/filter/ahrs/options.go]
tests: [x/math/filter/ahrs/ahrs_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "ahrs 9/9 pass; equations match Madgwick reference code; probe ahrsprobe converges to roll 30° (6/9-DOF)"
---
# Madgwick AHRS (gradient-descent orientation filter)

**What it does:** Fuses gyro + accelerometer (+ magnetometer) into a unit quaternion `[w,x,y,z]` with Madgwick's normalised gradient-descent correction (gain β = `GainP`).

**Where:** `x/math/filter/ahrs/madgwick.go`: `NewMadgwick` :23, `Update` :69-83 (input `mat.Matrix3x3` rows accel/gyro/mag), `calculate` (9-DOF) :97-204, `calculateWOMag` (6-DOF) :206-279; options `options.go:3-55` (`WithKP`, `WithMagnetometer`…).

**Inputs → outputs:** accel any unit (normalised), gyro rad/s, mag any unit; `timestep` s → quaternion via `Output()`. Defaults β=0.1. Zero allocations per update (fixed arrays).

## Completeness review

- Complete: yes (IMU and MARG variants). Missing vs reference: when the magnetometer vector is all-zero the 9-DOF path does not fall back to IMU update (madgwick.go:137-146); when accel is all-zero the correction is still applied with an un-normalised zero vector (:131) instead of being skipped. `GainI` unused (no gyro-bias estimation).
- Tested: `TestMadgwickCalculateWith/WithoutMagnetometerMatchesReference` compare one step against stored quaternions (golden values; origin of the reference not stated in the test), reset, input-preservation, panic-without-sensors. Run: go test -count=1 -json ./x/math/filter/ahrs (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok, 9 pass.
- Correct: yes. Line-by-line comparison of `s` and `qDot` terms with Madgwick's published C implementation (`MadgwickAHRSupdate`/`MadgwickAHRSupdateIMU`, index shift q0→q1) shows identical expressions (:153-188, :237-263). Convergence probe (`tmp/easyrobot/filters/ahrsprobe/main.go`: static roll 30°, zero gyro, 4000×10 ms, β=0.5): 6-DOF roll 29.62°, 9-DOF roll 30.00°.

## TODOs

- [ ] Skip correction on zero accel; fall back to IMU update on zero mag — madgwick.go:131,137 #p2
- [ ] Convergence/regression test with a known static orientation (probe above) — ahrs_test.go #p2
- [ ] Optional gyro-bias integral (use `GainI`) or remove `GainI` from Madgwick options #p3

## Links

- Module: [[filters]]
- Sibling: [[mahony-ahrs]]
