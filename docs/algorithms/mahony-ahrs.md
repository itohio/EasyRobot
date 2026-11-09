---
type: algorithm
status: partial
tags: [easyrobot, filter, ahrs, imu]
workstream: [W3, W7]
module: "[[filters]]"
summary: "6-DOF error sign flipped (converges upside down)"
source: [x/math/filter/ahrs/mahony.go]
tests: [x/math/filter/ahrs/ahrs_test.go]
test_result: pass
complete: yes
correct: no
evidence: "ahrs 9/9 pass; probe ahrsprobe: 6-DOF converges to roll -150° (upside down) instead of 30°; 9-DOF converges to 30°"
---
# Mahony AHRS (complementary PI filter)

**What it does:** Quaternion attitude filter: error = cross product of measured vs estimated gravity (and magnetic) direction, fed back to the gyro through PI gains (`GainP`, `GainI`).

**Where:** `x/math/filter/ahrs/mahony.go`: `NewMahony` :24, `Update` :72-86, `calculate` (9-DOF) :100-201, `calculateWOMag` (6-DOF) :203-262.

**Inputs → outputs:** same 3×3 input as [[madgwick-ahrs]] (rows accel, gyro rad/s, mag); output quaternion `[w,x,y,z]`. No allocations.

## Completeness review

- Complete: yes (IMU + MARG paths, integral term with anti-windup when `GainI==0`). Integral accumulates `e` without `dt` (:170-172, :231-233) — same as the x-io reference, but makes `GainI` sample-rate dependent.
- Tested: `TestMahonyCalculateWith/WithoutMagnetometerMatchesReference` compare one step to stored golden quaternions/integrals; they would pass with either error sign because the golden values were evidently produced by this code. Run: go test -count=1 -json ./x/math/filter/ahrs (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok.
- Correct: **no for 6-DOF**, yes for 9-DOF.
  - `mahony.go:225-229` computes `e = v × a` (estimated × measured) while the 9-DOF path (:164-168) and the x-io reference use `e = a × v`. The 6-DOF feedback therefore has the wrong sign. Probe (`tmp/easyrobot/filters/ahrsprobe/main.go`, static roll 30°, Kp=2, Ki=0, 40 s): 6-DOF ends at roll **−150.00°** (q≈[0.259,−0.966,0,0], gravity flipped); 9-DOF ends at 30.00°.

## TODOs

- [ ] Flip 6-DOF error to `a × v` (match :164-168) — mahony.go:225-229 #p1 #ER-019
- [ ] Convergence test from a tilted static pose for both paths — ahrs_test.go #p1 #ER-019
- [ ] Scale integral by dt (document gain units) — mahony.go:170,231 #p3

## Links

- Module: [[filters]]
