---
type: algorithm
status: tested
tags: [easyrobot, filter, ahrs, imu]
workstream: [W3, W7]
module: "[[filters]]"
summary: "6-DOF and 9-DOF converge to a static tilt; Euler(q) gives roll/pitch/yaw of the [w,x,y,z] output"
source: [x/math/filter/ahrs/mahony.go, x/math/filter/ahrs/euler.go]
tests: [x/math/filter/ahrs/ahrs_test.go, x/math/filter/ahrs/convergence_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race ./x/math/filter/ahrs 16/16 pass (2026-11-12): 6- and 9-DOF converge to roll 30°/pitch −20° (9-DOF also yaw 40°); a constant gyro bias is learned by the integral (GainI·eInt = −bias within 1e-3, attitude within 0.002; dropping the integral in either path fails); yaw rate integrates to +90°; zero accel and accel scale handled; Update allocates 0; red before the fix: 6-DOF settled at the mirror tilt (ER-SIMALGO-logs/mahony-fail-before.txt); mutants in ER-SIMALGO-logs/mahony-mutants-r2.txt all fail; review S2: Euler near pitch ±90° rebuilt within 0.067° of q over 20,000 poses (red before: 31° off; ER-SIMALGO-logs/s2-*.txt, 6 mutants fail)"
---
# Mahony AHRS (complementary PI filter)

**What it does:** Quaternion attitude filter: error = cross product of measured vs estimated gravity (and magnetic) direction, fed back to the gyro through PI gains (`GainP`, `GainI`).

**Where:** `x/math/filter/ahrs/mahony.go`: `NewMahony` :24, `Update` :72-86, `calculate` (9-DOF) :100-201, `calculateWOMag` (6-DOF) :203-264. `x/math/filter/ahrs/euler.go`: `Euler` :18.

**Inputs → outputs:** same 3×3 input as [[madgwick-ahrs]] (rows accel, gyro rad/s, mag); output quaternion `[w,x,y,z]`, the rotation from the sensor frame to the world frame (z up; x along the horizontal magnetic field with a magnetometer). `Euler(q)` gives roll, pitch, yaw (Z-Y-X; yaw counter-clockwise seen from above, relative to the start without a magnetometer). Within about 0.03° of pitch ±90° (|sin pitch| ≥ 0.99999988) only roll ∓ yaw is defined: `Euler` then returns roll 0 and the whole rotation about the vertical as yaw, so the angles still describe `q`. `vec.Quaternion`'s own `Roll/Pitch/Yaw` read w from index 3 and do not apply to this output. `Update(dt, &m)` with a `*mat.Matrix3x3` allocates nothing.

## Completeness review

- Complete: yes (IMU + MARG paths, integral term with anti-windup when `GainI==0`). Integral accumulates `e` without `dt` (:170-172, :233-235) — same as the x-io reference, but makes `GainI` sample-rate dependent.
- Tested: `TestMahonyCalculateWith/WithoutMagnetometerMatchesReference` compare one step with the test's own x-io reference step (its 6-DOF error was `v × a` like the bug; now `a × v`). `convergence_test.go`:
  - `TestMahonyConvergesToAStaticTilt`: both paths, 40 s at 100 Hz from identity; gravity estimate within 0.01 of the measured direction; 9-DOF roll/pitch/yaw within 0.02 rad.
  - `TestMahonyCancelsAGyroBias`: 10 min at a tilted pose, bias (0.12, −0.16, 0.1) rad/s, Kp 2, Ki 0.005. The attitude stays within 0.002, and `GainI·eInt` equals −bias within 1e-3: the part across gravity for 6-DOF, all of it for 9-DOF. Without the integral the tilt would be |bias|/Kp ≈ 0.1 rad.
  - `TestMahonyAccelEdgeCases`: a zero accel sample integrates the gyro alone without NaN; |a| = 1 and 9.81 give the same attitude.
  - `TestMahonyIntegratesTheYawRate`: +90°/s for 1 s → +90° ± 2°.
  - `TestEulerOfKnownOrientations`: incl. both gimbal-lock clamps.
  - `TestEulerAtThePitchPoles`: 20,000 poses within 0.1° of ±90° (a fifth exactly at the pole), random roll and yaw; the returned angles rebuilt into a quaternion are within 0.1° of `q` (worst 0.067°).
  - `TestMahonyUpdateDoesNotAllocate`.
  - Run: `go test -race -count=1 ./x/math/filter/ahrs/` pass, 2026-11-12.
- Correct: yes for 6-DOF and 9-DOF. The 6-DOF error was `e = v × a` (estimated × measured; the 9-DOF path and the x-io reference use `a × v`), so it settled at the mirror tilt (red run: gravity estimate −(0.34, 0.47, 0.81) for a measured +(0.34, 0.47, 0.81)); fixed to `a × v` (`mahony.go:224-231`).

## TODOs

- [x] Flip 6-DOF error to `a × v` (match :164-168) — mahony.go:225-229 #p1 #ER-019
- [x] Convergence test from a tilted static pose for both paths — ahrs_test.go #p1 #ER-019
- [ ] Scale integral by dt (document gain units) — mahony.go:170,233 #p3

## Links

- Module: [[filters]]
