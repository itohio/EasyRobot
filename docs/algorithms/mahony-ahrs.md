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

**Where:** `x/math/filter/ahrs/mahony.go`: `NewMahony` :26, `Update` :75-89, `calculate` (9-DOF) :103-204, `calculateWOMag` (6-DOF) :206-267. The 6-DOF update goes through `update6`: `calculateWOMag` by default (`mahony6_float.go`), fixed point with `-tags fixedmotion` (`mahony6_fixed.go`, `update6` :83). `x/math/filter/ahrs/euler.go`: `Euler` :18.

**Inputs → outputs:** same 3×3 input as [[madgwick-ahrs]] (rows accel, gyro rad/s, mag); output quaternion `[w,x,y,z]`, the rotation from the sensor frame to the world frame (z up; x along the horizontal magnetic field with a magnetometer). `Euler(q)` gives roll, pitch, yaw (Z-Y-X; yaw counter-clockwise seen from above, relative to the start without a magnetometer). Within about 0.03° of pitch ±90° (|sin pitch| ≥ 0.99999988) only roll ∓ yaw is defined: `Euler` then returns roll 0 and the whole rotation about the vertical as yaw, so the angles still describe `q`. `vec.Quaternion`'s own `Roll/Pitch/Yaw` read w from index 3 and do not apply to this output. `Update(dt, &m)` with a `*mat.Matrix3x3` allocates nothing.

## Completeness review

- Complete: yes (IMU + MARG paths, integral term with anti-windup when `GainI==0`). Integral accumulates `e` without `dt` (:173-175, :236-238) — same as the x-io reference, but makes `GainI` sample-rate dependent.
- Tested: `TestMahonyCalculateWith/WithoutMagnetometerMatchesReference` compare one step with the test's own x-io reference step (its 6-DOF error was `v × a` like the bug; now `a × v`). `convergence_test.go`:
  - `TestMahonyConvergesToAStaticTilt`: both paths, 40 s at 100 Hz from identity; gravity estimate within 0.01 of the measured direction; 9-DOF roll/pitch/yaw within 0.02 rad.
  - `TestMahonyCancelsAGyroBias`: 10 min at a tilted pose, bias (0.12, −0.16, 0.1) rad/s, Kp 2, Ki 0.005. The attitude stays within 0.002, and `GainI·eInt` equals −bias within 1e-3: the part across gravity for 6-DOF, all of it for 9-DOF. Without the integral the tilt would be |bias|/Kp ≈ 0.1 rad.
  - `TestMahonyAccelEdgeCases`: a zero accel sample integrates the gyro alone without NaN; |a| = 1 and 9.81 give the same attitude.
  - `TestMahonyIntegratesTheYawRate`: +90°/s for 1 s → +90° ± 2°.
  - `TestEulerOfKnownOrientations`: incl. both gimbal-lock clamps.
  - `TestEulerAtThePitchPoles`: 20,000 poses within 0.1° of ±90° (a fifth exactly at the pole), random roll and yaw; the returned angles rebuilt into a quaternion are within 0.1° of `q` (worst 0.067°).
  - `TestMahonyUpdateDoesNotAllocate`.
  - Run: `go test -race -count=1 ./x/math/filter/ahrs/` pass, 2026-11-12.
- Correct: yes for 6-DOF and 9-DOF. The 6-DOF error was `e = v × a` (estimated × measured; the 9-DOF path and the x-io reference use `a × v`), so it settled at the mirror tilt (red run: gravity estimate −(0.34, 0.47, 0.81) for a measured +(0.34, 0.47, 0.81)); fixed to `a × v` (`mahony.go:230-234`).
- **Fixed point for FPU-less cores (`-tags fixedmotion`, KERNELS-1).** Opt-in, never selected automatically; float32 stays the default and the reference. The 6-DOF update runs in integers: accelerometer in Q22 m/s², gyro in Q24 rad/s, unit vectors, attitude and integral in Q30, gains in Q28 (fpv-gate: in Q16, 0.005 was 0.1 % off, and a large integral carried that to 2.5e-4 rad), int64 products with round-to-nearest shifts (the gain products through 128 bits), the accelerometer normalised by an integer square root, the quaternion by three Newton steps of 1/√s. Inputs are read from the float's bits with symmetric rounding: a truncating conversion's bias moved the escape from the upside-down start by 1e-4 rad. The float state (`q`, `eInt`) stays the filter's state; it is taken up again whenever it differs from what the fixed path last wrote (a new filter, `Reset`, a state set directly). The 9-DOF path stays float32.
  - `TestFixedMahonyFollowsFloat` (tag `fixedmotion`) checks the fixed path within 1e-4 rad of the float path at every sample, over 30 scenarios: tilts 0–180° for 60 s, spins to ±6 rad/s, hand motion, zero accelerometer samples, the badge demo's gains Kp 2 and 10, and integral gains, Ki 0.005 from a large initial error among them. Worst: 5.1e-5 rad at the exact 180° start with Ki 0.005 (float32 itself is 1.2e-5 from a float64 reference at that start without Ki); about 2e-6 elsewhere. Also `TestFixedMahonyResetForgets` and `TestFixedMahonyDoesNotAllocate`.
  - fpv-gate's independent probe (`fpvgate_mahony_fixed_test.go`: MPU6050 noise, offsets and gyro bias, Ki switched on mid-run, 2000 °/s spins, ±15 g shaking, near free fall, a 20 Hz step, a mid-run Reset, 30 minutes of noisy rest with Ki) passes. Worst is 7.4e-5 rad in the 30-minute rest, which is mostly float32's own yaw drift: it is 6.3e-5 from float64 there, the fixed path 1.1e-5. The tilt part is 7.7e-7.
  - All tests pass with and without the tag. fpv-gate's strapdown harness (KP 2, KI 0, so 6-DOF) prints the same 142 lines in both builds.

## TODOs

- [x] Flip 6-DOF error to `a × v` (match :167-171) — mahony.go:230-234 #p1 #ER-019
- [x] Convergence test from a tilted static pose for both paths — ahrs_test.go #p1 #ER-019
- [ ] Scale integral by dt (document gain units) — mahony.go:173,236 #p3

## Links

- Module: [[filters]]
