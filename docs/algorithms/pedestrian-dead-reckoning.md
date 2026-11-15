---
type: algorithm
status: tested
tags: [easyrobot, navigation, imu, pedometer, tinygo]
workstream:
module: "[[navigation]]"
summary: "steps from |a| + heading of a chosen body axis + track; synthetic-signal tests, no hardware yet"
source: [x/math/nav/pdr/pdr.go]
tests: [x/math/nav/pdr/pdr_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race ./x/math/nav/pdr pass (13 tests incl. a badge-mounted walk with a Mahony heading, 2026-11-12, after review); 18 mutants all fail, plus the round-2 survivors N1–N4 and N8; Detector.Update 0 allocs, 14.6–60.8 ns/op on an Apple M5 (10 cores, go1.27.2 darwin/arm64) at load average 55 — timing depends on load, so it is no figure of merit; tinygo 0.43 tufty2040 build with mpu6050 + Mahony allocates only at construction"
---
# Pedestrian dead reckoning (step detector, heading, track)

**What it does:** counts steps from a body-worn accelerometer, gives the walking direction as the heading of a chosen body axis from an attitude filter's orientation, and advances a 2D position by one step length along it for each step.

**Where:** `x/math/nav/pdr/pdr.go` — `DetectorConfig`, `DefaultDetectorConfig` (:31-46), `Detector`, `NewDetector` (:53-85), `Update` (:88-118), `Steps` (:121), `Heading` (:123-147), `Track`, `Step` (:149-163).

**Inputs → outputs:**
- `Detector.Update(a [3]float32 m/s²) bool` once per sample → true on the sample that completes a step. Any orientation: only |a| is used. NaN/Inf samples are ignored; zero samples before the sensor wakes do not seed the filter.
- `Heading(q [4]float32, axis [3]float32) (rad, ok)`: q is the filter's `[w,x,y,z]` (sensor → world, z up, e.g. `[4]float32(mahony.Output())`). axis is the body axis that points where the wearer walks, in sensor coordinates. It returns that axis's horizontal direction, counter-clockwise from the world's x. `ok` is false within about 6° of vertical.
- `Track.Step(heading)`: x along heading 0, y along +π/2, metres.

**Badge mounting** (decided): screen right = IMU −X, up = +Y, out = −Z; the wearer walks along the screen-out axis, so `Heading(q, [0, 0, −1])`. The yaw of `ahrs.Euler` is the heading of the sensor's x axis and is undefined when that axis is near vertical. With the badge on its side the IMU's x axis is vertical, and Euler yaw stops being a direction. `Heading` works for any mounting.

**Method:** |a| → first-order low-pass (`Cutoff`, default 3 Hz) → minus a running baseline (time constant `Baseline`, default 2 s; it follows gravity and the accelerometer's scale error, starting from the first non-zero sample). A rise above `Threshold` (default 1 m/s²) arms the detector; the fall back through the baseline completes the step; arming is blocked for `MinInterval` (default 0.3 s) after the last step.

**Limits:**
- **Step length:** a constant (it changes with the walker and pace; a Weinberg-style length from the bounce amplitude is a TODO).
- **Heading drift:** without a magnetometer (MPU6050) the heading is relative to the start and drifts with the gyro bias about the vertical. A straight walk curves slowly; with a magnetometer the heading is absolute.
- **Bounce amplitude:** the default threshold needs a peak bounce of about 1.25 m/s² at 1–1.5 steps/s and 1.5 m/s² at 2–2.5 steps/s for a full count (measured sweep). A loosely hung badge may bounce less, which needs a lower `Threshold`.
- **Running:** at 3 steps/s or faster it is counted at about half the rate (`MinInterval`).
- **Jolts:** like any threshold detector, a single jolt that crosses the threshold (e.g. 30 ms at +15 m/s²) counts as one step.
- **Sample timing:** a missed sample stretches the detector's time base; it assumes a fixed rate.

## Completeness review

- Complete: yes for the badge demo's walking path.
- Tested (`pdr_test.go`, synthetic body-worn signals: gravity, bounce, deterministic noise):
  - **Step counting:**
    - `TestDetectorCountsSteps`: 1.2, 2 and 2.5 steps/s for 20 s, count ±1.
    - `TestDetectorIgnoresRestAndVibration`: rest 30 s and 15 Hz vibration give 0.
    - `TestDetectorFollowsAScaleError`: 0.9 g and 1.12 g accelerometers.
    - `TestDetectorAnyOrientation`: gravity along x, −y, z and two diagonals give identical counts.
    - `TestDetectorBaselineTimeConstant`: a +0.5 m/s² level step reaches 63 % in `Baseline` s ±5 %, with no steps.
    - `TestDetectorPrimingAndBadSamples`: zero samples before waking then 30 s of noisy rest give 0 steps; a NaN and a +Inf sample mid-walk leave the count right.
    - `TestDetectorMinInterval`.
    - `TestDetectorHysteresis`: with `MinInterval` 0, a bounce barely above `Threshold` with noise still counts 40±1 in 20 s at 2/s; firing on the fall through `Threshold` instead of the baseline counts 44–57.
    - `TestNewDetectorRefusesBadConfig`: includes ±Inf/NaN and a gap beyond int32.
  - **Heading:**
    - `TestHeading`: the badge's −Z at four turns, in portrait and with the IMU x axis vertical; the vertical axis is refused. The refusal cone, whatever the axis length (10): 5.5° from vertical refused, 6° accepted with heading 0; a zero axis refused.
    - `TestWalkWithMahonyHeading`: a chest-worn badge (IMU +Y up), 10 s settling, 11 steps, a 90° left turn about the IMU's y axis through a 6-axis Mahony, 11 steps. The legs come out ≈ 11 step lengths each and the turn +90° ±10°.
  - **Track and allocation:** `TestTrackStep`, `TestDetectorDoesNotAllocate`.
  - **Run:** `go test -race -count=1 ./x/math/nav/pdr/` pass, 2026-11-12.
- Correct: yes on these signals; each mutant fails a test. Real walking is unverified until the badge's MPU6050 is fitted.

## TODOs

- [ ] Step length from the bounce amplitude (Weinberg) instead of a constant — x/math/nav/pdr/pdr.go:159 #p3
- [ ] Walk with the badge: count against a tally, closed-loop track error, and the bounce amplitude a hung badge gives — x/math/nav/pdr/pdr.go:88 #p2

## Links

- Module: [[navigation]] · Attitude: [[mahony-ahrs]] · Sensor: [[mpu6050-imu-driver]]
