---
type: module
group: "Control and planning"
status: tested
tags: [easyrobot, navigation, dead-reckoning]
workstream:
packages: [x/math/nav/pdr, x/math/nav/strapdown, x/math/nav/calib, x/math/nav/eskf]
test_result: pass
---
# Navigation (x/math/nav)

**Purpose:** dead reckoning from body-worn or vehicle sensors: position from motion events and a heading, or from the accelerometer and an attitude. First pieces are for the Tufty badge:

- its walking path: MPU6050 → [[mahony-ahrs]] → steps → track;
- its 3D motion path: MPU6050 → Mahony → strapdown with zero-velocity updates;
- its startup calibration at rest: MPU6050 → calib → gyro bias for Mahony and strapdown, and the page's live statistics and forecast;
- its motion from one joint estimate: MPU6050 → [[error-state-kalman-filter]] (attitude, velocity, position and both biases with their covariance, in place of Mahony + strapdown); no application runs it yet.

**Packages:** `x/math/nav/pdr` (pdr.go 163, pdr_test.go 389), `x/math/nav/strapdown` (strapdown.go 990, strapdown_test.go 1531), `x/math/nav/calib` (calib.go 373, calib_test.go 500) and `x/math/nav/eskf` (eskf.go 802, eskf_test.go 592); no build tags.

## Public API

- `pdr.NewDetector(pdr.DefaultDetectorConfig(rateHz)) (Detector, error)`; `Detector.Update(a [3]float32) bool`; `Steps()`.
- `pdr.Heading(q [4]float32, axis [3]float32) (heading, ok)`: the horizontal direction of a body axis from an attitude quaternion; any mounting.
- `pdr.Track{StepLength}`; `Step(heading)`; fields `X, Y, Steps`.
- Composition (badge: walking axis = IMU −Z): `mpu6050.Read` → `ahrs.MahonyAHRS.Update(dt, &in)` → `pdr.Heading([4]float32(mahony.Output()), [3]float32{0, 0, -1})` → `Track.Step(heading)` when `Detector.Update(accel)` is true. The detector does not need the filter; any heading source works.
- `strapdown.New(strapdown.DefaultConfig()) (Nav, error)`; `Nav.Update(dt, q, accel, gyro)` with `q = [4]float32(mahony.Output())` and the MPU6050's m/s² and rad/s; `Reset()` returns to the origin. After every sample the `Nav` is the badge's whole motion state, one value for every consumer (path, paddle, water, fire, boids): `Q` (the attitude used: the filter's until the first rest, then the Nav's own, gyro-carried and levelled at rest), `P`, `V` (m, m/s, world, z up), `PB`, `VB` (the same in the badge frame: each sample's displacement rotated into that sample's badge frame and summed, so a tilt in place adds nothing; Pong's paddle follows `PB[1]`), `A` (world linear acceleration, gravity and bias removed, every sample), `F` (specific force, badge frame: gravity plus the badge's own acceleration; water and fire), `Still`. Gentle and slow strokes are tracked in every direction. Start at rest: the first rest teaches the biases, and nothing moves before it. Tilting in place does not move the path (60 tilt cases with an MPU6050's noise and offsets up to past the datasheet's: 0.0 mm). A turn's offset error is taken out of the motion at its rest, and a stroke made while turning that never shows itself jumps into place at the rest after it. Every stroke of 5 cm or more, and every 3 cm stroke of 1 s or less, registers. Known failures (ER-067, 144 runs; r11's targets): 3 cm over 1.5 s 5 times unregistered and 3 times read the wrong way (−109 %, −736 %, −781 %); 3 cm in 2 s with no turn ended at −39 % once; slow 2 s strokes jump back at their late rest (10 cm reads 48 % in one of three draws). These were measured as mostly design, not the sensor. Mahony's integral term no longer matters after the first rest.
- `strapdown.Nav.Seed(gyroBias)` before the first rest ends: the Nav keeps a startup calibration's gyro bias instead of learning one from its shorter first rest. Not used on the badge until strapdown r11: seeding was found to widen strokes made right after a turn (22 of 36 within 10 % against 26).
- `calib.New(calib.DefaultConfig()) (Calibrator, error)`; `Calibrator.Update(dt, accel, gyro)`; `State()` (waiting, measuring, moved, done), `Progress()` (0–1), `Accel()`, `Gyro()` (`Stats{Mean, Spread}`: per axis, mean and standard deviation over the last 0.5 s), `Thresholds()`, `AccelRest()` (±G on the axis that points up), `Up()`, `Result() (Result, bool)` (`GyroBias`, `AccelOffset`, `AccelNoise`, `GyroNoise`, `Forecast{HeadingDrift, HeadingNoise, HeadingOK, PosDrift, PosNoise, PosOK}` in °/min and cm/min), `Reset()`. Composition (badge's first page): feed every sample until `Result` is ok and show the page and the forecast; until strapdown r11, Mahony and the Nav take the raw samples and the Nav is not seeded. `AccelOffset` is for display: one pose cannot tell an offset from the table's tilt.

## Algorithms

<!-- vault:algorithms h=9851646e57be -->

Counts: 4 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[error-state-kalman-filter]] | tested | pass | error-state Kalman filter for a body's motion from a 6-axis IMU: attitude, velocity, position, gyro and accelerometer bias estimated jointly with their covariance (15 states, float32, sequential scalar updates); tilt from gravity gated on its innovation, zero velocity when the body isn't translating (rotating in place too), zero rate at rest gated so a slow turn stays out of the bias; a multi-pose calibration's fit as the prior; no alloc per sample |
| [[pedestrian-dead-reckoning]] | tested | pass | steps from \|a\| + heading of a chosen body axis + track; synthetic-signal tests, no hardware yet |
| [[startup-calibration]] | tested | pass | startup calibration of an IMU at rest: finds the face that is down, waits for stillness, averages for 3 s, restarts when moved; gyro bias, readings at rest, noise, live per-axis statistics and a forecast of heading and position drift checked against simulation; seeds strapdown's gyro bias |
| [[strapdown-inertial-navigation]] | tested | pass | strapdown translation from an IMU: the Nav carries its own attitude on the gyro after the first rest (levelled at rest, gyro bias learned at rest), rotates the accelerometer's force into the world, removes gravity and integrates to velocity and position; zero-velocity updates at rest in 3D; a rotation in place moves nothing while it lasts and its rest takes up any translation it carried; every turned motion's rest takes the turned offsets' error out of the whole motion; bias learning in the world and the sensor frame; the Nav is the badge's whole motion state (Q, P, V, PB, VB, A, F); float32, no alloc per sample |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, float32, `errors`; value types, no allocation per sample. `pdr`, `strapdown`, `calib` and `eskf` use `chewxy/math32`. TinyGo 0.43 `-target=tufty2040`:

- with mpu6050 + Mahony + pdr: allocations only at construction (2026-11-12);
- with mpu6050 + Mahony + strapdown: one allocation in `strapdown`, the 736-byte `Nav` from `New`; none per sample; 94,296 B flash (2026-11-13);
- with mpu6050 + Mahony + calib + seeded strapdown: no allocation in `calib` (the Calibrator is 212 bytes, a stack value); strapdown's `Nav` as above; 97,512 B flash (2026-11-13);
- eskf (`kernelbench`): one allocation, the 2864-byte `Filter` from `New`; none per sample; 121,756 B flash (2026-11-15).

## Tests and evidence

`go test -race -count=1 ./x/math/nav/...` pass, 2026-11-13; details in [[pedestrian-dead-reckoning]], [[strapdown-inertial-navigation]], [[startup-calibration]] and [[error-state-kalman-filter]].

## TODOs

None at module level.
